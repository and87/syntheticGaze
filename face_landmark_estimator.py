from typing import List, Tuple
import copy
import logging
import os

import cv2
import mediapipe as mp
import numpy as np

from ultralight_facedetector import findHumans_ultralight, get_ultralightWiderFaceRect
from omegaconf import DictConfig
from face_model import Face

logger = logging.getLogger(__name__)


def _resolve_face_mesh_constructor():
    # Some mediapipe builds expose FaceMesh at mp.solutions, others only under
    # mediapipe.python.solutions.
    try:
        return mp.solutions.face_mesh.FaceMesh
    except Exception:
        pass

    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh

        return mp_face_mesh.FaceMesh
    except Exception:
        return None


class LandmarkEstimator:
    def __init__(self, config: DictConfig):
        self.padding = float(config.face_detector.padding)  # padding for improving face detection of cropped images
        self.ultralight = bool(config.face_detector.ultralight)
        self.max_num_faces = int(config.face_detector.mediapipe_max_num_faces)
        self.static_image_mode = bool(config.face_detector.mediapipe_static_image_mode)
        self.min_det_conf = float(config.face_detector.mediapipe_min_det_conf)
        self.min_track_conf = float(config.face_detector.mediapipe_min_track_conf)

        self.backend = str(getattr(config.face_detector, "mediapipe_backend", "face_mesh")).strip().lower()
        self.delegate = str(getattr(config.face_detector, "mediapipe_delegate", "cpu")).strip().lower()
        self.task_model_path = str(getattr(config.face_detector, "mediapipe_task_model", "")).strip()

        self.detector = None
        self._using_tasks = False

        if self.backend == "tasks":
            self._init_tasks_with_fallback()
        else:
            self._init_face_mesh_detector()

    def _init_face_mesh_detector(self):
        face_mesh_ctor = _resolve_face_mesh_constructor()
        if face_mesh_ctor is None:
            raise RuntimeError(
                "FaceMesh API not available in current mediapipe installation. "
                "Use --mediapipe-backend tasks with a valid --mediapipe-task-model, "
                "or install a mediapipe build that includes FaceMesh."
            )

        self.detector = face_mesh_ctor(
            max_num_faces=self.max_num_faces,
            static_image_mode=self.static_image_mode,
            refine_landmarks=True,
            min_detection_confidence=self.min_det_conf,
            min_tracking_confidence=self.min_track_conf,
        )
        self._using_tasks = False
        logger.info("Landmark backend: mediapipe FaceMesh.")

    def _create_tasks_detector(self, delegate_name: str):
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
        except Exception as exc:
            raise RuntimeError(f"MediaPipe Tasks import failed: {exc}") from exc

        if not self.task_model_path:
            raise RuntimeError(
                "mediapipe_task_model is empty. Provide --mediapipe-task-model to use Tasks backend."
            )
        if not os.path.exists(self.task_model_path):
            raise RuntimeError(f"Task model not found: {self.task_model_path}")

        delegate = (
            mp_tasks.BaseOptions.Delegate.GPU
            if delegate_name == "gpu"
            else mp_tasks.BaseOptions.Delegate.CPU
        )
        base_options = mp_tasks.BaseOptions(
            model_asset_path=self.task_model_path,
            delegate=delegate,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_det_conf,
            min_face_presence_confidence=self.min_det_conf,
            min_tracking_confidence=self.min_track_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)

    def _init_tasks_with_fallback(self):
        preferred = "gpu" if self.delegate == "gpu" else "cpu"

        try:
            self.detector = self._create_tasks_detector(preferred)
            self._using_tasks = True
            logger.info(
                f"Landmark backend: mediapipe Tasks FaceLandmarker ({preferred.upper()})."
            )
            return
        except Exception as exc:
            logger.warning(f"Could not initialize Tasks backend ({preferred.upper()}): {exc}")

        if preferred == "gpu":
            try:
                self.detector = self._create_tasks_detector("cpu")
                self._using_tasks = True
                logger.info("Landmark backend: mediapipe Tasks FaceLandmarker (CPU fallback).")
                return
            except Exception as exc:
                logger.warning(f"Could not initialize Tasks CPU fallback: {exc}")

        logger.warning("Falling back to mediapipe FaceMesh backend.")
        try:
            self._init_face_mesh_detector()
        except Exception as exc:
            raise RuntimeError(
                "Tasks backend initialization failed and FaceMesh fallback is unavailable."
            ) from exc

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.padding > 0: #  Modify image with padding
            pad_ratio = self.padding
            h, w = image.shape[:2]
            pad_h, pad_w = int(h * pad_ratio), int(w * pad_ratio)
            padded_img = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

            faces = self._detect_faces_mediapipe(padded_img)
        
            valid_faces = []
            for face in faces: # Shift landmarks back accounting for padding
                face.landmarks -= np.array([pad_w, pad_h])
                face.landmarks_eyes -= np.array([pad_w, pad_h])
                face.bbox -= np.array([[pad_w, pad_h], [pad_w, pad_h]])
                valid_faces.append(face)
                
            return valid_faces
        else:
            return self._detect_faces_mediapipe(image)

    def _detect_landmarks(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        h, w = image.shape[:2]
        output: List[Tuple[np.ndarray, np.ndarray]] = []

        if self._using_tasks:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            for prediction in result.face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h) for pt in prediction], dtype=np.float64)
                pts3d = np.array([(pt.x, pt.y, pt.z) for pt in prediction], dtype=np.float64)
                if pts.shape[0] >= 478:
                    output.append((pts[:478], pts3d[:478]))
            return output

        predictions = self.detector.process(image[:, :, ::-1])
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h) for pt in prediction.landmark], dtype=np.float64)
                pts3d = np.array([(pt.x, pt.y, pt.z) for pt in prediction.landmark], dtype=np.float64)
                if pts.shape[0] >= 478:
                    output.append((pts[:478], pts3d[:478]))
        return output

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Face]:
        temp_image = copy.deepcopy(image)
        detected = []
        if self.ultralight:
            faces = findHumans_ultralight(temp_image)
            for face in faces:
                widerFace = get_ultralightWiderFaceRect(face,image.shape[1],image.shape[0])
                cropped_img = copy.deepcopy(image[widerFace['yw1']:widerFace['yw2'] + 1, widerFace['xw1']:widerFace['xw2'] + 1, :])
                candidates = self._detect_landmarks(cropped_img)
                for pts_local, pts3d in candidates:
                    pts = pts_local.copy()
                    pts[:, 0] += widerFace['xw1']
                    pts[:, 1] += widerFace['yw1']
                    pts3D = pts3d.copy()
                    if pts.shape[0] >= 478:
                        bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                        bbox = np.round(bbox).astype(np.int32)
                        detected.append(Face(bbox, pts[:468], pts[468:478], pts3D))
        else:
            candidates = self._detect_landmarks(image)
            for pts, pts3D in candidates:
                if pts.shape[0] >= 478:
                    bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                    bbox = np.round(bbox).astype(np.int32)
                    detected.append(Face(bbox, pts[:468], pts[468:478], pts3D))
        return detected
