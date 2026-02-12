import argparse
import csv
import io
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, Iterator, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml
from omegaconf import OmegaConf

try:
    from camera import Camera
    from face_landmark_estimator import LandmarkEstimator
    from face_model import FaceModel
    from normalization_utils import LandmarkNormalizer
except ImportError as exc:
    print(f"Error: Could not import local modules: {exc}")
    raise SystemExit(1)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Landmark indices (as requested)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_INNER_CORNER = 133
LEFT_OUTER_CORNER = 33
LEFT_EYE_CONTOUR = [LEFT_OUTER_CORNER, LEFT_INNER_CORNER, 159, 145]
RIGHT_INNER_CORNER = 362
RIGHT_OUTER_CORNER = 263
RIGHT_EYE_CONTOUR = [RIGHT_OUTER_CORNER, RIGHT_INNER_CORNER, 386, 374]
HEAD_ANCHORS = [1, 9]

LANDMARK_INDICES = LEFT_IRIS + LEFT_EYE_CONTOUR + RIGHT_IRIS + RIGHT_EYE_CONTOUR + HEAD_ANCHORS
GAZE_COLS = ["gaze_x", "gaze_y", "gaze_z", "gaze_yaw", "gaze_pitch"]
LANDMARK_COLS_X = [f"{idx}_x" for idx in LANDMARK_INDICES]
LANDMARK_COLS_Y = [f"{idx}_y" for idx in LANDMARK_INDICES]
FIELDNAMES = ["subject", "camera", "image_path"] + GAZE_COLS + LANDMARK_COLS_X + LANDMARK_COLS_Y


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


@dataclass
class AnnotationEntry:
    subject: str
    camera_id: int
    image_rel_path: str
    gaze_point_cam: Optional[np.ndarray]


SampleKey = Tuple[str, int, str]


def first_existing_path(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def parse_camera_ids(text: Optional[str]) -> Set[int]:
    out: Set[int] = set()
    if not text:
        return out
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            logger.warning(f"Ignoring invalid camera id '{token}' in --rotate-cams")
    return out


def normalize_subject_id(token: str) -> str:
    value = token.strip().lower()
    if not value:
        return value
    match = re.fullmatch(r"subject(\d+)", value)
    if match:
        return f"subject{int(match.group(1)):04d}"
    if value.isdigit():
        return f"subject{int(value):04d}"
    return value


def parse_subject_filter(subjects_text: Optional[str], subjects_file: Optional[str]) -> Optional[Set[str]]:
    selected: Set[str] = set()

    if subjects_text:
        for tok in re.split(r"[,\s;]+", subjects_text):
            tok = tok.strip()
            if not tok:
                continue
            selected.add(normalize_subject_id(tok))

    if subjects_file:
        try:
            with open(subjects_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    for tok in re.split(r"[,\s;]+", stripped):
                        tok = tok.strip()
                        if not tok:
                            continue
                        selected.add(normalize_subject_id(tok))
        except Exception as exc:
            raise SystemExit(f"Could not read --subjects-file '{subjects_file}': {exc}") from exc

    return selected if selected else None


def load_rotate_camera_ids_from_orientation(path: str) -> Set[int]:
    rotate_camera_ids: Set[int] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning(f"Could not read orientation config '{path}': {exc}")
        return rotate_camera_ids

    cameras = data.get("cameras", {})
    if not isinstance(cameras, dict):
        logger.warning(f"Invalid orientation config format in '{path}'")
        return rotate_camera_ids

    for cam_key, cam_cfg in cameras.items():
        match = re.search(r"cam(\d+)", str(cam_key).lower())
        if not match:
            continue
        cam_id = int(match.group(1))
        rotate_180 = False
        if isinstance(cam_cfg, dict):
            rotate_180 = bool(cam_cfg.get("rotate_180", False))
        elif isinstance(cam_cfg, bool):
            rotate_180 = cam_cfg
        if rotate_180:
            rotate_camera_ids.add(cam_id)

    return rotate_camera_ids


def parse_camera_id_from_name(name: str) -> int:
    match = re.search(r"cam(\d+)", name.lower())
    if match:
        return int(match.group(1))
    return -1


def parse_subject_from_path(path_str: str) -> str:
    match = re.search(r"(subject\d+)", path_str.lower())
    if match:
        return normalize_subject_id(match.group(1))
    first = path_str.replace("\\", "/").split("/")[0]
    return normalize_subject_id(first) if first else "subject_unknown"


def make_sample_key(subject: str, camera_id: int, image_rel_path: str) -> SampleKey:
    return (str(subject).strip().lower(), int(camera_id), image_rel_path.replace("\\", "/").strip().lower())


def get_last_csv_sample_key(csv_path: str) -> Optional[SampleKey]:
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            if pos <= 0:
                return None
            line = b""
            while pos > 0:
                pos -= 1
                f.seek(pos)
                char = f.read(1)
                if char == b"\n" and line:
                    break
                line = char + line
        last = line.decode("utf-8", errors="ignore").strip()
        if not last:
            return None
        if last.lower().startswith("subject;camera;image_path;"):
            return None
        parts = last.split(";")
        if len(parts) < 3:
            return None
        subject = parts[0].strip()
        camera_id = int(parts[1].strip())
        image_rel_path = parts[2].strip()
        return make_sample_key(subject, camera_id, image_rel_path)
    except Exception as exc:
        logger.warning(f"Could not parse last CSV row for resume from {csv_path}: {exc}")
        return None


def vector_to_pitch_yaw(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    norm = np.linalg.norm([x, y, z])
    if norm > 0:
        x, y, z = x / norm, y / norm, z / norm
    pitch = float(np.arcsin(np.clip(y, -1.0, 1.0)))
    yaw = float(np.arctan2(x, -z))
    return yaw, pitch


def fetch_url_text(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "SyntheticGaze/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def fetch_url_text_with_retries(url: str, timeout: int, retries: int = 3, base_sleep: float = 1.5) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return fetch_url_text(url, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = base_sleep * attempt
            logger.warning(
                f"Failed to fetch {url} (attempt {attempt}/{retries}): {exc}. Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_exc}")


def list_remote_files(base_url: str, timeout: int) -> List[str]:
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    html = fetch_url_text(base_url, timeout=timeout)
    parser = LinkParser()
    parser.feed(html)

    files: List[str] = []
    for href in parser.hrefs:
        if href in ("../", "./"):
            continue
        if href.startswith("?") or href.startswith("#"):
            continue
        if href.endswith("/"):
            continue
        files.append(urllib.parse.urljoin(base_url, href))
    files.sort()
    return files


def parse_annotation_line(
    line: str,
    subject_from_file: str,
    parse_gaze_point_cam: bool = True,
) -> Optional[AnnotationEntry]:
    stripped = line.strip()
    if not stripped:
        return None
    tokens = [tok for tok in re.split(r"[,\s;]+", stripped) if tok]
    if len(tokens) < 2:
        return None

    frame_folder = tokens[0]
    image_name = tokens[1]
    frame_norm = frame_folder.replace("\\", "/").strip("/")
    image_norm = image_name.replace("\\", "/").strip("/")
    if frame_norm.lower().startswith("subject"):
        image_rel_path = f"{frame_norm}/{image_norm}"
        subject = parse_subject_from_path(frame_norm)
    else:
        image_rel_path = f"{subject_from_file}/{frame_norm}/{image_norm}"
        subject = subject_from_file
    camera_id = parse_camera_id_from_name(image_name)
    if camera_id < 0:
        return None

    gaze_point_cam = None
    if parse_gaze_point_cam and len(tokens) >= 7:
        try:
            gaze_point_cam = np.array(
                [float(tokens[4]), float(tokens[5]), float(tokens[6])], dtype=np.float64
            )
        except ValueError:
            gaze_point_cam = None

    return AnnotationEntry(
        subject=subject,
        camera_id=camera_id,
        image_rel_path=image_rel_path,
        gaze_point_cam=gaze_point_cam,
    )


def iter_remote_annotation_entries(
    annotation_root_url: str,
    timeout: int,
    parse_gaze_point_cam: bool = True,
    allowed_subjects: Optional[Set[str]] = None,
) -> Iterator[AnnotationEntry]:
    files = list_remote_files(annotation_root_url, timeout=timeout)
    if not files:
        logger.warning(f"No annotation files found at {annotation_root_url}")
        return

    for file_url in files:
        subject_match = re.search(r"(subject\d+)\.csv$", file_url.lower())
        subject_from_file = normalize_subject_id(subject_match.group(1)) if subject_match else "subject_unknown"
        if allowed_subjects is not None and subject_from_file not in allowed_subjects:
            continue
        logger.info(f"Loading annotations from {file_url} (subject={subject_from_file})")
        try:
            text = fetch_url_text_with_retries(file_url, timeout=timeout, retries=3)
        except Exception as exc:
            logger.warning(f"Skipping annotation file {file_url}: {exc}")
            continue

        lines = [ln for ln in text.splitlines() if ln.strip()]
        logger.info(f"Loaded {len(lines)} annotation rows for {subject_from_file}")
        for line in lines:
            entry = parse_annotation_line(
                line,
                subject_from_file=subject_from_file,
                parse_gaze_point_cam=parse_gaze_point_cam,
            )
            if entry is not None:
                yield entry


class CalibrationProvider:
    def __init__(
        self,
        profile: str,
        cam_calibration_dir: Optional[str],
        camera_params_path: Optional[str],
    ):
        self.profile = profile
        self.per_camera: Dict[int, Tuple[np.ndarray, np.ndarray, int, int]] = {}
        self.source_camera: Optional[Camera] = None

        if profile == "eth_precise" and cam_calibration_dir and os.path.isdir(cam_calibration_dir):
            self.per_camera = self._load_per_camera(cam_calibration_dir)
            if self.per_camera:
                logger.info(
                    f"Loaded per-camera calibration for {len(self.per_camera)} cameras from {cam_calibration_dir}"
                )

        if profile == "eth_precise" and camera_params_path and os.path.exists(camera_params_path):
            self.source_camera = Camera(camera_params_path)
            logger.info(f"Loaded fallback camera params from {camera_params_path}")

    @staticmethod
    def _read_xml(xml_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            return None
        try:
            K = fs.getNode("Camera_Matrix").mat()
            dist = fs.getNode("Distortion_Coefficients").mat()
            w = int(fs.getNode("image_Width").real())
            h = int(fs.getNode("image_Height").real())
        finally:
            fs.release()
        if K is None or dist is None or w <= 0 or h <= 0:
            return None
        return (
            np.asarray(K, dtype=np.float64),
            np.asarray(dist, dtype=np.float64).reshape(-1, 1),
            w,
            h,
        )

    def _load_per_camera(self, cam_dir: str) -> Dict[int, Tuple[np.ndarray, np.ndarray, int, int]]:
        out: Dict[int, Tuple[np.ndarray, np.ndarray, int, int]] = {}
        for name in sorted(os.listdir(cam_dir)):
            match = re.search(r"cam(\d+)\.xml$", name.lower())
            if not match:
                continue
            cam_id = int(match.group(1))
            item = self._read_xml(os.path.join(cam_dir, name))
            if item is not None:
                out[cam_id] = item
        return out

    def get(self, camera_id: int, image_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image_shape[:2]

        if self.profile == "eth_precise" and camera_id in self.per_camera:
            K, dist, src_w, src_h = self.per_camera[camera_id]
            K = K.copy()
            if src_w > 0 and src_h > 0 and (src_w != w or src_h != h):
                sx = w / float(src_w)
                sy = h / float(src_h)
                K[0, 0] *= sx
                K[1, 1] *= sy
                K[0, 2] *= sx
                K[1, 2] *= sy
            return K, dist.copy()

        if self.profile == "eth_precise" and self.source_camera is not None:
            K = self.source_camera.camera_matrix.astype(np.float64).copy()
            src_w = float(self.source_camera.width)
            src_h = float(self.source_camera.height)
            if src_w > 0 and src_h > 0:
                sx = w / src_w
                sy = h / src_h
                K[0, 0] *= sx
                K[1, 1] *= sy
                K[0, 2] *= sx
                K[1, 2] *= sy
            dist = self.source_camera.dist_coefficients.astype(np.float64).copy()
            return K, dist

        focal = float(max(w, h))
        K = np.array(
            [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.zeros((5, 1), dtype=np.float64)
        return K, dist


class LandmarkPipeline:
    def __init__(
        self,
        config,
        calibration_provider: CalibrationProvider,
        normalized_camera_params_path: str,
        rotate_camera_ids: Set[int],
        equalize_luma: bool,
    ):
        self.config = config
        self.calibration_provider = calibration_provider
        self.rotate_camera_ids = rotate_camera_ids
        self.equalize_luma = equalize_luma

        logger.info("Initializing LandmarkEstimator...")
        self.landmark_estimator = LandmarkEstimator(self.config)
        self.face_model3d = FaceModel()

        self.normalized_camera = Camera(normalized_camera_params_path)
        self.normalizer = LandmarkNormalizer(self.normalized_camera)

    @staticmethod
    def _largest_face(faces):
        def area(face) -> float:
            return float((face.bbox[1, 0] - face.bbox[0, 0]) * (face.bbox[1, 1] - face.bbox[0, 1]))

        return max(faces, key=area)

    @staticmethod
    def _equalize_luma(image: np.ndarray) -> np.ndarray:
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def preprocess(self, image: np.ndarray, camera_id: int) -> Tuple[np.ndarray, bool]:
        out = image
        rotated = False
        if camera_id in self.rotate_camera_ids:
            out = cv2.rotate(out, cv2.ROTATE_180)
            rotated = True
        if self.equalize_luma:
            out = self._equalize_luma(out)
        return out, rotated

    def extract(
        self, image: np.ndarray, camera_id: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        camera_matrix, dist_coeffs = self.calibration_provider.get(camera_id, image.shape)
        faces = self.landmark_estimator.detect_faces(image)
        if not faces:
            return None
        face = self._largest_face(faces)

        success, rvec, tvec = cv2.solvePnP(
            self.face_model3d.LANDMARKS.astype(np.float64),
            face.landmarks.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        head_R = cv2.Rodrigues(rvec)[0]
        all_landmarks = np.concatenate([face.landmarks, face.landmarks_eyes], axis=0)
        if all_landmarks.shape[0] <= max(LANDMARK_INDICES):
            return None
        subset = all_landmarks[LANDMARK_INDICES]

        M, R_norm = self.normalizer.compute_normalization_matrix(head_R, tvec, camera_matrix)
        normalized_landmarks = self.normalizer.normalize_landmarks(subset, M)
        return normalized_landmarks, R_norm, tvec.reshape(3)


class CsvWriter:
    def __init__(self, output_csv_path: str, batch_size: int):
        self.output_csv_path = output_csv_path
        self.batch_size = max(1, int(batch_size))
        self.buffer: List[dict] = []

        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(output_csv_path):
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
                writer.writeheader()
            logger.info(f"Initialized CSV: {output_csv_path}")
        else:
            logger.info(f"Appending to existing CSV: {output_csv_path}")

    def add(self, row: dict):
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.output_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
            writer.writerows(self.buffer)
        self.buffer = []


def build_row(
    subject: str,
    camera_id: int,
    image_path: str,
    gaze_vector: np.ndarray,
    yaw: float,
    pitch: float,
    normalized_landmarks: np.ndarray,
) -> dict:
    row = {
        "subject": subject,
        "camera": camera_id,
        "image_path": image_path.replace("\\", "/"),
        "gaze_x": float(gaze_vector[0]),
        "gaze_y": float(gaze_vector[1]),
        "gaze_z": float(gaze_vector[2]),
        "gaze_yaw": float(yaw),
        "gaze_pitch": float(pitch),
    }
    for i, idx in enumerate(LANDMARK_INDICES):
        row[f"{idx}_x"] = float(normalized_landmarks[i, 0])
        row[f"{idx}_y"] = float(normalized_landmarks[i, 1])
    return row


def build_remote_image_url(train_root_url: str, image_rel_path: str) -> str:
    if not train_root_url.endswith("/"):
        train_root_url = train_root_url + "/"
    rel = image_rel_path.replace("\\", "/").strip("/")
    quoted = "/".join(urllib.parse.quote(part) for part in rel.split("/"))
    return urllib.parse.urljoin(train_root_url, quoted)


def download_file(url: str, local_path: str, timeout: int):
    req = urllib.request.Request(url, headers={"User-Agent": "SyntheticGaze/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(local_path, "wb") as f:
        shutil.copyfileobj(resp, f)


def process_local_images(
    pipeline: LandmarkPipeline,
    writer: CsvWriter,
    image_root: str,
    max_samples: Optional[int] = None,
    resume_after: Optional[SampleKey] = None,
    allowed_subjects: Optional[Set[str]] = None,
):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths: List[str] = []
    for root, _, files in os.walk(image_root):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                image_paths.append(os.path.join(root, name))
    image_paths.sort()

    processed = 0
    skipped = 0
    resuming = resume_after is not None
    resume_hit = False
    for path in image_paths:
        if max_samples is not None and processed >= max_samples:
            break
        rel = os.path.relpath(path, image_root).replace("\\", "/")
        subject = parse_subject_from_path(rel)
        if allowed_subjects is not None and subject not in allowed_subjects:
            continue
        camera_id = parse_camera_id_from_name(os.path.basename(path))
        current_key = make_sample_key(subject, camera_id, rel)
        if resuming:
            if current_key == resume_after:
                resuming = False
                resume_hit = True
                logger.info(f"Resume point reached at {rel}; continuing from next sample.")
            continue

        img = cv2.imread(path)
        if img is None:
            skipped += 1
            continue
        pre, _ = pipeline.preprocess(img, camera_id)
        result = pipeline.extract(pre, camera_id)
        if result is None:
            skipped += 1
            continue
        normalized_landmarks, _, _ = result

        gaze_vec = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        row = build_row(subject, camera_id, rel, gaze_vec, np.nan, np.nan, normalized_landmarks)
        writer.add(row)
        processed += 1

    writer.flush()
    if resume_after is not None and not resume_hit:
        logger.warning("Resume key not found in local dataset scan. No new rows were added.")
    logger.info(f"Local mode complete. Processed={processed}, Skipped={skipped}")


def process_remote_stream(
    pipeline: LandmarkPipeline,
    writer: CsvWriter,
    base_url: str,
    annotation_subdir: str,
    train_subdir: str,
    timeout: int,
    max_samples: Optional[int] = None,
    resume_after: Optional[SampleKey] = None,
    allowed_subjects: Optional[Set[str]] = None,
):
    annotation_root = urllib.parse.urljoin(base_url if base_url.endswith("/") else base_url + "/", annotation_subdir + "/")
    train_root = urllib.parse.urljoin(base_url if base_url.endswith("/") else base_url + "/", train_subdir + "/")
    parse_gaze_point_cam = annotation_subdir.replace("\\", "/").strip("/").lower() == "annotation_train"
    logger.info(
        f"Annotation split '{annotation_subdir}': "
        f"{'with gaze labels' if parse_gaze_point_cam else 'without gaze labels'}"
    )

    processed = 0
    skipped = 0
    resuming = resume_after is not None
    resume_hit = False
    with tempfile.TemporaryDirectory(prefix="xgaze_dl_") as tmp_dir:
        tmp_image_path = os.path.join(tmp_dir, "current_image.jpg")

        for entry in iter_remote_annotation_entries(
            annotation_root,
            timeout=timeout,
            parse_gaze_point_cam=parse_gaze_point_cam,
            allowed_subjects=allowed_subjects,
        ):
            if max_samples is not None and processed >= max_samples:
                break
            current_key = make_sample_key(entry.subject, entry.camera_id, entry.image_rel_path)
            if resuming:
                if current_key == resume_after:
                    resuming = False
                    resume_hit = True
                    logger.info(
                        f"Resume point reached at {entry.image_rel_path}; continuing from next sample."
                    )
                continue
            image_url = build_remote_image_url(train_root, entry.image_rel_path)

            try:
                download_file(image_url, tmp_image_path, timeout=timeout)
                img = cv2.imread(tmp_image_path)
                if img is None:
                    skipped += 1
                    continue

                pre, _ = pipeline.preprocess(img, entry.camera_id)
                result = pipeline.extract(pre, entry.camera_id)
                if result is None:
                    skipped += 1
                    continue
                normalized_landmarks, R_norm, head_t = result

                gaze_vec_norm = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                yaw = np.nan
                pitch = np.nan
                if entry.gaze_point_cam is not None:
                    gaze_vec_cam = entry.gaze_point_cam - head_t
                    norm = np.linalg.norm(gaze_vec_cam)
                    if norm > 0:
                        gaze_vec_cam = gaze_vec_cam / norm
                        gaze_vec_norm = R_norm @ gaze_vec_cam
                        norm2 = np.linalg.norm(gaze_vec_norm)
                        if norm2 > 0:
                            gaze_vec_norm = gaze_vec_norm / norm2
                            yaw, pitch = vector_to_pitch_yaw(gaze_vec_norm)

                row = build_row(
                    subject=entry.subject,
                    camera_id=entry.camera_id,
                    image_path=entry.image_rel_path,
                    gaze_vector=gaze_vec_norm,
                    yaw=yaw,
                    pitch=pitch,
                    normalized_landmarks=normalized_landmarks,
                )
                writer.add(row)
                processed += 1

                if processed % 100 == 0:
                    logger.info(f"Processed={processed}, Skipped={skipped}")

            except Exception as exc:
                skipped += 1
                logger.warning(f"Failed sample {entry.image_rel_path}: {exc}")
            finally:
                if os.path.exists(tmp_image_path):
                    try:
                        os.remove(tmp_image_path)
                    except OSError:
                        pass

    writer.flush()
    if resume_after is not None and not resume_hit:
        logger.warning("Resume key not found in remote annotation stream. No new rows were added.")
    logger.info(f"Remote mode complete. Processed={processed}, Skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ETH-XGaze CSV from local images or remote streaming download."
    )
    parser.add_argument("--mode", choices=["local", "remote"], default="remote")
    parser.add_argument(
        "--profile",
        choices=["eth_precise", "camera_agnostic"],
        default="eth_precise",
        help="eth_precise uses ETH camera calibration; camera_agnostic uses generic intrinsics.",
    )
    parser.add_argument("--config", default="benchmark_config.yaml")
    parser.add_argument("--dataset-root", default=os.path.join("ETH-GAZE DATASET", "train"))
    parser.add_argument(
        "--output-csv",
        default=os.path.join(
            "ETH-GAZE DATASET", "processed", "training_xgaze_dataset_landmarks_with_gaze.csv"
        ),
    )
    parser.add_argument("--normalized-camera-params", default="normalized_camera_params.yaml")
    parser.add_argument("--camera-params", default=os.path.join("configs", "camera_params.yaml"))
    parser.add_argument(
        "--cam-calibration-dir",
        default=os.path.join("ETH-GAZE DATASET", "calibration", "cam_calibration"),
    )
    parser.add_argument(
        "--orientation-config",
        default=os.path.join("ETH-GAZE DATASET", "camera_orientation.yaml"),
    )
    parser.add_argument(
        "--rotate-cams",
        default=None,
        help="Manual override: comma-separated camera ids. If omitted uses orientation-config.",
    )
    parser.add_argument("--equalize-luma", action="store_true")
    parser.add_argument("--det-conf", type=float, default=0.3)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument(
        "--mediapipe-backend",
        choices=["face_mesh", "tasks"],
        default="face_mesh",
        help="face_mesh: classic solution API. tasks: FaceLandmarker task API.",
    )
    parser.add_argument(
        "--mediapipe-delegate",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Delegate for tasks backend. GPU is typically available on Ubuntu.",
    )
    parser.add_argument(
        "--mediapipe-task-model",
        default=os.path.join("models", "face_landmarker.task"),
        help="Path to face_landmarker.task when --mediapipe-backend=tasks.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--base-url",
        default="https://dataset.ait.ethz.ch/downloads/T3fODqLSS1/eth-xgaze/raw/data/",
    )
    parser.add_argument("--annotation-subdir", default="annotation_train")
    parser.add_argument("--train-subdir", default="train")
    parser.add_argument(
        "--subjects",
        default=None,
        help="Optional subject filter: comma/space-separated values (e.g. subject0001,subject0002 or 1,2).",
    )
    parser.add_argument(
        "--subjects-file",
        default=None,
        help="Optional text file with subjects (one per line or comma-separated).",
    )
    parser.add_argument("--download-timeout", type=int, default=40)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last row already present in --output-csv.",
    )
    args = parser.parse_args()

    config_path = first_existing_path(
        [args.config, "benchmark_config.yaml", os.path.join("configs", "benchmark_config.yaml")]
    )
    if not config_path:
        raise SystemExit("Config file not found.")

    normalized_camera_params_path = first_existing_path(
        [
            args.normalized_camera_params,
            "normalized_camera_params.yaml",
            os.path.join("configs", "normalized_camera_params.yaml"),
        ]
    )
    if not normalized_camera_params_path:
        raise SystemExit("Normalized camera params not found.")

    orientation_config_path = first_existing_path(
        [args.orientation_config, os.path.join("ETH-GAZE DATASET", "camera_orientation.yaml")]
    )
    if args.rotate_cams is not None and str(args.rotate_cams).strip() != "":
        rotate_camera_ids = parse_camera_ids(args.rotate_cams)
        logger.info(f"Using --rotate-cams override: {sorted(rotate_camera_ids)}")
    elif orientation_config_path:
        rotate_camera_ids = load_rotate_camera_ids_from_orientation(orientation_config_path)
        logger.info(
            f"Loaded rotate_180 cameras from {orientation_config_path}: {sorted(rotate_camera_ids)}"
        )
    else:
        rotate_camera_ids = set()
        logger.warning("No orientation config found and no --rotate-cams override.")

    config = OmegaConf.load(config_path)
    config.face_detector.mediapipe_min_det_conf = float(args.det_conf)
    config.face_detector.padding = float(args.padding)
    config.face_detector.mediapipe_backend = str(args.mediapipe_backend)
    config.face_detector.mediapipe_delegate = str(args.mediapipe_delegate)
    config.face_detector.mediapipe_task_model = str(args.mediapipe_task_model)
    config.face_detector.ultralight = False
    config.face_detector.mediapipe_static_image_mode = True
    logger.info(
        "MediaPipe settings: "
        f"backend={config.face_detector.mediapipe_backend}, "
        f"delegate={config.face_detector.mediapipe_delegate}, "
        f"task_model={config.face_detector.mediapipe_task_model}"
    )

    cam_calibration_dir = first_existing_path(
        [args.cam_calibration_dir, os.path.join("ETH-GAZE DATASET", "calibration", "cam_calibration")]
    )
    camera_params_path = first_existing_path(
        [args.camera_params, os.path.join("configs", "camera_params.yaml"), "camera_params.yaml"]
    )

    calibration_provider = CalibrationProvider(
        profile=args.profile,
        cam_calibration_dir=cam_calibration_dir,
        camera_params_path=camera_params_path,
    )
    pipeline = LandmarkPipeline(
        config=config,
        calibration_provider=calibration_provider,
        normalized_camera_params_path=normalized_camera_params_path,
        rotate_camera_ids=rotate_camera_ids,
        equalize_luma=bool(args.equalize_luma),
    )
    writer = CsvWriter(output_csv_path=args.output_csv, batch_size=args.batch_size)
    allowed_subjects = parse_subject_filter(args.subjects, args.subjects_file)
    if allowed_subjects is not None:
        preview = ", ".join(sorted(allowed_subjects)[:10])
        suffix = " ..." if len(allowed_subjects) > 10 else ""
        logger.info(
            f"Subject filter enabled ({len(allowed_subjects)} subjects): {preview}{suffix}"
        )
    resume_after = get_last_csv_sample_key(args.output_csv) if args.resume else None
    if resume_after is not None:
        logger.info(
            f"Resuming after last CSV row: subject={resume_after[0]}, camera={resume_after[1]}, path={resume_after[2]}"
        )
    elif args.resume:
        logger.info("Resume requested but CSV has no resumable data; starting from beginning.")

    if args.mode == "local":
        if not os.path.isdir(args.dataset_root):
            raise SystemExit(f"Dataset root not found: {args.dataset_root}")
        process_local_images(
            pipeline=pipeline,
            writer=writer,
            image_root=args.dataset_root,
            max_samples=args.max_samples,
            resume_after=resume_after,
            allowed_subjects=allowed_subjects,
        )
    else:
        process_remote_stream(
            pipeline=pipeline,
            writer=writer,
            base_url=args.base_url,
            annotation_subdir=args.annotation_subdir,
            train_subdir=args.train_subdir,
            timeout=args.download_timeout,
            max_samples=args.max_samples,
            resume_after=resume_after,
            allowed_subjects=allowed_subjects,
        )


if __name__ == "__main__":
    main()
