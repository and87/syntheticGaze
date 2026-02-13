# SyntheticGaze

README operativo del progetto (stato aggiornato al 2026-02-11).

===============================================================================
1) OBIETTIVO DELLE MODIFICHE
===============================================================================
Il codice e' stato adattato per usare la pipeline di estrazione landmark da ETH-XGaze
raw, con focus su:
- generazione CSV landmark/gaze
- rotazione corretta per camera
- niente training modello
- supporto download streaming remoto (una immagine alla volta)
- gestione dataset molto grande (elaborazione + rimozione file temporaneo)
- supporto ripresa elaborazione (resume)
- variante Ubuntu con MediaPipe Tasks e tentativo GPU


===============================================================================
2) FILE PRINCIPALE MODIFICATO: GenerateNormalizedDataset4_Xgaze.py
===============================================================================

2.1 Modalita' di esecuzione
- Aggiunta gestione modalita':
  - local: legge immagini da cartella locale
  - remote: scarica da URL ETH e processa in streaming
- Default impostato su remote.

2.2 Profili operativi
- Introdotti due profili:
  - eth_precise
    Usa calibrazione camera ETH (camXX.xml) + fallback camera params.
  - camera_agnostic
    Usa intrinseci generici.
- Entrambi i profili possono usare camera_orientation.yaml per rotazione immagini.

2.3 Parsing annotation_train e costruzione path immagini
- Implementato parsing robusto delle righe annotation_train.
- Allineato al formato reale:
  - annotation: annotation_train/subjectXXXX.csv
  - immagini: train/subjectXXXX/frameYYYY/camZZ.JPG
- image_path nel CSV costruito come:
  subjectXXXX/frameYYYY/camZZ.JPG

2.4 Download remoto streaming e cleanup
- Aggiunti helper HTTP:
  - fetch URL pagina indice
  - parsing link HTML
  - download file singolo
- Pipeline remota:
  - legge annotation file per file
  - scarica una sola immagine per volta in cartella temporanea
  - processa landmark/gaze
  - cancella il file temporaneo
- In questo modo non si accumulano immagini su disco.

2.5 Rotazione immagini per camera
- Caricamento automatico camere ruotate da:
  ETH-GAZE DATASET/camera_orientation.yaml
- Possibile override manuale con --rotate-cams.
- Rotazione applicata in preprocess prima del detector.

2.6 Landmark e gaze nel CSV
- Colonne CSV mantenute nel formato richiesto:
  subject;camera;image_path;gaze_x;gaze_y;gaze_z;gaze_yaw;gaze_pitch;...landmark...
- Landmark usati (x e y): indici occhi/iridi + anchor testa
  (468..477, 33,133,159,145,263,362,386,374,1,9).
- Per i sample con annotation valida:
  - calcolo vettore gaze in camera coords
  - normalizzazione nel sistema normalizzato
  - calcolo yaw/pitch.

2.7 Calibrazione camera
- Aggiunto provider calibrazione:
  - caricamento per-camera da XML in ETH-GAZE DATASET/calibration/cam_calibration
  - scaling intrinseci in base alla risoluzione immagine corrente
  - fallback su camera_params.yaml se necessario
  - fallback finale su intrinseci generici.

2.8 Parametri detector e CLI
- Aggiunti/tenuti parametri CLI per controllare pipeline:
  --mode, --profile, --output-csv, --max-samples, --download-timeout,
  --annotation-subdir, --train-subdir, --base-url, --padding, --det-conf, ecc.

2.9 Resume elaborazione (ripartenza)
- Aggiunto --resume:
  - legge ultima riga valida del CSV di output
  - costruisce chiave sample (subject, camera, image_path)
  - in local/remote salta fino al sample gia' scritto
  - riprende dal successivo
- Utile per run molto lunghi o interrotti.


===============================================================================
3) FILE MODIFICATO: face_landmark_estimator.py
===============================================================================

3.1 Backend MediaPipe selezionabile
- Introduzione backend configurabile:
  - face_mesh (API classica MediaPipe)
  - tasks (MediaPipe Tasks FaceLandmarker)

3.2 Delegate CPU/GPU per Tasks
- Per backend tasks, supporto delegate:
  - cpu
  - gpu
- Impostazione letta da config/CLI.

3.3 Fallback robusto automatico
- Se tasks GPU non parte:
  - tenta tasks CPU
- Se tasks CPU non parte:
  - fallback a FaceMesh
- Questo evita blocchi quando modello Tasks o GPU non sono disponibili.

3.4 Uniformazione output landmark
- Normalizzazione output detector a formato coerente:
  - punti 0..467 come landmarks viso
  - punti 468..477 come landmarks iridi
- Compatibilita' mantenuta con la pipeline esistente (Face object).


===============================================================================
4) FILE DI CONFIG AGGIUNTO
===============================================================================

4.1 configs/benchmark_config_ubuntu_gpu.yaml
- Config pronta per macchina Ubuntu:
  - mediapipe_backend: tasks
  - mediapipe_delegate: gpu
  - mediapipe_task_model: models/face_landmarker.task
- Serve come variante alternativa per ambiente Linux con GPU.


===============================================================================
5) SCRIPT AGGIUNTI
===============================================================================

5.1 run_xgaze_remote_ubuntu_gpu.sh
- Script di lancio per Ubuntu, modalita' remote full-dataset con resume.
- Imposta automaticamente backend tasks+gpu.

5.2 scripts/download_mediapipe_face_landmarker_model.sh
- Script per scaricare il modello ufficiale:
  models/face_landmarker.task
- Usa curl o wget.


===============================================================================
6) COMPORTAMENTO ESECUZIONE FINALE (STATO ATTUALE)
===============================================================================

- Il codice produce CSV landmark+gaze senza avviare training.
- In remote mode processa annotation_train e train via URL ETH.
- Gestisce grandi volumi: una immagine alla volta + cancellazione temporaneo.
- Usa orientamento camere da camera_orientation.yaml.
- Supporta resume per riprendere run interrotti.
- Ha variante Ubuntu/GPU con fallback automatico se tasks GPU non disponibile.


===============================================================================
7) NOTE IMPORTANTI OPERATIVE
===============================================================================

- Per usare MediaPipe Tasks servono:
  - pacchetto mediapipe compatibile
  - file modello .task presente nel path configurato
- Se il modello non c'e', la pipeline fa fallback a FaceMesh.
- In run Windows gia' testati: pipeline standard funziona e processa sample.


===============================================================================
8) ISTRUZIONI DI UTILIZZO (WINDOWS E UBUNTU)
===============================================================================

8.1 Windows (PowerShell) - avvio ambiente
- Posizionarsi nella root progetto:
  C:\workspace\SyntheticGaze
- Attivare virtual env:
  .\env\Scripts\Activate.ps1

8.2 Windows - test rapido (4 immagini)
- Esegue pipeline remota e si ferma a 4 sample:
  python .\GenerateNormalizedDataset4_Xgaze.py `
    --mode remote `
    --profile eth_precise `
    --max-samples 4 `
    --output-csv "ETH-GAZE DATASET\processed\smoke4.csv"

8.3 Windows - esecuzione full dataset (remote)
- Nessun limite campioni: non passare --max-samples.
  python .\GenerateNormalizedDataset4_Xgaze.py `
    --mode remote `
    --profile eth_precise `
    --output-csv "ETH-GAZE DATASET\processed\training_xgaze_dataset_landmarks_with_gaze.csv" `
    --resume

8.4 Windows - variante camera_agnostic
  python .\GenerateNormalizedDataset4_Xgaze.py `
    --mode remote `
    --profile camera_agnostic `
    --output-csv "ETH-GAZE DATASET\processed\training_xgaze_camera_agnostic.csv" `
    --resume

8.5 Windows - modalita' local (immagini gia' su disco)
  python .\GenerateNormalizedDataset4_Xgaze.py `
    --mode local `
    --dataset-root "ETH-GAZE DATASET\train" `
    --profile eth_precise `
    --output-csv "ETH-GAZE DATASET\processed\training_local.csv" `
    --resume

8.6 Ubuntu - setup base
- Posizionarsi nella root progetto.
- Attivare venv (esempio):
  source env/bin/activate

8.7 Ubuntu - scaricare modello MediaPipe Tasks
  chmod +x scripts/download_mediapipe_face_landmarker_model.sh
  ./scripts/download_mediapipe_face_landmarker_model.sh

8.8 Ubuntu - esecuzione GPU (Tasks)
  chmod +x run_xgaze_remote_ubuntu_gpu.sh
  ./run_xgaze_remote_ubuntu_gpu.sh

8.9 Ubuntu - comando esplicito equivalente (senza script helper)
  python GenerateNormalizedDataset4_Xgaze.py \
    --mode remote \
    --profile eth_precise \
    --config configs/benchmark_config_ubuntu_gpu.yaml \
    --mediapipe-backend tasks \
    --mediapipe-delegate gpu \
    --mediapipe-task-model models/face_landmarker.task \
    --output-csv "ETH-GAZE DATASET/processed/training_xgaze_dataset_landmarks_with_gaze.csv" \
    --resume

8.10 Verifica backend detector nei log
- Se GPU attiva correttamente:
  "Landmark backend: mediapipe Tasks FaceLandmarker (GPU)."
- Se GPU non disponibile o modello mancante:
  fallback automatico a CPU o FaceMesh (messaggio warning/info a log).

8.11 Esecuzione test set (CSV separato)
- Windows (PowerShell):
  .\run_xgaze_remote_test.ps1

- Ubuntu:
  chmod +x run_xgaze_remote_test.sh
  ./run_xgaze_remote_test.sh

- Comando equivalente:
  python GenerateNormalizedDataset4_Xgaze.py \
    --mode remote \
    --profile eth_precise \
    --annotation-subdir annotation_test \
    --train-subdir test \
    --output-csv "ETH-GAZE DATASET/processed/test_xgaze_dataset_landmarks.csv" \
    --resume


===============================================================================
9) LISTA COMPLETA PARAMETRI CLI (GenerateNormalizedDataset4_Xgaze.py)
===============================================================================

Parametri principali:
- --mode {local,remote}
  Default: remote
  Seleziona sorgente immagini.

- --profile {eth_precise,camera_agnostic}
  Default: eth_precise
  Scelta calibrazione camera.

- --config <path>
  Default: benchmark_config.yaml
  Config generale detector/pipeline.

- --dataset-root <path>
  Default: ETH-GAZE DATASET\train
  Root immagini in modalita' local.

- --output-csv <path>
  Default: ETH-GAZE DATASET\processed\training_xgaze_dataset_landmarks_with_gaze.csv
  CSV di output.

- --normalized-camera-params <path>
  Default: normalized_camera_params.yaml
  Parametri camera normalizzata usati dalla normalizzazione landmark.

- --camera-params <path>
  Default: configs\camera_params.yaml
  Fallback camera params.

- --cam-calibration-dir <path>
  Default: ETH-GAZE DATASET\calibration\cam_calibration
  Cartella XML calibrazione per-camera.

- --orientation-config <path>
  Default: ETH-GAZE DATASET\camera_orientation.yaml
  File orientamento camere (rotate_180).

- --rotate-cams <lista_csv>
  Default: non impostato
  Override manuale camere da ruotare (esempio: 3,6,13).

- --equalize-luma
  Default: false
  Abilita equalizzazione luminanza in preprocess.

- --det-conf <float>
  Default: 0.3
  Detection confidence MediaPipe.

- --padding <float>
  Default: 0.0
  Padding immagine prima della detection.

Parametri MediaPipe backend:
- --mediapipe-backend {face_mesh,tasks}
  Default: face_mesh
  Se tasks, usa MediaPipe Tasks FaceLandmarker.

- --mediapipe-delegate {cpu,gpu}
  Default: cpu
  Delegate per backend tasks.

- --mediapipe-task-model <path>
  Default: models\face_landmarker.task
  Modello .task usato dal backend tasks.

Parametri performance/controllo run:
- --batch-size <int>
  Default: 100
  Numero righe bufferizzate prima del flush CSV.

- --max-samples <int>
  Default: non impostato
  Limita numero sample processati (utile per test smoke).

- --download-timeout <int>
  Default: 40
  Timeout HTTP in secondi per listing/download.

Parametri URL/struttura dataset remoto:
- --base-url <url>
  Default: https://dataset.ait.ethz.ch/downloads/T3fODqLSS1/eth-xgaze/raw/data/
  Base URL dataset.

- --annotation-subdir <nome_cartella>
  Default: annotation_train
  Sottocartella annotation.

- --train-subdir <nome_cartella>
  Default: train
  Sottocartella immagini.

Resume:
- --resume
  Default: false
  Riprende dal sample successivo all'ultima riga valida gia' presente nel CSV.


===============================================================================
10) NOTE PRATICHE CONSIGLIATE
===============================================================================

- Per run lunghi usare sempre --resume.
- Per test veloci usare --max-samples 1/4/10.
- In Windows usare virgolette attorno ai path con spazi.
- Per Ubuntu GPU verificare driver/NVIDIA e disponibilita' delegate.


===============================================================================
11) PARALLELIZZAZIONE PER SOGGETTI (NUOVO)
===============================================================================

Nuovi parametri in GenerateNormalizedDataset4_Xgaze.py:
- --subjects
  Filtro soggetti inline (esempio: --subjects "subject0001,subject0002" oppure --subjects "1,2").
- --subjects-file
  File di soggetti (uno per riga oppure separati da virgola/spazio).

Nuovi script:
- scripts/make_subject_shards_from_remote.py
  Scarica la lista subject*.csv da annotation e genera shard_00.txt ... shard_07.txt.
- run_xgaze_remote_train_parallel_8.sh
  Avvia 8 processi in parallelo (uno per shard) con CSV partizionati.
- scripts/merge_csv_parts.py
  Unisce i CSV partizionati in un unico CSV finale.
