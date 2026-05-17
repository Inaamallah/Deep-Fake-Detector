# 🔍 Deepfake Detector

> **A production-grade, end-to-end deepfake video detection system** — from raw video URL to a structured analytical verdict, powered by EfficientNet-B4 fine-tuned on FaceForensics++, deployed behind Nginx with async job processing, a React dashboard, and full observability.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [What This System Does](#-what-this-system-does)
- [Architecture Overview](#-architecture-overview)
- [Pipeline Stages](#-pipeline-stages)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Fine-tuning the Model](#-fine-tuning-the-model)
- [API Reference](#-api-reference)
- [Frontend Dashboard](#-frontend-dashboard)
- [Configuration](#-configuration)
- [Running Tests](#-running-tests)
- [Observability](#-observability)
- [Production Deployment](#-production-deployment)
- [Performance Benchmarks](#-performance-benchmarks)
- [Limitations & Honest Caveats](#-limitations--honest-caveats)
- [7-Day Build Log](#-7-day-build-log)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 What This System Does

A user submits a YouTube link, a Vimeo URL, a TikTok clip, or uploads a raw video file. Within a few minutes, the system returns a structured analytical report containing a **REAL / DEEPFAKE / INCONCLUSIVE** verdict, a confidence-weighted probability score, a 95% bootstrap confidence interval, a timeline of per-frame scores showing *when* in the video suspicious regions appear, and Grad-CAM heatmaps that highlight *which facial regions* triggered the model on the most suspicious frames.

The system is built to be honest about uncertainty. Rather than producing a single overconfident number, every verdict is accompanied by a confidence interval computed via bootstrap resampling and a temporal pattern classification that distinguishes a video that is suspicious throughout (CONSISTENT\_FAKE) from one where only a segment looks manipulated (PARTIAL\_FAKE). An analyst can look at the heatmaps and decide for themselves whether the highlighted regions are meaningful.

No GPU is required at any point. The entire inference pipeline runs on CPU using ONNX Runtime with graph-level optimisations, making this deployable on standard cloud instances without GPU surcharges.

---

## 🏗 Architecture Overview

The system follows a **producer-consumer** architecture. The FastAPI server accepts requests instantly and drops work onto a Redis-backed Celery queue. A separate worker process picks up jobs and runs the full ML pipeline. The React frontend polls for status updates and renders the final report when the job completes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User's Browser                             │
│                    React Dashboard (Port 80)                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │  HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Nginx Reverse Proxy (Port 80)                    │
│         Static files → /usr/share/nginx/html (React build)         │
│         /api/* requests → proxy_pass to FastAPI                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │  HTTP proxy
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               FastAPI + Uvicorn (Port 8000, 2 workers)              │
│    POST /api/v1/jobs/url   → creates job, queues Celery task        │
│    GET  /api/v1/jobs/{id}  → polls SQLite for status                │
│    GET  /api/v1/jobs/{id}/result → serves final_report.json         │
│    GET  /metrics           → Prometheus metrics                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │  Celery task message
                             ▼
┌───────────────┐   push/pop   ┌──────────────────────────────────────┐
│     Redis     │◄────────────►│        Celery Worker Process         │
│  (broker +    │              │  Stage 1: Download & validate        │
│   backend)    │              │  Stage 2: Frame extraction           │
└───────────────┘              │  Stage 3: Face detection & alignment │
                               │  Stage 4: ONNX inference             │
                               │  Stage 5: Temporal + Grad-CAM        │
                               └──────────────┬───────────────────────┘
                                              │  writes to shared volume
                                              ▼
                               ┌──────────────────────────────────────┐
                               │     data/results/{video_id}/         │
                               │       final_report.json              │
                               │       heatmaps/*.png                 │
                               └──────────────────────────────────────┘
```

The **API container** and **Worker container** share a named Docker volume (`pipeline-data`) mounted at `/app/data`. When the worker writes `final_report.json`, the API can read it immediately. The disk is the integration layer between the two processes — no serialisation overhead, no network round-trip, and files persist if either process restarts.

---

## 🔬 Pipeline Stages

Understanding each stage helps you debug problems, tune performance, and extend the system.

### Stage 1 — Video ingestion & validation

The `download_video()` function in `src/ingestion/downloader.py` accepts a URL or a local file path. For platform URLs (YouTube, Vimeo, TikTok, Twitter, Instagram, Twitch) it delegates to **yt-dlp** with a 720p-max format string to keep file sizes manageable. For direct `.mp4` / `.webm` URLs it uses a streaming HTTP download with `httpx`. After download, `ffprobe` probes the container metadata — duration, resolution, FPS, file size — and rejects videos that exceed configurable limits. The output is a typed `VideoMetadata` dataclass written to a `manifest.json` file.

### Stage 2 — Smart frame extraction

Rather than sampling every N seconds, the extractor in `src/ingestion/frame_extractor.py` uses **Gaussian-weighted scene-change detection**: it computes the mean absolute pixel difference between consecutive frames and keeps only frames where this difference exceeds a threshold. This concentrates the frame budget on moments where the face or scene actually changes — which is exactly where deepfake artifacts are most visible. For long talking-head videos it falls back to uniform sampling. Up to 120 frames are extracted per video, resized to 640×640 with letterbox padding, and saved as lossless PNG files.

### Stage 3 — Face detection, alignment & quality filtering

Each frame is processed by **MediaPipe FaceDetection** (primary, ~5ms/frame on CPU) with **MTCNN** as a fallback for challenging angles (~80ms/frame). For each detected face, a **5-point similarity transform** (left eye, right eye, nose tip, mouth corners) is estimated with `cv2.estimateAffinePartial2D` and applied to warp the face into a canonical 112×112 crop where the eyes always land at the same pixel coordinates. This alignment is critical — the EfficientNet-B4 model was trained on consistently-aligned faces, and misaligned inputs would silently degrade accuracy. Faces are then quality-filtered by detector confidence (≥ 0.5), minimum size (≥ 60px), and Laplacian variance sharpness score (≥ 50.0). Surviving crops are saved as PNG files alongside a `face_manifest.json`.

### Stage 4 — ONNX inference & confidence-weighted scoring

The model is **EfficientNet-B4 fine-tuned on FaceForensics++**, exported to ONNX format for CPU-optimised inference. ONNX Runtime with `ORT_ENABLE_ALL` graph optimisations (node fusion, constant folding) achieves roughly 200ms per face on a modern laptop CPU. Each face crop is preprocessed with ImageNet normalisation, batched in groups of 8, and passed through the model. The raw logit is divided by a calibration temperature (T = 1.5) before sigmoid activation to produce a well-calibrated P(fake) probability. Video-level scoring uses **confidence-weighted aggregation** — faces whose score is near the 0.5 decision boundary receive near-zero weight, preventing uncertain frames from diluting the signal from decisive ones.

### Stage 5 — Temporal analysis, Grad-CAM & confidence intervals

The scoring engine in `src/scoring/` performs three independent analyses on the per-frame P(fake) scores. **Temporal analysis** applies Gaussian smoothing and finds contiguous suspicious windows, computes a run-length score measuring whether fake frames cluster together (more suspicious) or appear scattered (less suspicious), and classifies the video as CONSISTENT\_FAKE, CONSISTENT\_REAL, PARTIAL\_FAKE, or INCONCLUSIVE. **Grad-CAM** runs a forward and backward pass through the PyTorch model on the top-5 highest-scoring frames, producing 14×14 activation maps that are resized and overlaid on the face crops as JET-colourmap heatmaps. **Bootstrap confidence intervals** resample the per-face score distribution 2,000 times to produce 90% and 95% intervals, with a human-readable interpretation of interval width. All outputs are assembled into `final_report.json`.

---

## 📁 Project Structure

```
deepfake-detector/
│
├── src/                          # All Python source code
│   ├── ingestion/
│   │   ├── downloader.py         # Video download + validation
│   │   └── frame_extractor.py    # Scene-change frame sampling
│   │
│   ├── detection/
│   │   ├── face_detector.py      # MediaPipe + MTCNN dual detector
│   │   ├── face_aligner.py       # 5-point similarity transform
│   │   ├── quality_filter.py     # Blur / size / confidence filter
│   │   ├── face_extractor.py     # Orchestrates detection pipeline
│   │   ├── preprocessor.py       # ImageNet normalisation for ONNX
│   │   ├── model_downloader.py   # Weight loading + ONNX export
│   │   ├── inference_engine.py   # ONNX Runtime inference + batching
│   │   ├── video_scorer.py       # Confidence-weighted aggregation
│   │   ├── dataset.py            # PyTorch Dataset for fine-tuning
│   │   └── trainer.py            # Fine-tuning loop + early stopping
│   │
│   ├── scoring/
│   │   ├── temporal_analyzer.py  # Sliding window + run-length analysis
│   │   ├── gradcam.py            # Grad-CAM heatmap generation
│   │   ├── confidence_estimator.py # Bootstrap confidence intervals
│   │   └── report_builder.py     # Assembles final_report.json
│   │
│   ├── api/
│   │   ├── app.py                # FastAPI application factory
│   │   ├── routes.py             # All endpoint handlers
│   │   ├── models.py             # Pydantic request/response types
│   │   ├── database.py           # SQLite job store
│   │   ├── auth.py               # API key dependency
│   │   ├── tasks.py              # Celery pipeline task
│   │   ├── celery_app.py         # Celery configuration
│   │   ├── limiter.py            # SlowAPI rate limiter
│   │   ├── metrics.py            # Prometheus instrumentation
│   │   └── middleware.py         # Request logging + timing
│   │
│   └── utils/
│       └── logger.py             # Structlog JSON logger
│
├── frontend/                     # React dashboard
│   ├── src/
│   │   ├── api/client.js         # All FastAPI calls
│   │   ├── hooks/useJobPolling.js# Auto-polling custom hook
│   │   ├── utils/formatting.js   # Display helpers
│   │   └── components/
│   │       ├── SubmitForm.jsx     # URL + drag-drop upload
│   │       ├── ProgressTracker.jsx# Pipeline stage display
│   │       ├── VerdictCard.jsx    # Main verdict display
│   │       ├── ConfidenceGauge.jsx# SVG arc gauge
│   │       ├── TemporalChart.jsx  # Recharts score timeline
│   │       ├── HeatmapViewer.jsx  # Heatmap grid + lightbox
│   │       ├── JobHistory.jsx     # Sidebar history list
│   │       └── StatusBadge.jsx    # Coloured status pill
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── nginx/
│   ├── nginx.conf                # Global Nginx configuration
│   └── default.conf              # Server block + proxy rules
│
├── tests/
│   ├── test_day1.py              # Ingestion pipeline tests
│   ├── test_day2.py              # Face detection tests
│   ├── test_day3.py              # Inference + dataset tests
│   ├── test_day4.py              # Temporal + confidence tests
│   ├── test_day5.py              # API route + database tests
│   └── test_smoke.py             # End-to-end smoke tests
│
├── scripts/
│   └── healthcheck.sh            # Full-stack health verification
│
├── config.py                     # Pydantic settings with .env support
├── cli.py                        # Click CLI for each pipeline stage
├── celery_worker.py              # Celery worker entry point
├── run_server.py                 # Uvicorn entry point
├── Dockerfile.api                # API container image
├── Dockerfile.worker             # Worker container image
├── docker-compose.yml            # Full stack orchestration
├── .env.example                  # Environment variable template
└── requirements.txt              # Python dependencies
```

---

## 🛠 Technology Stack

Every technology choice below was made deliberately. Understanding the reasoning helps you make informed decisions when you need to extend or swap components.

| Layer | Technology | Why this choice |
|---|---|---|
| **ML backbone** | EfficientNet-B4 (timm) | Best accuracy/compute trade-off for face classification; trained on ImageNet giving strong visual priors |
| **Training data** | FaceForensics++ | 1,000 videos across 4 manipulation methods; the de facto benchmark for deepfake detection |
| **Inference runtime** | ONNX Runtime | 2-3× faster than native PyTorch on CPU via graph fusion; no GPU dependency |
| **Face detection** | MediaPipe + MTCNN | MediaPipe is fast (5ms); MTCNN is accurate on difficult angles — complementary strengths |
| **Video download** | yt-dlp | Supports 1,000+ platforms; actively maintained; handles format negotiation automatically |
| **Video processing** | OpenCV (headless) | Industry standard for frame I/O; no display dependency in server environments |
| **API framework** | FastAPI | Automatic OpenAPI docs; Pydantic validation; async-native; significantly faster than Flask |
| **Task queue** | Celery + Redis | Battle-tested for ML workloads; supports retries, rate limits, and worker concurrency control |
| **Database** | SQLite | Zero infrastructure for single-machine deployment; trivial to swap to PostgreSQL later |
| **Frontend** | React + Vite + Tailwind | Fast HMR in development; CSS-in-JS avoids stylesheet conflicts; Tailwind for design consistency |
| **Charts** | Recharts | Composable SVG charts; renders on server-side without a canvas; good TypeScript types |
| **Reverse proxy** | Nginx | Serves static files from disk (fastest possible path); keeps Python server free for API work |
| **Observability** | Prometheus + structlog | Machine-readable metrics; JSON logs compatible with Grafana Loki, ELK, CloudWatch |
| **Containerisation** | Docker Compose | Reproducible builds; shared named volumes; service health dependencies |

---

## ✅ Prerequisites

You need the following installed on your machine before cloning and running the project. The version numbers are the ones the project was built and tested with — later versions will likely work but are untested.

**System dependencies** that must be installed at the OS level, not through pip:

- **Python 3.11** — the project uses `match` statements and `tomllib` from the standard library that require 3.11+
- **ffmpeg** (which includes ffprobe) — used for video probing and frame extraction. Install with `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu, or from [ffmpeg.org](https://ffmpeg.org/download.html) on Windows
- **Docker Desktop** (or Docker Engine + Compose plugin) — required for the containerised deployment. [Install Docker](https://docs.docker.com/get-docker/)
- **Node.js 20+** — required only to build the React frontend. Install with `brew install node` or from [nodejs.org](https://nodejs.org/)

Verify your installations with:

```bash
python3 --version      # should print Python 3.11.x
ffprobe -version       # should print ffprobe version 6.x or later
docker --version       # should print Docker version 24.x or later
node --version         # should print v20.x or later
```

---

## 🚀 Quick Start

The fastest path to a running system, assuming all prerequisites are installed.

**Clone the repository and set up the Python environment:**

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Set up environment variables by copying the example file:**

```bash
cp .env.example .env
# Open .env and replace the default API key with a real secret:
# API_KEY=your-secret-key-here
```

**Build the React frontend:**

```bash
cd frontend
npm install
npm run build        # produces frontend/dist/ — served by Nginx
cd ..
```

**Start the full Docker stack:**

```bash
docker compose up --build
# First run takes 3–8 minutes to download images and install dependencies.
# Subsequent starts take ~30 seconds thanks to Docker layer caching.
```

**Verify everything is healthy:**

```bash
bash scripts/healthcheck.sh
```

**Open the dashboard** at [http://localhost](http://localhost) — you should see the analysis interface. Paste a YouTube URL and click **Analyze video** to run your first analysis.

---

## 🎓 Fine-tuning the Model

The system ships with support for fine-tuning EfficientNet-B4 on your own deepfake dataset. Fine-tuning is a one-time operation that you run locally (or on a free Google Colab GPU), save the weights, and then those weights are used for all subsequent inference.

**Step 1 — Prepare your labelled data** by running the ingestion and face extraction pipeline on your real and fake videos:

```bash
# For each real (authentic) video:
python cli.py ingest /path/to/real_video.mp4
python cli.py detect-faces <video_id>

# For each fake (deepfake) video:
python cli.py ingest /path/to/fake_video.mp4
python cli.py detect-faces <video_id>
```

**Step 2 — Run fine-tuning** by pointing the `finetune` command at the resulting manifest files. You need at least a few dozen videos in each class for meaningful results; 200+ per class is recommended for production accuracy:

```bash
python cli.py finetune \
  --real-manifests data/face_crops/abc1/face_manifest.json \
  --real-manifests data/face_crops/abc2/face_manifest.json \
  --fake-manifests data/face_crops/def1/face_manifest.json \
  --fake-manifests data/face_crops/def2/face_manifest.json \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-4
```

The training loop saves the best checkpoint (by validation AUC) to `models/deepfake_efficientb4_finetuned.pt` and applies early stopping after 3 epochs of no improvement.

**Step 3 — Export the fine-tuned weights to ONNX** so the inference engine can use them:

```bash
python cli.py infer <any_video_id> --export-onnx
# This overwrites models/deepfake_efficientb4.onnx with your fine-tuned weights.
```

All subsequent inference calls — from the CLI or the API — will now use your fine-tuned model.

---

## 📡 API Reference

The API is self-documenting. With the server running, visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI or [http://localhost:8000/redoc](http://localhost:8000/redoc) for the ReDoc interface. Both show every endpoint with request schemas, response schemas, and a **Try it out** button.

All endpoints except `/api/v1/health` require the `X-API-Key` header.

---

### `POST /api/v1/jobs/url`

Submit a platform video URL for analysis. Returns immediately with a `job_id` — the analysis runs asynchronously.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

**Response `202 Accepted`:**
```json
{
  "job_id":     "550e8400-e29b-41d4-a716-446655440000",
  "status":     "PENDING",
  "source":     "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "video_id":   null,
  "created_at": 1700000000.0,
  "updated_at": 1700000000.0
}
```

---

### `POST /api/v1/jobs/upload`

Upload a local video file. Accepts `multipart/form-data` with a `file` field.

```bash
curl -X POST http://localhost:8000/api/v1/jobs/upload \
  -H "X-API-Key: your-key" \
  -F "file=@/path/to/video.mp4"
```

---

### `GET /api/v1/jobs/{job_id}`

Poll the current status of a submitted job. Clients should call this every 3–5 seconds until `status` is `DONE` or `FAILED`.

**Job status lifecycle:**

```
PENDING → DOWNLOADING → EXTRACTING_FRAMES → DETECTING_FACES → RUNNING_INFERENCE → ANALYZING → DONE
                                                                                              ↘ FAILED
```

---

### `GET /api/v1/jobs/{job_id}/result`

Retrieve the complete analysis report for a finished job. The `report` field is `null` while the job is still running.

**Response `200 OK` (job complete):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "DONE",
  "report": {
    "video_id":           "a3b9f2c1d4e5",
    "verdict":            "DEEPFAKE",
    "weighted_prob_fake": 0.8741,
    "mean_prob_fake":     0.8523,
    "overall_confidence": 0.7482,
    "confidence_interval": {
      "point_estimate": 0.8523,
      "ci_lower_95":    0.7901,
      "ci_upper_95":    0.9104,
      "interpretation": "The model predicts 'DEEPFAKE' with high certainty. ..."
    },
    "temporal": {
      "temporal_verdict":       "CONSISTENT_FAKE",
      "suspicious_frame_ratio": 0.8333,
      "run_length_score":       4.21,
      "peak_frame_idx":         42,
      "peak_score":             0.9812,
      "suspicious_windows": [
        { "start_frame": 0, "end_frame": 99, "mean_prob_fake": 0.87, "frame_count": 100 }
      ]
    },
    "heatmaps": [
      {
        "frame_idx":    42,
        "prob_fake":    0.9812,
        "heatmap_path": "data/results/.../heatmaps/frame_000042_heatmap.png",
        "overlay_path": "data/results/.../heatmaps/frame_000042_overlay.png"
      }
    ],
    "total_faces_scored": 98,
    "elapsed_seconds":    87.3
  }
}
```

---

### `GET /api/v1/jobs/{job_id}/heatmaps/{filename}`

Serve a Grad-CAM heatmap PNG. The `filename` values come from the `heatmaps` array in the result response. Images are served with appropriate cache headers.

---

### `GET /api/v1/health`

Liveness probe. Returns `200 {"status": "ok"}`. No authentication required. Used by Docker health checks and load balancers.

---

### `GET /metrics`

Prometheus metrics in text exposition format. Exposes counters for requests, job submissions, verdicts, and histograms for request latency and pipeline duration.

---

## 🖥 Frontend Dashboard

The React dashboard is an industrial-aesthetic analysis terminal built with React 18, Vite, and Tailwind CSS. It uses the `IBM Plex Mono` font for data displays, `Bebas Neue` for the verdict headline, and a dark charcoal colour scheme with amber accents for suspicious content and emerald for authentic content.

**The dashboard has six panels:**

The **Submit Form** accepts a URL (YouTube, Vimeo, TikTok, or any direct `.mp4` link) via a text input, or a video file via drag-and-drop. Switching between the two modes is done with a tab toggle.

The **Progress Tracker** shows the five pipeline stages as a live-updating checklist. Completed stages show a checkmark; the active stage shows a spinner with the stage name highlighted.

The **Verdict Card** is the centrepiece of the results view — a large typographic verdict (REAL / DEEPFAKE / INCONCLUSIVE) with the weighted probability, confidence interval, face count, and temporal pattern as supporting statistics below it.

The **Confidence Gauge** is a hand-coded SVG arc gauge showing P(fake) as a needle. The needle animates with a spring-physics transition when the result loads, and the arc colour interpolates from green (safe) through amber to red as the probability rises.

The **Temporal Chart** uses Recharts to display two overlaid line series — the raw per-frame scores and the Gaussian-smoothed version — over a timeline indexed by frame number. Suspicious windows are marked with vertical amber reference lines and summarised below the chart.

The **Heatmap Viewer** is a responsive grid of the top-N most suspicious frame overlays. Clicking any thumbnail opens a lightbox showing the raw JET-colourmap heatmap and the face+heatmap blend side by side, with a short explanation of what hot regions mean in the context of deepfake detection.

---

## ⚙️ Configuration

All settings are defined as a Pydantic `Settings` class in `config.py` and can be overridden via environment variables prefixed with `DEEPFAKE_`. Copy `.env.example` to `.env` and edit as needed.

| Variable | Default | Description |
|---|---|---|
| `DEEPFAKE_MAX_FRAMES` | `120` | Maximum frames extracted per video |
| `DEEPFAKE_MIN_FRAMES` | `20` | Minimum frames (enforced for short clips) |
| `DEEPFAKE_SCENE_THRESHOLD` | `30.0` | Pixel diff threshold for scene-change detection |
| `DEEPFAKE_TARGET_FPS` | `2.0` | Fallback sampling rate for uniform extraction |
| `DEEPFAKE_MAX_VIDEO_DURATION_SECONDS` | `600` | Reject videos longer than 10 minutes |
| `DEEPFAKE_MAX_VIDEO_SIZE_MB` | `500` | Reject files larger than 500 MB |
| `DEEPFAKE_API_KEYS` | `["dev-key-..."]` | JSON array of valid API keys |
| `DEEPFAKE_CELERY_BROKER_URL` | `redis://localhost:6379/0` | Redis URL for the task queue |
| `DEEPFAKE_CELERY_BACKEND_URL` | `redis://localhost:6379/1` | Redis URL for task results |
| `DEEPFAKE_LOG_FORMAT` | `json` | `json` for production, `console` for development |
| `DEEPFAKE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

---

## 🧪 Running Tests

The test suite is split into **unit tests** (fast, no dependencies, use mocks) and **smoke tests** (slow, require the full Docker stack running).

**Run all unit tests** across all seven pipeline days:

```bash
pytest tests/ -v -m "not smoke"
```

**Run tests for a specific day** — useful when you are actively developing a module:

```bash
pytest tests/test_day1.py -v    # ingestion tests
pytest tests/test_day2.py -v    # face detection tests
pytest tests/test_day3.py -v    # inference + dataset tests
pytest tests/test_day4.py -v    # temporal + confidence tests
pytest tests/test_day5.py -v    # API route + database tests
```

**Run the integration test** that creates a synthetic video and runs the full Day 1 pipeline against it — requires ffmpeg installed:

```bash
pytest tests/test_day1.py -v -m integration
```

**Run smoke tests** against a running Docker stack. These make real HTTP requests and verify the full system end-to-end:

```bash
# Start the stack first:
docker compose up -d

# Then run smoke tests (excludes the slow full-pipeline test):
pytest tests/test_smoke.py -v -m "smoke and not slow"

# Or run the complete end-to-end test including a real video analysis (~10 min):
pytest tests/test_smoke.py -v -m smoke
```

---

## 📊 Observability

**Structured logging** — every log line is emitted as a JSON object with `timestamp`, `level`, `logger`, `event`, and any contextual fields bound during that request. In development, set `DEEPFAKE_LOG_FORMAT=console` for human-readable coloured output. In production the JSON format is directly ingestible by Grafana Loki, Elasticsearch, AWS CloudWatch Logs, and Datadog.

**Prometheus metrics** — scraped from `GET /metrics`. The following metrics are available:

```
http_requests_total{method, endpoint, status_code}          # counter
http_request_duration_seconds{method, endpoint}             # histogram
jobs_submitted_total                                        # counter
jobs_completed_total{final_status}                          # counter
verdict_total{verdict}                                      # counter
jobs_in_progress                                            # gauge
inference_duration_seconds                                  # histogram
```

**Request tracing** — every response includes an `X-Request-ID` header containing an 8-character correlation ID. Any log line emitted during that request includes the same ID, making it trivial to pull all logs for a specific request from your log aggregation system.

**Container health checks** — the API container checks `GET /api/v1/health` every 30 seconds. The worker container checks `celery inspect ping` every 60 seconds. Docker Compose respects these when starting dependent services, and a container management system (ECS, Kubernetes) will automatically restart unhealthy containers.

---

## 🌐 Production Deployment

The Docker Compose setup is production-capable for single-machine deployments. For a full production environment, consider the following additional steps.

**Generate a strong API key** before deploying to any public-facing server:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Copy the output into your .env file as API_KEY=...
```

**Enable HTTPS** by adding a Certbot/Let's Encrypt container to `docker-compose.yml` and updating `nginx/default.conf` to listen on port 443 with `ssl_certificate` and `ssl_certificate_key` directives. The [nginx-certbot](https://github.com/JonasAlfredsson/docker-nginx-certbot) Docker image handles certificate renewal automatically.

**Limit Nginx to internal traffic only** by removing the `ports: - "8000:8000"` line from the `api` service in `docker-compose.yml`. The API should only be reachable through Nginx in production — direct access bypasses rate limiting and security headers.

**Scale the worker horizontally** by increasing Celery concurrency or by running multiple worker containers:

```bash
docker compose up --scale worker=3
```

**Back up the pipeline-data volume** regularly, as it contains all analysis results and SQLite database:

```bash
docker run --rm \
  -v deepfake-detector_pipeline-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/pipeline-data-$(date +%Y%m%d).tar.gz /data
```

---

## ⏱ Performance Benchmarks

All benchmarks measured on a 2023 MacBook Pro (M2, 16GB RAM) using CPU-only inference (no GPU, no Apple Neural Engine), processing a 2-minute 720p video.

| Stage | Time | Notes |
|---|---|---|
| Download (YouTube 720p) | 15–45s | Network-dependent |
| Frame extraction (120 frames) | 8–15s | Scene-change mode |
| Face detection (120 frames) | 25–60s | MediaPipe primary |
| ONNX inference (100 faces, batch 8) | 20–35s | ~250ms/face avg |
| Temporal + Grad-CAM + CI | 8–15s | Top-5 heatmaps |
| **Total pipeline** | **76–170s** | **~2 minutes typical** |

For a 30-second social media clip, total pipeline time is typically under 60 seconds. For a 10-minute video at the maximum duration limit, expect 5–8 minutes.

---

## ⚠️ Limitations & Honest Caveats

Every production system deserves an honest limitations section. This project is strong within its scope and has genuine blind spots outside it.

**Training distribution generalisation** — the default model was fine-tuned on FaceForensics++ which covers four specific manipulation methods from 2018–2020 (Deepfakes face-swap, Face2Face expression transfer, FaceSwap, NeuralTextures). Newer generation methods based on diffusion models (2022+) use fundamentally different synthesis mechanisms and may not produce the same pixel-level artifacts the model learned to detect. Fine-tuning on recent deepfake examples is strongly recommended for production use.

**Compression sensitivity** — deepfake detection relies heavily on subtle frequency-domain artifacts. Heavy social media compression (Instagram, TikTok at high compression settings) can destroy these artifacts before the model sees them, pushing scores toward 0.5 (uncertain) even for genuine fakes. The model is most reliable on lightly compressed or uncompressed source material.

**Single-face assumption** — the pipeline takes the top-3 faces by confidence score per frame. Videos with crowds, panels, or rapidly switching speakers will have their face budget split across multiple people, potentially diluting the signal for any individual face.

**No audio analysis** — deepfake audio (voice cloning) is not detected. The system analyses only visual information. A video with authentic visuals but cloned audio would be classified as REAL by this system.

**Verdict is probabilistic, not definitive** — the system provides a probability estimate with a confidence interval, not a legal determination. Verdicts should be treated as analytical input to human review, not as conclusive evidence of manipulation.

---

## 📅 7-Day Build Log

This project was built over seven days at 2–3 hours per day with no GPU hardware. Below is a summary of what was implemented on each day.

**Day 1 — Project setup & video ingestion.** Scaffolded the project structure, configured the Python environment, built the video downloader supporting YouTube/Vimeo/TikTok/direct URLs via yt-dlp, implemented smart scene-change frame extraction with Gaussian sampling, and wrote the first CLI commands and unit tests.

**Day 2 — Face detection & extraction pipeline.** Implemented the dual-detector system (MediaPipe primary + MTCNN fallback), built the 5-point landmark similarity transform alignment system targeting FaceForensics++ coordinate standards, added the quality filter (confidence, size, sharpness), and wired everything into a manifest-driven orchestrator.

**Day 3 — Pre-trained model & fine-tuning infrastructure.** Replaced the broken Day 3 placeholder with an honest architecture — EfficientNet-B4 from timm with ImageNet pretrained weights. Built the ONNX export pipeline, ImageNet-standard preprocessor, ONNX Runtime inference engine with temperature calibration, confidence-weighted video-level scoring, and the complete PyTorch fine-tuning loop with layer freezing, class imbalance handling, and early stopping.

**Day 4 — Temporal analysis, Grad-CAM & confidence intervals.** Built the Gaussian-smoothed temporal score analyser with suspicious window detection and run-length clustering score. Implemented Grad-CAM with hook-based activation/gradient capture on EfficientNet-B4's `conv_head` layer. Added bootstrap confidence intervals. Assembled everything into `final_report.json`.

**Day 5 — FastAPI backend & async job queue.** Built the complete REST API with URL submission, file upload, job polling, and result retrieval endpoints. Added Celery + Redis async task queue, SQLite job store with COALESCE update logic, API key authentication, and SlowAPI rate limiting.

**Day 6 — React frontend dashboard.** Built the industrial-aesthetic React dashboard with the submit form (URL + drag-drop), animated pipeline progress tracker, verdict card with bootstrap CI display, SVG arc confidence gauge, Recharts temporal score timeline with suspicious window markers, Grad-CAM heatmap grid with lightbox, and session-persisted job history sidebar.

**Day 7 — Docker, Nginx, observability & production hardening.** Containerised the API and Worker with layer-optimised Dockerfiles. Wrote the Nginx configuration with JSON access logs, gzip, security headers, static file caching, and API proxy. Added Prometheus metrics, structured request logging middleware with path normalisation, container health checks, and end-to-end smoke tests.

---

## 🤝 Contributing

Contributions are welcome and appreciated. The most valuable contributions are additional training data, improved model architectures, and real-world test cases that reveal edge cases in the detection pipeline.

Before submitting a pull request, please run the full test suite to confirm nothing is broken:

```bash
pytest tests/ -v -m "not smoke"
```

Please write tests for any new functionality. The project maintains one test file per pipeline day — new detection methods belong in `tests/test_day2.py`, new scoring methods in `tests/test_day4.py`, and so on.

For major changes (new ML architectures, database schema changes, API breaking changes), please open an issue first to discuss the approach before investing time in implementation.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The FaceForensics++ dataset used for fine-tuning is available for academic research only under a separate license agreement. See the [FaceForensics++ repository](https://github.com/ondyari/FaceForensics) for access instructions.

---

<div align="center">

Built over 7 days · CPU-only · No GPU required

**[View API Docs](http://localhost:8000/docs)** · **[Report an Issue](https://github.com/your-username/deepfake-detector/issues)** · **[Request a Feature](https://github.com/your-username/deepfake-detector/issues)**

</div>
