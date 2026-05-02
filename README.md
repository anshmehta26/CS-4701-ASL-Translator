# Real-Time American Sign Language Recognition

CS 4701 — Practicum in Artificial Intelligence (Spring 2026)

**Team:** Shashank Kalyanaraman (ssk252), Saurav Tewari (st782), Ansh Mehta (am2555)

A real-time ASL recognizer that translates fingerspelled letters and a
~50-sign vocabulary from live webcam input. The system extracts MediaPipe
hand landmarks, models temporal dynamics with an LSTM (and a Transformer
baseline for comparison), and runs end-to-end at < 300 ms latency on a
laptop CPU.

---

## Project layout

```
asl-translator/
├── configs/
│   └── config.yaml               # all hyperparameters in one place
├── scripts/
│   ├── prepare_dataset.py        # build manifest.csv from WLASL JSON
│   ├── extract_landmarks.py      # run MediaPipe over videos -> .npz
│   ├── make_synthetic_data.py    # generate fake data for smoke tests
│   ├── train.py                  # train LSTM or Transformer
│   ├── evaluate.py               # test-set top-k + confusion matrix
│   ├── benchmark_latency.py      # per-prediction timing
│   └── realtime_demo.py          # live webcam demo
├── src/
│   ├── data/
│   │   ├── wlasl.py              # parse WLASL_v0.3.json -> manifest
│   │   ├── landmarks.py          # MediaPipe extractor + preprocessing
│   │   ├── augmentation.py       # flip / noise / time-warp / drop
│   │   └── dataset.py            # PyTorch Dataset + pad_collate
│   ├── models/
│   │   ├── lstm.py               # biLSTM classifier
│   │   ├── transformer.py        # Transformer encoder classifier
│   │   └── factory.py            # build_model("lstm" | "transformer")
│   ├── training/
│   │   ├── trainer.py            # main training loop (CE or CTC)
│   │   └── metrics.py            # top-k, confusion matrix, running avg
│   ├── inference/
│   │   └── realtime.py           # webcam loop + sliding window predictor
│   ├── evaluation/
│   │   ├── evaluate.py           # full test-split evaluation report
│   │   └── latency.py            # forward-pass timing benchmark
│   └── utils/                    # config loader, logging, seeding
├── tests/                        # unit tests (pytest-compatible)
├── requirements.txt
└── README.md
```

---

## Setup

Requires Python 3.10 or newer.

```bash
git clone <repo-url>
cd asl-translator

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **MediaPipe note**: MediaPipe wheels are available for Python 3.9–3.12 on
> macOS / Linux / Windows. Apple Silicon Macs need the `mediapipe` wheel
> (not `mediapipe-silicon`); recent versions support arm64 natively.

---

## Quick smoke test (no real data needed)

To verify the whole pipeline without downloading WLASL:

```bash
python scripts/make_synthetic_data.py --classes 10 --per-class 30
python scripts/train.py --model lstm
python scripts/evaluate.py --checkpoint checkpoints/lstm_best.pt
python scripts/benchmark_latency.py --checkpoint checkpoints/lstm_best.pt
```

The synthetic data has class-specific signatures embedded, so a working
LSTM will reach > 95% top-1 within a few epochs.

---

## Full pipeline (real WLASL data)

### 1. Get the data

Download `WLASL_v0.3.json` from the [WLASL repo](https://github.com/dxli94/WLASL)
and place it at `data/raw/WLASL_v0.3.json`. Run their video downloader (or
your own) so video files end up under `data/raw/videos/{video_id}.mp4`.

### 2. Build the manifest

```bash
python scripts/prepare_dataset.py
```

This selects the top 50 most-frequent glosses (configurable in
`configs/config.yaml`) that have at least 7 instances each, and writes
`data/processed/manifest.csv` plus `label_map.json`.

### 3. Extract landmarks

```bash
python scripts/extract_landmarks.py
```

Caches MediaPipe landmark sequences to `data/processed/landmarks/{vid}.npz`.
This is the slowest stage — it only runs once. Pass `--limit 50` for a quick
trial.

### 4. Train

```bash
# LSTM (proposal's primary architecture)
python scripts/train.py --model lstm

# Transformer (comparison architecture)
python scripts/train.py --model transformer

# Optional: train with CTC loss instead of cross-entropy
python scripts/train.py --model lstm --ctc
```

Checkpoints land in `checkpoints/`, training history in `logs/`, and curves
in `figures/`.

### 5. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/lstm_best.pt
python scripts/evaluate.py --checkpoint checkpoints/transformer_best.pt
```

Produces top-1 / top-5 accuracy, per-class accuracy, a confusion-matrix PNG
under `figures/`, the raw confusion matrix CSV in `logs/`, and the 10
most-confused class pairs in stdout.

### 6. Latency benchmark

```bash
python scripts/benchmark_latency.py --checkpoint checkpoints/lstm_best.pt
```

Reports mean / median / p95 / p99 forward-pass time and whether the model
meets the < 300 ms target.

### 7. Live webcam demo

```bash
python scripts/realtime_demo.py --checkpoint checkpoints/lstm_best.pt
```

Press **q** to quit. The display shows the top-3 predictions, smoothed
probabilities, current FPS, and per-prediction latency.

---

## How it works (one paragraph per stage)

**Landmark extraction.** Each video frame is passed through MediaPipe Hands,
which returns 21 3D keypoints per detected hand. We keep up to two hands per
frame, so each frame becomes a 126-dim vector (`2 * 21 * 3`). Frames where
MediaPipe detects nothing leave NaNs; we linearly interpolate them along the
time axis. We drop a clip entirely if more than 40% of its frames had no
detection.

**Per-clip normalization.** Each hand's wrist (joint 0) is shifted to the
origin and the hand is scaled by the diagonal of its 3D bounding box. This
makes the features invariant to the signer's distance from the camera and
their position in the frame. A small centered moving-average smooths out
MediaPipe's per-frame jitter.

**Augmentation.** During training only: horizontal flip (mirrors x and swaps
left/right hand slots), small Gaussian noise, time warping (resample faster
or slower), and random per-frame drops. All augmentations preserve the
sequence's class.

**Variable-length training.** Sequences are kept at their natural length
(up to a 96-frame cap) and padded only at batch boundaries. The LSTM uses
`pack_padded_sequence` so padding is fully ignored by the recurrence; the
Transformer uses a key-padding mask. Clip embeddings come from a masked
mean-pool over time.

**CTC mode.** Optional alternative training mode for sequence-level
fingerspelling outputs, where each sign is treated as a single label and the
model learns to align over time without per-frame supervision. Toggle with
`training.ctc_mode: true` in the config or `--ctc` on the command line.

**Real-time inference.** A sliding window of the last 32 landmark vectors
is maintained as the webcam streams. Every 4 frames we re-run preprocessing
on the window and predict. Predicted probabilities are exponentially
smoothed across windows so the displayed label doesn't flicker. MediaPipe
runs once per frame and its output is cached, so the only repeated work is
the cheap normalization + the model forward pass — well under the 300 ms
budget on CPU for a 256-hidden biLSTM.

---

## Evaluation methodology (per the proposal)

1. **Top-1 / Top-5 accuracy** on the held-out test split (15% of clips,
   stratified by class).
2. **Confusion matrix analysis** to understand which visually-similar signs
   the model confuses (saved as both PNG and CSV).
3. **Latency benchmarking** (mean / median / p95) of the forward pass, used
   as a tie-breaker between the LSTM and Transformer architectures.
4. **Cross-subject generalization** can be enabled by setting
   `dataset.cross_subject: true` (requires signer-id metadata in the manifest)
   to ensure the model isn't memorizing individual signers.

---

## Tests

```bash
pytest tests/ -v
```

Tests cover augmentation, landmark preprocessing helpers, dataset/collate
correctness, and forward-pass shape invariants for both models.

---

## License & acknowledgments

Course project for CS 4701 (Cornell, Spring 2026). Uses the
[WLASL](https://dxli94.github.io/WLASL/) dataset and
[MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).
