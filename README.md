# Real-Time American Sign Language Recognition

CS 4701 — Practicum in Artificial Intelligence (Spring 2026)

**Team:** Shashank Kalyanaraman (ssk252), Saurav Tewari (st782), Ansh Mehta (am2555)

A real-time ASL recognizer that translates fingerspelled letters (A–Z plus
space / delete / nothing) from a live webcam feed, with the option to extend
to dynamic word-level signs using WLASL.

The system extracts MediaPipe hand landmarks per frame, models them with a
biLSTM (with a Transformer baseline for comparison), and runs end-to-end at
< 300 ms latency on a laptop CPU.

---

## Two project tracks

The repo supports two datasets out of the box, controlled by which config you pass:

| Track | Dataset | Config | What it recognizes |
|---|---|---|---|
| **Primary — fingerspelling** | [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | `configs/alphabet.yaml` | Static letters A–Z plus `space`, `del`, `nothing` |
| **Stretch — dynamic signs** | [WLASL](https://github.com/dxli94/WLASL) | `configs/config.yaml` | Word-level dynamic signs (50-class subset) |

Start with the alphabet track — it's faster to set up, demos cleanly as a
"sign-to-text typer", and aligns with the primary milestone in our status
report.

---

## Project layout

```
asl-translator/
├── configs/
│   ├── alphabet.yaml             # fingerspelling config (start here)
│   └── config.yaml               # WLASL config (stretch goal)
├── scripts/
│   ├── prepare_alphabet.py       # alphabet: manifest + landmark extraction in one go
│   ├── prepare_dataset.py        # WLASL: build manifest from WLASL_v0.3.json
│   ├── extract_landmarks.py      # WLASL: run MediaPipe over videos
│   ├── make_synthetic_data.py    # generate fake data for smoke tests
│   ├── train.py                  # train LSTM or Transformer
│   ├── evaluate.py               # test-set top-k + confusion matrix
│   ├── benchmark_latency.py      # per-prediction timing
│   ├── realtime_demo.py          # generic top-K webcam display
│   └── realtime_typing.py        # fingerspelling-to-text webcam demo
├── src/
│   ├── data/
│   │   ├── alphabet.py           # Kaggle ASL Alphabet loader
│   │   ├── wlasl.py              # WLASL JSON parser
│   │   ├── landmarks.py          # MediaPipe extractor (video + image)
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
├── tests/                        # 27 unit tests (pytest)
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
> (recent versions support arm64 natively).

---

## Smoke test (no real data needed)

To verify the whole pipeline without downloading anything real:

```bash
python scripts/make_synthetic_data.py --classes 10 --per-class 30
python scripts/train.py --model lstm
python scripts/evaluate.py --checkpoint checkpoints/lstm_best.pt
python scripts/benchmark_latency.py --checkpoint checkpoints/lstm_best.pt
```

Reaches > 95% top-1 within a few epochs on the synthetic class signatures.

---

## Track 1 — Fingerspelling (primary)

### 1. Get the data

Download from
[Kaggle: ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
(requires a free Kaggle account — easiest is the "Download" button on the
dataset page). Unzip into the project so you end up with:

```
data/raw/asl_alphabet_train/
    A/A1.jpg, A/A2.jpg, ...
    B/...
    ...
    space/...
    del/...
    nothing/...
```

The Kaggle ZIP is sometimes nested as `asl_alphabet_train/asl_alphabet_train/`;
the prepare script auto-detects either layout.

### 2. Build manifest + extract landmarks

```bash
python scripts/prepare_alphabet.py --config configs/alphabet.yaml
```

By default this caps to 500 images per class (≈ 14k total). MediaPipe
processes ~30–50 images/sec on a laptop CPU, so this takes 5–10 minutes.
Bump the cap with `--cap-per-class 1000` for a bigger training set.

### 3. Train

```bash
python scripts/train.py --model lstm --config configs/alphabet.yaml
```

Expected: > 95% top-1 on the test split within 25 epochs.

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/lstm_best.pt --config configs/alphabet.yaml
```

Saves a confusion matrix PNG under `figures/` and a JSON summary under `logs/`.

### 5. Live typing demo

```bash
python scripts/realtime_typing.py --checkpoint checkpoints/lstm_best.pt
```

Point your webcam at your hand and fingerspell. Letters commit to the typed
buffer once the model has stably predicted them across multiple frames.
Drop your hand briefly between letters as a separator (the `nothing` class
acts as an automatic word boundary).

**Keys:** `q` quit, `c` clear buffer, `b` backspace.

---

## Track 2 — Dynamic signs (stretch goal, WLASL)

### 1. Get the data

Download `WLASL_v0.3.json` from
[the WLASL repo](https://github.com/dxli94/WLASL) and place it at
`data/raw/WLASL_v0.3.json`. Run their video downloader (or your own) so
video files end up under `data/raw/videos/{video_id}.mp4`.

> Heads up: WLASL videos are scraped from YouTube, and a meaningful fraction
> are now unavailable. Expect to end up with ~60–80% of the dataset.

### 2. Build the manifest

```bash
python scripts/prepare_dataset.py
```

Selects the top 50 most-frequent glosses (configurable in
`configs/config.yaml`) that have at least 7 instances each.

### 3. Extract landmarks (slow — runs once)

```bash
python scripts/extract_landmarks.py
```

### 4. Train, evaluate, demo

```bash
python scripts/train.py --model lstm
python scripts/evaluate.py --checkpoint checkpoints/lstm_best.pt
python scripts/realtime_demo.py --checkpoint checkpoints/lstm_best.pt
```

The generic `realtime_demo.py` shows top-K predictions; it works for both
tracks but is a better fit for word-level signs (where typing-style commit
logic doesn't apply).

---

## How it works (one paragraph per stage)

**Landmark extraction.** Each video frame (or image, for the alphabet track)
is passed through MediaPipe Hands, which returns 21 3D keypoints per
detected hand. The alphabet track keeps one hand per frame (63 dims); the
WLASL track keeps up to two hands (126 dims). For video, frames where
MediaPipe fails are linearly interpolated along the time axis; for stills,
samples with no detection are simply dropped.

**Per-clip normalization.** Each hand's wrist (joint 0) is shifted to the
origin and the hand is scaled by the diagonal of its 3D bounding box. This
makes features invariant to the signer's distance from the camera and
position in the frame. For video, a small centered moving-average smooths
out MediaPipe's per-frame jitter.

**Augmentation.** During training only: horizontal flip (mirrors x and
swaps left/right hand slots when 2-hand mode is on), small Gaussian noise,
time warping (resample faster or slower), and random per-frame drops. All
augmentations preserve the sample's class. The alphabet config disables
time warping and frame drops since the samples are static.

**Variable-length training.** Sequences are kept at their natural length up
to a per-config cap and padded only at batch boundaries. The LSTM uses
`pack_padded_sequence` so padding is fully ignored by the recurrence; the
Transformer uses a key-padding mask. Clip embeddings come from a masked
mean-pool over time. For the alphabet track, single-frame samples are
replicated up to `min_seq_len` so the model sees the same input
distribution at train- and inference-time (where the sliding window will
contain mostly identical frames as the user holds a letter).

**CTC mode.** Optional alternative training mode for sequence-level
fingerspelling outputs, where the model learns to align over time without
per-frame supervision. Toggle via `training.ctc_mode: true` or `--ctc`.

**Real-time inference.** A sliding window of the last N landmark vectors is
maintained as the webcam streams. Every `stride` frames we re-run
preprocessing on the window and predict. Predicted probabilities are
exponentially smoothed across windows so the displayed label doesn't
flicker. MediaPipe runs once per frame and its output is cached, so the
only repeated work is the cheap normalization + the model forward pass —
under 15 ms on CPU for both architectures.

**Typing demo logic.** For fingerspelling, a letter only commits to the
typed string once the same prediction is the top-1 for several consecutive
windows AND its smoothed confidence is above a threshold. The `nothing`
class acts as a separator (drop your hand to "release" the current letter),
and a cooldown prevents accidentally typing duplicates while you hold a
single letter for too long. The state machine is unit-tested.

---

## Evaluation methodology (per the proposal)

1. **Top-1 / Top-5 accuracy** on the held-out test split (10–15% of samples,
   stratified by class).
2. **Confusion matrix analysis** to understand which visually-similar signs
   the model confuses (saved as both PNG and CSV). Useful here for finding
   alphabet pairs like `M / N` or `R / U` that are easy to confuse.
3. **Latency benchmarking** (mean / median / p95) of the forward pass, used
   as a tie-breaker between the LSTM and Transformer architectures.
4. **Cross-subject generalization** can be enabled by setting
   `dataset.cross_subject: true` (requires signer-id metadata) on the WLASL
   track to verify the model isn't memorizing individual signers.

---

## Tests

```bash
pytest tests/ -v
```

Tests cover augmentation, landmark preprocessing helpers, dataset/collate
correctness, both model forward-pass shapes, alphabet manifest building,
and the typing state machine.

```
27 passed
```

---

## License & acknowledgments

Course project for CS 4701 (Cornell, Spring 2026). Uses the
[Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
and [WLASL](https://dxli94.github.io/WLASL/) datasets, plus
[MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).
