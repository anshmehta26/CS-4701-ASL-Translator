# Real-Time American Sign Language Fingerspelling Recognition

A real-time webcam-based ASL fingerspelling translator that recognizes the manual alphabet (A–Z) from live video and types out the recognized letters as text. Cornell **CS 4701 — Practicum in AI**, Spring 2026.

**Team:** Shashank Kalyanaraman (ssk252) · Saurav Tewari (st782) · Ansh Mehta (am2555)

---

## Headline result

**90.87% top-1 accuracy** on a held-out **cross-signer** test split (signers never seen during training or validation), with **0.97 ms median model latency** on Apple Silicon — over 300× under our 300 ms real-time target.

| Model | Test top-1 | Test top-5 | Latency (median) | Params |
|---|---:|---:|---:|---:|
| LSTM (single-signer baseline) | 94.93% | 99.18% | 4.89 ms | 674k |
| LSTM (cross-signer) | 75.48% | 98.08% | 4.89 ms | 674k |
| **Transformer (cross-signer, deployed)** | **90.87%** | **96.90%** | **0.97 ms** | **409k** |

The single-signer 94.93% number is misleading — train, val, and test all came from one person. The **cross-signer 90.87%** is the real generalization measurement: it tests whether the model can recognize fingerspelling from people the model has never seen. See the [final report](docs/CS4701_Final_Report.pdf) for the methodology.

**21 of 26 letters achieve at least 95% accuracy.** The remaining error budget is concentrated in five letters (X at 0%, T at 52%, D at 50%, R at 81.5%, P at 89.5%) — all attributable to two known structural limitations of landmark-only static-frame approaches (occlusion blindness and motion blindness).

---

## How it works

The pipeline is: **webcam frame → MediaPipe Hands → 21 keypoints in 3D → wrist-relative + scale normalization → sliding window of recent landmark vectors → Transformer encoder (3 layers, 4 heads) → softmax over 26 letters → typing buffer.**

**Why landmarks instead of raw pixels:** lightweight (63 floats per frame), nearly invariant to lighting and background, and run sub-millisecond on a laptop CPU. The trade-off — that MediaPipe discards occlusion information — turned out to be the system's primary accuracy bottleneck.

**Why a Transformer:** beat the LSTM on every dimension we measured — 15.4 percentage points higher cross-signer top-1, 39% fewer parameters, ~5× lower inference latency.

**Cross-signer evaluation:** trained on signers P1–P6, validated on P7–P8, tested on P9–P10 — all from the [ASL-HG dataset](https://data.mendeley.com/datasets/j4y5w2c8w9/1) (10 distinct volunteers). No signer ever appears in two splits.

---

## Setup

Requires **Python 3.11** (MediaPipe is broken on 3.13 as of writing).

```bash
git clone https://github.com/anshmehta26/CS-4701-ASL-Translator.git
cd CS-4701-ASL-Translator

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quick start

### 1. Download ASL-HG

Get the dataset from [Mendeley Data](https://data.mendeley.com/datasets/j4y5w2c8w9/1) (free signup). Unzip into `data/raw/asl-hg/`. The filename prefix `Pn_` is the signer ID (P1–P10), used for the by-signer split.

### 2. Build manifest + extract landmarks (~10 minutes)

```bash
python scripts/prepare_asl_hg.py --config configs/asl_hg.yaml
```

### 3. Train (~30 min on M-series Mac)

```bash
python scripts/train.py --model transformer --config configs/asl_hg.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/transformer_best.pt --config configs/asl_hg.yaml
```

### 5. Live demo

```bash
python scripts/realtime_typing.py --checkpoint checkpoints/transformer_best.pt --config configs/asl_hg.yaml
```

Point your webcam at your hand and fingerspell. Letters commit to the typed buffer once the model has stably predicted them across multiple frames; drop your hand briefly between letters as a separator.

**Keys:** `q` quit, `c` clear buffer, `b` backspace.

---

## Project layout

- `configs/` — YAML configs for ASL-HG (the real pipeline), the single-signer Kaggle baseline, and the WLASL stretch goal.
- `scripts/` — `prepare_asl_hg.py`, `train.py`, `evaluate.py`, `benchmark_latency.py`, `realtime_typing.py`, and equivalents for the other tracks.
- `src/data/` — dataset loaders, MediaPipe extractor, augmentation.
- `src/models/` — biLSTM and Transformer encoder.
- `src/training/` — trainer with cross-entropy and CTC support.
- `src/inference/` — webcam loop with sliding-window predictor.
- `src/evaluation/` — full test-split evaluation and latency benchmarking.
- `tests/` — 32 unit tests.
- `docs/CS4701_Final_Report.pdf` — 14-page final report.

---

## Tests

```bash
pytest tests/ -v
```

32 unit tests covering augmentation, landmark preprocessing, by-signer split logic, both model forward-pass shapes, manifest builders, and the typing state machine.

---

## What we'd do next

A two-stream architecture combining the current landmark encoder with a small CNN over a tightly-cropped hand image, processed across genuine temporal sequences (real video frames, not replicated single frames). Would address both the closed-fist occlusion problem (CNN sees occlusion patterns directly) and the dynamic-letter motion problem (J, Z, and the X failure case). This is the standard approach in the sign-language recognition literature and was used by all the top-placing teams in the [2023 Kaggle ASL Fingerspelling competition](https://www.kaggle.com/competitions/asl-fingerspelling).

---

## Acknowledgments

- [ASL-HG dataset](https://data.mendeley.com/datasets/j4y5w2c8w9/1) (Mendeley Data)
- [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/)

Course project for **Cornell CS 4701 (Spring 2026)**, taught by Prof. Kevin Ellis.