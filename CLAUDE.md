# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Workflow Rules

- Break all instructions into numbered tasks; execute only one at a time.
- After completing each task, confirm completion and explicitly request the next task number before proceeding.
- Stop immediately and request input if a command fails, an error occurs, required input is missing, or instructions are ambiguous.
- Do not guess at fixes. Always surface the exact error output before taking any action.
- Keep responses concise. Use code blocks for all commands.

## Code Guidelines

From `CODEGUIDELINES`: Don't ruin the existing architecture. Follow the same logic and brevity. Don't abstract code into large classes if you only save a few lines in one place — this prevents programmers from understanding it quickly. Prioritize developer comprehension over compactness.

## Environment Setup

**Python 3.10 required** (3.11+ breaks TF 2.4 compatibility).

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements-cuda.txt
```

Verify GPU/CUDA:
```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

No build step, no test suite, no linter — run scripts directly via `main.py`.

## Pipeline Commands

All operations go through `main.py`:

```bash
# Extract video frames
python main.py videoed extract-video --input-file workspace/data_src/<video> --output-dir workspace/data_src
python main.py videoed extract-video --input-file workspace/data_dst/<video> --output-dir workspace/data_dst

# Detect and align faces
python main.py extract --input-dir workspace/data_src --output-dir workspace/data_src/aligned
python main.py extract --input-dir workspace/data_dst --output-dir workspace/data_dst/aligned

# Sort faces by quality
python main.py sort --input-dir workspace/data_dst/aligned --by blur

# Train model (runs until user presses Enter)
python main.py train --training-data-src-dir workspace/data_src/aligned --training-data-dst-dir workspace/data_dst/aligned --model-dir workspace/model

# Merge (apply trained model to target frames)
python main.py merge --input-dir workspace/data_dst --output-dir workspace/merged --model-dir workspace/model

# Reconstruct final video
python main.py videoed video-from-sequence --input-dir workspace/merged --output-file workspace/result.mp4

# XSeg segmentation editor (Qt GUI)
python main.py xseg editor --input-dir workspace/data_dst/aligned
```

## Architecture Overview

This is a **pipeline-based deepfake system**: extract → sort → train → merge → export. Each stage's output feeds the next.

### Entry Point & Flow
`main.py` routes subcommands to `mainscripts/` handlers. The pipeline is intentionally sequential — don't skip stages.

### Key Modules

**`mainscripts/`** — One file per pipeline stage: `Extractor.py`, `Trainer.py`, `Merger.py`, `Sorter.py`, `VideoEd.py`, `XSegUtil.py`, `ExportDFM.py`.

**`models/`** — Four model types, each in its own folder:
- `Model_SAEHD` — Primary production model (Selective Attention autoencoder)
- `Model_Quick96` — Fast 96px model for quick iteration
- `Model_AMP` — Attention mechanism variant
- `Model_XSeg` — Segmentation-only model
All inherit from `models/ModelBase.py`.

**`core/leras/`** — Custom TensorFlow 2.4 wrapper (not Keras). Manages GPU/CPU device configuration, implements custom layers (Conv2D, Dense, AdaIN, BatchNorm, etc.), architectures (`archis/DeepFakeArchi.py`), and optimizers (AdaBelief, RMSprop). Used instead of Keras for full control of TF graph operations.

**`facelib/`** — Face detection and processing:
- `S3FDExtractor.py` — Face detection (pre-trained `.npy` weights bundled)
- `FANExtractor.py` — 2D/3D facial landmark extraction (68 or 98 points)
- `LandmarksProcessor.py` — Alignment, cropping, transformation matrices
- `FaceType.py` — Enum defining crop regions (HALF, FULL, HEAD, WHOLE_FACE, etc.)
- `XSegNet.py` — Segmentation network for face masking

**`samplelib/`** — Dataset management. `Sample` is the core data structure (filename, shape, landmarks, xseg mask, pose angles). `SampleLoader` feeds batches to training. `SampleProcessor` handles augmentation (warp, HSV, blur, color transfer). `PackedFaceset` is a compressed dataset format.

**`core/joblib/`** — Multiprocessing framework. `SubprocessorBase` enables parallel workers for extraction and merging. `MPFunc`/`MPClassFuncOnDemand` wrap functions for subprocess execution.

**`DFLIMG/`** — Custom image format: `.jpg` files with embedded DFL metadata (landmarks, face type, segmentation polygons, pose angles) stored as a dict appended to the JPEG binary.

**`merger/`** — Frame blending logic. `MergeMasked.py` handles masked blending with color correction and edge feathering. `InteractiveMergerSubprocessor.py` enables interactive parameter adjustment during merging.

### Data Flow (Training Loop)

```
SRC faces → SampleLoader → SampleProcessor (augment) → Encoder
                                                              ↓
DST faces → SampleLoader → SampleProcessor (augment) → Decoder_DST → swapped face
                                                              ↓
                                          Loss = reconstruction + adversarial + perceptual
                                                              ↓
                                                   AdaBelief optimizer backprop
```

### Backend (GUI wrapper)

`backend/` contains the FastAPI control plane. It has its own venv (Python 3.10) separate from DFL's venv (Python 3.7).

**`backend/command_registry.py` is the single source of truth for all paths and subprocess commands.** Never scatter path definitions elsewhere.

To start the backend:
```bat
cd backend
.venv\Scripts\uvicorn main:app --host 127.0.0.1 --port 8000 --no-access-log
```

To run the test suite (server must be running first):
```bat
cd backend
.venv\Scripts\python test_backend.py
```

### Workspace Layout

```
C:\Users\Rod\Documents\DFL\
├── main.py
├── venv\                        # Python 3.7.9 virtualenv
├── backend\                     # FastAPI wrapper
│   ├── command_registry.py      # ALL paths and stage commands live here
│   ├── process_manager.py
│   └── main.py
└── workspace\
    ├── data_src\                # Raw source frames
    │   └── aligned\             # Extracted + aligned source faces
    ├── data_dst\                # Raw destination frames
    │   └── aligned\             # Extracted + aligned destination faces
    ├── model\                   # Training checkpoints (saved every 25 min)
    └── merged\                  # Per-frame merge output
```

Model checkpoints: `<name>_options.pkl` (hyperparams) + TF checkpoint files + `<name>_backup.zip`.
