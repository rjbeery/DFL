# DeepFaceLab Setup & Workflow Guide

---

## 1. Project Overview

**DeepFaceLab** is an open-source deep learning tool for face replacement (face swap) in images and video.

**Goal:** Run a complete, end-to-end face swap pipeline locally — from raw source footage to merged output video.

---

## 2. Repository Setup

**Clone the repository:**

```bash
git clone https://github.com/iperov/DeepFaceLab.git
cd DeepFaceLab
```

**Post-clone folder structure (key items):**

```
DeepFaceLab/
├── _internal/          # Core DFL runtime and models
├── scripts/            # Batch/shell scripts for each pipeline step
├── mainscripts/        # Python entrypoints for each stage
├── models/             # Model architecture definitions
└── workspace/          # Working directory for your project data (created manually)
```

> The `workspace/` folder is not created by the clone — you create it and populate it with your own data.

---

## 3. Getting or Refreshing the Repository

### 3.1 First-Time Clone

```bash
git clone https://github.com/iperov/DeepFaceLab.git
```

### 3.2 Enter the Repo Folder

```bash
cd DeepFaceLab
```

### 3.3 Refresh an Existing Local Copy

```bash
git fetch origin
git pull
```

### 3.4 Optional Status Check

```bash
git status
```

### 3.5 Notes

- Use `clone` only once when setting up a new local copy.
- Use `fetch` / `pull` at the start of any session when working from an existing repo.
- Keep all project work inside the `DeepFaceLab/` folder.

---

## 4. Inspecting the Project After Pulling

### 4.1 Goal

- Verify the repository pulled correctly.
- Identify top-level files and folders before running anything.

### 4.2 List Files

**Windows Command Prompt:**

```cmd
dir
```

**PowerShell:**

```powershell
Get-ChildItem
```

### 4.3 What to Look For

| Item | Type |
|------|------|
| `main.py` | File — main DFL entrypoint |
| `mainscripts/` | Folder — per-stage Python scripts |
| `models/` | Folder — model architecture definitions |
| `samplelib/` | Folder — data sampling utilities |
| `requirements*.txt` | File(s) — dependency lists (if present) |
| `workspace/` | Folder — your data directory (created manually) |

### 4.4 Notes

- Do not edit any source files at this stage.
- This step only confirms the repo structure is present and sane.
- If key files or folders are missing, the repo may not have pulled correctly — re-run `git pull` or re-clone.

---

## 5. Environment Setup

### 5.1 Python Version

- **Target:** Python 3.10
- Newer versions (3.11+) may break compatibility with DFL dependencies.

### 5.2 Virtual Environment

**Create:**

```cmd
python -m venv venv
```

**Activate (Windows):**

```cmd
venv\Scripts\activate
```

### 5.3 Verification

```cmd
python --version
```

Expected output: `Python 3.10.x`

### 5.4 Notes

- Always activate the venv before running any DFL script or install command.
- Keep all dependencies isolated to this project — do not install into the global Python environment.

---

## 6. Installing Project Dependencies

### 6.1 Prerequisite

Virtual environment must be activated before running any install commands.

```cmd
venv\Scripts\activate
```

### 6.2 Optional: Upgrade pip First

```cmd
python -m pip install --upgrade pip
```

### 6.3 Install Dependencies

```cmd
pip install -r requirements-cuda.txt
```

> Use `requirements-cuda.txt` for a local NVIDIA/CUDA-based setup. `requirements-colab.txt` is for Colab-style environments and should not be used here.

### 6.4 Verification

```cmd
pip list
```

Confirms all packages installed and their versions.

### 6.5 Notes

- Run all commands from inside the `DeepFaceLab/` project folder.
- If installation fails, stop and capture the exact error message before taking any action.
- Do not guess at fixes or apply random solutions.

---

## 7. GPU and CUDA Setup

### 7.1 Requirements

- NVIDIA GPU required
- CUDA support required
- CPU-only is not practical for DeepFaceLab — training times are prohibitive without GPU acceleration.

### 7.2 PyTorch CUDA Check

```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

Expected: `True`

### 7.3 GPU Detection Check

```cmd
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected: Name of your installed GPU (e.g., `NVIDIA GeForce RTX 3080`)

### 7.4 Failure Handling

If result is `False`:

**Likely causes:**
- CUDA not installed or not on PATH
- PyTorch installed without CUDA support (CPU-only build)
- GPU drivers missing or outdated

**Action:** Stop here and resolve before continuing. Do not proceed to training with a failed CUDA check.

---

## 8. Workspace Structure

```
workspace/
├── data_src/     # Source face footage (the face you are extracting FROM)
├── data_dst/     # Destination footage (the video you are swapping INTO)
├── model/        # Saved model checkpoints and training state
└── merged/       # Output frames after merging (pre-final video)
```

| Folder | Purpose |
|--------|---------|
| `data_src/` | Raw or preprocessed frames of the **source** face |
| `data_dst/` | Raw or preprocessed frames of the **target** video |
| `model/` | Active model being trained; persists between sessions |
| `merged/` | Per-frame merge results; assembled into final video |

---

## 9. Execution Philosophy

- **This is a pipeline, not a single script.** Each stage (extract → align → train → merge) must be run in order, and outputs feed the next step.
- **Data quality > settings.** Clean, well-lit, varied source footage will outperform any hyperparameter tuning on poor data.
- **Iteration is required.** Training is not one-and-done. Expect to review results, adjust, and continue training across multiple sessions.

---

## 10. Step 1: Extracting Frames from Video

### 10.1 Goal

Convert source and destination videos into individual image frames for processing.

### 10.2 Input Locations

| Folder | Contents |
|--------|----------|
| `workspace/data_src/` | Source video (the face being extracted) |
| `workspace/data_dst/` | Destination video (the target footage) |

### 10.3 Commands

**Extract source frames:**

```cmd
python main.py videoed extract-video --input-file workspace/data_src/<video_name> --output-dir workspace/data_src
```

**Extract destination frames:**

```cmd
python main.py videoed extract-video --input-file workspace/data_dst/<video_name> --output-dir workspace/data_dst
```

### 10.4 Expected Result

Both folders populated with sequentially numbered image frames (e.g., `000001.png`, `000002.png`, ...).

### 10.5 Notes

- Replace `<video_name>` with the actual filename including extension (e.g., `source.mp4`).
- Do not proceed if extraction fails or folders remain empty.
- This step must complete successfully before moving to face extraction.

---

## 11. Step 2: Extracting and Aligning Faces

### 11.1 Goal

- Detect faces in the extracted frames.
- Produce aligned face images ready for model training.

### 11.2 Commands

**Extract and align source faces:**

```cmd
python main.py extract --input-dir workspace/data_src --output-dir workspace/data_src/aligned
```

**Extract and align destination faces:**

```cmd
python main.py extract --input-dir workspace/data_dst --output-dir workspace/data_dst/aligned
```

### 11.3 Expected Result

- `workspace/data_src/aligned/` and `workspace/data_dst/aligned/` populated with cropped, aligned face images.
- DFL detects and normalizes face orientation across all frames.

### 11.4 Notes

- Requires successful frame extraction (Step 1) before running.
- Poor detections or missed faces will directly reduce final output quality.
- Do not proceed if extraction fails or produces obviously bad/misaligned results — review output before continuing.

---

## 12. Step 3: Training the Model

### 12.1 Goal

Train a face-swapping model using the aligned source and destination face sets.

### 12.2 Command

```cmd
python main.py train --training-data-src-dir workspace/data_src/aligned --training-data-dst-dir workspace/data_dst/aligned --model-dir workspace/model
```

### 12.3 Expected Result

- Training UI or preview window opens.
- Model files begin appearing in `workspace/model/`.
- Loss values update over time as training progresses.

### 12.4 Notes

- Training may take hours or longer depending on hardware and data size.
- Stop and inspect if the preview looks obviously broken early on.
- Better data matters more than adjusting settings — do not chase random parameter changes.
- Do not proceed to merge until model output looks usable.

---

## 13. Step 4: Merging the Result

### 13.1 Goal

Apply the trained model to the destination frames to produce merged output frames with the swapped face.

### 13.2 Command

```cmd
python main.py merge --input-dir workspace/data_dst --output-dir workspace/merged --model-dir workspace/model
```

### 13.3 Expected Result

- Merged image frames appear in `workspace/merged/`.
- The swapped face is visible on the destination footage frames.

### 13.4 Notes

- Do not run merge unless training output looked usable first.
- If the merge looks wrong, return to training — fix data quality or continue training before retrying.
- Merging is not a substitute for good source and destination data.

---

## 14. Step 5: Exporting the Final Video

### 14.1 Goal

Convert the merged output frames back into a final video file.

### 14.2 Command

```cmd
python main.py videoed video-from-sequence --input-dir workspace/merged --output-file workspace/result.mp4
```

### 14.3 Expected Result

- `workspace/result.mp4` is created.
- The face swap is visible in the rendered video.

### 14.4 Notes

- Requires completed merge output in `workspace/merged/` before running.
- If the final video looks bad, the issue is upstream — check extraction, alignment, or training quality first.
- Export settings are not the primary quality lever.

---

## 15. Quality Control and Stop Conditions

### 15.1 Purpose

Prevent wasting time by catching bad data or bad outputs before pushing them further down the pipeline.

### 15.2 Checkpoints

**After frame extraction:**
- Image frames exist in `data_src/` and `data_dst/`
- Both source and destination videos extracted without error

**After face extraction:**
- Aligned face images exist in `data_src/aligned/` and `data_dst/aligned/`
- Faces are centered and mostly correctly cropped
- Stop if many faces are missed, badly cropped, or misaligned

**After training starts:**
- Preview is visibly changing over time
- Model files are appearing in `workspace/model/`
- Stop if previews look obviously broken and do not improve

**After merge:**
- Swapped face is visible in `workspace/merged/` frames
- Stop if output is badly distorted, misplaced, or unusable

**After export:**
- `workspace/result.mp4` was created
- Final video accurately reflects the merged frames

### 15.3 General Rule

Do not continue to the next stage when the current stage is clearly bad. Fix upstream problems first.

---

## 16. Run Log

### 16.1 Purpose

Track real pipeline attempts — what was run, what happened, and what to do next. Eliminates guesswork when resuming work between sessions.

### 16.2 Entry Template

```
Date:
Source input:
Destination input:
Step reached:
Result:
Error (exact text if any):
Next action:
```

### 16.3 Notes

- Keep entries short and factual.
- Paste exact error text — do not paraphrase.
- Record blockers immediately when they occur.

### 16.4 Log

<!-- Add run entries below this line -->

---

## 17. Claude Workflow Rules

### 17.1 Task Structure

- All instructions must be broken into numbered tasks.
- Only one task is executed at a time.

### 17.2 Control Flow

After completing each task, Claude must:
- Confirm the task is complete.
- Explicitly request the next task number before proceeding.

### 17.3 Stop Conditions

Claude must stop immediately and request input if:
- A command fails.
- An error occurs.
- Required input is missing.
- Instructions are ambiguous.

### 17.4 No Assumptions Rule

- Do not guess at fixes.
- Do not proceed past errors.
- Always surface the exact error output before taking any action.

### 17.5 Output Style

- Keep responses concise.
- Use code blocks for all commands.
- Avoid unnecessary explanation.

---

---

## 18. Docker Wrapper & Cloud Deployment

The DFL pipeline is also wrapped in a FastAPI + Docker control panel that runs
DFL stages as subprocesses and provides a browser UI for launch/stop/logs/backup.

### Local quick-start

```bash
cp .env.example .env             # create local config (defaults are fine for local use)
docker compose up --build        # first run — builds the image
docker compose up                # subsequent runs
# UI: http://localhost  (nginx on port 80)
```

### Deploying to a cloud GPU VM

See **[docs/FIRST_CLOUD_DEPLOY.md](docs/FIRST_CLOUD_DEPLOY.md)** for the exact step-by-step deployment guide:

- VM bootstrap (Docker, NVIDIA Container Toolkit, storage mount)
- Environment variable configuration (`.env.example` → `.env`)
- `docker compose up -d --build`
- Smoke tests (GPU check, login, backup, SSE log stream)
- Phase 2: enable HTTPS with Let's Encrypt

See **[docs/CLOUD_RUNTIME.md](docs/CLOUD_RUNTIME.md)** for architecture reference:

- Container architecture diagram
- Port and volume mapping tables
- nginx reverse proxy and TLS configuration reference
- Troubleshooting (GPU not visible, port unreachable, permission errors, SSE issues)
- Known limitations

---

_Last updated: 2026-04-15_
