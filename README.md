# CoachWorld

CoachWorld is the action-conditioned video world model used by RoboCoach. This
repository contains the model, Wan2.2 adaptation, video-latent data interfaces,
training entry point, and focused tests. The Python package is `coachworld`.

## Contents

| Path | Purpose |
| --- | --- |
| `coachworld/world_model/` | Conditioning, inference, and training |
| `coachworld/wan/` | Modified Wan2.2 model and VAE components |
| `coachworld/data/` | Video-latent datasets and action/camera conventions |
| `coachworld/calibration/` | Camera and robot geometry helpers |
| `coachworld/evaluator/` | Metrics and video output used during training |
| `scripts/training/train_world_model.py` | Distributed training entry point |
| `configs/training/coachworld.yaml` | Example training configuration |
| `tests/` | Model and data contract tests |

## Setup

Use Python 3.10 or 3.11 in a GPU environment compatible with PyTorch and
Wan2.2. From this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q tests
```

Wan2.2 model weights, video-latent datasets, and training checkpoints must be
obtained separately under their own terms. They are not included here. Set
`COACHWORLD_MODEL_ROOT`, `COACHWORLD_DATA_ROOT`, `COACHWORLD_OUTPUT_ROOT`, and
`COACHWORLD_MAX_TRAIN_STEPS` for your run. Dataset roots require prepared
video-latent collections with manifests, indexes, and normalization statistics.

```bash
torchrun --nproc_per_node="$NPROC" scripts/training/train_world_model.py \
  --config configs/training/coachworld.yaml
```

Set `NPROC` to the number of training processes. To resume an existing run,
pass `--resume /path/to/checkpoint`.
The dataset readers accept the format described in
`coachworld/data/video_latent.py`. The included
`coachworld/data/video_latent_release.py` can package an existing collection; it
does not download or generate one.

## Scope and licensing

This repository covers CoachWorld source code; datasets and model weights are
distributed separately. The root MIT license applies to RoboCoach-authored code.
Modified Wan2.2 components retain Apache-2.0 terms; PRoPE has its own MIT
header. See `THIRD_PARTY.md`.
