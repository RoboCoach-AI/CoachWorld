# CoachWorld

**An action-conditioned video world model for robot manipulation.**

[Paper](https://arxiv.org/abs/2609.39685) | [Project page and videos](https://robocoach-ai.github.io/) | [Pretrained checkpoint](https://huggingface.co/JEdward/CoachWorld)

[Jiajun Liu](https://jedward225.github.io/), [Yifan Chen](https://cyf-24.github.io/), Yichao Liu, [Jiayi Zhang](https://zhangjiayi24.github.io/), [Ruoqu Chen](https://ruoqu.cc/), Shaoxuan Xie, [Guocai Yao](https://scholar.google.com/citations?user=FLkv_vIAAAAJ), [Mengdi Xu](https://www.mengdixu.me/), [Sen Cui](https://sencui-thu.github.io/), [Changshui Zhang](https://www.au.tsinghua.edu.cn/en/info/1078/3205.htm)

## Overview

CoachWorld is the world-model component of **RoboCoach: World Models as Active Coaches for Compositional Robot Skills**. It predicts future visual observations from sparse visual history and robot actions, and supports autoregressive interaction with skill policies.

RoboCoach uses these imagined rollouts in a Route-Imagine-Diagnose-Improve loop: a separate progress judge identifies unfinished subtasks, and recurring failures guide which demonstrations to collect and which skill experts to update. **This repository implements CoachWorld, not the router, progress judge, or complete coaching pipeline.** See the [paper](https://arxiv.org/abs/2609.39685) and [project page](https://robocoach-ai.github.io/) for the full method and videos.

In the paper's real-robot experiments, the complete coaching method raises task success from **13.3% to 75.0% on Franka** and **40.0% to 83.8% on AgileX** with 150 additional subtask demonstrations. These are results for RoboCoach as a whole, not standalone CoachWorld benchmark scores.

## What is released

| Component | Where |
| --- | --- |
| Action-conditioned world model and training code | [`coachworld/world_model/`](coachworld/world_model/) |
| Modified Wan2.2 backbone and VAE components | [`coachworld/wan/`](coachworld/wan/) |
| Video-latent data and action/camera contracts | [`coachworld/data/`](coachworld/data/), [`coachworld/calibration/`](coachworld/calibration/) |
| Distributed training entry point and example config | [`scripts/training/train_world_model.py`](scripts/training/train_world_model.py), [`configs/training/coachworld.yaml`](configs/training/coachworld.yaml) |
| Model and data contract tests | [`tests/`](tests/) |
| CoachWorld pretraining weights | [Hugging Face](https://huggingface.co/JEdward/CoachWorld) |

The released checkpoint is the **CoachWorld pretrain** model. It is not the target-only ablation or a platform-specific post-trained model from the paper. Model weights and datasets are distributed separately from this source repository.

## Installation

Use Python 3.10 or 3.11 on a GPU machine with a PyTorch/CUDA setup suitable for Wan2.2. From the repository root:

```bash
git clone https://github.com/RoboCoach-AI/CoachWorld.git
cd CoachWorld
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q tests
```

The test command checks the included code and data contracts; it does not reproduce the paper's training or evaluation results.

## Pretrained checkpoint

Download the [CoachWorld pretrain release](https://huggingface.co/JEdward/CoachWorld/tree/main):

```bash
python -m pip install -U huggingface_hub
hf download JEdward/CoachWorld --local-dir models/CoachWorld
```

The download includes `dit_model.safetensors` and release metadata. The checkpoint uses this repository's custom model and conditioning code; it is not a drop-in checkpoint for a generic Transformers or Diffusers pipeline. Action coordinate frames, normalization, camera calibration, and temporal sampling must match the data contracts in [`coachworld/data/`](coachworld/data/). Raw robot commands cannot be substituted directly for prepared model actions.

## Training

Training starts from the Wan2.2 TI2V-5B base assets and a **prepared** CoachWorld `video_latent` dataset. The example config expects:

```text
<COACHWORLD_MODEL_ROOT>/Wan2.2-TI2V-5B/
<COACHWORLD_DATA_ROOT>/video_latent/
  manifest.json
  train.index.jsonl
  val.index.jsonl
  stats/arm_slot_eef_pose.json
  ... latent and signal shards ...
```

The validation split is required by the default config. See [`coachworld/data/video_latent.py`](coachworld/data/video_latent.py) for the manifest and index schema. [`coachworld/data/video_latent_release.py`](coachworld/data/video_latent_release.py) packages an existing collection; it does **not** download or create the dataset.

Set the asset roots and a positive training length, then launch from the repository root:

```bash
export COACHWORLD_MODEL_ROOT=/path/to/base-models
export COACHWORLD_DATA_ROOT=/path/to/prepared-data
export COACHWORLD_OUTPUT_ROOT=/path/to/outputs
export COACHWORLD_MAX_TRAIN_STEPS=10000

torchrun --nproc_per_node=5 scripts/training/train_world_model.py \
  --config configs/training/coachworld.yaml
```

`COACHWORLD_MODEL_ROOT` points to the **parent of the Wan2.2 directory**, not to the CoachWorld checkpoint downloaded above. Adjust the GPU/process count and configuration for your hardware.

To start a new training run from the released CoachWorld weights, add `--init_from models/CoachWorld`. This argument takes the **directory containing** `dit_model.safetensors` and does not restore optimizer state or the training step. Use `--resume /path/to/checkpoint-N` only for a checkpoint saved by this trainer when you need to restore model, optimizer, and step together. Configuration values can also be overridden with dotlist arguments such as `world_model.learning_rate=1e-5`.

## Scope and limitations

- This release contains the CoachWorld implementation and checkpoint, not the full RoboCoach coaching system.
- Training requires separately prepared video latents, action signals, camera metadata, and the Wan2.2 base assets.
- The repository contains model inference components but no standalone, one-command inference demo for arbitrary robot videos.

## Citation

If you use CoachWorld, please cite the RoboCoach paper:

```bibtex
@article{liu2026robocoach,
  title={RoboCoach: World Models as Active Coaches for Compositional Robot Skills},
  author={Liu, Jiajun and Chen, Yifan and Liu, Yichao and Zhang, Jiayi
          and Chen, Ruoqu and Xie, Shaoxuan and Yao, Guocai
          and Xu, Mengdi and Cui, Sen and Zhang, Changshui},
  journal={arXiv preprint arXiv:2609.39685},
  year={2026},
  url={https://arxiv.org/abs/2609.39685}
}
```

## License and third-party code

RoboCoach-authored code is released under the root [MIT license](LICENSE). Modified Wan2.2 components retain Apache-2.0 terms, and PRoPE retains its MIT header. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for attribution. Model weights and datasets have their own terms.
