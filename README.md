<p align="center">
  <img src="https://robocoach-ai.github.io/assets/figures/robocoach_mark.png" width="88" alt="RoboCoach logo" />
</p>

<h1 align="center">CoachWorld</h1>

<p align="center"><strong>The action-conditioned world model behind RoboCoach</strong></p>
<p align="center">RoboCoach: World Models as Active Coaches for Compositional Robot Skills</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.39685"><img src="https://img.shields.io/badge/arXiv-2609.39685-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Paper on arXiv" /></a>
  <a href="https://robocoach-ai.github.io/"><img src="https://img.shields.io/badge/Project-Page-00897B?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Project page" /></a>
  <a href="https://huggingface.co/JEdward/CoachWorld"><img src="https://img.shields.io/badge/Hugging_Face-Model-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" alt="Pretrained model on Hugging Face" /></a>
  <a href="https://robocoach-ai.github.io/#teaser"><img src="https://img.shields.io/badge/Video-Watch_Teaser-455A64?style=for-the-badge" alt="Watch the teaser video" /></a>
</p>

<p align="center">
  <a href="https://jedward225.github.io/">Jiajun&nbsp;Liu</a>,
  <a href="https://cyf-24.github.io/">Yifan&nbsp;Chen</a>,
  <a href="https://github.com/RoboCoach-AI/CoachWorld">Yichao&nbsp;Liu</a>,
  <a href="https://zhangjiayi24.github.io/">Jiayi&nbsp;Zhang</a>,
  <a href="https://ruoqu.cc/">Ruoqu&nbsp;Chen</a>,<br />
  <a href="https://github.com/RoboCoach-AI/CoachWorld">Shaoxuan&nbsp;Xie</a>,
  <a href="https://scholar.google.com/citations?user=FLkv_vIAAAAJ">Guocai&nbsp;Yao</a>,
  <a href="https://www.mengdixu.me/">Mengdi&nbsp;Xu</a>,
  <a href="https://sencui-thu.github.io/">Sen&nbsp;Cui</a>,
  <a href="https://scholar.google.com/citations?user=GL9M37YAAAAJ">Changshui&nbsp;Zhang</a>
</p>

## 🌟 Overview

**RoboCoach turns imagined failures into targeted supervision for robot skills.** In its **Route–Imagine–Diagnose–Improve** loop, skill policies act inside CoachWorld, a progress judge identifies the first unfinished subtask, and aggregated failures guide demonstration collection and expert updates.

[![RoboCoach overview: routing, world-model rollout, progress judging, and targeted improvement](https://robocoach-ai.github.io/assets/figures/teaser-green.png)](https://robocoach-ai.github.io/#overview)

**CoachWorld makes these interactions possible** with an adapted Wan2.2 TI2V-5B backbone:

- **Action-conditioned prediction:** generate future observations from sparse visual history and robot end-effector actions.
- **Single-arm and bimanual robots:** use a shared two-slot action representation with camera-aware conditioning.
- **Autoregressive rollout:** feed predicted observations back into the policy across successive video chunks.

With **150 additional subtask demonstrations**, the full RoboCoach system improves success from **13.3% to 75.0% on Franka** and **40.0% to 83.8% on AgileX**. See the [paper](https://arxiv.org/abs/2609.39685) for evaluation details and the [project page](https://robocoach-ai.github.io/) for interactive examples.

## 📦 Release and Code

This repository provides the **CoachWorld model, inference components, and training tools**. The router, progress judge, and expert-update pipeline are separate parts of RoboCoach.

| Component | Where |
| --- | --- |
| Action-conditioned world model and training code | [`coachworld/world_model/`](coachworld/world_model/) |
| Modified Wan2.2 backbone and VAE components | [`coachworld/wan/`](coachworld/wan/) |
| Video-latent data and action/camera contracts | [`coachworld/data/`](coachworld/data/), [`coachworld/calibration/`](coachworld/calibration/) |
| Distributed training entry point and example config | [`scripts/training/train_world_model.py`](scripts/training/train_world_model.py), [`configs/training/coachworld.yaml`](configs/training/coachworld.yaml) |
| Model and data contract tests | [`tests/`](tests/) |
| CoachWorld pretraining weights | [Hugging Face](https://huggingface.co/JEdward/CoachWorld) |

The released checkpoint is the **CoachWorld pretrain** model, trained on heterogeneous single-arm and dual-arm robot data. The target-only ablation and platform-specific post-trained models are distinct from this release.

## ⚙️ Installation

Use Python 3.10 or 3.11 on a GPU machine with a PyTorch/CUDA setup suitable for Wan2.2. From the repository root:

```bash
git clone https://github.com/RoboCoach-AI/CoachWorld.git
cd CoachWorld
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q tests
```

The tests cover model components and data contracts. All commands below run from the repository root.

## 🤗 Pretrained Model

Download the [CoachWorld pretrain release](https://huggingface.co/JEdward/CoachWorld/tree/main):

```bash
python -m pip install -U huggingface_hub
hf download JEdward/CoachWorld --local-dir models/CoachWorld
```

The release includes `dit_model.safetensors`, `training_config.yaml`, and model metadata. Use the custom [`coachworld`](coachworld/) implementation to load the checkpoint; the [model card](https://huggingface.co/JEdward/CoachWorld) describes the release.

**Preparing inputs:** match the action coordinate frames, normalization, camera calibration, and temporal sampling specified in [`coachworld/data/`](coachworld/data/). The inference components require these prepared inputs; a standalone inference CLI is not included.

## 🏋️ Training

### Prepare the assets

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

### Launch training

Set the asset roots and a positive training length. The step count below is an example:

```bash
export COACHWORLD_MODEL_ROOT=/path/to/base-models
export COACHWORLD_DATA_ROOT=/path/to/prepared-data
export COACHWORLD_OUTPUT_ROOT=/path/to/outputs
export COACHWORLD_MAX_TRAIN_STEPS=10000

torchrun --nproc_per_node=5 scripts/training/train_world_model.py \
  --config configs/training/coachworld.yaml
```

`COACHWORLD_MODEL_ROOT` points to the **parent of the Wan2.2 directory**, not to the CoachWorld checkpoint downloaded above. Adjust the GPU/process count and configuration for your hardware.

### Warm-start or resume

Append the appropriate argument to the training command:

| Goal | Argument | Restores |
| --- | --- | --- |
| Start from the released CoachWorld weights | `--init_from models/CoachWorld` | Model weights; training starts at step 0 |
| Continue an existing training run | `--resume /path/to/checkpoint-N` | Model, optimizer, and training step |

`--init_from` takes the **directory containing** `dit_model.safetensors`. A resumable checkpoint must have been saved by this trainer, including optimizer state. Configuration values can also be overridden with dotlist arguments such as `world_model.learning_rate=1e-5`.

## 📝 Citation

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

## 🙏 Acknowledgments and License

RoboCoach-authored code is released under the root [MIT license](LICENSE). Modified Wan2.2 components retain Apache-2.0 terms, and PRoPE retains its MIT header. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for attribution. Model weights and datasets have their own terms.
