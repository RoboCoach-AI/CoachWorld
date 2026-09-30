"""Trainer for the experimental minWM-style causal Wan backend."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from coachworld.config import WMTrainingConfig, WorldModelConfig
from coachworld.wan.action.temporal_grouping import wan_causal_group_indices
from coachworld.wan import CausalWanModel

logger = logging.getLogger(__name__)


def _is_causal_action_key(name: str) -> bool:
    return (
        name.startswith("action_encoder.")
        or name.startswith("eef_projection_encoder.")
        or name.startswith("proprio_decoder.")
        or ".cross_attn.k_action." in name
        or ".cross_attn.v_action." in name
        or ".cross_attn.norm_k_action." in name
    )


def _infinite_loader(dataloader: DataLoader):
    epoch = 0
    while True:
        sampler = dataloader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from dataloader
        epoch += 1


def _wm_collate_causal(batch: list[dict]) -> dict:
    latents = torch.stack([item["latent"] for item in batch])
    actions = [item["action"] for item in batch]
    action_shapes = {tuple(action.shape[1:]) for action in actions}
    if len(action_shapes) != 1:
        raise RuntimeError(f"Mixed action trailing shapes in causal batch: {sorted(action_shapes)}")
    max_action_len = max(int(action.shape[0]) for action in actions)
    padded_actions = actions[0].new_zeros((len(actions), max_action_len, *actions[0].shape[1:]))
    for idx, action in enumerate(actions):
        padded_actions[idx, : action.shape[0]] = action
    texts = [str(item.get("text", "")) for item in batch]
    domain_ids = torch.tensor([int(item["domain_id"]) for item in batch], dtype=torch.long)
    out = {"latent": latents, "action": padded_actions, "text": texts, "domain_id": domain_ids}
    if "viewmats" in batch[0]:
        out["viewmats"] = torch.stack([item["viewmats"] for item in batch])
        out["Ks"] = torch.stack([item["Ks"] for item in batch])
    if "eef_uv" in batch[0]:
        out["eef_uv"] = torch.stack([item["eef_uv"] for item in batch])
        out["eef_depth"] = torch.stack([item["eef_depth"] for item in batch])
        out["eef_valid"] = torch.stack([item["eef_valid"] for item in batch])
        out["eef_image_hw"] = torch.stack([item["eef_image_hw"] for item in batch])
        if "eef_heatmap" in batch[0]:
            out["eef_heatmap"] = torch.stack([item["eef_heatmap"] for item in batch])
    return out


class CausalWanTrainer:
    """Small, explicit trainer for ``world_model.backbone=wan2.2_causal``.

    This is the first formal training path for the causal backend. It supports
    single-view video_latent roots, the existing action K/V contract, and the
    initial clean-context self-forcing schedule.
    """

    VAE_Z_DIM = 48
    PATCH_SIZE = (1, 2, 2)

    def __init__(self, training_config: WMTrainingConfig, model_config: WorldModelConfig):
        self.tcfg = training_config
        self.mcfg = model_config
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16 if self.mcfg.mixed_precision == "bf16" else torch.float32
        if self.mcfg.mixed_precision == "fp16":
            self.dtype = torch.float16

        self.model: Optional[CausalWanModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.global_step = 0
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main = self.rank == 0

    @property
    def latent_frames(self) -> int:
        return int(self.mcfg.history_length) + int(self.mcfg.pred_frames)

    @property
    def latent_height(self) -> int:
        return int(self.mcfg.num_cameras) * int(self.mcfg.latent_height_per_view)

    @property
    def latent_width(self) -> int:
        return int(self.mcfg.latent_width)

    @property
    def seq_len(self) -> int:
        return (
            self.latent_frames
            * (self.latent_height // self.PATCH_SIZE[1])
            * (self.latent_width // self.PATCH_SIZE[2])
        )

    def setup(self, resume_from: str | None = None, init_from: str | None = None) -> None:
        if self.world_size > 1 and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "CausalWanTrainer DDP requires torch.distributed to be initialized. "
                "Launch with torchrun, not plain python."
            )
        if bool(self.mcfg.text_conditioning_enabled) or bool(self.tcfg.text_conditioning_enabled):
            raise ValueError(
                "CausalWanTrainer currently supports text-free training only. "
                "Set world_model.text_conditioning_enabled=false and "
                "wm_training.text_conditioning_enabled=false."
            )
        if int(self.mcfg.num_cameras) != 1:
            raise ValueError(
                "CausalWanTrainer initial integration expects single-view latents. "
                f"Got num_cameras={self.mcfg.num_cameras}."
            )
        if bool(self.mcfg.causal_prope_enabled) and not bool(
            getattr(self.tcfg, "video_latent_camera_conditioning", False)
        ):
            raise ValueError(
                "world_model.causal_prope_enabled=true requires "
                "wm_training.video_latent_camera_conditioning=true so real "
                "viewmats/Ks are present in the batch."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CausalWanTrainer requires CUDA.")
        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cuda", int(torch.cuda.current_device()))

        action_config = {
            "action_dim": int(self.mcfg.action_dim),
            "action_schema": str(self.mcfg.action_schema),
            "max_arm_slots": int(self.mcfg.max_arm_slots),
            "action_max_time_steps": int(self.mcfg.action_max_time_steps),
            "action_kv_enabled": bool(self.mcfg.action_kv_enabled),
            "action_v_init_scale": float(self.mcfg.action_v_init_scale),
            "action_num_domains": int(self.mcfg.action_num_domains),
            "action_domain_prompt_tokens": int(self.mcfg.action_domain_prompt_tokens),
            "domain_aware_action_projection_enabled": bool(
                self.mcfg.domain_aware_action_projection_enabled
            ),
            "eef_projection_kv_enabled": bool(
                getattr(self.mcfg, "eef_projection_kv_enabled", False)
            ),
            "num_cameras": int(self.mcfg.num_cameras),
            "proprio_head_enabled": bool(self.mcfg.causal_proprio_head_enabled),
            "proprio_mid_dim": int(self.mcfg.causal_proprio_mid_dim),
        }
        base_model = CausalWanModel(
            model_type="t2v",
            patch_size=self.PATCH_SIZE,
            text_len=int(self.mcfg.causal_text_tokens),
            in_dim=self.VAE_Z_DIM,
            out_dim=self.VAE_Z_DIM,
            dim=int(self.mcfg.causal_dim),
            ffn_dim=int(self.mcfg.causal_ffn_dim),
            freq_dim=int(self.mcfg.causal_freq_dim),
            text_dim=int(self.mcfg.text_dim),
            num_heads=int(self.mcfg.causal_num_heads),
            num_layers=int(self.mcfg.causal_num_layers),
            local_attn_size=int(self.mcfg.causal_local_attn_size),
            sink_size=int(self.mcfg.causal_sink_size),
            cross_attn_norm=True,
            action_config=action_config,
        )
        base_model.num_frame_per_block = int(self.mcfg.causal_num_frame_per_block)
        base_model.gradient_checkpointing = bool(self.mcfg.gradient_checkpoint)
        if bool(self.mcfg.causal_prope_enabled):
            base_model.enable_prope(zero_init=bool(self.mcfg.causal_prope_zero_init))
        if str(self.mcfg.checkpoint):
            if self.is_main:
                logger.info("Loading causal Wan backbone from %s", self.mcfg.checkpoint)
            base_model = base_model.to(dtype=self.dtype)
            self._load_wan_backbone_weights(base_model, Path(self.mcfg.checkpoint))
            base_model._warm_start_action_kv()
        elif not bool(self.mcfg.causal_preserve_zero_head):
            torch.nn.init.xavier_uniform_(base_model.head.head.weight)
            if base_model.head.head.bias is not None:
                torch.nn.init.zeros_(base_model.head.head.bias)
        base_model = base_model.to(device=self.device, dtype=self.dtype)
        if self.world_size > 1:
            self.model = DDP(
                base_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model = base_model
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.mcfg.learning_rate),
            weight_decay=0.0,
        )

        if init_from:
            self._load_model_only(init_from)
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(
            "CausalWanTrainer ready: rank=%d/%d device=%s dtype=%s latent=(48,%d,%d,%d) seq_len=%d dim=%d layers=%d",
            self.rank,
            self.world_size,
            self.device,
            self.dtype,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            self.seq_len,
            int(self.mcfg.causal_dim),
            int(self.mcfg.causal_num_layers),
        )

    def _model_module(self) -> CausalWanModel:
        assert self.model is not None
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _load_wan_backbone_weights(self, model: CausalWanModel, checkpoint: Path) -> None:
        """Load matching Wan DiT tensors into the causal backend.

        This intentionally loads only exact same-name, same-shape tensors from
        the Wan safetensors checkpoint. Newly added action modules remain
        initialized by CausalWanModel and are warm-started from loaded text K/V
        after this method returns.
        """
        checkpoint = checkpoint.expanduser().resolve()
        if checkpoint.is_file():
            shard_paths = [checkpoint]
            weight_map = None
        else:
            index_path = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
            single_path = checkpoint / "diffusion_pytorch_model.safetensors"
            if index_path.exists():
                index = json.loads(index_path.read_text())
                weight_map = dict(index["weight_map"])
                shard_paths = sorted({checkpoint / name for name in weight_map.values()})
            elif single_path.exists():
                shard_paths = [single_path]
                weight_map = None
            else:
                raise FileNotFoundError(
                    "Causal Wan init expects a Wan DiT safetensors checkpoint: "
                    f"{index_path} or {single_path}"
                )

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("safetensors is required for causal Wan pretrained init") from exc

        target = model.state_dict()
        copied: list[str] = []
        shape_mismatch: list[str] = []
        source_only = 0
        target_seen: set[str] = set()
        with torch.no_grad():
            for shard_path in shard_paths:
                if not shard_path.exists():
                    raise FileNotFoundError(shard_path)
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    shard_keys = f.keys()
                    for key in shard_keys:
                        if weight_map is not None and Path(weight_map[key]).name != shard_path.name:
                            continue
                        if key not in target:
                            source_only += 1
                            continue
                        tensor = f.get_tensor(key)
                        if tuple(tensor.shape) != tuple(target[key].shape):
                            shape_mismatch.append(
                                f"{key}: checkpoint={tuple(tensor.shape)} model={tuple(target[key].shape)}"
                            )
                            continue
                        target[key].copy_(tensor.to(dtype=target[key].dtype))
                        copied.append(key)
                        target_seen.add(key)

        missing = sorted(set(target.keys()) - target_seen)
        missing_non_action = [
            key for key in missing
            if not _is_causal_action_key(key)
            and not key.startswith("img_emb.")
            and "prope" not in key
        ]
        if shape_mismatch:
            raise RuntimeError(
                "Wan pretrained init found shape mismatches. The causal config "
                "must match the Wan checkpoint architecture. First mismatches:\n  - "
                + "\n  - ".join(shape_mismatch[:20])
            )
        if missing_non_action:
            raise RuntimeError(
                "Wan pretrained init did not cover non-action causal parameters. "
                "This usually means the causal config/model_type does not match "
                "the Wan checkpoint. First missing keys:\n  - "
                + "\n  - ".join(missing_non_action[:40])
            )
        if len(copied) < 100:
            raise RuntimeError(
                f"Wan pretrained init copied only {len(copied)} tensors from {checkpoint}; "
                "refusing to train from an effectively random model."
            )
        if self.is_main:
            logger.info(
                "Causal Wan pretrained init loaded %d tensors from %d shard(s); "
                "missing_action_or_optional=%d source_only=%d",
                len(copied),
                len(shard_paths),
                len(missing),
                source_only,
            )

    def _barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier(device_ids=[self.local_rank])

    def _context(self, batch_size: int) -> list[torch.Tensor]:
        return [
            torch.zeros(
                int(self.mcfg.causal_text_tokens),
                int(self.mcfg.text_dim),
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(batch_size)
        ]

    def _check_latent_shape(self, latent: torch.Tensor) -> None:
        expected = (self.VAE_Z_DIM, self.latent_frames, self.latent_height, self.latent_width)
        if tuple(latent.shape[1:]) != expected:
            raise RuntimeError(
                "Unexpected latent batch shape for CausalWanTrainer: "
                f"got {tuple(latent.shape)}, expected (B,{','.join(map(str, expected))})"
            )

    def _action_grad_l2(self) -> float:
        assert self.model is not None
        total = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if (
                "action_encoder" not in name
                and "eef_projection_encoder" not in name
                and "k_action" not in name
                and "v_action" not in name
                and "norm_k_action" not in name
            ):
                continue
            grad = param.grad.detach().float()
            total += float(torch.sum(grad * grad).item())
        return total**0.5

    def _eef_projection_kwargs(self, batch: dict) -> dict:
        if not bool(getattr(self.mcfg, "eef_projection_kv_enabled", False)):
            return {}
        required = ("eef_uv", "eef_depth", "eef_valid", "eef_image_hw")
        missing = [key for key in required if key not in batch]
        if missing:
            raise RuntimeError(
                "world_model.eef_projection_kv_enabled=true requires EEF projection "
                f"fields in the batch; missing {missing}. Set "
                "wm_training.video_latent_eef_projection_roots."
            )
        return {
            "eef_uv": batch["eef_uv"].to(device=self.device, dtype=self.dtype),
            "eef_depth": batch["eef_depth"].to(device=self.device, dtype=self.dtype),
            "eef_valid": batch["eef_valid"].to(device=self.device),
            "eef_image_hw": batch["eef_image_hw"].to(device=self.device),
        }

    @torch.no_grad()
    def _self_forced_clean_context(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        domain_ids: torch.Tensor,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        eef_projection_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        assert self.model is not None
        history = int(self.mcfg.history_length)
        replace_frames = int(self.mcfg.causal_self_forcing_history_frames)
        replace_frames = max(0, min(replace_frames, history))
        if replace_frames == 0:
            return latent, {"self_forcing_active": 0.0, "self_forcing_history_frames": 0.0}
        sigma_value = float(self.mcfg.causal_self_forcing_sigma)
        if not (0.0 < sigma_value <= 1.0):
            raise ValueError(
                "world_model.causal_self_forcing_sigma must be in (0,1], "
                f"got {sigma_value}"
            )

        bsz = latent.shape[0]
        sigma = torch.full((bsz, 1, 1, 1, 1), sigma_value, device=self.device, dtype=self.dtype)
        noise = torch.randn_like(latent)
        noisy = (1.0 - sigma) * latent + sigma * noise
        timestep = (sigma.flatten() * 1000.0).view(bsz, 1).expand(bsz, self.latent_frames)

        velocity = self.model(
            x=noisy,
            t=timestep,
            context=self._context(bsz),
            seq_len=self.seq_len,
            clean_x=None,
            action_seq=action if bool(self.mcfg.action_kv_enabled) else None,
            action_domain_ids=domain_ids,
            viewmats=viewmats,
            Ks=Ks,
            **(eef_projection_kwargs or {}),
        )
        denoised = noisy - sigma * velocity
        clean_context = latent.clone()
        start = history - replace_frames
        clean_context[:, :, start:history] = denoised[:, :, start:history]
        delta = (clean_context[:, :, start:history] - latent[:, :, start:history]).float()
        return clean_context.detach(), {
            "self_forcing_active": 1.0,
            "self_forcing_history_frames": float(replace_frames),
            "self_forcing_sigma": sigma_value,
            "self_forcing_context_delta": float(delta.abs().mean().item()),
        }

    def _compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        assert self.model is not None
        latent = batch["latent"].to(device=self.device, dtype=self.dtype)
        latent = latent.permute(0, 2, 1, 3, 4).contiguous()
        self._check_latent_shape(latent)

        bsz = latent.shape[0]
        action = batch["action"].to(device=self.device, dtype=self.dtype)
        domain_ids = batch["domain_id"].to(device=self.device, dtype=torch.long)
        viewmats = None
        Ks = None
        if "viewmats" in batch:
            # Dataset returns (B,F,V,*,*) so keep the causal single-view slice.
            viewmats = batch["viewmats"].to(device=self.device, dtype=self.dtype)[:, :, 0]
            Ks = batch["Ks"].to(device=self.device, dtype=self.dtype)[:, :, 0]
        eef_projection_kwargs = self._eef_projection_kwargs(batch)
        clean_context = latent
        sf_logs = {"self_forcing_active": 0.0, "self_forcing_history_frames": 0.0}
        if (
            bool(self.mcfg.causal_self_forcing_enabled)
            and self.global_step >= int(self.mcfg.causal_self_forcing_start_step)
            and torch.rand((), device=self.device).item() < float(self.mcfg.causal_self_forcing_prob)
        ):
            clean_context, sf_logs = self._self_forced_clean_context(
                latent,
                action,
                domain_ids,
                viewmats=viewmats,
                Ks=Ks,
                eef_projection_kwargs=eef_projection_kwargs,
            )
        sigma = torch.rand(bsz, device=self.device, dtype=self.dtype).view(bsz, 1, 1, 1, 1)
        noise = torch.randn_like(latent)
        noisy = (1.0 - sigma) * latent + sigma * noise
        target = noise - latent
        timestep = (sigma.flatten() * 1000.0).view(bsz, 1).expand(bsz, self.latent_frames)

        want_proprio = self._model_module().proprio_decoder is not None
        model_out = self.model(
            x=noisy,
            t=timestep,
            context=self._context(bsz),
            seq_len=self.seq_len,
            clean_x=clean_context,
            action_seq=action if bool(self.mcfg.action_kv_enabled) else None,
            action_domain_ids=domain_ids,
            viewmats=viewmats,
            Ks=Ks,
            **eef_projection_kwargs,
            return_proprio=want_proprio,
        )
        if want_proprio:
            output, proprio_pred = model_out
        else:
            output = model_out
            proprio_pred = None
        history = int(self.mcfg.history_length)
        pred_future = output[:, :, history:, :, :]
        target_future = target[:, :, history:, :, :]
        main_loss = F.mse_loss(pred_future.float(), target_future.float()) * float(self.tcfg.loss_scale)
        loss = main_loss
        proprio_loss_val = 0.0
        if proprio_pred is not None:
            proprio_loss = self._compute_proprio_loss(
                proprio_pred,
                action,
                T_latent=self.latent_frames,
                history_frames=history,
            )
            proprio_loss_val = float(proprio_loss.detach().item())
            loss = loss + float(getattr(self.tcfg, "causal_proprio_loss_weight", 0.0)) * proprio_loss
        logs = {
            "loss": float(loss.detach().item()),
            "loss_main": float(main_loss.detach().item()),
            "loss_proprio": proprio_loss_val,
            "sigma_mean": float(sigma.float().mean().item()),
            "output_std": float(output.detach().float().std().item()),
            "action_std": float(action.detach().float().std().item()),
            "prope_active": float(viewmats is not None),
            **sf_logs,
        }
        return loss, logs

    def _latent_frame_condition_targets(
        self,
        condition: torch.Tensor,
        T_latent: int,
    ) -> torch.Tensor:
        if str(getattr(self.tcfg, "action_condition_timestep_rate", "raw")) == "latent":
            target = condition[:, :T_latent, ...]
        else:
            indices = wan_causal_group_indices(
                T_latent,
                device=condition.device,
            )[:, -1]
            needed = int(indices.max().item()) + 1
            if condition.shape[1] < needed:
                pad_shape = list(condition.shape)
                pad_shape[1] = needed - condition.shape[1]
                pad = condition[:, -1:, ...].expand(*pad_shape)
                condition = torch.cat([condition, pad], dim=1)
            target = condition.index_select(1, indices)
        if target.shape[1] < T_latent:
            pad_shape = list(target.shape)
            pad_shape[1] = T_latent - target.shape[1]
            pad = target[:, -1:, ...].expand(*pad_shape)
            target = torch.cat([target, pad], dim=1)
        elif target.shape[1] > T_latent:
            target = target[:, :T_latent, ...]
        return target

    def _compute_proprio_loss(
        self,
        proprio_pred,
        condition: torch.Tensor,
        *,
        T_latent: int,
        history_frames: int,
    ) -> torch.Tensor:
        schema = str(getattr(self.mcfg, "action_schema", "fixed"))
        target = self._latent_frame_condition_targets(condition, T_latent)
        if schema == "fixed":
            if not torch.is_tensor(proprio_pred):
                raise TypeError("fixed causal proprio head must return a tensor")
            target = target.to(device=proprio_pred.device, dtype=proprio_pred.dtype)
            per_frame = F.mse_loss(proprio_pred, target, reduction="none").mean(dim=-1)
            if bool(getattr(self.tcfg, "causal_proprio_future_only", True)):
                frame_keep = torch.ones_like(per_frame)
                frame_keep[:, :history_frames] = 0.0
                return (per_frame * frame_keep).sum() / frame_keep.sum().clamp(min=1.0)
            return per_frame.mean()

        if schema == "arm_slot":
            if not isinstance(proprio_pred, dict):
                raise TypeError("arm-slot causal proprio head must return a dict")
            values_pred = proprio_pred["values"]
            mask_logits = proprio_pred["mask_logits"]
            action_dim = int(self.mcfg.action_dim)
            target = target.to(device=values_pred.device, dtype=values_pred.dtype)
            values_target = target[..., :action_dim]
            mask_target = target[..., action_dim : action_dim + 1].clamp(0.0, 1.0)
            if values_pred.shape != values_target.shape:
                raise ValueError(
                    f"proprio value shape mismatch: pred={tuple(values_pred.shape)} "
                    f"target={tuple(values_target.shape)}"
                )
            slot_weight = mask_target.squeeze(-1)
            if bool(getattr(self.tcfg, "causal_proprio_future_only", True)):
                frame_keep = torch.ones(
                    slot_weight.shape[:2],
                    device=slot_weight.device,
                    dtype=slot_weight.dtype,
                )
                frame_keep[:, :history_frames] = 0.0
                slot_weight = slot_weight * frame_keep[:, :, None]
            pose_mse = F.mse_loss(values_pred, values_target, reduction="none").mean(dim=-1)
            pose_loss = (pose_mse * slot_weight).sum() / slot_weight.sum().clamp(min=1.0)
            mask_bce = F.binary_cross_entropy_with_logits(
                mask_logits.float(),
                mask_target.float(),
                reduction="none",
            ).squeeze(-1)
            if bool(getattr(self.tcfg, "causal_proprio_future_only", True)):
                mask_bce = mask_bce * frame_keep[:, :, None]
                mask_denom = frame_keep.sum().clamp(min=1.0) * mask_bce.shape[-1]
            else:
                mask_denom = torch.tensor(
                    mask_bce.numel(),
                    device=mask_bce.device,
                    dtype=mask_bce.dtype,
                ).clamp(min=1.0)
            mask_loss = mask_bce.sum() / mask_denom
            return pose_loss + float(getattr(self.tcfg, "causal_proprio_mask_weight", 0.05)) * mask_loss.to(
                dtype=pose_loss.dtype
            )

        raise ValueError(f"Unsupported causal proprio action_schema={schema!r}")

    def train(self, train_dataset, val_dataset=None) -> None:
        assert self.model is not None and self.optimizer is not None
        sampler = DistributedSampler(train_dataset, shuffle=True) if self.world_size > 1 else None
        dataloader = DataLoader(
            train_dataset,
            batch_size=int(self.mcfg.batch_size),
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=int(os.environ.get("COACHWORLD_CAUSAL_NUM_WORKERS", "0")),
            pin_memory=True,
            drop_last=True,
            collate_fn=_wm_collate_causal,
        )
        loader_iter = _infinite_loader(dataloader)

        output_dir = Path(self.tcfg.output_dir)
        if self.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()
        max_steps = int(self.mcfg.max_train_steps)
        grad_accum = max(1, int(self.mcfg.grad_accum_steps))
        self.optimizer.zero_grad(set_to_none=True)

        logger.info(
            "Starting causal training: max_steps=%d batch=%d grad_accum=%d lr=%.2e",
            max_steps,
            int(self.mcfg.batch_size),
            grad_accum,
            float(self.mcfg.learning_rate),
        )

        while self.global_step < max_steps:
            self.global_step += 1
            batch = next(loader_iter)
            loss, logs = self._compute_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite causal loss at step {self.global_step}: {loss.item()}")
            (loss / grad_accum).backward()
            if self.global_step == 1 or self.global_step % int(self.tcfg.log_interval) == 0:
                logs["grad_l2_action"] = self._action_grad_l2()
            if self.global_step % grad_accum == 0:
                if float(self.tcfg.gradient_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.tcfg.gradient_clip))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.global_step == 1 or self.global_step % int(self.tcfg.log_interval) == 0:
                logger.info(
                    "Causal step %d/%d loss=%.6f main=%.6f proprio=%.6f sigma=%.3f out_std=%.4f action_std=%.4f prope=%d grad_l2_action=%.4g",
                    self.global_step,
                    max_steps,
                    logs["loss"],
                    logs.get("loss_main", logs["loss"]),
                    logs.get("loss_proprio", 0.0),
                    logs["sigma_mean"],
                    logs["output_std"],
                    logs["action_std"],
                    int(logs.get("prope_active", 0.0)),
                    logs.get("grad_l2_action", 0.0),
                )
                if logs.get("self_forcing_active", 0.0):
                    logger.info(
                        "Causal self-forcing step %d: history_frames=%d sigma=%.3f context_delta=%.5f",
                        self.global_step,
                        int(logs.get("self_forcing_history_frames", 0.0)),
                        float(logs.get("self_forcing_sigma", 0.0)),
                        float(logs.get("self_forcing_context_delta", 0.0)),
                    )

            if (
                val_dataset is not None
                and int(self.tcfg.val_interval) > 0
                and self.global_step % int(self.tcfg.val_interval) == 0
            ):
                val_loss = self.validate(val_dataset)
                if self.is_main:
                    logger.info("Causal validation at step %d: avg_loss=%.6f", self.global_step, val_loss)

            if int(self.tcfg.save_interval) > 0 and self.global_step % int(self.tcfg.save_interval) == 0:
                self.save_checkpoint(self.global_step, output_dir)

        if max_steps > 0 and (
            int(self.tcfg.save_interval) <= 0 or self.global_step % int(self.tcfg.save_interval) != 0
        ):
            self.save_checkpoint(self.global_step, output_dir)
        logger.info("Causal training complete: %d steps", self.global_step)

    @torch.no_grad()
    def validate(self, val_dataset) -> float:
        assert self.model is not None
        was_training = self.model.training
        self.model.eval()
        sampler = DistributedSampler(val_dataset, shuffle=False) if self.world_size > 1 else None
        dataloader = DataLoader(
            val_dataset,
            batch_size=int(self.mcfg.batch_size),
            sampler=sampler,
            shuffle=False,
            num_workers=int(os.environ.get("COACHWORLD_CAUSAL_NUM_WORKERS", "0")),
            pin_memory=True,
            drop_last=True,
            collate_fn=_wm_collate_causal,
        )
        max_batches = max(1, int(self.tcfg.val_video_num))
        losses = []
        for idx, batch in enumerate(dataloader):
            if idx >= max_batches:
                break
            loss, _ = self._compute_loss(batch)
            losses.append(float(loss.detach().item()))
        local_sum = torch.tensor(
            [sum(losses), len(losses)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        if was_training:
            self.model.train()
        return float((local_sum[0] / local_sum[1].clamp_min(1)).item())

    def save_checkpoint(self, step: int, output_dir: Path) -> None:
        assert self.model is not None and self.optimizer is not None
        self._barrier()
        if not self.is_main:
            self._barrier()
            return
        ckpt_dir = output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_state = {
            key: value.detach().cpu()
            for key, value in self._model_module().state_dict().items()
        }
        torch.save(model_state, ckpt_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        state = {
            "step": int(step),
            "backbone": str(self.mcfg.backbone),
            "latent_frames": int(self.latent_frames),
            "latent_height": int(self.latent_height),
            "latent_width": int(self.latent_width),
            "seq_len": int(self.seq_len),
        }
        (ckpt_dir / "trainer_state.json").write_text(json.dumps(state, indent=2, sort_keys=True))
        logger.info("Causal checkpoint saved: %s", ckpt_dir)
        self._barrier()

    def _load_model_only(self, path: str | Path) -> None:
        assert self.model is not None
        ckpt_dir = Path(path)
        model_path = ckpt_dir / "model.pt" if ckpt_dir.is_dir() else ckpt_dir
        if not model_path.exists():
            raise FileNotFoundError(f"Causal init checkpoint missing: {model_path}")
        state = torch.load(model_path, map_location="cpu")
        self._model_module().load_state_dict(state, strict=True)
        logger.info("Loaded causal model weights from %s", model_path)

    def load_checkpoint(self, path: str | Path) -> None:
        assert self.model is not None and self.optimizer is not None
        ckpt_dir = Path(path)
        model_path = ckpt_dir / "model.pt"
        optim_path = ckpt_dir / "optimizer.pt"
        state_path = ckpt_dir / "trainer_state.json"
        if not model_path.exists() or not optim_path.exists() or not state_path.exists():
            raise FileNotFoundError(f"Invalid causal checkpoint directory: {ckpt_dir}")
        self._model_module().load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        self.optimizer.load_state_dict(torch.load(optim_path, map_location=self.device))
        state = json.loads(state_path.read_text())
        self.global_step = int(state["step"])
        logger.info("Resumed causal checkpoint %s at step %d", ckpt_dir, self.global_step)
