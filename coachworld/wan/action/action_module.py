"""Action encoder for WAN 2.2 DiT action conditioning.

Action path:
  arm-slot robot_action storage (B, T, slots, action_dim + exists_mask)
    → split continuous values from exists metadata
    → domain-aware per-frame action context tokens
    → Wan causal grouped action modulation tokens
    → dedicated action K/V plus Wan timestep/AdaLN modulation.

The concat-with-text approach puts hundreds of text tokens and a short action
sequence in one softmax, giving action weak attention mass. CoachWorld instead uses
a modality-dedicated action K/V path: visual Q/O are shared with text, text K/V
stay pretrained, and action K/V are separate modules warm-started from text K/V.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DomainResidualLinear(nn.Module):
    """Shared linear projection plus a zero-init per-domain residual.

    Cross-embodiment action conditioning needs domain-specific corrections, but
    replacing one shared projector with independently initialized per-domain
    projectors makes small mixed-domain runs cold-start three separate action
    spaces.  This layer starts exactly as a shared Linear and lets each domain
    learn only a residual adapter.
    """

    def __init__(self, num_domains: int, input_dim: int, output_dim: int):
        super().__init__()
        self.num_domains = int(num_domains)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.num_domains <= 0:
            raise ValueError(f"num_domains must be positive, got {self.num_domains}")
        self.shared = nn.Linear(self.input_dim, self.output_dim)
        self.delta_weight = nn.Parameter(
            torch.zeros(self.num_domains, self.input_dim, self.output_dim)
        )
        self.delta_bias = nn.Parameter(torch.zeros(self.num_domains, self.output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.shared.weight)
        nn.init.zeros_(self.shared.bias)
        nn.init.zeros_(self.delta_weight)
        nn.init.zeros_(self.delta_bias)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"DomainResidualLinear expects x=(N,D), got {tuple(x.shape)}")
        domain_ids = domain_ids.to(device=x.device, dtype=torch.long).view(-1)
        if domain_ids.numel() != x.shape[0]:
            raise ValueError(
                f"domain_ids length {domain_ids.numel()} does not match x batch {x.shape[0]}"
            )
        if int(domain_ids.min()) < 0 or int(domain_ids.max()) >= self.num_domains:
            raise ValueError(
                f"domain_ids must be in [0,{self.num_domains}), got "
                f"min={int(domain_ids.min())} max={int(domain_ids.max())}"
            )
        shared = self.shared(x)
        delta_weight = self.delta_weight.index_select(0, domain_ids)
        delta_bias = self.delta_bias.index_select(0, domain_ids)
        delta = torch.bmm(x.unsqueeze(1), delta_weight).squeeze(1) + delta_bias
        return shared + delta


class ActionEncoder(nn.Module):
    """Project per-frame robot conditions to DiT hidden-dim tokens, LayerNormed.

    Args:
        action_dim: Per-frame condition dimensionality.
        hidden_dim: DiT hidden dim — MUST match WAN 2.2's dim (3072 for 5B).
        mid_dims: Intermediate MLP widths. Defaults to (128, 256), chosen small
            so the encoder is a cheap projector rather than a deep processor.

    Output statistics:
        Final LayerNorm (affine weight=1, bias=0 at init) brings each action
        token to unit variance regardless of upstream weight scale. This
        matches the effective magnitude of text_tokens (T5 through WAN's
        pretrained 4096→3072 MLP outputs ~unit variance), so when the
        dedicated action cross-attn in each DiT block warm-starts its K/V
        projections from the text K/V weights, the action attention output
        comes out at a comparable magnitude rather than 30× smaller.
    """

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 3072,
        mid_dims: tuple[int, int] = (128, 256),
        max_time_steps: int = 512,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_time_steps = int(max_time_steps)
        if self.max_time_steps < 1:
            raise ValueError(f"max_time_steps must be >= 1, got {self.max_time_steps}")
        self.time_embedding = nn.Parameter(torch.zeros(self.max_time_steps, hidden_dim))

        self.net = nn.Sequential(
            nn.Linear(action_dim, mid_dims[0]),
            nn.GELU(),
            nn.Linear(mid_dims[0], mid_dims[1]),
            nn.GELU(),
            nn.Linear(mid_dims[1], hidden_dim),
            # LayerNorm — the magnitude-equalizer. Without it, Xavier-init
            # through the 3-MLP produces action_tokens ~0.03 per feature,
            # ~30× smaller than text_tokens. LayerNorm with default affine
            # (γ=1, β=0) scales outputs to unit variance.
            nn.LayerNorm(hidden_dim),
        )

        # Xavier init on the three linears. The LayerNorm uses default affine
        # (weight=1, bias=0) from PyTorch, which gives unit-variance output.
        # We do NOT zero-init the final Linear: V1.0 tried this and hit NaN
        # loss under bf16 backward (the degenerate zero state prevents grads
        # from escaping the encoder).
        for layer in (self.net[0], self.net[2], self.net[4]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

    def forward(self, action: torch.Tensor, domain_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            action: (B, T, action_dim) robot conditions, normalized by dataset stats.

        Returns:
            (B, T, hidden_dim) unit-variance action tokens ready to feed into
            the dedicated action K/V of each DiT block's cross-attention.
        """
        if action.shape[1] > self.max_time_steps:
            raise ValueError(
                f"action sequence length {action.shape[1]} exceeds "
                f"max_time_steps={self.max_time_steps}"
            )
        tokens = self.net(action)
        time = self.time_embedding[: action.shape[1]].to(
            device=tokens.device, dtype=tokens.dtype
        ).view(1, action.shape[1], self.hidden_dim)
        return tokens + time


class EefProjectionEncoder(nn.Module):
    """Encode image-space EEF projections as action K/V context tokens.

    Inputs are aligned to latent frames with shape ``(B,F,V,*)`` or
    ``(B,F,V,S,*)``:
    ``uv`` in image pixels, ``depth`` in camera coordinates, ``valid`` as the
    in-image visibility flag, and ``image_hw`` as ``(height,width)``. Missing
    projections should carry NaN uv/depth; they are converted to a zero feature
    with ``present=0`` while finite but out-of-bounds projections keep their
    normalized coordinates and use ``valid=0``.
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        mid_dims: tuple[int, int] = (64, 128),
        max_time_steps: int = 512,
        max_views: int = 8,
        max_slots: int = 4,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_time_steps = int(max_time_steps)
        self.max_views = int(max_views)
        self.max_slots = int(max_slots)
        if self.max_time_steps < 1:
            raise ValueError(f"max_time_steps must be >= 1, got {self.max_time_steps}")
        if self.max_views < 1:
            raise ValueError(f"max_views must be >= 1, got {self.max_views}")
        if self.max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {self.max_slots}")
        self.time_embedding = nn.Parameter(torch.zeros(self.max_time_steps, hidden_dim))
        self.view_embedding = nn.Parameter(torch.zeros(self.max_views, hidden_dim))
        self.slot_embedding = nn.Parameter(torch.zeros(self.max_slots, hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(5, mid_dims[0]),
            nn.GELU(),
            nn.Linear(mid_dims[0], mid_dims[1]),
            nn.GELU(),
            nn.Linear(mid_dims[1], hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        for layer in (self.net[0], self.net[2], self.net[4]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.view_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.slot_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        uv: torch.Tensor,
        depth: torch.Tensor,
        valid: torch.Tensor,
        image_hw: torch.Tensor,
    ) -> torch.Tensor:
        if uv.dim() == 4 and uv.shape[-1] == 2:
            uv = uv.unsqueeze(3)
            depth = depth.unsqueeze(3)
            valid = valid.unsqueeze(3)
            image_hw = image_hw.unsqueeze(3)
        elif uv.dim() != 5 or uv.shape[-1] != 2:
            raise ValueError(
                f"eef_uv must have shape (B,F,V,2) or (B,F,V,S,2), got {tuple(uv.shape)}"
            )
        if depth.shape != uv.shape[:-1]:
            raise ValueError(
                f"eef_depth must have shape {tuple(uv.shape[:-1])}, got {tuple(depth.shape)}"
            )
        if valid.shape != uv.shape[:-1]:
            raise ValueError(
                f"eef_valid must have shape {tuple(uv.shape[:-1])}, got {tuple(valid.shape)}"
            )
        if image_hw.dim() == 4 and image_hw.shape == uv.shape[:3] + uv.shape[-1:]:
            image_hw = image_hw.unsqueeze(3).expand(*uv.shape)
        if image_hw.shape != uv.shape:
            raise ValueError(
                f"eef_image_hw must have shape {tuple(uv.shape)} or "
                f"{tuple(uv.shape[:3] + uv.shape[-1:])}, got {tuple(image_hw.shape)}"
            )
        B, F, V, S, _ = uv.shape
        if F > self.max_time_steps:
            raise ValueError(
                f"EEF projection sequence length {F} exceeds max_time_steps={self.max_time_steps}"
            )
        if V > self.max_views:
            raise ValueError(f"EEF projection views {V} exceeds max_views={self.max_views}")
        if S > self.max_slots:
            raise ValueError(f"EEF projection slots {S} exceeds max_slots={self.max_slots}")

        device = self.time_embedding.device
        dtype = self.time_embedding.dtype
        uv = uv.to(device=device, dtype=dtype)
        depth = depth.to(device=device, dtype=dtype)
        image_hw = image_hw.to(device=device, dtype=dtype).clamp_min(1.0)
        present = torch.isfinite(uv).all(dim=-1) & torch.isfinite(depth)
        uv = torch.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)

        height = image_hw[..., 0].clamp_min(2.0)
        width = image_hw[..., 1].clamp_min(2.0)
        x_norm = (uv[..., 0] / (width - 1.0)) * 2.0 - 1.0
        y_norm = (uv[..., 1] / (height - 1.0)) * 2.0 - 1.0
        present_f = present.to(dtype=dtype)
        valid_f = (valid.to(device=uv.device).bool() & present).to(dtype=dtype)
        features = torch.stack(
            [
                x_norm * present_f,
                y_norm * present_f,
                depth * present_f,
                valid_f,
                present_f,
            ],
            dim=-1,
        )
        tokens = self.net(features)
        time = self.time_embedding[:F].to(device=tokens.device, dtype=tokens.dtype)
        view = self.view_embedding[:V].to(device=tokens.device, dtype=tokens.dtype)
        slot = self.slot_embedding[:S].to(device=tokens.device, dtype=tokens.dtype)
        tokens = (
            tokens
            + time.view(1, F, 1, 1, self.hidden_dim)
            + view.view(1, 1, V, 1, self.hidden_dim)
            + slot.view(1, 1, 1, S, self.hidden_dim)
        )
        return tokens.reshape(B, F * V * S, self.hidden_dim)


class ArmSlotActionEncoder(nn.Module):
    """BWM-style grouped action encoder with domain-aware projection.

    Storage input remains ``(B,T,S,D+1)`` so existing video_latent files stay
    usable.  The model contract is stricter:

    * ``values = action[..., :D]`` are the only continuous action values.
    * ``exists = action[..., D]`` is metadata/gating, never projected as an
      action coordinate.
    * cross-attention context keeps one token per time/slot pair, so the model
      can distinguish left/right or single/dual-arm slots structurally.
    * per-latent modulation tokens use Wan/BWM 4-frame causal grouping:
      latent 0 sees ``[frame0, frame0, frame0, frame0]``; later latents see
      ``[(i-1)*4+1, ..., i*4]``.
    """

    def __init__(
        self,
        action_dim: int = 15,
        hidden_dim: int = 3072,
        max_arm_slots: int = 2,
        mid_dims: tuple[int, int] = (128, 256),
        max_time_steps: int = 512,
        num_domains: int = 1,
        domain_prompt_tokens: int = 0,
        domain_aware_projection: bool = False,
        domain_aware_group_projection: bool = False,
        enable_group_modulation: bool = True,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_arm_slots = int(max_arm_slots)
        self.num_domains = int(num_domains)
        self.domain_prompt_tokens = int(domain_prompt_tokens)
        self.domain_aware_projection = bool(domain_aware_projection)
        self.domain_aware_group_projection = bool(domain_aware_group_projection)
        self.enable_group_modulation = bool(enable_group_modulation)
        if self.max_arm_slots < 1:
            raise ValueError(f"max_arm_slots must be >= 1, got {self.max_arm_slots}")
        if self.num_domains < 1:
            raise ValueError(f"num_domains must be >= 1, got {self.num_domains}")
        if self.domain_prompt_tokens < 0:
            raise ValueError(
                f"domain_prompt_tokens must be >= 0, got {self.domain_prompt_tokens}"
            )
        self.max_time_steps = int(max_time_steps)
        if self.max_time_steps < 1:
            raise ValueError(f"max_time_steps must be >= 1, got {self.max_time_steps}")

        self.frame_input_dim = self.max_arm_slots * self.action_dim + self.max_arm_slots
        self.group_input_dim = self.frame_input_dim * 4
        self.slot_embedding = nn.Parameter(torch.zeros(self.max_arm_slots, hidden_dim))
        self.time_embedding = nn.Parameter(torch.zeros(self.max_time_steps, hidden_dim))
        if self.domain_aware_projection:
            self.slot_projection = DomainResidualLinear(
                self.num_domains, self.action_dim, mid_dims[0]
            )
        else:
            self.slot_projection = nn.Linear(self.action_dim, mid_dims[0])
        self.context_projection = None
        self.group_projection = None
        self.group_net = None
        if self.enable_group_modulation:
            if self.domain_aware_projection:
                self.context_projection = DomainResidualLinear(
                    self.num_domains, self.frame_input_dim, mid_dims[0]
                )
            else:
                self.context_projection = nn.Linear(self.frame_input_dim, mid_dims[0])
            if self.domain_aware_group_projection:
                if not self.domain_aware_projection:
                    raise ValueError(
                        "domain_aware_group_projection requires domain_aware_projection"
                    )
                self.group_projection = DomainResidualLinear(
                    self.num_domains, self.group_input_dim, hidden_dim * 4
                )
            else:
                self.group_projection = nn.Linear(self.group_input_dim, hidden_dim * 4)
            self.group_net = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        else:
            if self.domain_aware_group_projection:
                raise ValueError("domain_aware_group_projection requires enable_group_modulation")
        self.slot_net = nn.Sequential(
            nn.GELU(),
            nn.Linear(mid_dims[0], mid_dims[1]),
            nn.GELU(),
            nn.Linear(mid_dims[1], hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        if self.domain_prompt_tokens:
            self.domain_prompt = nn.Embedding(
                self.num_domains,
                self.domain_prompt_tokens * hidden_dim,
            )
        else:
            self.domain_prompt = None

        for projection in (self.slot_projection, self.context_projection, self.group_projection):
            if projection is None:
                continue
            if isinstance(projection, nn.Linear):
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)
        for layer in (self.slot_net[1], self.slot_net[3]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.slot_embedding, mean=0.0, std=0.02)
        if self.group_net is not None:
            nn.init.ones_(self.group_net[0].weight)
            nn.init.zeros_(self.group_net[0].bias)
            nn.init.zeros_(self.group_net[2].weight)
            nn.init.zeros_(self.group_net[2].bias)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        if self.domain_prompt is not None:
            nn.init.normal_(self.domain_prompt.weight, mean=0.0, std=0.02)

    def _require_domain_ids(self, domain_ids: torch.Tensor | None, batch_size: int, device) -> torch.Tensor:
        if (
            self.domain_aware_projection
            or self.domain_aware_group_projection
            or self.domain_prompt is not None
        ):
            if domain_ids is None:
                raise ValueError(
                    "domain_ids are required when domain-aware projection or "
                    "domain prompts are enabled"
                )
        if domain_ids is None:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        domain_ids = domain_ids.to(device=device, dtype=torch.long).view(-1)
        if domain_ids.numel() != batch_size:
            raise ValueError(
                f"domain_ids must have shape (B,), got {tuple(domain_ids.shape)} "
                f"for batch_size={batch_size}"
            )
        if int(domain_ids.min()) < 0 or int(domain_ids.max()) >= self.num_domains:
            raise ValueError(
                f"domain_ids must be in [0,{self.num_domains}), got "
                f"min={int(domain_ids.min())} max={int(domain_ids.max())}"
            )
        return domain_ids

    def _split_values_exists(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action.dim() != 4:
            raise ValueError(
                "arm-slot action must have shape (B,T,S,D+1), "
                f"got {tuple(action.shape)}"
            )
        if action.shape[2] != self.max_arm_slots:
            raise ValueError(
                f"arm-slot count mismatch: got {action.shape[2]}, "
                f"expected {self.max_arm_slots}"
            )
        if action.shape[3] != self.action_dim + 1:
            raise ValueError(
                f"arm-slot feature dim mismatch: got {action.shape[3]}, "
                f"expected action_dim+1={self.action_dim + 1}"
            )
        values = action[..., : self.action_dim]
        exists = action[..., self.action_dim].to(dtype=values.dtype).clamp(0.0, 1.0)
        return values, exists

    def frame_inputs(self, action: torch.Tensor) -> torch.Tensor:
        """Return canonical per-frame action input ``(B,T,S*D+S)``."""
        values, exists = self._split_values_exists(action)
        values = values * exists[..., None]
        return torch.cat(
            [
                values.reshape(values.shape[0], values.shape[1], -1),
                exists,
            ],
            dim=-1,
        )

    def _project_slots(
        self,
        values: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T, S, D = values.shape
        flat = values.reshape(B * T * S, D)
        if self.domain_aware_projection:
            flat_domain_ids = domain_ids.view(B, 1, 1).expand(B, T, S).reshape(B * T * S)
            projected = self.slot_projection(flat, flat_domain_ids)
        else:
            projected = self.slot_projection(flat)
        return projected.reshape(B, T, S, -1)

    def _project_context(
        self,
        frame_input: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.context_projection is None:
            raise RuntimeError("context_projection is disabled")
        B, T, D = frame_input.shape
        flat = frame_input.reshape(B * T, D)
        if self.domain_aware_projection:
            flat_domain_ids = domain_ids.view(B, 1).expand(B, T).reshape(B * T)
            projected = self.context_projection(flat, flat_domain_ids)
        else:
            projected = self.context_projection(flat)
        return projected.reshape(B, T, -1)

    def _project_group(
        self,
        grouped_input: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.group_projection is None:
            raise RuntimeError("group_projection is disabled")
        B, T, D = grouped_input.shape
        flat = grouped_input.reshape(B * T, D)
        if self.domain_aware_group_projection:
            flat_domain_ids = domain_ids.view(B, 1).expand(B, T).reshape(B * T)
            projected = self.group_projection(flat, flat_domain_ids)
        else:
            projected = self.group_projection(flat)
        return projected.reshape(B, T, -1)

    def forward(self, action: torch.Tensor, domain_ids: torch.Tensor | None = None) -> torch.Tensor:
        values, exists = self._split_values_exists(action)
        B, T, S, _ = values.shape
        domain_ids = self._require_domain_ids(domain_ids, B, values.device)
        if T > self.max_time_steps:
            raise ValueError(
                f"action sequence length {T} exceeds max_time_steps={self.max_time_steps}"
            )

        projected = self._project_slots(values, domain_ids)
        tokens = self.slot_net(projected)
        tokens = tokens + self.slot_embedding.to(device=tokens.device, dtype=tokens.dtype).view(
            1, 1, S, self.hidden_dim
        )
        tokens = tokens + self.time_embedding[:T].to(
            device=tokens.device, dtype=tokens.dtype
        ).view(1, T, 1, self.hidden_dim)
        tokens = tokens * exists[..., None]
        tokens = tokens.reshape(B, T * S, self.hidden_dim)
        if self.domain_prompt is not None:
            prompts = self.domain_prompt(domain_ids).to(dtype=tokens.dtype)
            prompts = prompts.view(B, self.domain_prompt_tokens, self.hidden_dim)
            tokens = torch.cat([prompts, tokens], dim=1)
        return tokens

    def encode_group_modulation(
        self,
        action: torch.Tensor,
        *,
        num_latent_frames: int,
        vae_temporal_stride: int = 4,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-Wan-latent action modulation ``(B,T_latent,H)``."""
        if self.group_net is None:
            raise RuntimeError("group modulation is disabled for this ArmSlotActionEncoder")
        from .temporal_grouping import group_wan_causal_actions

        frame_input = self.frame_inputs(action)
        B = frame_input.shape[0]
        domain_ids = self._require_domain_ids(domain_ids, B, frame_input.device)
        groups = group_wan_causal_actions(
            frame_input,
            num_latent_frames=int(num_latent_frames),
            vae_temporal_stride=int(vae_temporal_stride),
        )
        grouped = groups.chunks
        if grouped.shape[-1] != self.group_input_dim:
            raise ValueError(
                f"grouped action width mismatch: got {grouped.shape[-1]}, "
                f"expected {self.group_input_dim}"
            )
        projected = self._project_group(grouped, domain_ids)
        return self.group_net(projected)


class ActionDecoder(nn.Module):
    """Inverse of ActionEncoder — predicts per-frame actions from DiT visual
    tokens. Used purely for an auxiliary MSE loss that forces the DiT to route
    action-relevant information through its visual representations.

    Without this auxiliary signal, the DiT's cross-attn gate on the action
    path converges toward zero (the model learns to ignore action) because
    text + history already over-determine the next-frame prediction on most
    training samples. LDA-1B (arXiv 2602.12215) uses the same mechanism —
    a separate action prediction head — to keep action representations
    informative. Ctrl-World side-steps this differently by making action the
    sole conditioning input; we have a parallel text path so we need the
    explicit auxiliary loss.

    Args:
        hidden_dim: DiT hidden dim (must match WanModelAction.dim).
        action_dim: Robot action dimensionality (7 Franka, 14 AGIBot).
        mid_dim: Middle MLP width. 512 is enough; this decoder is small.

    Forward:
        visual_tokens: (B, L_vis, hidden_dim) — post-last-block, pre-head.
            L_vis = Fp * Hp * Wp where (Fp, Hp, Wp) = grid_sizes.
        grid_sizes: tensor([Fp, Hp, Wp]).

    Returns:
        (B, Fp, action_dim) — predicted action per latent frame. Caller
        is responsible for mask / loss reduction.
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        action_dim: int = 7,
        mid_dim: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, action_dim),
        )
        # Xavier on the two Linears. LayerNorm keeps default (γ=1, β=0).
        for m in (self.net[1], self.net[3]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self, visual_tokens: torch.Tensor, grid_sizes
    ) -> torch.Tensor:
        """Pool visual tokens per latent frame, then MLP → action.

        Spatial pooling (mean over Hp*Wp) discards spatial layout — we only
        need the per-frame action-relevant content. Keeps the decoder tiny
        (~2M params) and alignment with the action target at (B, T, D)
        trivial.
        """
        B, L_vis, H = visual_tokens.shape
        # grid_sizes may be torch.Tensor or tuple; normalize.
        if torch.is_tensor(grid_sizes):
            Fp, Hp, Wp = [int(v) for v in grid_sizes.tolist()]
        else:
            Fp, Hp, Wp = int(grid_sizes[0]), int(grid_sizes[1]), int(grid_sizes[2])
        assert L_vis == Fp * Hp * Wp, (
            f"L_vis={L_vis} != Fp*Hp*Wp={Fp*Hp*Wp}"
        )
        # (B, L_vis, H) → (B, Fp, Hp*Wp, H) → mean over Hp*Wp → (B, Fp, H)
        tokens = visual_tokens.view(B, Fp, Hp * Wp, H).mean(dim=2)
        return self.net(tokens)


class ArmSlotActionDecoder(nn.Module):
    """Slot-aware auxiliary decoder for arm-slot action conditioning.

    Predicts active-slot EEF values and slot presence from final visual tokens.
    The trainer owns the loss reduction because it needs access to CFG drop
    masks and future-only masking.
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        action_dim: int = 10,
        max_arm_slots: int = 2,
        mid_dim: int = 512,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.max_arm_slots = int(max_arm_slots)
        if self.max_arm_slots < 1:
            raise ValueError(f"max_arm_slots must be >= 1, got {self.max_arm_slots}")
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
        )
        self.value_head = nn.Linear(mid_dim, self.max_arm_slots * self.action_dim)
        self.mask_head = nn.Linear(mid_dim, self.max_arm_slots)
        for m in (self.trunk[1], self.value_head, self.mask_head):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, visual_tokens: torch.Tensor, grid_sizes) -> dict[str, torch.Tensor]:
        B, L_vis, H = visual_tokens.shape
        if torch.is_tensor(grid_sizes):
            Fp, Hp, Wp = [int(v) for v in grid_sizes.tolist()]
        else:
            Fp, Hp, Wp = int(grid_sizes[0]), int(grid_sizes[1]), int(grid_sizes[2])
        if L_vis != Fp * Hp * Wp:
            raise ValueError(f"L_vis={L_vis} != Fp*Hp*Wp={Fp*Hp*Wp}")
        pooled = visual_tokens.view(B, Fp, Hp * Wp, H).mean(dim=2)
        features = self.trunk(pooled)
        values = self.value_head(features).view(B, Fp, self.max_arm_slots, self.action_dim)
        mask_logits = self.mask_head(features).view(B, Fp, self.max_arm_slots, 1)
        return {"values": values, "mask_logits": mask_logits}
