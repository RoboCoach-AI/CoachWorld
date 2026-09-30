# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import importlib.metadata as _coachworld_importlib_metadata
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
# Some cluster venvs have broken torch dist-info where
# importlib.metadata.version("torch") returns None. diffusers imports
# accelerate during ModelMixin import, and accelerate parses that value.
# Prefer torch.__version__ for this one broken metadata case.
_coachworld_orig_metadata_version = _coachworld_importlib_metadata.version
if not getattr(_coachworld_importlib_metadata.version, "_coachworld_torch_none_guard", False):
    def _coachworld_metadata_version(name):
        try:
            version = _coachworld_orig_metadata_version(name)
        except _coachworld_importlib_metadata.PackageNotFoundError:
            if name == "torch":
                return str(getattr(torch, "__version__", "0"))
            raise
        if name == "torch" and version is None:
            return str(getattr(torch, "__version__", "0"))
        return version

    _coachworld_metadata_version._coachworld_torch_none_guard = True
    _coachworld_importlib_metadata.version = _coachworld_metadata_version
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.distributed.device_mesh import DeviceMesh
try:
    from torch.distributed.fsdp import fully_shard
except ImportError:
    from torch.distributed._composable.fsdp import fully_shard
from .attention import flash_attention
from .action.temporal_grouping import group_wan_causal_actions

__all__ = ['WanModelAction']


def apply_dense_relative_action_chunks(
    chunks: torch.Tensor,
    baseline: torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    """Apply dense relative action semantics.

    Structured CoachWorld 14D actions are [absolute_state, delta]. Relative dense
    conditioning should remove absolute pose offsets, but must not turn the
    explicit delta half into a second derivative.
    """
    if action_dim % 2 == 0 and action_dim >= 14:
        state_dim = action_dim // 2
        out = chunks.clone()
        out[..., :state_dim] = (
            out[..., :state_dim] - baseline[:, :, None, :state_dim]
        )
        return out
    return chunks - baseline[:, :, None, :]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, *, num_cameras: int = 1, position_mode: str = "global"):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    # print(grid_sizes.shape, len(grid_sizes.tolist()), grid_sizes.tolist()[0])
    f, h, w = grid_sizes.tolist()
    num_cameras = int(num_cameras)
    position_mode = str(position_mode)
    if position_mode not in {"global", "view_local"}:
        raise ValueError(f"Unsupported multi-view position mode: {position_mode!r}")
    if position_mode == "view_local":
        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}")
        if h % num_cameras != 0:
            raise ValueError(
                f"Patch-grid height {h} is not divisible by num_cameras={num_cameras}; "
                "cannot reset view-local RoPE coordinates."
            )
        h_per_view = h // num_cameras
        h_index = torch.arange(h, device=freqs[1].device) % h_per_view
        h_freqs = freqs[1].index_select(0, h_index)
    else:
        h_freqs = freqs[1][:h]
    for i in range(len(x)):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_freqs.view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        x_float = x.float()
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return torch_F.layer_norm(
            x_float,
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        ).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        *,
        num_cameras: int = 1,
        position_mode: str = "global",
        viewmats=None,
        Ks=None,
        prope_image_width: int | None = None,
        prope_image_height: int | None = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x_rope = flash_attention(
            q=rope_apply(
                q,
                grid_sizes,
                freqs,
                num_cameras=num_cameras,
                position_mode=position_mode,
            ),
            k=rope_apply(
                k,
                grid_sizes,
                freqs,
                num_cameras=num_cameras,
                position_mode=position_mode,
            ),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        prope_enabled = viewmats is not None and hasattr(self, "prope_o")
        if prope_enabled:
            from .prope import prope_qkv

            f, h, w = [int(vv) for vv in grid_sizes.tolist()]
            if int(num_cameras) <= 0 or h % int(num_cameras) != 0:
                raise ValueError(
                    f"Invalid PRoPE view layout: H={h}, num_cameras={num_cameras}"
                )
            q_p, k_p, v_p, apply_fn_o = prope_qkv(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                viewmats=viewmats,
                Ks=Ks,
                patches_x=w,
                patches_y=h // int(num_cameras),
                image_width=prope_image_width,
                image_height=prope_image_height,
            )
            q_p = q_p.permute(0, 2, 1, 3)
            k_p = k_p.permute(0, 2, 1, 3)
            v_p = v_p.permute(0, 2, 1, 3)
            x_prope = flash_attention(
                q=q_p,
                k=k_p,
                v=v_p,
                k_lens=seq_lens,
                window_size=self.window_size,
            )
            x_prope = apply_fn_o(x_prope.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # output
        x = self.o(x_rope.flatten(2))
        if prope_enabled:
            x = x + self.prope_o(x_prope.flatten(2))
        return x



class WanI2VCrossAttention(WanSelfAttention):
    """Cross-attention with a text path (WAN pretrained) and an optional
    action path (dedicated K/V branch, LDA-1B-inspired but gate-free).

    Design:
      * Q, O, norm_q: shared between text and action paths. "What the visual
        tokens query" and "how attention output is written back" don't
        depend on modality, so these stay pretrained.
      * K, V, norm_k: separate per modality. Text uses the WAN pretrained
        modules. Action introduces ``k_action`` / ``v_action`` /
        ``norm_k_action``, warm-started from the pretrained text K/V so the
        attention pattern is sensible from step 0.
      * ``v_action`` is additionally scaled down by
        ``WanModelAction.V_ACTION_INIT_SCALE`` after warm-start, so the
        initial ``attn_action`` magnitude is smaller than ``attn_text``. The
        action contribution is summed in directly:
        ``combined = attn_text + attn_action``. No learnable residual gate.
      * Rationale: an earlier learnable ``alpha_action`` Parameter (init
        0.1) collapsed toward 0 under main-loss gradient pressure on the
        video-prediction objective — the optimizer's easiest way to reduce
        loss was to set the gate to 0, since DROID text + history already
        over-determine the next frame. Removing the gate removes that
        absorbing state; the magnitude of action's contribution is now
        controlled by ``v_action.weight`` itself, which the optimizer must
        grow if (and only if) the task genuinely benefits from action.
    """

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        # Action-specific K/V branch. Same shape as text K/V so warm-start
        # (copy pretrained weights) is a direct tensor copy.
        # ``v_action.weight`` is additionally scaled down by
        # V_ACTION_INIT_SCALE in post_load_warm_start_action_kv to keep the
        # initial ``attn_action`` magnitude tiny — a hard-coded residual gate
        # alternative to a learnable ``alpha_action`` Parameter (the learnable
        # gate collapsed toward 0 in smoke tests; see CLAUDE.md 2026-04-23).
        # ``k_action`` keeps full warm-start magnitude so the attention
        # *pattern* is sensible from step 0; ``V_a`` starting small means the
        # output pattern is sensible but the contribution is small.
        self.k_action = nn.Linear(dim, dim)
        self.v_action = nn.Linear(dim, dim)
        self.norm_k_action = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, action_context=None, action_keep_mask=None):
        r"""Two-path cross-attention: text (pretrained) and action (new).

        Args:
            x: (B, L_vis, C) visual queries.
            context: (B, L_text, C) text tokens.
            context_lens: unused (kept for signature compat). The V1 varlen
                path was removed to avoid bf16 flash_attn NaN edge cases.
            action_context: Optional (B, L_act, C) action tokens. None skips
                the action path entirely (CFG uncond / text-only branch).
            action_keep_mask: Optional (B,) float tensor with 1.0 for kept
                samples and 0.0 for CFG-dropped samples. When provided,
                ``attn_action`` is multiplied by this mask per-sample AFTER
                attention, so dropped samples get an exact zero action
                contribution — matching inference's ``action_context=None``
                uncond branch. Zeroing ``action_context`` at input is NOT
                sufficient: ``k_action`` / ``v_action`` have biases, so
                ``k_action(0) = bias_k`` and ``v_action(0) = bias_v``, which
                leak a constant bias vector through attention.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Shared Q (WAN pretrained).
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        # Text path (WAN pretrained K/V).
        k_t = self.norm_k(self.k(context)).view(b, -1, n, d)
        v_t = self.v(context).view(b, -1, n, d)
        attn_text = flash_attention(q, k_t, v_t, k_lens=None)

        if action_context is not None:
            # Action path (new K/V, warm-started from text K/V at module init,
            # with ``v_action`` rescaled by V_ACTION_INIT_SCALE so initial
            # contribution is tiny — no learnable gate in front).
            k_a = self.norm_k_action(self.k_action(action_context)).view(b, -1, n, d)
            v_a = self.v_action(action_context).view(b, -1, n, d)
            attn_action = flash_attention(q, k_a, v_a, k_lens=None)
            if action_keep_mask is not None:
                # (B,) → (B, 1, 1, 1) for broadcast over (L_vis, n_heads, d_head).
                keep = action_keep_mask.to(
                    device=attn_action.device, dtype=attn_action.dtype
                ).view(-1, 1, 1, 1)
                attn_action = attn_action * keep
            combined = attn_text + attn_action
        else:
            combined = attn_text

        # Shared O (WAN pretrained).
        combined = combined.flatten(2)
        return self.o(combined)


WAN_CROSSATTENTION_CLASSES = {
    'i2v_cross_attn': WanI2VCrossAttention,
}
def mul_add(x, y, z):
    return x.float() + y.float() * z.float()


def mul_add_add(x, y, z):
    return x.float() * (1 + y) + z

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 dense_action_film=False,
                 action_control_scale=0.05,
                 action_dense_rank=128,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.dense_action_film = dense_action_film
        self.action_control_scale = float(action_control_scale)
        self.action_dense_rank = int(action_dense_rank)
        self.eps = eps
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        if dense_action_film:
            # Low-rank LingBot-style dense control. The old full-rank variant
            # added four dim x dim projections per block (~1.1B params at
            # dim=3072, 30 blocks) and OOMed 4xA800 on full-window training.
            # This keeps per-token scale/shift but factorizes the block-local
            # transform through a small rank.
            rank = max(1, int(action_dense_rank))
            self.action_injector_norm = nn.LayerNorm(rank)
            self.action_injector_layer1 = nn.Linear(rank, rank)
            self.action_scale_shift_layer = nn.Linear(rank, dim * 2)
        else:
            self.action_injector_norm = None
            self.action_injector_layer1 = None
            self.action_scale_shift_layer = None
        self._last_action_scale_abs_mean = None
        self._last_action_shift_abs_mean = None

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens=None,
        num_cameras=1,
        position_mode="global",
        viewmats=None,
        Ks=None,
        prope_image_width: int | None = None,
        prope_image_height: int | None = None,
        action_context=None,
        action_keep_mask=None,
        action_dense_context=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] — visual tokens (query).
            e(Tensor): Shape [B, 6, C] — time-embedding modulation (AdaLN-Zero).
            seq_lens(Tensor): Shape [B], length of each visual sequence.
            grid_sizes(Tensor): Shape [3]: (F, H, W) patch grid.
            freqs(Tensor): RoPE freqs, shape [1024, C / num_heads / 2].
            context(Tensor): Shape [B, L_text, C] — text cross-attn K/V only.
                Action is passed SEPARATELY via ``action_context`` (LDA-1B-style
                modality-dedicated cross-attn, V1.1). The V1.0 "concat text+action
                into one context" approach starved action of attention mass when
                L_text(512) >> L_action(17).
            context_lens: Unused (kept for signature compat). Varlen path was
                removed after bf16 flash_attn NaN in V1.0.
            action_context: Optional (B, L_act, C) action tokens. None → skip
                the action path (CFG uncond / text-only branch).
            action_keep_mask: Optional (B,) float tensor used for per-sample
                CFG dropout — forwarded to the cross-attn to zero out the
                action contribution exactly (avoids Linear-bias leakage).
            action_dense_context: Optional (B, L_vis, C) dense action/control
                feature aligned with visual tokens. When enabled, this applies
                LingBot-style feature-wise scale/shift modulation before text
                cross-attn and FFN.
        """
        if e.dim() == 3:
            e = (self.modulation + e).chunk(6, dim=1)
        elif e.dim() == 4:
            modulation = self.modulation.unsqueeze(2)  # 1, 6, 1, dim
            e = (modulation + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]

        # self-attention (time-AdaLN-modulated)
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs,
            num_cameras=num_cameras,
            position_mode=position_mode,
            viewmats=viewmats,
            Ks=Ks,
            prope_image_width=prope_image_width,
            prope_image_height=prope_image_height,
        )
        x = x + y * e[2]

        if self.action_scale_shift_layer is not None and action_dense_context is not None:
            action_h = self.action_injector_norm(action_dense_context.to(dtype=x.dtype))
            action_h = torch_F.silu(self.action_injector_layer1(action_h)) + action_h
            action_h = action_h + action_dense_context.to(dtype=x.dtype)
            scale_shift = self.action_scale_shift_layer(action_h)
            scale, shift = scale_shift.chunk(2, dim=-1)
            scale = scale * self.action_control_scale
            shift = shift * self.action_control_scale
            self._last_action_scale_abs_mean = scale.detach().float().abs().mean()
            self._last_action_shift_abs_mean = shift.detach().float().abs().mean()
            x = (1.0 + scale) * x + shift

        # cross-attention: text + action run through the dual-path
        # WanI2VCrossAttention. Normed x is reused for both paths (the same
        # query layer-norm is applied before both text and action K/V).
        dtype = context.dtype
        x = x + self.cross_attn(
            self.norm3(x.to(dtype)), context, context_lens,
            action_context=action_context, action_keep_mask=action_keep_mask,
        )

        # FFN (time-AdaLN-modulated)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x



class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        if e.dim() == 2:
                modulation = self.modulation  # 1, 2, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            modulation = self.modulation.unsqueeze(2)  # 1, 2, seq, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim),
            nn.GELU(), nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class WanModelAction(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 img_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 inject_sample_info=False,
                 action_config=None,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        if action_config is None:
            action_config = {}
        assert model_type in ['i2v', 'ti2v']
        self.model_type = model_type

        # Action config schema:
        #   action_dim:   canonical per-arm condition width
        #   mid_dims:     ActionEncoder MLP hidden widths, e.g. [128, 256]
        #   vae_temporal_stride: raw video-frame to latent-frame stride for
        #     dense action alignment.
        self._action_cfg = {
            "action_dim": action_config.get("action_dim", 7),
            "mid_dims": tuple(action_config.get("mid_dims", (128, 256))),
            "action_schema": str(action_config.get("action_schema", "fixed")),
            "max_arm_slots": int(action_config.get("max_arm_slots", 2)),
            "action_max_time_steps": int(action_config.get("action_max_time_steps", 512)),
            "action_kv_enabled": bool(action_config.get("action_kv_enabled", True)),
            "action_v_init_scale": float(action_config.get("action_v_init_scale", 0.1)),
            "action_num_domains": int(action_config.get("action_num_domains", 1)),
            "action_domain_prompt_tokens": int(action_config.get("action_domain_prompt_tokens", 0)),
            "domain_aware_action_projection_enabled": bool(
                action_config.get("domain_aware_action_projection_enabled", False)
            ),
            "domain_aware_group_projection_enabled": bool(
                action_config.get("domain_aware_group_projection_enabled", False)
            ),
            "dense_action_film_enabled": bool(
                action_config.get("dense_action_film_enabled", False)
            ),
            "action_adaln_modulation_enabled": bool(
                action_config.get("action_adaln_modulation_enabled", False)
            ),
            "action_adaln_scale": float(action_config.get("action_adaln_scale", 1.0)),
            "vae_temporal_stride": action_config.get("vae_temporal_stride", 4),
            "action_control_scale": float(
                action_config.get("action_control_scale", 0.05)
            ),
            "action_dense_rank": int(action_config.get("action_dense_rank", 128)),
            "action_dense_layernorm_enabled": bool(
                action_config.get("action_dense_layernorm_enabled", False)
            ),
            "action_dense_relative_chunks": bool(
                action_config.get("action_dense_relative_chunks", False)
            ),
            "eef_projection_kv_enabled": bool(
                action_config.get("eef_projection_kv_enabled", False)
            ),
        }
        self._view_cfg = {
            "num_cameras": int(action_config.get("num_cameras", 1)),
            "multi_view_position_mode": str(
                action_config.get("multi_view_position_mode", "global")
            ),
            "camera_id_embedding_enabled": bool(
                action_config.get("camera_id_embedding_enabled", False)
            ),
            "latent_height_per_view": int(
                action_config.get(
                    "latent_height_per_view",
                    max(1, int(action_config.get("latent_H", 12)) // max(1, int(action_config.get("num_cameras", 1)))),
                )
            ),
            "latent_width": int(action_config.get("latent_width", action_config.get("latent_W", 20))),
            "vae_spatial_stride": int(action_config.get("vae_spatial_stride", 16)),
        }
        if self._view_cfg["num_cameras"] <= 0:
            raise ValueError(
                f"num_cameras must be positive, got {self._view_cfg['num_cameras']}"
            )
        if self._view_cfg["multi_view_position_mode"] not in {"global", "view_local"}:
            raise ValueError(
                "Unsupported multi_view_position_mode="
                f"{self._view_cfg['multi_view_position_mode']!r}"
            )
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = -1

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if self._view_cfg["camera_id_embedding_enabled"]:
            self.camera_id_embedding = nn.Parameter(
                torch.zeros(self._view_cfg["num_cameras"], dim)
            )
        else:
            self.camera_id_embedding = None
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks (vanilla WAN attention blocks — action is no longer injected
        # per-block; it enters via cross-attn K/V context, built by _forward).
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm,
                              dense_action_film=self._action_cfg[
                                  "dense_action_film_enabled"
                              ],
                              action_control_scale=self._action_cfg[
                                  "action_control_scale"
                              ],
                              action_dense_rank=self._action_cfg[
                                  "action_dense_rank"
                              ],
                              eps=eps)
            for _ in range(num_layers)
        ])

        # Action encoder: robot condition sequence -> DiT-dim action tokens.
        # Text and action use separate K/V branches in cross-attention.
        # Lazy import avoids circular dep at module load.
        action_schema = self._action_cfg["action_schema"]
        if action_schema not in {"fixed", "arm_slot"}:
            raise ValueError(f"Unsupported action_schema={action_schema!r}")

        from .action.action_module import (
            ActionDecoder,
            ActionEncoder,
            ArmSlotActionDecoder,
            ArmSlotActionEncoder,
            EefProjectionEncoder,
        )
        if action_schema == "arm_slot":
            self.action_encoder = ArmSlotActionEncoder(
                action_dim=self._action_cfg["action_dim"],
                hidden_dim=dim,
                max_arm_slots=self._action_cfg["max_arm_slots"],
                mid_dims=self._action_cfg["mid_dims"],
                max_time_steps=self._action_cfg["action_max_time_steps"],
                num_domains=self._action_cfg["action_num_domains"],
                domain_prompt_tokens=self._action_cfg["action_domain_prompt_tokens"],
                domain_aware_projection=self._action_cfg[
                    "domain_aware_action_projection_enabled"
                ],
                domain_aware_group_projection=self._action_cfg[
                    "domain_aware_group_projection_enabled"
                ],
            )
        else:
            self.action_encoder = ActionEncoder(
                action_dim=self._action_cfg["action_dim"],
                hidden_dim=dim,
                mid_dims=self._action_cfg["mid_dims"],
                max_time_steps=self._action_cfg["action_max_time_steps"],
            )
        self.eef_projection_encoder = None
        if bool(self._action_cfg["eef_projection_kv_enabled"]):
            if not bool(self._action_cfg["action_kv_enabled"]):
                raise ValueError("eef_projection_kv_enabled requires action_kv_enabled")
            self.eef_projection_encoder = EefProjectionEncoder(
                hidden_dim=dim,
                mid_dims=self._action_cfg["mid_dims"],
                max_time_steps=self._action_cfg["action_max_time_steps"],
                max_views=int(self._view_cfg["num_cameras"]),
                max_slots=int(self._action_cfg.get("max_arm_slots", 4)),
            )
        # Action decoder (LDA-1B-style auxiliary head): predicts per-frame
        # action from the last block's visual tokens. Used only during
        # training to produce an auxiliary MSE loss that forces the DiT to
        # route action information through its visual representations.
        # Without it, the text cross-attn path over-determines the next
        # frame on most samples and the model learns to ignore action.
        if action_schema == "arm_slot":
            self.action_decoder = ArmSlotActionDecoder(
                hidden_dim=dim,
                action_dim=self._action_cfg["action_dim"],
                max_arm_slots=self._action_cfg["max_arm_slots"],
                mid_dim=512,
            )
        else:
            self.action_decoder = ActionDecoder(
                hidden_dim=dim,
                action_dim=self._action_cfg["action_dim"],
                mid_dim=512,
            )
        action_condition_width = self._action_condition_width()
        if self._action_cfg["action_adaln_modulation_enabled"] and action_schema != "arm_slot":
            adaln_action_dim = action_condition_width * int(
                self._action_cfg["vae_temporal_stride"]
            )
            self.action_adaln_encoder = nn.Sequential(
                nn.Linear(adaln_action_dim, dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim),
            )
        else:
            self.action_adaln_encoder = None
        if self._action_cfg["dense_action_film_enabled"]:
            dense_action_dim = action_condition_width * int(
                self._action_cfg["vae_temporal_stride"]
            )
            dense_rank = int(self._action_cfg["action_dense_rank"])
            dense_layers = [
                nn.Linear(dense_action_dim, dense_rank),
                nn.GELU(approximate='tanh'),
                nn.Linear(dense_rank, dense_rank),
            ]
            if self._action_cfg["action_dense_layernorm_enabled"]:
                dense_layers.append(nn.LayerNorm(dense_rank))
            self.action_dense_encoder = nn.Sequential(*dense_layers)
        else:
            self.action_dense_encoder = None
        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE frequencies — intentionally NOT a registered buffer because:
        # (1) register_buffer forces dtype conversion on .to() which corrupts freq values
        # (2) freqs are deterministically recomputed from config, not learned
        # They are manually moved to the correct device in _forward().
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v' or model_type == 'ti2v':
            # self.img_emb = nn.Sequential(
            # nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.img_emb = nn.Sequential(
                                nn.LayerNorm(img_dim),
                                nn.Linear(img_dim, img_dim),
                                nn.GELU(),
                                nn.Linear(img_dim, dim),
                                nn.LayerNorm(dim)) # MLPProj(1280, dim)
        else:
            self.img_emb = None

        # Initialize weights (Xavier on all Linear, zero-init on output head).
        # We deliberately do NOT re-apply zero-init on action_encoder.net[-1]
        # after init_weights: empirically (500-step smoke-test on 8×A800) the
        # zero-init-last-layer + bf16 backward path produced NaN loss every
        # step, while Xavier-init on the same layer gave finite loss (~0.87).
        # Small Xavier init acts as a symmetry-breaking seed; zero-init
        # created a degenerate state that bf16 gradients could not escape.
        self.init_weights()
        self._init_multiview_embeddings()
        self._init_root_action_conditioners()
        self._init_dense_action_control()

        # Warm-start the action K/V in every block's cross-attn from the
        # text K/V (WAN pretrained). This seeds the action attention with the
        # same pattern as the pretrained text attention, so step-0 attention
        # is meaningful rather than random — then the gradient pushes them to
        # specialize. Must run AFTER from_pretrained() has loaded text K/V,
        # but from_pretrained calls __init__ first and then load_state_dict,
        # so we call this here defensively; it's idempotent (identity copy
        # after state_dict loads real pretrained weights). The real copy
        # happens the FIRST time the caller invokes load_state_dict() too —
        # see WanModelAction.post_load_warm_start_action_kv for that.
        self._warm_start_action_kv_from_text()

        self.gradient_checkpointing = False

    def _init_multiview_embeddings(self):
        """Initialize explicit camera/view identity embeddings."""
        if self.camera_id_embedding is not None:
            nn.init.normal_(self.camera_id_embedding, mean=0.0, std=0.02)

    def enable_prope(self, zero_init: bool = True):
        """Attach PRoPE residual projections to all self-attention blocks."""
        from .prope import add_prope_parameters

        add_prope_parameters(self, zero_init=zero_init)

    def _init_root_action_conditioners(self):
        """Initialize root action encoders as live, input-sensitive paths.

        These modules are newly introduced on top of the WAN checkpoint. They
        must not start as zero maps: a zero dense/AdaLN encoder makes every
        action variant produce the same conditioning tensor, so the DiT can
        only react to action present/absent, not action content.
        """
        if getattr(self, "action_encoder", None) is not None:
            for projection_name in (
                "input_projection",
                "slot_projection",
                "context_projection",
                "group_projection",
            ):
                projection = getattr(self.action_encoder, projection_name, None)
                if isinstance(projection, nn.Linear):
                    nn.init.xavier_uniform_(projection.weight)
                    nn.init.zeros_(projection.bias)
                elif hasattr(projection, "reset_parameters"):
                    projection.reset_parameters()
            for module_name in ("net", "slot_net", "context_net", "group_net"):
                module = getattr(self.action_encoder, module_name, None)
                if module is None:
                    continue
                if module_name == "group_net":
                    for layer in module.modules():
                        if isinstance(layer, nn.LayerNorm) and layer.elementwise_affine:
                            nn.init.ones_(layer.weight)
                            nn.init.zeros_(layer.bias)
                    final_linear = next(
                        layer for layer in reversed(list(module.modules()))
                        if isinstance(layer, nn.Linear)
                    )
                    nn.init.zeros_(final_linear.weight)
                    if final_linear.bias is not None:
                        nn.init.zeros_(final_linear.bias)
                    continue
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    elif isinstance(layer, nn.LayerNorm) and layer.elementwise_affine:
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
            time_embedding = getattr(self.action_encoder, "time_embedding", None)
            if time_embedding is not None:
                nn.init.normal_(time_embedding, mean=0.0, std=0.02)
            slot_embedding = getattr(self.action_encoder, "slot_embedding", None)
            if slot_embedding is not None:
                nn.init.normal_(slot_embedding, mean=0.0, std=0.02)
            domain_prompt = getattr(self.action_encoder, "domain_prompt", None)
            if domain_prompt is not None:
                nn.init.normal_(domain_prompt.weight, mean=0.0, std=0.02)

        for module_name in ("action_dense_encoder", "action_adaln_encoder"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm) and layer.elementwise_affine:
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _init_dense_action_control(self):
        """Initialize low-rank dense action control as a small live path."""
        for block in self.blocks:
            if getattr(block, "action_scale_shift_layer", None) is not None:
                for layer_name in (
                    "action_injector_layer1",
                    "action_scale_shift_layer",
                ):
                    layer = getattr(block, layer_name)
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def expand_input_channels(self, extra_channels: int, zero_init: bool = True):
        """Append extra input channels to the 3D patch embedding.

        Used for an explicit condition-frame mask channel.
        The pretrained channels are copied exactly; new channels start at zero
        so enabling the mask is initially non-disruptive.
        """
        if extra_channels <= 0:
            return
        old = self.patch_embedding
        new = nn.Conv3d(
            old.in_channels + extra_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
            padding_mode=old.padding_mode,
        )
        new = new.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:, :old.in_channels].copy_(old.weight)
            if zero_init:
                new.weight[:, old.in_channels:].zero_()
            else:
                nn.init.xavier_uniform_(new.weight[:, old.in_channels:].flatten(1))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.patch_embedding = new
        self.in_dim = old.in_channels + extra_channels

    # Scale applied to v_action weights/bias after warm-start so the initial
    # ``attn_action`` magnitude is small relative to attn_text. Replaces the
    # previous learnable ``alpha_action`` gate, which collapsed toward 0
    # under main-loss gradient pressure on video-prediction tasks. Keeping
    # it as a class constant rather than a per-module Parameter means the
    # optimizer can't push it toward 0 — the action path's magnitude is
    # controlled by ``v_action.weight`` itself, which must learn to grow
    # (or not) based on whether the task actually needs action signal.
    V_ACTION_INIT_SCALE = 0.1

    def _warm_start_action_kv_from_text(self):
        """Copy text cross-attn K/V weights into the action K/V branch.

        K path is a direct copy (so the attention *pattern* starts the same
        as pretrained text attention). V path is copied then scaled down by
        V_ACTION_INIT_SCALE, so initial attn_action output has similar
        shape but a smaller magnitude — small enough not to disrupt
        pretrained behavior, non-zero so gradient flows.

        Called once after init_weights (Xavier). On from_pretrained, the
        text K/V weights will later be overwritten by the real pretrained
        values, so callers should invoke this again after load. See
        post_load_warm_start_action_kv().
        """
        scale = float(self._action_cfg.get("action_v_init_scale", self.V_ACTION_INIT_SCALE))
        with torch.no_grad():
            for block in self.blocks:
                ca = block.cross_attn
                # K: full warm-start — preserves attention pattern.
                ca.k_action.weight.data.copy_(ca.k.weight.data)
                if ca.k.bias is not None and ca.k_action.bias is not None:
                    ca.k_action.bias.data.copy_(ca.k.bias.data)
                if hasattr(ca.norm_k, "weight") and hasattr(ca.norm_k_action, "weight"):
                    ca.norm_k_action.weight.data.copy_(ca.norm_k.weight.data)
                # V: warm-start then scale. Scale both weight and bias so
                # ``v_action(x)`` magnitude is uniformly reduced — bias alone
                # wouldn't change the slope of the learned mapping.
                ca.v_action.weight.data.copy_(ca.v.weight.data).mul_(scale)
                if ca.v.bias is not None and ca.v_action.bias is not None:
                    ca.v_action.bias.data.copy_(ca.v.bias.data).mul_(scale)

    def post_load_warm_start_action_kv(self):
        """Call after from_pretrained / load_state_dict to refresh warm-start.

        The `_warm_start_action_kv_from_text` in __init__ seeds action K/V
        from Xavier-init text K/V. After pretrained weights load, text K/V
        become the real WAN values — re-copy so action K/V reflect those.
        Diffusers may initialize missing checkpoint keys after __init__, so
        also refresh the root action encoders here. Otherwise newly added
        action MLPs can be left as zero/constant maps, making action content
        invisible while action present/absent still changes the forward pass.

        Idempotent for base-model construction. Do not call this after loading
        a finetuned action checkpoint unless intentionally reinitializing its
        action-side modules.
        """
        self._init_root_action_conditioners()
        self._warm_start_action_kv_from_text()
        self._init_dense_action_control()

    def _action_condition_width(self) -> int:
        action_dim = int(self._action_cfg["action_dim"])
        schema = str(self._action_cfg["action_schema"])
        if schema == "fixed":
            return action_dim
        if schema == "arm_slot":
            return int(self._action_cfg["max_arm_slots"]) * action_dim
        raise ValueError(f"Unsupported action_schema={schema!r}")

    def _flatten_action_condition(self, action_seq: torch.Tensor) -> torch.Tensor:
        schema = str(self._action_cfg["action_schema"])
        action_dim = int(self._action_cfg["action_dim"])
        if schema == "fixed":
            if action_seq.dim() != 3:
                raise ValueError(
                    "fixed action must have shape (B,T,D), "
                    f"got {tuple(action_seq.shape)}"
                )
            if action_seq.shape[-1] != action_dim:
                raise ValueError(
                    f"fixed action dim mismatch: got {action_seq.shape[-1]}, "
                    f"expected {action_dim}"
                )
            return action_seq
        if schema == "arm_slot":
            slots = int(self._action_cfg["max_arm_slots"])
            if action_seq.dim() != 4:
                raise ValueError(
                    "arm-slot action must have shape (B,T,S,D+1), "
                    f"got {tuple(action_seq.shape)}"
                )
            if action_seq.shape[2] != slots or action_seq.shape[3] != action_dim + 1:
                raise ValueError(
                    "arm-slot action shape mismatch: "
                    f"got {tuple(action_seq.shape)}, expected (B,T,{slots},{action_dim + 1})"
                )
            values = action_seq[..., :action_dim]
            mask = action_seq[..., action_dim : action_dim + 1].to(dtype=values.dtype)
            return (values * mask).reshape(action_seq.shape[0], action_seq.shape[1], -1)
        raise ValueError(f"Unsupported action_schema={schema!r}")

    def _chunk_actions_for_latent_frames(self, action_seq, num_latent_frames: int):
        """Group raw-frame actions into one chunk per latent frame.

        WAN 2.2 TI2V uses a causal temporal VAE: latent 0 anchors video frame
        0, while latent i>0 summarizes frames ``[(i-1)*stride+1, i*stride]``.
        Dense control therefore receives the raw action chunk aligned to that
        causal latent range, flattened to ``stride * action_dim``. Callers must
        provide enough raw-frame actions for this exact Wan causal grouping.
        """
        action_seq = self._flatten_action_condition(action_seq)
        stride = int(self._action_cfg.get("vae_temporal_stride", 4))
        action_dim = int(action_seq.shape[-1])
        groups = group_wan_causal_actions(
            action_seq,
            num_latent_frames=num_latent_frames,
            vae_temporal_stride=stride,
        )
        chunks = groups.chunks.reshape(
            action_seq.shape[0], num_latent_frames, stride, action_dim
        )
        if bool(self._action_cfg.get("action_dense_relative_chunks", False)):
            # Remove pose-like offsets from absolute state dimensions while
            # preserving explicit delta/displacement dimensions. This branch is
            # off in the current arm-slot EEF-pose mainline; keep it explicit so
            # any future relative-action condition must opt in deliberately.
            baseline_idx = torch.zeros(
                num_latent_frames, device=action_seq.device, dtype=torch.long
            )
            if num_latent_frames > 1:
                baseline_idx[1:] = (
                    torch.arange(
                        num_latent_frames - 1,
                        device=action_seq.device,
                        dtype=torch.long,
                    )
                    * stride
                )
            baseline = action_seq.index_select(1, baseline_idx)
            chunks = apply_dense_relative_action_chunks(chunks, baseline, action_dim)
        return chunks.reshape(action_seq.shape[0], num_latent_frames, stride * action_dim)

    def encode_action_adaln_chunks(
        self,
        action_seq,
        num_latent_frames: int,
        action_domain_ids: torch.Tensor | None = None,
    ):
        """Return per-latent-frame action AdaLN modulation, or None."""
        if (
            str(self._action_cfg["action_schema"]) == "arm_slot"
            and bool(self._action_cfg["action_adaln_modulation_enabled"])
        ):
            modulation = self.action_encoder.encode_group_modulation(
                action_seq.to(dtype=self.patch_embedding.weight.dtype),
                num_latent_frames=int(num_latent_frames),
                vae_temporal_stride=int(self._action_cfg.get("vae_temporal_stride", 4)),
                domain_ids=action_domain_ids,
            )
            return modulation * float(self._action_cfg.get("action_adaln_scale", 1.0))
        if self.action_adaln_encoder is None:
            return None
        chunks = self._chunk_actions_for_latent_frames(action_seq, num_latent_frames)
        modulation = self.action_adaln_encoder(
            chunks.to(dtype=self.patch_embedding.weight.dtype)
        )
        return modulation * float(self._action_cfg.get("action_adaln_scale", 1.0))

    def encode_dense_action_chunks(self, action_seq, num_latent_frames: int):
        """Return raw action chunks and encoded per-latent-frame control."""
        if self.action_dense_encoder is None:
            return None, None
        chunks = self._chunk_actions_for_latent_frames(action_seq, num_latent_frames)
        encoded = self.action_dense_encoder(
            chunks.to(dtype=self.patch_embedding.weight.dtype)
        )
        return chunks, encoded

    def expand_dense_action_tokens(self, action_frame_tokens, hp: int, wp: int):
        """Broadcast per-frame dense action tokens to visual token layout."""
        if action_frame_tokens is None:
            return None
        token_dim = action_frame_tokens.shape[-1]
        return action_frame_tokens[:, :, None, :].expand(
            -1, -1, hp * wp, -1
        ).reshape(
            action_frame_tokens.shape[0],
            action_frame_tokens.shape[1] * hp * wp,
            token_dim,
        )

    def _add_camera_id_embedding(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Add per-view identity to height-stacked visual tokens.

        Visual tokens are flattened in Conv3d output order: (F, H, W). For a
        height-stacked multi-camera latent, H is laid out as
        [view0_h, view1_h, ...]. This method maps every H row to its view id
        and adds a learned embedding before the DiT blocks.
        """
        if self.camera_id_embedding is None:
            return x
        f, h, w = [int(v) for v in grid_sizes.tolist()]
        num_cameras = int(self._view_cfg["num_cameras"])
        if h % num_cameras != 0:
            raise ValueError(
                f"Patch-grid height {h} is not divisible by num_cameras={num_cameras}; "
                "cannot add camera_id_embedding."
            )
        h_per_view = h // num_cameras
        view_ids = torch.arange(h, device=x.device, dtype=torch.long) // h_per_view
        emb = self.camera_id_embedding.to(device=x.device, dtype=x.dtype).index_select(
            0, view_ids
        )
        emb = emb.view(1, h, 1, self.dim).expand(f, h, w, self.dim)
        emb = emb.reshape(1, f * h * w, self.dim)
        return x + emb

    def fully_shard(self, mesh: DeviceMesh):
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)
        fully_shard(self.head, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.time_embedding, mesh=mesh, reshard_after_forward=True)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        *args,
        **kwargs
    ):
        # if kwargs.get('classify_mode', False) is True:
        # kwargs.pop('classify_mode')
        # return self._forward_classify(*args, **kwargs)
        # else:
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        text_embedding,
        seq_len,
        cond_concat=None,
        action_seq=None,
        action_dense_seq=None,
        action_dense_zero_first_chunk=False,
        action_dense_zero_prefix_chunks=0,
        action_drop_mask=None,
        action_domain_ids=None,
        viewmats=None,
        Ks=None,
        eef_uv=None,
        eef_depth=None,
        eef_valid=None,
        eef_image_hw=None,
        return_aux=False,
    ):
        r"""Forward pass of the WAN 2.2 DiT with cross-attn action conditioning.

        Args:
            x: List of (in_dim, F, H, W) latent tensors, or stacked (B, C, F, H, W).
            t: Diffusion timesteps, shape (B,).
            text_embedding: T5-XXL text embeddings, shape (B, L_text, 4096).
                Projected to (B, L_text, dim) via the pretrained WAN 2.2
                text projection.
            seq_len: Max visual sequence length after patching (unused — kept
                for signature compat with older call sites).
            cond_concat: Optional conditioning concat'd along channel dim.
            action_seq: Robot action tensor, either fixed ``(B,T_act,D)`` or
                arm-slot ``(B,T_act,slots,D+1)`` where the last feature is
                slot-exists metadata.
                None → no action conditioning (unconditional or text-only branch
                of CFG). Non-None → encoded to action tokens for the dedicated
                action K/V path parallel to the pretrained text K/V path.
            action_dense_seq: Optional raw action tensor used only by the dense
                FiLM / action-delta paths. This lets callers zero condition-frame
                actions for sparse K/V while still computing Cosmos-style
                relative dense chunks against the true previous action.
            action_dense_zero_first_chunk: Alias for
                ``action_dense_zero_prefix_chunks=1``.
            action_dense_zero_prefix_chunks: Number of leading latent-frame
                action chunks to zero after chunk encoding. Use this to apply
                the same future-only causal contract to dense FiLM/AdaLN as the
                K/V action-token path: condition/history chunks are state
                anchors and must not receive action control.
            action_drop_mask: Optional bool tensor (B,). True where action is
                dropped for that sample (per-sample CFG dropout during training).
                Effective context length for that sample is text_len only.
            action_domain_ids: Optional int tensor (B,) selecting domain-aware
                action projection and domain soft prompts.
            return_aux: If True, also compute and return the auxiliary action
                prediction from the last-block visual tokens. Used only during
                training for the LDA-style auxiliary action reconstruction loss.

        Returns:
            List[Tensor] if ``return_aux=False``: denoised latent per batch
                element, each shape (C_out, F, H/8, W/8).
            (List[Tensor], Tensor) if ``return_aux=True``: same denoised
                latents PLUS aux_action_pred of shape (B, Fp, action_dim).
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        n_dense_zero_chunks = max(0, int(action_dense_zero_prefix_chunks or 0))
        if action_dense_zero_first_chunk:
            n_dense_zero_chunks = max(n_dense_zero_chunks, 1)

        if isinstance(x, list):
            x = torch.stack(x).to(device)

        if cond_concat is not None:
            x = torch.cat([x, cond_concat], dim=1)

        if viewmats is not None:
            viewmats = viewmats.to(device=device, dtype=x.dtype)
            if viewmats.dim() != 5 or viewmats.shape[-2:] != (4, 4):
                raise ValueError(
                    "WanModelAction PRoPE expects viewmats with shape (B,F,V,4,4), "
                    f"got {tuple(viewmats.shape)}"
                )
            if Ks is not None:
                Ks = Ks.to(device=device, dtype=x.dtype)
                if Ks.shape != viewmats.shape[:-2] + (3, 3):
                    raise ValueError(
                        "WanModelAction PRoPE expects Ks with shape (B,F,V,3,3), "
                        f"got Ks={tuple(Ks.shape)}, viewmats={tuple(viewmats.shape)}"
                    )

        # --- patch embed ---
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)  # (B, L_vis, dim)
        x = self._add_camera_id_embedding(x, grid_sizes)
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)

        # --- time embedding (WAN pretrained AdaLN-Zero path) ---
        e_base = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))

        # --- text tokens (WAN pretrained projection) ---
        # text_embedding: (B, L_text, text_dim=4096)
        # self.text_embedding (nn.Sequential of Linear+GELU+Linear) is pretrained
        # by WAN 2.2 on T5-XXL embeddings. Route T5 through this, NOT img_emb
        # (img_emb is a CLIP-image projection kept dormant for checkpoint compat).
        text_tokens = self.text_embedding(text_embedding)  # (B, L_text, dim)
        B, L_text, _ = text_tokens.shape

        # --- action tokens (V1.1: dedicated action cross-attn path) ---
        # Text and action go through SEPARATE cross-attn K/V projections
        # inside each block (WanI2VCrossAttention's text / action_context
        # paths). No concat. Per-sample CFG dropout is applied as an
        # ``action_keep_mask`` tensor forwarded to each block's cross-attn;
        # dropped samples get ``attn_action *= 0`` POST-attention, so the
        # final contribution is an exact zero — matching inference's
        # ``action_context=None`` uncond branch. Zeroing action_tokens at
        # input would NOT give a true null: ``k_action`` / ``v_action``
        # have biases, so ``V(0) = bias_v`` leaks through attention.
        if action_seq is not None and bool(self._action_cfg["action_kv_enabled"]):
            action_tokens = self.action_encoder(
                action_seq.to(dtype=text_tokens.dtype),
                domain_ids=action_domain_ids,
            )
        else:
            action_tokens = None
        if self.eef_projection_encoder is not None and eef_uv is not None:
            if eef_depth is None or eef_valid is None or eef_image_hw is None:
                raise ValueError(
                    "EEF projection conditioning requires eef_uv, eef_depth, "
                    "eef_valid, and eef_image_hw"
                )
            eef_tokens = self.eef_projection_encoder(
                eef_uv.to(device=device, dtype=text_tokens.dtype),
                eef_depth.to(device=device, dtype=text_tokens.dtype),
                eef_valid.to(device=device),
                eef_image_hw.to(device=device),
            ).to(dtype=text_tokens.dtype)
            action_tokens = eef_tokens if action_tokens is None else torch.cat([action_tokens, eef_tokens], dim=1)
        if action_tokens is not None:
            if action_drop_mask is not None:
                action_keep_mask = (~action_drop_mask.to(device=device).bool()).to(
                    dtype=action_tokens.dtype
                )
            else:
                action_keep_mask = None
        else:
            action_tokens = None
            action_keep_mask = None

        e0 = None
        action_adaln = None
        head_e = e_base
        adaln_source = action_dense_seq if action_dense_seq is not None else action_seq
        group_mod_enabled = (
            str(self._action_cfg["action_schema"]) == "arm_slot"
            and bool(self._action_cfg["action_adaln_modulation_enabled"])
        )
        if (
            (self.action_adaln_encoder is not None or group_mod_enabled)
            and adaln_source is not None
        ):
            Fp, Hp, Wp = [int(v) for v in grid_sizes.tolist()]
            action_adaln = self.encode_action_adaln_chunks(
                adaln_source,
                Fp,
                action_domain_ids=action_domain_ids,
            )
            if n_dense_zero_chunks > 0 and action_adaln.shape[1] > 0:
                n_zero = min(n_dense_zero_chunks, action_adaln.shape[1])
                action_adaln = action_adaln.clone()
                action_adaln[:, :n_zero, :] = 0
            if action_drop_mask is not None:
                keep = (~action_drop_mask.to(device=device).bool()).to(
                    dtype=action_adaln.dtype
                ).view(-1, 1, 1)
                action_adaln = action_adaln * keep
            action_adaln = action_adaln[:, :, None, :].expand(
                -1, -1, Hp * Wp, -1
            ).reshape(action_adaln.shape[0], Fp * Hp * Wp, self.dim)
            e_tokens = e_base[:, None, :] + action_adaln.to(
                device=e_base.device, dtype=e_base.dtype
            )
            head_e = e_tokens
            e0 = self.time_projection(e_tokens).unflatten(2, (6, self.dim))
            e0 = e0.permute(0, 2, 1, 3)

        if e0 is None:
            e0 = self.time_projection(e_base).unflatten(1, (6, self.dim))

        action_dense_tokens = None
        dense_source = action_dense_seq if action_dense_seq is not None else action_seq
        if self.action_dense_encoder is not None and dense_source is not None:
            Fp, Hp, Wp = [int(v) for v in grid_sizes.tolist()]
            _, action_frame_tokens = self.encode_dense_action_chunks(dense_source, Fp)
            if n_dense_zero_chunks > 0 and action_frame_tokens.shape[1] > 0:
                n_zero = min(n_dense_zero_chunks, action_frame_tokens.shape[1])
                action_frame_tokens = action_frame_tokens.clone()
                action_frame_tokens[:, :n_zero, :] = 0
            action_dense_tokens = self.expand_dense_action_tokens(
                action_frame_tokens, Hp, Wp
            ).to(dtype=text_tokens.dtype)
            if action_drop_mask is not None:
                keep = (~action_drop_mask.to(device=device).bool()).to(
                    dtype=action_dense_tokens.dtype
                ).view(-1, 1, 1)
                action_dense_tokens = action_dense_tokens * keep

        kwargs = dict(
            e=e0,
            grid_sizes=grid_sizes,
            seq_lens=seq_lens,
            freqs=self.freqs,
            num_cameras=int(self._view_cfg["num_cameras"]),
            position_mode=str(self._view_cfg["multi_view_position_mode"]),
            viewmats=viewmats,
            Ks=Ks,
            prope_image_width=int(self._view_cfg["latent_width"]) * int(
                self._view_cfg["vae_spatial_stride"]
            ),
            prope_image_height=int(self._view_cfg["latent_height_per_view"]) * int(
                self._view_cfg["vae_spatial_stride"]
            ),
            context=text_tokens,
            context_lens=None,
            action_context=action_tokens,
            action_keep_mask=action_keep_mask,
            action_dense_context=action_dense_tokens,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # Auxiliary action prediction from last-block visual tokens.
        # Compute BEFORE self.head so the aux path sees the pure
        # transformer output, not the channel-reduced head output.
        # Small decoder (~2M params); branches cleanly from the main path
        # via the already-materialized ``x`` tensor so no extra forward
        # cost on the DiT trunk.
        aux_action_pred = None
        if return_aux:
            aux_action_pred = self.action_decoder(x, grid_sizes)

        x_hidden = x
        x = self.head(x_hidden, head_e)
        x = self.unpatchify(x, grid_sizes)
        outputs = [u.float() for u in x]
        if return_aux:
            return outputs, aux_action_pred
        return outputs
    def unpatchify(self, x, grid_sizes): # TODO check grid sizes
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return x

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # ---- generic init for all Linear layers ----
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # Ensure any LayerNorm with affine params is sensible
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, WanRMSNorm):
                nn.init.ones_(m.weight)  # already ones by default, keep explicit

        # ---- embeddings ----
        # Conv3d patch embed: Xavier on flattened kernel (preserves fan-in/out)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)

        # time embedding MLP: match common diffusion init (small std for stability)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)




        # ---- img_emb (MLPProj) ----
        # MLPProj = [LayerNorm(in), Linear, GELU, Linear, LayerNorm(out)]
        if getattr(self, "img_emb", None) is not None:
            for m in self.img_emb.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    # Both input and output LayerNorms
                    if m.elementwise_affine:
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
        # ---- output head ----
        nn.init.zeros_(self.head.head.weight)
        if getattr(self.head.head, "bias", None) is not None:
            nn.init.zeros_(self.head.head.bias)

        # ActionEncoder does its own (zero-init final linear) in its __init__.
