"""CPU-only, evaluation-only SHuBERT feature encoder; no acquisition or pickle load.

Adapted from ShesterG/SHuBERT commit cc192932 (full revision in acquisition
manifest), examples/shubert/models/shubert.py and vendored fairseq wav2vec2.py.
Fairseq portions: Copyright (c) Facebook, Inc. and its affiliates, MIT license.
See CPU_ADAPTER_NOTES.txt and vendor/shubert/fairseq/LICENSE for attribution.

Only the unmasked transformer feature path is implemented. A constructed model
has RANDOM weights until load_exact_state_dict is called. Default configuration
matches the inspected source defaults, NOT proof of checkpoint configuration.
No DINO, landmark extraction, attribute classifier, training, or network access.
"""

from dataclasses import asdict, dataclass, fields
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CPUShubertConfig:
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    activation_fn: str = "gelu"
    layer_type: str = "transformer"
    layer_norm_first: bool = False
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    dropout_features: float = 0.0
    encoder_layerdrop: float = 0.0
    conv_pos: int = 32
    conv_pos_groups: int = 16
    pos_conv_depth: int = 1
    required_seq_len_multiple: int = 2
    channels_embed_dim: int = 384
    channels_pose_embed_dim: int = 14
    intermediate_dim: int = 1024
    codebook_size: int = 256

    @classmethod
    def from_mapping(cls, model_cfg: Mapping[str, Any]):
        """Use a separately audited primitive cfg.model mapping, not pickle objects.

        Non-inference training fields are ignored. Missing fields use inspected
        source defaults; the caller must record/verify these against checkpoint
        metadata. Tensor shapes alone cannot establish LayerNorm order or heads.
        """
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in model_cfg.items() if k in names}
        return cls(**values)

    def __post_init__(self):
        if self.layer_type != "transformer" or self.pos_conv_depth != 1:
            raise ValueError("Only transformer / pos_conv_depth=1 was ported")
        if self.activation_fn != "gelu":
            raise ValueError("Only the official GELU inference configuration was ported")
        for name in ("encoder_layers", "encoder_embed_dim", "encoder_ffn_embed_dim",
                     "encoder_attention_heads", "conv_pos", "conv_pos_groups",
                     "required_seq_len_multiple", "channels_embed_dim",
                     "channels_pose_embed_dim", "intermediate_dim", "codebook_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.layer_norm_first, bool):
            raise ValueError("layer_norm_first must be a boolean")
        if self.encoder_embed_dim % self.encoder_attention_heads:
            raise ValueError("encoder_embed_dim must divide into attention heads")
        if self.encoder_embed_dim % self.conv_pos_groups:
            raise ValueError("encoder_embed_dim must divide into conv_pos_groups")
        if self.intermediate_dim % 4:
            raise ValueError("intermediate_dim must divide into four channels")
        for name in ("dropout", "attention_dropout", "activation_dropout",
                     "dropout_input", "dropout_features", "encoder_layerdrop"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")


class _SamePad(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        return x[:, :, :-self.remove] if self.remove else x


class _SelfAttention(nn.Module):
    """The vendored fairseq CPU fast path, retaining separate projection keys."""

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.encoder_embed_dim
        self.num_heads = cfg.encoder_attention_heads
        self.dropout = cfg.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, padding_mask=None):
        return F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=self.embed_dim, num_heads=self.num_heads,
            in_proj_weight=None,
            in_proj_bias=torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k=None, bias_v=None, add_zero_attn=False,
            dropout_p=self.dropout, out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias, training=False,
            key_padding_mask=padding_mask, need_weights=False, attn_mask=None,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )


class _TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.encoder_embed_dim
        self.self_attn = _SelfAttention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_embed_dim, d)
        self.final_layer_norm = nn.LayerNorm(d)
        self.layer_norm_first = cfg.layer_norm_first
        # Evaluation makes these identities; modules preserve the source order.
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.activation_dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

    def forward(self, x, padding_mask=None):
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(x, padding_mask)
            x = residual + self.dropout1(x)
            residual = x
            x = self.final_layer_norm(x)
            x = F.gelu(self.fc1(x).float()).type_as(x)
            x = self.fc2(self.dropout2(x))
            layer_result = x  # Official layer[-1], before residual and dropout3.
            x = residual + self.dropout3(x)
        else:
            x, attn = self.self_attn(x, padding_mask)
            x = residual + self.dropout1(x)
            x = self.self_attn_layer_norm(x)
            residual = x
            x = F.gelu(self.fc1(x).float()).type_as(x)
            x = self.fc2(self.dropout2(x))
            layer_result = x
            x = residual + self.dropout3(x)
            x = self.final_layer_norm(x)
        return x, (attn, layer_result)


class _TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, k = cfg.encoder_embed_dim, cfg.conv_pos
        conv = nn.Conv1d(d, d, k, padding=k // 2, groups=cfg.conv_pos_groups)
        # Legacy API deliberately retains official weight_g/weight_v key names.
        conv = nn.utils.weight_norm(conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(conv, _SamePad(k), nn.GELU())
        self.layers = nn.ModuleList(_TransformerLayer(cfg) for _ in range(cfg.encoder_layers))
        self.layer_norm = nn.LayerNorm(d)
        self.layer_norm_first = cfg.layer_norm_first
        self.required_seq_len_multiple = cfg.required_seq_len_multiple
        self.dropout = cfg.dropout

    def forward(self, x, padding_mask=None, layer=None):
        if padding_mask is not None:
            x = x.clone()
            x[padding_mask] = 0  # CPU equivalent of fairseq.utils.index_put.
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        remainder = (-x.shape[1]) % self.required_seq_len_multiple
        if remainder:
            x = F.pad(x, (0, 0, 0, remainder), value=0)
            if padding_mask is None:
                padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
                padding_mask[:, -remainder:] = True
            else:
                padding_mask = F.pad(padding_mask, (0, remainder), value=True)
        x = F.dropout(x, p=self.dropout, training=False).transpose(0, 1)
        results = []
        for index, encoder_layer in enumerate(self.layers):
            x, (attn, ffn_output) = encoder_layer(x, padding_mask)
            results.append((x, attn, ffn_output))
            if index == layer:
                break
        x = x.transpose(0, 1)
        if remainder:
            x = x[:, :-remainder]
            results = [(a[:-remainder], None if b is None else b[:-remainder],
                        c[:-remainder]) for a, b, c in results]
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, results


class CPUShubertAdapter(nn.Module):
    """Full SHuBERT state key layout, restricted to unmasked CPU FP32 features.

    Source items have face/left_hand/right_hand [T,384], body_posture [T,14].
    All items must have the same T; caller-supplied padding masks describe pads.
    Optional training labels are ignored because feature extraction does not use
    them. Missing detections must be handled exactly by the upstream pipeline.
    """

    def __init__(self, cfg: CPUShubertConfig):
        super().__init__()
        self.cfg = cfg
        d, channel, intermediate = cfg.encoder_embed_dim, cfg.channels_embed_dim, cfg.intermediate_dim
        self.post_extract_proj = nn.Linear(intermediate, d)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.mask_emb = nn.Parameter(torch.empty(1, 1, 1, intermediate // 4).uniform_())
        self.encoder = _TransformerEncoder(cfg)
        self.layer_norm = nn.LayerNorm(channel * 4)  # Present but unused in official feature path.
        self.layer_norm_face = nn.LayerNorm(channel)
        self.layer_norm_lhand = nn.LayerNorm(channel)
        self.layer_norm_rhand = nn.LayerNorm(channel)
        self.layer_norm_body = nn.LayerNorm(cfg.channels_pose_embed_dim)
        self.heads = nn.ModuleList(nn.Linear(d, cfg.codebook_size) for _ in range(4))
        self.face_proj = nn.Linear(channel, intermediate // 4)
        self.left_hand_proj = nn.Linear(channel, intermediate // 4)
        self.right_hand_proj = nn.Linear(channel, intermediate // 4)
        self.body_posture_proj = nn.Linear(cfg.channels_pose_embed_dim, intermediate // 4)
        self.exact_state_loaded = False
        self.eval()

    def load_exact_state_dict(self, state: Mapping[str, torch.Tensor]):
        """Accept already safely decoded tensors; does NOT establish provenance.

        Keep all official keys, including unused pretraining heads. No prefix
        guessing, partial load, unpickling, downloading, or optimizer restore.
        """
        expected = self.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        invalid = {}
        for key in set(expected) & set(state):
            value = state[key]
            if not isinstance(value, torch.Tensor):
                invalid[key] = "not a tensor"
            elif value.layout != torch.strided or not value.is_floating_point():
                invalid[key] = "expected a dense floating tensor"
            elif tuple(value.shape) != tuple(expected[key].shape):
                invalid[key] = {"expected": list(expected[key].shape), "found": list(value.shape)}
        if missing or unexpected or invalid:
            raise ValueError({"missing_keys": missing, "unexpected_keys": unexpected,
                              "invalid_tensors": invalid})
        self.load_state_dict(state, strict=True)
        self.exact_state_loaded = True
        return {"missing_keys": [], "unexpected_keys": [], "tensor_count": len(expected),
                "model_config": asdict(self.cfg), "checkpoint_provenance_verified": False}

    @torch.inference_mode()
    def forward(self, source: Sequence[Mapping[str, torch.Tensor]], padding_mask=None,
                mask=False, features_only=True, layer=None, kmeans_labels=None):
        if self.training or any(m.training for m in self.modules()):
            raise RuntimeError("CPU adapter supports eval() only")
        if mask or not features_only or kmeans_labels is not None:
            raise ValueError("Only unmasked feature extraction is implemented")
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int)
                                  or not 0 <= layer < self.cfg.encoder_layers):
            raise ValueError("layer must be a valid zero-based encoder index")
        if not source:
            raise ValueError("At least one source item is required")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("CPU FP32 model parameters are required")
        dimensions = {"face": self.cfg.channels_embed_dim, "left_hand": self.cfg.channels_embed_dim,
                      "right_hand": self.cfg.channels_embed_dim,
                      "body_posture": self.cfg.channels_pose_embed_dim}
        length = None
        for sample in source:
            for name, dimension in dimensions.items():
                value = sample.get(name)
                if (not isinstance(value, torch.Tensor) or value.ndim != 2
                        or value.shape[1] != dimension or value.shape[0] < 1):
                    raise ValueError(f"{name} must have shape [T>0,{dimension}]")
                if value.device.type != "cpu" or value.dtype != torch.float32:
                    raise ValueError(f"{name} must be a CPU float32 tensor")
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name} contains non-finite values")
                length = value.shape[0] if length is None else length
                if value.shape[0] != length:
                    raise ValueError("All four streams and batch items must have identical T")
        if padding_mask is not None:
            if (padding_mask.device.type != "cpu" or padding_mask.dtype != torch.bool
                    or tuple(padding_mask.shape) != (len(source), length)):
                raise ValueError("padding_mask must be CPU bool [B,T]")
            if padding_mask.all(dim=1).any():
                raise ValueError("An entirely padded sample has no valid attention keys")
        parts = []
        for name, normalization, projection in (
            ("face", self.layer_norm_face, self.face_proj),
            ("left_hand", self.layer_norm_lhand, self.left_hand_proj),
            ("right_hand", self.layer_norm_rhand, self.right_hand_proj),
            ("body_posture", self.layer_norm_body, self.body_posture_proj),
        ):
            parts.append(projection(normalization(torch.stack([s[name] for s in source]))))
        x = torch.stack(parts, dim=2)
        x = self.dropout_input(x).view(len(source), length, -1)
        x = self.post_extract_proj(x)
        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)
        return {"x": x, "padding_mask": padding_mask, "layer_results": layer_results}

    def extract_features(self, source, padding_mask=None, kmeans_labels=None, mask=False, layer=None):
        return self.forward(source, padding_mask=padding_mask, kmeans_labels=kmeans_labels,
                            mask=mask, features_only=True, layer=layer)


def stack_official_features(result):
    """Match features/shubert_inference.py: [layers,T,D], one clip only.

    layer_results[i][-1] is FFN output before residual/normalization, NOT x.
    """
    layers = result["layer_results"]
    if not layers or any(item[-1].ndim != 3 or item[-1].shape[1] != 1 for item in layers):
        raise ValueError("Official saved feature layout requires a single-clip batch")
    return torch.stack([item[-1].squeeze(1) for item in layers], dim=0)
