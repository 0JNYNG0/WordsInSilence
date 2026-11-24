"""TinyNeXt-T frozen image features, isolated research adapter.

Derived from Fanhong Zeng's MIT TinyNeXt, commit
3eb30a847f8e5916b975f139d101a0da1f0d7e67. See TINYNEXT_LICENSE.txt.
Changes: remove unused einops/timm registry dependencies, use torch's truncated
normal initializer, expose pre-pool and pooled features, strict hash-checked
weights loading and official evaluation preprocessing. No training, network
access, temporal smoothing or operating-model changes occur on import.

Input: BCHW float32 RGB, official 224x224 evaluation uses resize256/crop224 and
ToTensor then mean128/255, std1/255. This is NOT ImageNet mean/std.
Output: forward -> Bx1000 original ImageNet logits; forward_features -> Bx192x7x7
at224; forward_embedding -> Bx192. These are not hand-shape probabilities.
"""
import torch
import torch.nn as nn


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, identity, fx):
        return identity + fx


class Mul(nn.Module):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, act1, act2):
        return act1 * act2


class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, act1, act2):
        return torch.matmul(act1, act2)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


class Stem(nn.Sequential):
    def __init__(self, inp, oup):
        super(Stem, self).__init__(
            ConvBNReLU(inp, oup // 2, 3, 2),
            ConvBNReLU(oup // 2, oup, 3, 2)
        )


class MV2Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_channel):
        super(MV2Block, self).__init__()
        self.shortcut = stride == 1 and in_channel == out_channel
        self.layers = nn.Sequential(
            ConvBNReLU(in_channel, expand_channel, kernel_size=1),
            ConvBNReLU(expand_channel, expand_channel, stride=stride, groups=expand_channel),
            nn.Conv2d(expand_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.add = Add() if self.shortcut else None

    def forward(self, x):
        if self.shortcut:
            return self.add(x, self.layers(x))
        else:
            return self.layers(x)


class Embed(nn.Sequential):
    def __init__(self, inp, oup):
        super(Embed, self).__init__(MV2Block(inp, oup, 2, oup))


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.scale = dim ** -0.5
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.matmul1 = MatMul()
        self.matmul2 = MatMul()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-1, -2).contiguous()
        attn = self.matmul1(self.linear1(x), x.transpose(-2, -1))
        attn = self.softmax(attn * self.scale)
        x = self.matmul2(attn, self.linear2(x))
        x = x.transpose(-1, -2).view(B, C, H, W).contiguous()
        return x


class Mlp(nn.Sequential):
    def __init__(self, in_channels, ratio):
        super(Mlp, self).__init__(
            nn.Conv2d(in_channels, int(ratio * in_channels), kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(int(ratio * in_channels), in_channels, kernel_size=1, bias=True),
        )


class FormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super(FormerBlock, self).__init__()
        self.attention = nn.Sequential(nn.BatchNorm2d(dim), Attention(dim))
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, 1, groups=dim, bias=False)
        self.mlp = nn.Sequential(nn.BatchNorm2d(dim), Mlp(dim, mlp_ratio))
        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()

    def forward(self, x):
        x = self.add1(x, self.attention(x))
        x = self.add2(x, self.local(x))
        x = self.add3(x, self.mlp(x))
        return x


class SeModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SeModule, self).__init__()
        hidden_channel = max(channel // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, hidden_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, channel, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        self.mul = Mul()

    def forward(self, x):
        return self.mul(x, self.se(x))


class SeBlock(nn.Module):
    def __init__(self, in_chs, mlp_ratio, reduction=4):
        super(SeBlock, self).__init__()
        self.se = nn.Sequential(nn.BatchNorm2d(in_chs), SeModule(in_chs, reduction))
        self.local = nn.Conv2d(in_chs, in_chs, 3, 1, 1, 1, groups=in_chs, bias=False)
        self.mlp = nn.Sequential(nn.BatchNorm2d(in_chs), Mlp(in_chs, mlp_ratio))
        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()

    def forward(self, x):
        x = self.add1(x, self.se(x))
        x = self.add2(x, self.local(x))
        x = self.add3(x, self.mlp(x))
        return x


def gen_block(name, channels, ratio):
    expand_channel = int(ratio * channels)
    if name == "mv2":
        return MV2Block(in_channel=channels, out_channel=channels, stride=1, expand_channel=expand_channel)
    elif name == "former":
        return FormerBlock(dim=channels, mlp_ratio=ratio)
    elif name == "se":
        return SeBlock(in_chs=channels, mlp_ratio=ratio, reduction=4)
    else:
        raise ValueError("invalid block type")


class TinyNeXt(nn.Module):
    def __init__(self, num_classes, cfg=None, distillation=False):
        super(TinyNeXt, self).__init__()
        self.cfg = cfg
        input_channel = self.cfg[0][1]
        self.embeds = nn.ModuleList([Stem(3, input_channel)])
        self.stages = nn.ModuleList([])
        for i in range(4):
            name, width, depth, ratio = self.cfg[i]
            if i > 0:
                self.embeds.append(Embed(input_channel, width))
            stage = nn.Sequential(
                *[gen_block(name, width, ratio) for j in range(depth)]
            )
            self.stages.append(stage)
            input_channel = width
        self.norm = nn.BatchNorm2d(input_channel)
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.class_head = nn.Linear(input_channel, num_classes)
        self.distillation = distillation
        if self.distillation:
            self.dist_head = nn.Linear(input_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.embeds[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.global_pool(x)
        if self.distillation:
            cls_out = self.class_head(x), self.dist_head(x)
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.class_head(x)
        return cls_out


def tinynext_t(pretrained=False, num_classes=1000, distillation=False):
    # block width depth ratio
    cfg = [
        ["mv2", 32, 3, 2],
        ["mv2", 64, 3, 2],
        ["former", 96, 6, 2],
        ["se", 192, 2, 2],
    ]
    model = TinyNeXt(num_classes=num_classes, cfg=cfg, distillation=distillation)
    return model


def tinynext_s(pretrained=False, num_classes=1000, distillation=False):
    # block width depth ratio
    cfg = [
        ["mv2", 32, 3, 2],
        ["mv2", 64, 3, 2],
        ["former", 96, 8, 2],
        ["se", 192, 3, 2],
    ]
    model = TinyNeXt(num_classes=num_classes, cfg=cfg, distillation=distillation)
    return model


def tinynext_m(pretrained=False, num_classes=1000, distillation=False):
    # block width depth ratio
    cfg = [
        ["mv2", 32, 4, 2],
        ["mv2", 64, 4, 2],
        ["former", 128, 9, 2],
        ["se", 256, 4, 1.5],
    ]
    model = TinyNeXt(num_classes=num_classes, cfg=cfg, distillation=distillation)
    return model

from pathlib import Path
import argparse
import hashlib

UPSTREAM_COMMIT = "3eb30a847f8e5916b975f139d101a0da1f0d7e67"
CHECKPOINT_SHA256 = "dc4a9334c05f0d739ec34cdd52795d13395688bb5f2997bf3335095f9720f61e"
INPUT_SIZE = 224
FEATURE_DIM = 192


class TinyNeXtFeatureBackbone(TinyNeXt):
    def __init__(self):
        super().__init__(num_classes=1000, cfg=[
            ["mv2", 32, 3, 2], ["mv2", 64, 3, 2],
            ["former", 96, 6, 2], ["se", 192, 2, 2],
        ], distillation=False)

    def forward_features(self, x):
        for i in range(4):
            x = self.embeds[i](x)
            x = self.stages[i](x)
        return self.norm(x)

    def forward_embedding(self, x):
        return self.global_pool(self.forward_features(x))


def load_tinynext_t(checkpoint=None, freeze=True):
    """Load official fixed checkpoint; reject altered files or unexpected pickle globals."""
    checkpoint = Path(checkpoint) if checkpoint else Path(__file__).with_name("tinynext_t.pth")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint does not match the reviewed official SHA256")
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    if set(unsafe) - {"argparse.Namespace"}:
        raise ValueError(f"Unexpected checkpoint globals: {unsafe}")
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    if not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Checkpoint model entries must be finite tensors")
    model = TinyNeXtFeatureBackbone()
    model.load_state_dict(state, strict=True)
    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model


def official_eval_transform():
    """Exact upstream PIL evaluation geometry and normalization for input_size224."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(INPUT_SIZE), transforms.ToTensor(),
        transforms.Normalize((128.0 / 255.0,) * 3, (1.0 / 255.0,) * 3),
    ])


def normalize_rgb01(rgb01):
    """Only pixel normalization for an already geometrically prepared [0,1] RGB tensor.

    This does NOT perform official resize/center crop. Caller must record its
    geometric transform if this lower-level API is used for a common crop.
    """
    if not torch.is_floating_point(rgb01):
        raise TypeError("Expected floating-point RGB in [0,1]")
    return (rgb01 - 128.0 / 255.0) / (1.0 / 255.0)
