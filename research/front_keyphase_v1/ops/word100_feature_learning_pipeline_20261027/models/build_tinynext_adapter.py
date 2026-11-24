"""Create an auditable standalone adapter from the pinned MIT upstream files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "upstream/classification/models"
modules = (source / "modules.py").read_text(encoding="utf-8")
modules = modules.replace("from einops import rearrange\n", "")  # Unused upstream import.
network = (source / "tinynext.py").read_text(encoding="utf-8")
network = network.replace("from torch import nn\n", "")
network = network.replace("from timm.models.layers import trunc_normal_\n", "")
network = network.replace("from timm.models.registry import register_model\n", "")
network = network.replace("from models.modules import Stem, Embed, MV2Block, FormerBlock, SeBlock\n", "")
network = network.replace("@register_model\n", "")
network = network.replace("trunc_normal_(", "nn.init.trunc_normal_(")
header = '''"""TinyNeXt-T frozen image features, isolated research adapter.

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
'''
api = '''

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
'''
(ROOT / "tinynext_adapter.py").write_text(header + modules + network + api, encoding="utf-8")
(ROOT / "TINYNEXT_LICENSE.txt").write_bytes((ROOT / "upstream/LICENSE").read_bytes())
print("Standalone torch adapter created; original pinned sources retained.")
