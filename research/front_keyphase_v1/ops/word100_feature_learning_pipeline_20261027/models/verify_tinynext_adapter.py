"""Synthetic-input parity against pinned upstream, no dataset/model training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "deps"))
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from PIL import Image
import tinynext_adapter as adapter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference():
    # No imported source is fetched at runtime. The unused einops import is
    # removed only in memory; the pinned original file remains unchanged.
    source = ROOT / "upstream/classification/models"
    package = types.ModuleType("models")
    package.__path__ = [str(source)]
    sys.modules["models"] = package
    module = types.ModuleType("models.modules")
    code = (source / "modules.py").read_text(encoding="utf-8")
    code = code.replace("from einops import rearrange\n", "")
    exec(compile(code, str(source / "modules.py"), "exec"), module.__dict__)
    sys.modules["models.modules"] = module
    spec = importlib.util.spec_from_file_location("reference_tinynext", source / "tinynext.py")
    reference = importlib.util.module_from_spec(spec)
    sys.modules["reference_tinynext"] = reference
    spec.loader.exec_module(reference)
    return reference.tinynext_t(num_classes=1000, distillation=False).eval()


def main():
    torch.set_num_threads(8)
    report = {"at": datetime.now(timezone.utc).isoformat(), "training_runs": 0,
              "inputs": "fixed synthetic random RGB only; no public or user imagery",
              "checks": [], "prior_errors": [], "limitations": [
                  "Same Codex, separate implementation parity; not independent physical hand truth.",
                  "No ImageNet accuracy reproduction, hand-feature accuracy or CPU benchmark claim.",
                  "Reference uses current scoped timm1.0.22 and removes an unused einops import in memory; no old-runtime reproduction."]}
    try:
        model = adapter.load_tinynext_t()
        reference = load_reference()
        with torch.serialization.safe_globals([argparse.Namespace]):
            ckpt = torch.load(ROOT / "tinynext_t.pth", map_location="cpu", weights_only=True)
        state = ckpt["model"]
        reference.load_state_dict(state, strict=True)
        assert set(reference.state_dict()) == set(model.state_dict()) == set(state)
        assert all(torch.equal(model.state_dict()[k], state[k]) for k in state)
        assert all(torch.equal(reference.state_dict()[k], state[k]) for k in state)
        report["checkpoint_summary"] = {"entries": len(state), "parameter_count": sum(p.numel() for p in model.parameters()),
                                        "checkpoint_epoch": ckpt["epoch"], "model_name": ckpt["args"].model,
                                        "source_input_size": ckpt["args"].input_size, "source_distillation": ckpt["args"].distillation_type}
        assert not any(p.requires_grad for p in model.parameters())
        assert not model.training
        report["checks"].append({"check": "strict_load_327_entries_finite_frozen_eval", "pass": True})
        for seed, batch in [(73, 1), (74, 2), (75, 1)]:
            generator = torch.Generator().manual_seed(seed)
            rgb = torch.rand((batch, 3, 224, 224), generator=generator)
            x = adapter.normalize_rgb01(rgb)
            with torch.inference_mode():
                expected = reference(x)
                actual = model(x)
                spatial = model.forward_features(x)
                embedding = model.forward_embedding(x)
                reconstructed = model.class_head(embedding)
            assert actual.shape == (batch, 1000)
            assert spatial.shape == (batch, 192, 7, 7)
            assert embedding.shape == (batch, 192)
            assert all(torch.isfinite(value).all() for value in [expected, actual, spatial, embedding, reconstructed])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, reconstructed, rtol=0, atol=0)
            report["checks"].append({"check": "upstream_logits_and_features", "seed": seed, "batch": batch,
                                     "max_abs_difference": float((actual - expected).abs().max()),
                                     "feature_shape": list(spatial.shape), "embedding_shape": list(embedding.shape), "pass": True})
        spec = importlib.util.spec_from_file_location("reference_dataset_transforms", ROOT / "upstream/classification/util/datasets.py")
        data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data)
        official = data.build_transform(False, types.SimpleNamespace(input_size=224))
        ours = adapter.official_eval_transform()
        for shape in [(256, 256), (240, 320), (320, 240)]:
            array = np.random.default_rng(871).integers(0, 256, (shape[0], shape[1], 3), dtype=np.uint8)
            image = Image.fromarray(array)
            expected, actual = official(image), ours(image)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            report["checks"].append({"check": "official_eval_preprocessing", "input_hw": list(shape), "output_shape": list(actual.shape), "max_abs_difference": 0.0, "pass": True})
        # Hash rejection is part of reproducible loading, not a claim about model quality.
        try:
            adapter.load_tinynext_t(ROOT / "TINYNEXT_LICENSE.txt")
        except ValueError as error:
            assert "SHA256" in str(error)
            report["checks"].append({"check": "wrong_checkpoint_hash_rejected", "pass": True})
        else:
            raise AssertionError("Wrong checkpoint hash accepted")
        report["status"] = "PASS_STRICT_LOAD_LOGITS_AND_PREPROCESSING_PARITY"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["pins"] = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "tinynext_adapter.py", ROOT / "tinynext_t.pth", ROOT / "TINYNEXT_LICENSE.txt"]}
        destination = ROOT / "TINYNEXT_ADAPTER_VERIFY.json"
        if destination.exists():
            old = json.loads(destination.read_text(encoding="utf-8"))
            report["prior_errors"] = old.get("prior_errors", []) + [old]
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance_file = ROOT / "TINYNEXT_PROVENANCE.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    provenance["adapter"] = {"path": str(ROOT / "tinynext_adapter.py"), "sha256": sha(ROOT / "tinynext_adapter.py"),
                             "api": ["load_tinynext_t()", "official_eval_transform()", "model.forward_features(BCHW)", "model.forward_embedding(BCHW)", "model(BCHW) original1000logits"],
                             "feature_dim":192, "input_size":224, "global_install_changes":0,
                             "verification": str(destination), "verification_sha256":sha(destination)}
    provenance_file.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]), "checkpoint_summary": report["checkpoint_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
