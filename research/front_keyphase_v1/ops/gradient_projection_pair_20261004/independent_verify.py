"""Codex independent synthetic audit. No training, checkpoints, data, camera, or network.

Loads only the authored public projection helper. Its module does not load historical
code/models at import. Expectations are computed independently using NumPy float64.
All writes are limited to INDEPENDENT_PREFLIGHT*.json beside this verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", type=Path, default=HERE / "projected_train.py")
    ap.add_argument("--output", type=Path, default=HERE / "INDEPENDENT_PREFLIGHT.json")
    args = ap.parse_args()
    path, out = args.module.resolve(), args.output.resolve()
    if out.parent != HERE or not out.name.startswith("INDEPENDENT_PREFLIGHT") or out.suffix != ".json":
        raise ValueError("output must be INDEPENDENT_PREFLIGHT*.json beside this verifier")
    hashes_before = {"projected_train.py": sha(path), "independent_verify.py": sha(__file__)}
    spec = importlib.util.spec_from_file_location("codex_independent_projection_subject", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = module.project_extra_gradients
    records = []

    def check(name, fn):
        try:
            details = fn()
            records.append({"name": name, "status": "PASS", "details": details or {}})
        except Exception as exc:
            records.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})

    def tensors(vals, dtype=torch.float32):
        return [torch.tensor(v, dtype=dtype) for v in vals]

    def flat(gs):
        return np.concatenate([g.detach().cpu().numpy().reshape(-1).astype(np.float64) for g in gs])

    def unchanged(gs, snapshots):
        for actual, before in zip(gs, snapshots):
            assert actual.dtype == before.dtype and actual.shape == before.shape
            assert actual.detach().cpu().numpy().tobytes() == before.detach().cpu().numpy().tobytes()

    def reference(old, extra, enabled):
        o, e = flat(old), flat(extra)
        n, d = float(np.dot(o, o)), float(np.dot(o, e))
        applied = bool(enabled and d < 0 and n > 0)
        expected = e - (d / n) * o if applied else e.copy()
        split, offset = [], 0
        for template in extra:
            count = template.numel()
            # The public API returns the adjusted EXTRA gradient in the parameter dtype.
            split.append(expected[offset:offset + count].reshape(tuple(template.shape)).astype(
                template.detach().cpu().numpy().dtype))
            offset += count
        rounded = np.concatenate([s.reshape(-1).astype(np.float64) for s in split])
        info = {"old_norm": math.sqrt(n), "new_norm": float(np.linalg.norm(e)),
                "dot_before": d, "dot_after": float(np.dot(o, rounded)),
                "removed_norm": float(np.linalg.norm(e - rounded)), "projected": float(applied)}
        return split, info

    def numeric(old_values, extra_values, enabled=True, dtype=torch.float32):
        old, extra = tensors(old_values, dtype), tensors(extra_values, dtype)
        old_before, extra_before = [g.clone() for g in old], [g.clone() for g in extra]
        expected, info_expected = reference(old, extra, enabled)
        actual, info = project(old, extra, enabled)
        assert len(actual) == len(extra)
        for a, e, template in zip(actual, expected, extra):
            assert a.dtype == template.dtype and a.shape == template.shape
            assert a.device == template.device and torch.isfinite(a).all()
            # NumPy/Torch evaluate the same double-precision formula; tight tolerance
            # permits final floating arithmetic ordering, not a float32 inner product.
            np.testing.assert_allclose(a.detach().cpu().numpy(), e, rtol=1e-7, atol=0)
            assert a.data_ptr() != template.data_ptr(), "returned extra aliases input"
        unchanged(old, old_before)
        unchanged(extra, extra_before)
        # NumPy and Torch may reduce the float64 products in different orders.
        # Bound that accumulation difference by gamma_n * sum(abs(products));
        # near-zero post-projection dots need an absolute bound, not only rtol.
        o64, e64, a64 = flat(old), flat(extra), flat(actual)
        gamma = (len(o64) * np.finfo(np.float64).eps) / (1 - len(o64) * np.finfo(np.float64).eps)
        dot_bounds = {"dot_before": 2 * gamma * float(np.abs(o64 * e64).sum()),
                      "dot_after": 2 * gamma * float(np.abs(o64 * a64).sum())}
        for key, expected_value in info_expected.items():
            assert key in info and math.isfinite(float(info[key])), f"non-finite/missing {key}"
            if expected_value == 0 and key not in dot_bounds:
                assert float(info[key]) == 0, f"{key}: expected exact zero, got {info[key]}"
            else:
                assert math.isclose(float(info[key]), expected_value, rel_tol=5e-13,
                                    abs_tol=dot_bounds.get(key, 0)), (
                    key, info[key], expected_value)
        # Independent check that dot_after is measured AFTER the dtype round trip.
        after = float(np.dot(o64, a64))
        assert math.isclose(float(info["dot_after"]), after, rel_tol=0, abs_tol=dot_bounds["dot_after"]), (
            "post-roundtrip dot", info["dot_after"], after, dot_bounds["dot_after"])
        return {"enabled": enabled, "dtype": str(dtype), "global_old_norm_sq":
                float(np.dot(flat(old), flat(old))), "diagnostics": {k: float(info[k]) for k in info_expected},
                "input_preserved": True}

    cases = [
        ("positive", [[[1, 2], [3, 4]], [5, 6]], [[[0.5, 1], [1.5, 2]], [2.5, 3]], torch.float32),
        ("orthogonal_global", [[[1, 0]], [1]], [[[1, 3]], [-1]], torch.float32),
        ("negative", [[[1, 2]], [3]], [[[-2, 1]], [-4]], torch.float32),
        ("opposite", [[[1, 2]], [3]], [[[-2, -4]], [-6]], torch.float32),
        ("old_zero", [[[0, 0]], [0]], [[[2, -3]], [4]], torch.float32),
        ("extra_zero", [[[1, 2]], [3]], [[[0, 0]], [0]], torch.float32),
        ("both_zero", [[[0, 0]], [0]], [[[0, 0]], [0]], torch.float32),
        ("global_negative_weight_positive_bias_negative", [[[1]], [2]], [[[3]], [-2]], torch.float32),
        ("global_positive_weight_negative_bias_positive", [[[2]], [2]], [[[-2]], [4]], torch.float32),
        ("global_weight_bias_norm", [[[3, 4], [0, 0]], [12]], [[[-3, 0], [0, 0]], [-1]], torch.float32),
        ("tiny_norm_no_epsilon", [[[1e-30]], [0]], [[[-1e-30]], [0]], torch.float32),
        ("subnormal_old_no_float32_square", [[[1e-40]], [0]], [[[-1]], [0]], torch.float32),
        ("float64_cancellation_negative_dot", [[[1e8, 1]], [-1e8]], [[[1, -1]], [1]], torch.float32),
        ("float64_inner_avoids_float32_overflow", [[[1e20]], [2e20]], [[[-1e20]], [-2e20]], torch.float32),
        ("float32_roundtrip_dot_after", [[[3]], [1]], [[[-1]], [1]], torch.float32),
        ("double_input", [[[1, 2]], [3]], [[[-2, 1]], [-4]], torch.float64),
        ("double_tiny_positive_norm", [[[1e-100]], [0]], [[[-1e-100]], [0]], torch.float64),
    ]
    for name, old, extra, dtype in cases:
        for enabled in (False, True):
            check(f"{name}/{'ON' if enabled else 'OFF'}", lambda o=old, e=extra, en=enabled, dt=dtype:
                  numeric(o, e, en, dt))

    def nonfinite(which, value, enabled):
        old, extra = tensors([[[1, 2]], [3]]), tensors([[[2, 3]], [4]])
        (old if which == "old" else extra)[0][0, 1] = value
        snapshots = [g.clone() for g in old + extra]
        try:
            project(old, extra, enabled)
        except RuntimeError as exc:
            unchanged(old + extra, snapshots)
            return {"exception": str(exc), "input_preserved": True}
        raise AssertionError("non-finite input was accepted")

    for which in ("old", "extra"):
        for label, value in (("nan", float("nan")), ("positive_inf", float("inf")),
                             ("negative_inf", -float("inf"))):
            for enabled in (False, True):
                check(f"reject_{which}_{label}/{'ON' if enabled else 'OFF'}",
                      lambda w=which, v=value, en=enabled: nonfinite(w, v, en))

    def output_overflow():
        old, extra = tensors([[[1]], [2]]), tensors([[[3e38]], [-3e38]])
        snapshots = [g.clone() for g in old + extra]
        try:
            project(old, extra, True)
        except RuntimeError as exc:
            unchanged(old + extra, snapshots)
            return {"exception": str(exc), "input_preserved": True}
        raise AssertionError("finite float64 projection overflowing returned float32 was accepted")
    check("reject_dtype_roundtrip_overflow", output_overflow)

    def internal_nonfinite(case, enabled):
        # Finite float64 inputs can still overflow an internal norm/dot. This
        # exercises the public helper's fail-closed contract, not realistic
        # float32 head gradients (whose products fit float64 at this head size).
        if case == "norm":
            old, extra = tensors([[[1e200]], [0]], torch.float64), tensors([[[-1e-200]], [0]], torch.float64)
        else:
            old, extra = tensors([[[1e154]], [0]], torch.float64), tensors([[[-1e155]], [0]], torch.float64)
        snapshots = [g.clone() for g in old + extra]
        try:
            _, info = project(old, extra, enabled)
        except RuntimeError as exc:
            unchanged(old + extra, snapshots)
            return {"exception": str(exc), "input_preserved": True,
                    "float64_only_contract_edge": True}
        unchanged(old + extra, snapshots)
        raise AssertionError(f"non-finite internal {case} was not rejected: " + repr(info))

    for which in ("norm", "dot"):
        for enabled in (False, True):
            check(f"reject_internal_float64_{which}_overflow/{'ON' if enabled else 'OFF'}",
                  lambda w=which, en=enabled: internal_nonfinite(w, en))

    def analytic_off():
        # Synthetic two-objective linear/quadratic losses; no model, optimizer or learner.
        w = torch.tensor([[0.3, -0.4], [0.5, 0.2]], requires_grad=True)
        b = torch.tensor([0.7, -0.1], requires_grad=True)
        a = torch.tensor([[2., -1.], [0.5, 3.]])
        c = torch.tensor([[1., 0.25], [-2., 4.]])
        q = torch.tensor([1.5, -0.5])
        r = torch.tensor([-3., 2.])
        old_loss = (a * w).sum() + 0.5 * (w * w).sum() + (q * b).sum()
        extra_loss = 0.25 * ((c * w).sum() + (r * b).sum() + 0.5 * (b * b).sum())
        gold = torch.autograd.grad(old_loss, (w, b), retain_graph=True)
        gextra = torch.autograd.grad(extra_loss, (w, b), retain_graph=True)
        adjusted, info = project(gold, gextra, False)
        combined = [o + e for o, e in zip(gold, adjusted)]
        total = torch.autograd.grad(old_loss + extra_loss, (w, b))
        analytic = [a + w.detach() + 0.25 * c, q + 0.25 * (r + b.detach())]
        max_diff = 0.
        for got, summed, exact in zip(combined, total, analytic):
            max_diff = max(max_diff, float(torch.max(torch.abs(got - summed))),
                           float(torch.max(torch.abs(got - exact))))
            torch.testing.assert_close(got, summed, rtol=0, atol=1e-6)
            torch.testing.assert_close(got, exact, rtol=0, atol=1e-6)
        assert info["projected"] == 0
        return {"maximum_absolute_difference": max_diff, "tolerance": 1e-6,
                "synthetic_analytic_and_total_loss_gradients_match": True}
    check("disabled_sum_matches_analytic_and_total_backward", analytic_off)

    def randomized():
        rng = np.random.default_rng(20261004)
        maximum_dot_error = 0.
        for index in range(32):
            o = [rng.normal(size=(7, 11)).tolist(), rng.normal(size=7).tolist()]
            e = [rng.normal(size=(7, 11)).tolist(), rng.normal(size=7).tolist()]
            for enabled in (False, True):
                detail = numeric(o, e, enabled)
                if enabled and detail["diagnostics"]["projected"]:
                    maximum_dot_error = max(maximum_dot_error, abs(detail["diagnostics"]["dot_after"]))
        return {"seed": 20261004, "pairs": 32, "enabled_states_per_pair": 2,
                "maximum_absolute_post_roundtrip_dot_when_projected": maximum_dot_error}
    check("seeded_synthetic_weight_bias_properties", randomized)

    def empty_rejected():
        try:
            project([], [], True)
        except ValueError as exc:
            return {"exception": str(exc)}
        raise AssertionError("empty gradient lists accepted")
    check("empty_lists_rejected", empty_rejected)

    def shapes_rejected():
        try:
            project([torch.zeros(2)], [torch.zeros(3)], True)
        except ValueError as exc:
            return {"exception": str(exc)}
        raise AssertionError("shape mismatch accepted")
    check("shape_mismatch_rejected", shapes_rejected)

    hashes_after = {"projected_train.py": sha(path), "independent_verify.py": sha(__file__)}
    stable = hashes_before == hashes_after
    failed = [r["name"] for r in records if r["status"] != "PASS"]
    result = {"schema": "codex_independent_projection_synthetic_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "RECHECK_REQUIRED_SOURCE_CHANGED" if not stable else ("FAILED" if failed else "PASSED"),
              "scope": "synthetic public project_extra_gradients only; no train_arm execution or real inputs",
              "module_path": str(path), "verifier_path": str(Path(__file__).resolve()),
              "hashes_before": hashes_before, "hashes_after": hashes_after, "source_hashes_stable": stable,
              "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
              "passed": len(records) - len(failed), "failed": len(failed), "failed_names": failed,
              "checks": records,
              "limits": ["Does not validate six training runs or empirical retention gates.",
                         "Does not claim a non-increasing old loss under AdamW.",
                         "Float32 returned gradients may have a small negative post-projection dot.",
                         "No real data/model/camera/optimizer used; import does not invoke load_experiment."]}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": result["passed"], "failed": failed,
                      "module_sha256": hashes_before["projected_train.py"], "output": str(out)}))
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
