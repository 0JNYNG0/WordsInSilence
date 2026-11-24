"""Exact-rational, synthetic-only audit of analytic precision bounds and unchanged experiment code."""
import ast
import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "current_teacher_pair_20261004"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # No old modules/models/data loaded by module-level definitions.
    return module


def rational(value):
    return Q(float(value))


def gamma(d, dtype):
    u, k = rational(np.finfo(dtype).eps) / 2, 2 * d + 1
    return k * u / (1 - k * u)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [HERE / "teacher_pair.py", HERE / "test_teacher.py",
             HERE / "claude_code_only/teacher_pair.py", HERE / "claude_code_only/test_teacher.py",
             OLD / "teacher_pair.py", OLD / "test_teacher.py"]
    hashes = lambda: {str(p): sha(p) for p in paths}
    before = hashes()
    tp = load(HERE / "teacher_pair.py", "independent_precision_subject")
    old = load(OLD / "teacher_pair.py", "independent_original_teacher_subject")
    torch.set_num_threads(2)
    checks = []

    def check(name, fn):
        try:
            checks.append({"name": name, "status": "PASS", "details": fn() or {}})
        except Exception as exc:
            checks.append({"name": name, "status": "FAIL", "error": repr(exc)})

    def unchanged_experiment():
        def functions(path):
            source = path.read_text(encoding="utf-8").replace("current_teacher_precision_pair", "current_teacher_pair")
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        a, b = functions(OLD / "teacher_pair.py"), functions(HERE / "teacher_pair.py")
        names = ["make_teacher", "encoder_checks", "run_arm", "check_plan_inputs", "load_teacher_scores",
                 "check_teacher_scores", "id_overlap_audit", "load_base_rows", "build_teachers", "build_832",
                 "check_out", "train", "changes_between", "evaluate"]
        different = [name for name in names if a[name] != b[name]]
        assert not different, different
        consts = ["PARENT_SHA", "REF_SHA", "TRAIN_KW", "LOSS_COLUMNS", "STEPS", "EXTRA_W", "N_FOLDS",
                  "HALF", "N_BLOCKS", "KSL_PER_STEP", "SUPPORT_PER_STEP", "N_BASE", "N_CTX", "N_CLASSES",
                  "FOLD_TRAIN", "FOLD_HELD", "HEAD_KEYS", "GROUP_N", "ANNOTATION_MIN", "GATE_PRIMARY_TOTAL",
                  "TOL_TEACHER", "TOL_832", "TOL_NEW", "HELDOUT_GROUPS", "TRAINING_EXPOSED_GROUPS"]
        assert all(getattr(tp, name) == getattr(old, name) for name in consts)
        return {"function_AST_identical_after_unit_name_normalization": names, "constants_identical": consts}
    check("learner_teacher_evaluation_and_gates_unchanged", unchanged_experiment)

    def constants():
        for d in (1, 3, 64):
            c = tp.precision_constants(d)
            expected = {"gamma32": gamma(d, np.float32), "gamma64": gamma(d, np.float64),
                        "inv_one_minus_gamma64": 1 / (1 - gamma(d, np.float64)),
                        "k_tiny32": (2 * d + 1) * rational(np.finfo(np.float32).tiny),
                        "k_tiny64": (2 * d + 1) * rational(np.finfo(np.float64).tiny)}
            for key, exact in expected.items():
                got = float(c[key])
                assert rational(got) >= exact, (d, key, "down-rounded")
                assert rational(np.nextafter(got, -np.inf)) < exact, (d, key, "not minimal upper float")
        return {"dimensions": [1, 3, 64], "all_constants_smallest_float64_upper_bound": True}
    check("constants_checked_against_independent_exact_fractions", constants)

    def head_logits(p, w, b):
        head = nn.Linear(p.shape[1], w.shape[0])
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(w)); head.bias.copy_(torch.from_numpy(b))
            parts = [head(torch.from_numpy(p[i:i + 32].copy())).numpy().copy() for i in range(0, len(p), 32)]
        got = np.concatenate(parts)
        assert np.array_equal(got, tp.head32_batched(head, p, batch_size=32))
        return got

    def exact_bounds(o, p, w, b):
        d, k = o.shape[1], 2 * o.shape[1] + 1
        g32, g64 = gamma(d, np.float32), gamma(d, np.float64)
        t32, t64 = rational(np.finfo(np.float32).tiny), rational(np.finfo(np.float64).tiny)
        result = {name: [] for name in ("numpy64_vs_torch64_old_x", "prefix32_vs_numpy64_prefix_x",
                                       "prefix32_vs_old_x_float64_teacher")}
        for i in range(len(o)):
            for c in range(len(w)):
                ro, rp, rw = [rational(v) for v in o[i]], [rational(v) for v in p[i]], [rational(v) for v in w[c]]
                so = sum((abs(x * y) for x, y in zip(ro, rw)), Q()) + abs(rational(b[c]))
                sp = sum((abs(x * y) for x, y in zip(rp, rw)), Q()) + abs(rational(b[c]))
                delta = sum((abs(x - y) * abs(z) for x, y, z in zip(rp, ro, rw)), Q())
                result["numpy64_vs_torch64_old_x"].append(2 * (g64 * so + k * t64))
                result["prefix32_vs_numpy64_prefix_x"].append(g32 * sp + k * t32 + g64 * sp + k * t64)
                result["prefix32_vs_old_x_float64_teacher"].append(delta + g32 * sp + k * t32 + g64 * so + k * t64)
        return result

    def exercise(o, p, w, b):
        keep = [x.tobytes() for x in (o, p, w, b)]
        logits = head_logits(p, w, b)
        record, arrays = tp.precision_checks(o, p, w, b, logits)
        assert record["passed"], record
        required = exact_bounds(o, p, w, b)
        minimum_relative, maximum_relative = float("inf"), 0.
        for name, qs in required.items():
            got = arrays[f"precision__{name}__bound"].ravel()
            for bound, exact in zip(got, qs):
                assert rational(bound) >= exact, (name, "bound below exact symbolic requirement")
                # An anti-fudge check: directed-rounding overhead stays tiny; this is NOT
                # a production score tolerance or a fitted numerical acceptance criterion.
                ratio = float(rational(bound) / exact)
                minimum_relative, maximum_relative = min(minimum_relative, ratio), max(maximum_relative, ratio)
                assert ratio <= 1 + 1e-10, (name, "bound inflated beyond rounding allowance", ratio)
        assert keep == [x.tobytes() for x in (o, p, w, b)]
        return {"elements_per_check": len(o) * len(w), "bound_over_exact_symbolic_min": minimum_relative,
                "bound_over_exact_symbolic_max": maximum_relative, "details": record}

    rng = np.random.default_rng(614)
    o = rng.normal(size=(35, 64)).astype(np.float32)
    p = o.copy(); p[0, 0] = np.nextafter(p[0, 0], np.float32(np.inf)); p[34, 61] += np.float32(1e-6)
    w, b = rng.normal(size=(5, 64)).astype(np.float32), rng.normal(size=5).astype(np.float32)
    check("d64_35_rows_partial32_batch_and_exact_bounds", lambda: exercise(o, p, w, b))
    zero = np.zeros((2, 64), np.float32)
    check("all_zero_tiny_absolute_term", lambda: exercise(zero, zero.copy(), np.zeros((3, 64), np.float32), np.zeros(3, np.float32)))
    tiny = np.full((2, 64), np.float32(1e-30))
    check("normal_operands_subnormal_products", lambda: exercise(tiny, tiny.copy(), np.full((3, 64), np.float32(1e-10)), np.zeros(3, np.float32)))
    cancellation = np.zeros((2, 64), np.float32)
    cancellation[:, :3] = [1e8, 1, -1e8]
    check("large_cancellation_and_centered_small_answer", lambda: exercise(cancellation, cancellation.copy(),
          np.ones((3, 64), np.float32), np.array([0, 1, -1], np.float32)))

    def bad_outputs():
        logits = head_logits(p, w, b)
        teacher = tp.make_teacher(o, w, b)
        shifted = teacher.copy(); shifted[0, 0] += 1e-10
        r, _ = tp.precision_checks(o, p, w, b, logits, teacher=shifted)
        assert not r["passed"] and not r["teacher_bitwise_equal_numpy64_affine"]
        damaged = logits.copy(); damaged[0, 0] += 1
        r, _ = tp.precision_checks(o, p, w, b, damaged)
        assert not r["passed"] and r["checks"]["prefix32_vs_numpy64_prefix_x"]["violations"] > 0
        permuted, _ = tp.precision_checks(o, p, w, b, logits[:, ::-1].copy())
        assert not permuted["passed"]
        return {"modified_teacher_logits_and_class_permutation_rejected": True}
    check("deliberate_teacher_logit_and_class_corruption_rejected", bad_outputs)

    def invalid():
        logits = head_logits(p, w, b)
        not_repr = o.astype(np.float64); not_repr[0, 0] = 0.1
        subnormal = o.copy(); subnormal[0, 0] = np.nextafter(np.float32(0), np.float32(1))
        nonfinite = o.copy(); nonfinite[0, 0] = np.nan
        overflow_x, overflow_w = np.full_like(o, 1e30), np.full_like(w, 1e20)
        cases = [(not_repr, p, w, b, logits), (subnormal, p, w, b, logits), (nonfinite, p, w, b, logits),
                 (o, p[:-1], w, b, logits), (o, p, w[:, :-1], b, logits),
                 (overflow_x, overflow_x, overflow_w, b, np.zeros_like(logits))]
        for args in cases:
            try:
                tp.precision_checks(*args)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("invalid numeric input accepted")
        return {"cases_rejected": len(cases)}
    check("dtype_shape_nonfinite_subnormal_overflow_rejected", invalid)

    after = hashes()
    matched = all(before[str(HERE / n)] == before[str(HERE / "claude_code_only" / n)] for n in ("teacher_pair.py", "test_teacher.py"))
    failures = [c["name"] for c in checks if c["status"] != "PASS"]
    result = {"status": "PASSED" if not failures and before == after and matched else "FAILED",
              "created_utc": datetime.now(timezone.utc).isoformat(), "passed": len(checks) - len(failures),
              "failed": failures, "source_hashes_before": before, "source_hashes_after": after,
              "source_unchanged": before == after, "applied_matches_mirror": matched, "checks": checks,
              "actual_data_model_camera_reads": 0, "training_runs": 0,
              "verifier_sha256": sha(Path(__file__)),
              "limits": ["Synthetic checks of the frozen analytic formula; not a hardware floating-point proof.",
                         "Real preflight and teacher/train/evaluation provenance require separate checks."]}
    out = HERE / "INDEPENDENT_PRECISION_SYNTHETIC.json"
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "passed", "failed", "source_unchanged", "applied_matches_mirror")}))
    return int(result["status"] != "PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
