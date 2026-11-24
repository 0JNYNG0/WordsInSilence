"""Pure synthetic check of teacher-source and overlap repair. No real NPZ/model reads."""
import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "claude_code_only"


def main():
    paths = [MIRROR / "teacher_pair.py", MIRROR / "test_teacher.py", HERE / "teacher_pair.py", HERE / "test_teacher.py"]
    hashes = lambda: {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    spec = importlib.util.spec_from_file_location("independent_teacher_repair_subject", MIRROR / "teacher_pair.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)  # Definitions/imports only; load_modules is never called.
    checks = []

    def check(name, fn):
        try:
            checks.append({"name": name, "status": "PASS", "details": fn()})
        except Exception as exc:
            checks.append({"name": name, "status": "FAIL", "error": repr(exc)})

    def reject(fn):
        try:
            fn()
        except RuntimeError as exc:
            return {"rejected": str(exc)}
        raise AssertionError("invalid synthetic input accepted")

    class Archive:
        def __init__(self, arrays):
            self.arrays, self.files = arrays, list(arrays)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def __getitem__(self, key):
            return self.arrays[key]

    ids, classes = ["a", "b", "c", "d"], ["c0", "c1", "c2"]
    parent = np.arange(12, dtype=np.float64).reshape(4, 3) / 8
    arrays = dict(parent=parent, control=parent + 10, treatment=parent - 20,
                  ids=np.array(ids), classes=np.array(classes))
    selected, cached = ["a", "c", "d"], parent[[0, 2, 3]].copy()
    maxabs = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    def load(values):
        # Mock only the archive reader; no synthetic or real file is created/read.
        with patch.object(tp.np, "load", return_value=Archive(values)) as call:
            result = tp.load_teacher_scores(Path("synthetic_teacher.npz"))
            assert call.call_args.kwargs == {"allow_pickle": False}
            return result

    ts = load(arrays)
    def authoritative():
        assert ts["score_key"] == "parent" and np.array_equal(ts["scores"], parent)
        result = tp.check_teacher_scores(ts, selected, classes, cached, maxabs)
        assert result["parent_compared"] and result["parent_vs_cache_teacher_max_abs"] == 0
        assert not result["identical_order"] and result["relative_order_preserved"]
        return result
    check("parent_key_and_order_preserving_selection", authoritative)
    check("missing_parent_no_control_fallback", lambda: reject(lambda: load({k: v for k, v in arrays.items() if k != "parent"})))
    check("control_scores_cannot_replace_parent", lambda: reject(lambda:
          tp.check_teacher_scores(ts, selected, classes, cached + 10, maxabs)))
    check("class_permutation_rejected", lambda: reject(lambda:
          tp.check_teacher_scores(ts, selected, classes[::-1], cached[:, ::-1], maxabs)))
    check("row_permutation_rejected_even_matching_values", lambda: reject(lambda:
          tp.check_teacher_scores(ts, ["c", "a", "d"], classes, cached[[1, 0, 2]], maxabs)))
    check("missing_base_id_rejected", lambda: reject(lambda:
          tp.check_teacher_scores(ts, ["a", "z", "d"], classes, cached, maxabs)))
    check("duplicate_teacher_ids_rejected", lambda: reject(lambda:
          tp.check_teacher_scores(dict(ts, ids=["a", "b", "c", "a"]), selected, classes, cached, maxabs)))
    check("duplicate_base_ids_rejected", lambda: reject(lambda:
          tp.check_teacher_scores(ts, ["a", "a", "d"], classes, cached, maxabs)))
    check("wrong_score_shape_rejected", lambda: reject(lambda:
          tp.check_teacher_scores(dict(ts, scores=parent[:, :2]), selected, classes, cached, maxabs)))
    check("wrong_cache_shape_rejected", lambda: reject(lambda:
          tp.check_teacher_scores(ts, selected, classes, cached[:-1], maxabs)))
    nan = cached.copy()
    nan[0, 1] = np.nan
    check("nonfinite_comparison_rejected", lambda: reject(lambda:
          tp.check_teacher_scores(ts, selected, classes, nan, maxabs)))
    mismatch = cached.copy()
    mismatch[2, 1] += 1e-4
    check("above_frozen_tolerance_rejected", lambda: reject(lambda:
          tp.check_teacher_scores(ts, selected, classes, mismatch, maxabs)))

    meta = dict(annotation_dev=[{"id": "dev_a"}], whole_dev=[{"id": "dev_w"}],
                public=[{"id": "a", "condition": "intact"}, {"id": "a", "condition": "masked"}, {"id": "b"}, {"id": "public_only"}],
                mp=[{"id": "d", "condition": "whole"}, {"id": "d", "condition": "annotation"}, {"id": "mp_only"}])
    meta0 = copy.deepcopy(meta)
    def exposure():
        result = tp.id_overlap_audit(ids, meta, ["ksl0", "ksl1"], ["a", "sup0"])
        assert result["heldout_must_be_zero"] == dict(annotation_dev=0, whole_dev=0, new36=0)
        p, m = (result["training_exposed_regression_reference"][key] for key in ("public", "mp"))
        assert (p["rows"], p["unique_ids"], p["overlap_ids"], p["overlapping_ids"]) == (4, 3, 2, ["a", "b"])
        assert (m["rows"], m["unique_ids"], m["overlap_ids"], m["overlapping_ids"]) == (3, 2, 1, ["d"])
        assert result["support113_informational"] == 1 and meta == meta0
        assert "not an independent test" in result["training_exposed_note"]
        return result
    check("public_mp_exposure_counts_distinguish_rows_and_ids", exposure)
    for group in ("annotation_dev", "whole_dev"):
        bad = copy.deepcopy(meta)
        bad[group].append({"id": "a"})
        check(f"new_overlap_in_{group}_rejected", lambda bad=bad: reject(lambda:
              tp.id_overlap_audit(ids, bad, ["ksl0"], [])))
    check("new36_overlap_rejected", lambda: reject(lambda:
          tp.id_overlap_audit(ids, meta, ["a"], [])))
    check("missing_group_rejected", lambda: reject(lambda:
          tp.id_overlap_audit(ids, {k: v for k, v in meta.items() if k != "whole_dev"}, ["ksl0"], [])))
    after = hashes()
    matched = all(before[f"claude_code_only/{n}"] == before[n] for n in ("teacher_pair.py", "test_teacher.py"))
    failed = [r["name"] for r in checks if r["status"] != "PASS"]
    result = dict(status="PASSED" if not failed and before == after and matched else "FAILED",
                  created_utc=datetime.now(timezone.utc).isoformat(), passed=len(checks) - len(failed), failed=failed,
                  hashes_before=before, hashes_after=after, source_unchanged=before == after,
                  applied_matches_mirror=matched, checks=checks,
                  real_arrays_models_camera_reads=0, training_runs=0,
                  limitations=["Repair helpers only; source imports do not load historical modules.",
                               "Does not override the observed preflight numerical-tolerance failure.",
                               "Public/MP overlap is audited against pinned metadata by preflight; not treated as a new independent test."])
    output = HERE / "INDEPENDENT_REPAIR_SYNTHETIC_V2.json"
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "passed", "failed", "source_unchanged", "applied_matches_mirror")}))
    return int(result["status"] != "PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
