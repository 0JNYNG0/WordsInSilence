"""Check new-input guards without reattaching completed public comparisons."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from runtime_comparison import RuntimeGuideComparator
from guide_comparison import read, sha

U = Path(__file__).resolve().parent


def main():
    dst = U / "RUNTIME_INPUT_CHECKS.json"
    assert not dst.exists()
    wrapper = RuntimeGuideComparator()
    rows = read(U / "SAVED_PUBLIC_COMPARISON.json")["rows"]
    manifest = read(U.parent / "word100_partial_order_live_20261102/SAVED_CASES_MANIFEST.json")
    checks = []
    sample = None
    for row in rows:
        base = read(row["base_path"])
        m = manifest["cases"][row["word_id"]][row["condition"]]
        assert sha(row["base_path"]) == row["base_sha256"]
        assert sha(row["arrays_path"]) == row["arrays_sha256"]
        with np.load(row["arrays_path"], allow_pickle=False) as a:
            ff, tt = a["frame_id"].copy(), a["timestamp_us"].copy()
        wrapper.preflight(base, ff, tt, expected_word=row["word_id"], expected_source_sha256=m["video_sha256"])
        checks.append(row["word_id"] + ":" + row["condition"] + ":preflight_only")
        if sample is None:
            sample = base, ff, tt, row["word_id"], m["video_sha256"]
    base, ff, tt, word, digest = sample
    for name, mutate in [
        ("word", lambda b: b.update(word_id="WORD1133")),
        ("video", lambda b: b.update(source_sha256="f" * 64)),
        ("frame_count", lambda b: b.update(frames=len(ff) - 1)),
        ("duplicate_clause", lambda b: b["source_components"][0]["source_clauses"].append(deepcopy(b["source_components"][0]["source_clauses"][0]))),
    ]:
        invalid = deepcopy(base)
        mutate(invalid)
        try:
            wrapper.preflight(invalid, ff, tt, expected_word=word, expected_source_sha256=digest)
        except ValueError:
            checks.append(name + ":rejected")
        else:
            raise AssertionError("Accepted " + name)
    result = dict(status="PASS_NEW_INPUT_GUARDS_NO_PUBLIC_RESULT_REATTACHMENT", checks=checks,
        count=len(checks), public_cases_preflight_only=16, outputs_recomputed=0,
        new_model_inference=0, new_training=0, production_wiring=False,
        pins={str(p): sha(p) for p in [U / "runtime_comparison.py", U / "guide_comparison.py", Path(__file__)]})
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], checks=len(checks))))


if __name__ == "__main__":
    main()
