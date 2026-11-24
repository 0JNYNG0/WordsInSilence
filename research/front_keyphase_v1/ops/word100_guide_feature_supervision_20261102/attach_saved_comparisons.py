"""Append guide comparisons to frozen public outputs, never infer or rescore."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from guide_comparison import GuideComparator, read, sha

U = Path(__file__).resolve().parent
O = U.parent
A = O / "word100_partial_order_live_20261102"


def main():
    manifest_path = A / "SAVED_CASES_MANIFEST.json"
    manifest = read(manifest_path)
    comp = GuideComparator()
    output = U / "saved_comparisons"
    if output.exists():
        raise FileExistsError("Existing derived results are immutable; inspect completion first")
    output.mkdir()
    rows = []
    frozen = dict(contract_sha256=comp.contract_sha, code_sha256=sha(U / "guide_comparison.py"),
        driver_sha256=sha(Path(__file__)), manifest_sha256=sha(manifest_path),
        new_model_inference=0, new_training=0, old_scores_recomputed=False)
    (U / "SAVED_ATTACHMENT_FROZEN.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    for word, cases in manifest["cases"].items():
        for kind, entry in cases.items():
            base_path = Path(entry["result"])
            array_path = Path(entry["analysis"])
            assert base_path.resolve().is_relative_to(A.resolve() / "saved_cases")
            assert sha(base_path) == entry["result_sha256"]
            assert sha(array_path) == entry["analysis_sha256"]
            base = read(base_path)
            with np.load(array_path, allow_pickle=False) as arrays:
                ff = arrays["frame_id"].copy()
                tt = arrays["timestamp_us"].copy()
            out = comp.attach(base, ff, tt)
            dst = output / (word + "_" + kind + ".json")
            dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            addon = out["guide_feature_comparison"]
            items = addon["items"]
            states = Counter(p["observation"] for i in items for p in i["native_points"])
            old_points = {c["component_id"]: c.get("point_evidence", []) for c in base["source_components"]}
            for i in items:
                if i["native_points"]:
                    assert len(i["native_points"]) == len(old_points[i["component_id"]])
                    assert all(p["guide_violation"] is None for p in i["native_points"])
                assert not i["correction_enabled"]
            assert all(out[key] == value for key, value in base.items())
            rows.append(dict(word_id=word, condition=kind, base_path=str(base_path),
                base_sha256=entry["result_sha256"], arrays_path=str(array_path), arrays_sha256=entry["analysis_sha256"],
                result=str(dst), result_sha256=sha(dst),
                native_axis_sha256=hashlib.sha256(ff.tobytes() + tt.tobytes()).hexdigest(),
                old_fields_preserved=True, old_points_preserved=True, items=len(items),
                point_states=dict(states),
                component_summaries=[dict(clause_id=i["clause_id"], component_id=i["component_id"],
                    summary=i["summary"]["status"], frames=len(i["frame_ids"]),
                    time_binding=i["time_binding"], correction_enabled=i["correction_enabled"]) for i in items],
                comparison_seconds=addon["comparison_seconds"]))
            # Save progress after each append so an interrupted run is not restarted.
            (U / "SAVED_ATTACHMENT_PROGRESS.json").write_text(json.dumps(dict(completed=len(rows), rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    assert len(rows) == 16
    aggregate = Counter()
    for row in rows:
        aggregate.update(row["point_states"])
    report = dict(status="PASS_ADDITIVE_PUBLIC_COMPARISON_NOT_ACCURACY_EVALUATION", rows=rows,
        cases=len(rows), point_states=dict(aggregate),
        comparison_seconds_median=float(np.median([r["comparison_seconds"] for r in rows])),
        previous_evaluation_and_source_files_changed=False, new_model_inference=0, new_training=0,
        old_scores_recomputed=False, normal_crossing_preservation_scope="Exact old neural fields, source points and partial order outputs retained; no new detector/recognition accuracy test",
        edited_cases_are_semantic_error_gold=False, accuracy_gain_established=False,
        end_to_end_latency_measured=False, whole_goal_complete=False,
        code_sha256=frozen["code_sha256"], contract_sha256=comp.contract_sha,
        manifest_sha256=frozen["manifest_sha256"])
    (U / "SAVED_PUBLIC_COMPARISON.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "cases", "point_states", "comparison_seconds_median"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
