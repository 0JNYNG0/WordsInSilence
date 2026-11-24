"""Lossless merge of actual reviewers' sampled context labels; no inference."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

U = Path(__file__).resolve().parent
AE = U.parent / "word100_guide_feature_supervision_20261102"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dst = U / "CONTEXT_LABELS.json"
    assert not dst.exists()
    queue = read(AE / "CONTEXT_REVIEW_QUEUE.json")
    packets = {r["clip_id"]: r for r in queue["rows"]}
    claims = {c["claim_id"]: c for r in queue["rows"] for c in r["claims"]}
    expected = {(c["claim_id"], f): tt for c in claims.values() for f, tt in zip(c["witness_frames"], c["witness_timestamps_us"])}
    paths = [U / f"CONTEXT_{w}.json" for w in ["0739", "1133", "1567", "2523"]]
    flat = []
    clips = []
    checks = 0
    stats = defaultdict(Counter)
    per_person = defaultdict(lambda: defaultdict(Counter))
    seen = set()
    for path in paths:
        data = read(path)
        for ri, r in enumerate(data["rows"]):
            q = packets[r["clip_id"]]
            assert r["person"] == q["registered_id"] and r["role"] == "fit"
            assert r["word_id"] == q["word_id"]
            sheet = r.get("reviewed_sheet", r.get("sheet"))
            assert sheet["path"] == q["sheet"]["path"] and sheet["sha256"] == q["sheet"]["sha256"]
            assert sha(sheet["path"]) == sheet["sha256"]
            displayed = r["actual_displayed_frames"]
            assert len(set(displayed)) == len(displayed)
            views = r.get("sheet_display_count", r.get("sheet_view_count", r.get("sheet_open_count_this_pass")))
            assert views == 1
            assert len(r["zoom_views"]) <= 2
            extra = {v["frame"] for v in r["zoom_views"]}
            clips.append(dict(clip_id=r["clip_id"], sheet=sheet, displayed_frames=displayed,
                zoom_views=r["zoom_views"], review_path=str(path), review_row=ri))
            checks += 7
            for ci, c in enumerate(r["claims"]):
                original = claims[c["claim_id"]]
                assert c["clause_id"] == original["clause_id"]
                assert c["source"] == original["source"] and c["source_text"] == original["source_text"]
                assert [p["frame"] for p in c["context_points"]] == original["witness_frames"]
                checks += 3
                for pi, p in enumerate(c["context_points"]):
                    key = c["claim_id"], p["frame"]
                    assert key not in seen and expected[key] == p["timestamp_us"]
                    assert p["frame"] in set(displayed) | extra
                    assert p["context"] in [0, 1, None]
                    assert p["tested_feature_used_to_assign_context"] is False
                    assert p["reason_ko"] and p["independent_body_or_trajectory_evidence"]
                    assert all(f in displayed for f in p["preceding_frames"] + p["following_frames"])
                    assert all(f < p["frame"] for f in p["preceding_frames"])
                    assert all(f > p["frame"] for f in p["following_frames"])
                    assert p.get("educational_error") is None and p.get("whole_sign_correctness") is None
                    seen.add(key)
                    value = p["context"]
                    name = "unknown" if value is None else "applicable_context" if value else "other_context"
                    stats[c["clause_id"]][name] += 1
                    per_person[c["clause_id"]][r["person"]][name] += 1
                    flat.append(dict(id=f"{c['claim_id']}:f{p['frame']}", clip_id=r["clip_id"],
                        word_id=r["word_id"], person=r["person"], role="fit",
                        clause_id=c["clause_id"], source=c["source"], source_text=c["source_text"],
                        frame=p["frame"], timestamp_us=p["timestamp_us"], context=value,
                        mask=value is not None, reason_ko=p["reason_ko"],
                        independent_body_or_trajectory_evidence=p["independent_body_or_trajectory_evidence"],
                        preceding_frames=p["preceding_frames"], following_frames=p["following_frames"],
                        observed_query_kind=c["query_kind"],
                        provenance=dict(path=str(path), sha256=sha(path), pointer=f"/rows/{ri}/claims/{ci}/context_points/{pi}"),
                        scope="Directly reviewed sampled context only; not guide violation, dense phase or physical feature truth",
                        whole_sign_correctness=None, educational_error=None, exact_event_boundary_truth=None))
                    checks += 10
    assert set(expected) == seen and len(flat) == 129 and len(clips) == 12
    counts = Counter("unknown" if r["context"] is None else "applicable_context" if r["context"] else "other_context" for r in flat)
    timing = read(U / "NATIVE_TIME_AUDIT.json")
    assert timing["status"] == "PASS_NATIVE_AXIS_DISPLAY_FPS_DIFFERENCE_DIAGNOSED"
    readiness = {}
    for cid, values in per_person.items():
        paired = [p for p, c in values.items() if c["applicable_context"] > 0 and c["other_context"] > 0]
        readiness[cid] = dict(per_person={str(p): dict(c) for p, c in values.items()},
            records_with_both_contexts=paired, both_context_coverage_condition_met=len(paired) >= 2,
            observed_feature_wrong_in_same_context_control_proven=False,
            model_shortcut_robustness_tested=False, training_ready=False)
    result = dict(status="DIRECT_CONTEXT_PILOT_COMPLETE_NO_AUTOMATIC_TRAINING_RELEASE",
        rows=flat, clips=clips, context_counts=dict(counts), per_clause={k: dict(v) for k, v in stats.items()},
        readiness=readiness, checks=checks, claims=len(claims), context_queries=len(flat),
        unique_clip_frames=len({(r["clip_id"], r["frame"]) for r in flat}),
        actual_sheet_cells=sum(len(r["displayed_frames"]) for r in clips), actual_zoom_views=sum(len(r["zoom_views"]) for r in clips),
        no_dense_label_propagation=True, unknown_is_not_negative=True,
        new_model_inference=0, new_training=0, independent_KSL_expert_validation=False,
        whole_goal_complete=False,
        pins={str(p): sha(p) for p in paths + [AE / "CONTEXT_REVIEW_QUEUE.json", AE / "NEXT_TIME_BINDING_REVIEW.json", U / "SCOPE.json", U / "NATIVE_TIME_AUDIT.json", Path(__file__)]})
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "context_queries", "context_counts", "actual_sheet_cells", "checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
