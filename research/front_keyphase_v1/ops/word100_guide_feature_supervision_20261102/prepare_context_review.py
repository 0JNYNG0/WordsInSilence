"""Prepare a bounded event-context review without copying physical labels."""
from pathlib import Path
from collections import Counter
import hashlib
import json

U = Path(__file__).resolve().parent
Q = U.parent / "word100_predicate_supervision_20261102"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dst = U / "CONTEXT_REVIEW_QUEUE.json"
    assert not dst.exists()
    plan_path = U / "NEXT_TIME_BINDING_REVIEW.json"
    plan = read(plan_path)
    paths = [Q / n for n in ["BINDING_WITNESSES.json", "DIRECT_BINDINGS_ROOT.json", "DIRECT_BINDINGS_1567.json", "DIRECT_BINDINGS_2523.json"]]
    witness = read(paths[0])["rows"]
    sheets = {}
    for path in paths[1:]:
        for i, row in enumerate(read(path)["rows"]):
            if row["person"] in [1, 4, 7]:
                assert row["role"] == "fit"
                sheets[row["clip_id"]] = dict(metadata_path=str(path), metadata_pointer=f"/rows/{i}",
                    image=row["reviewed_sheet"], word_id=row["word_id"], person=row["person"])
    assert len(sheets) == 12
    rows = []
    for clip, sheet in sorted(sheets.items()):
        assert sha(sheet["image"]["path"]) == sheet["image"]["sha256"]
        claims = []
        for i, r in enumerate(witness):
            if r["clip_id"] != clip:
                continue
            assert r["role"] == "fit" and r["person"] in [1, 4, 7]
            # Intentionally omit r.component/support/masks/unsupported notes:
            # these can reveal the old physical judgement of the same image.
            claims.append(dict(claim_id=r["id"], provenance_pointer=f"/rows/{i}",
                clause_id=r["clause_id"], source_text=r["source_text"], source=r["source"],
                query_kind=r["kind"], witness_frames=r["witness_frames"],
                witness_timestamps_us=r["witness_timestamps_us"],
                available_zoom_images=r["inspected_images"],
                context_labels=[dict(frame=f, timestamp_us=t, event_context=None,
                    context_reason_without_target_feature=None, source_action_match=None)
                    for f, t in zip(r["witness_frames"], r["witness_timestamps_us"])],
                copied_physical_label=False, dense_label_propagation=False))
        rows.append(dict(clip_id=clip, word_id=sheet["word_id"], registered_id=sheet["person"],
            role="fit", sheet=sheet["image"], sheet_metadata_path=sheet["metadata_path"],
            sheet_metadata_pointer=sheet["metadata_pointer"], claims=claims,
            review_status="PENDING", maximum_extra_zoom_views=2))
    assert sum(len(r["claims"]) for r in rows) == 39
    out = dict(status="PREPARED_NOT_CONTEXT_LABELLED", scope="12 prespecified FIT videos, four expressions by registered IDs01/04/07",
        rows=rows, clips=len(rows), claims=39,
        witness_queries=sum(len(c["witness_frames"]) for r in rows for c in r["claims"]),
        label_schema=["ACTION_CONTEXT", "OTHER_CONTEXT", "CONTEXT_UNCERTAIN"],
        labels_are_educational_correctness=False, labels_are_framewise_shape_truth=False,
        review_rules=[
            "Review the chronological sheet once; at most two existing large witness images per clip when needed.",
            "Decide event context from body-relative action and before/after flow, not whether the required handshape is correct.",
            "A changed/missing required feature must not itself move the query outside the action's comparison scope.",
            "Separate the sampled point context from sequence context. Do not label unsampled frames or exact onset/offset.",
            "Keep source-action applicability uncertain when variant or linguistic timing cannot be established.",
            "After all12, close this batch and assess trainable context support and same-feature/different-context controls; no indefinite review expansion.",
        ],
        old_support_values_and_model_scores_excluded=True,
        historical_results_known=True, independent_blinded_human_review=False,
        preinspection=dict(sheets=1, clip="NIA_SL_WORD0739_REAL01_F", direct_image_view=True,
            new_context_labels=0, note="Root inspected this sheet once before queue compilation; include it in the12-sheet budget, do not claim blind review"),
        reviewed_plan_status=plan["status"], new_training=0, new_inference=0,
        whole_goal_complete=False, pins={str(p): sha(p) for p in [plan_path, *paths, Path(__file__)]})
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["status", "clips", "claims", "witness_queries"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
