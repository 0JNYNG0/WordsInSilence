"""Record root's actual sheet review, not labels inferred from prior targets."""
from pathlib import Path
from collections import Counter
import hashlib
import json

U = Path(__file__).resolve().parent
AE = U.parent / "word100_guide_feature_supervision_20261102"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


OBSERVED = {
    1: dict(frames=[0,40,43,45,48,51,53,55,56,59,60,62,64,67,68,70,72,75,76,78,80,81,83,86,87,89,91,94,97,99,102,119],
        face=[48,55,60], lateral=[64,67,68,70,72], elbow_change=[75,78,81], presented=[81,86,89,91], returning=[94,97,99,102]),
    4: dict(frames=[0,63,66,70,73,77,80,84,86,87,91,94,98,100,101,103,105,108,112,114,115,119,121,122,126,128,129,133,136,140,143,167],
        face=[73,84,87], lateral=[94,98,100,101,103], elbow_change=[105,108,112], presented=[114,115,119,126,129], returning=[133,136,140,143]),
    7: dict(frames=[0,32,35,37,40,42,45,47,48,50,52,53,55,58,61,63,66,69,71,74,76,79,80,82,84,87,89,92,121],
        face=[40,47,50,52], lateral=[58,61,63], elbow_change=[66,69,71], presented=[71,74,76,79], returning=[80,82,84,87,89,92]),
}


def main():
    dst = U / "CONTEXT_0739.json"
    assert not dst.exists()
    queue = read(AE / "CONTEXT_REVIEW_QUEUE.json")
    plan = read(AE / "NEXT_TIME_BINDING_REVIEW.json")
    sources = {r["clip_id"]: r["source_sha256"] for r in plan["pilot_packets"]}
    rows = []
    counts = Counter()
    for packet in queue["rows"]:
        if packet["word_id"] != "WORD0739":
            continue
        person = packet["registered_id"]
        visual = OBSERVED[person]
        assert sha(packet["sheet"]["path"]) == packet["sheet"]["sha256"]
        claims = []
        for claim in packet["claims"]:
            points = []
            for frame, ts in zip(claim["witness_frames"], claim["witness_timestamps_us"]):
                assert frame in visual["frames"]
                if claim["clause_id"] == "obedience_closed_hand":
                    if frame in visual["face"]:
                        value = 0
                        reason = "이 시점은 손목이 얼굴 앞에 놓이고 오른팔꿈치가 아래에 있는 앞선 얼굴 제시 동작이다. 이후 전완이 가슴 앞에 가로로 놓이고 팔꿈치가 옆으로 나오는 동작과 구별된다. 손가락이 펴졌는지를 기준으로 구분하지 않았다."
                        cues = ["wrist near face", "elbow below wrist", "later transverse forearm presentation is a different surrounding action"]
                    else:
                        value = 1
                        reason = "앞선 얼굴 제시 뒤 오른전완을 몸 앞에 가로로 두었다가 팔꿈치가 낮아지고 손목이 같은 쪽 어깨 가까이 놓이는 팔 동작의 맥락이다. 오른손이 펴져 있거나 가려져도 이 팔 배치와 앞뒤 순서만으로 같은 맥락이어야 한다."
                        cues = ["after face-presentation action", "lateral elbow/transverse forearm then lower elbow", "wrist remains near upper torso while left arm rests"]
                    circular = False
                else:
                    value = None
                    reason = "넓은 오른팔 제시 맥락과 팔꿈치 배치 변화는 보인다. 그러나 검사하려는 ‘옆으로 들었다가 내리는’ 움직임 자체를 제외하고 이 문구의 정확한 적용 시점을 독립적으로 가르는 근거는 부족하다. 해당 움직임이 없었다는 뜻은 아니며, 움직임 완료·단계 경계 정답으로 삼지 않는다."
                    cues = ["right arm remains in the later torso/shoulder presentation", "same broad episode includes a lateral pose, elbow change and later hold", "specific source-motion applicability cannot be separated without the tested motion"]
                    circular = "would_be_circular_if_forced"
                points.append(dict(frame=frame, timestamp_us=ts, context=value, reason_ko=reason,
                    independent_body_or_trajectory_evidence=cues,
                    preceding_frames=[f for f in visual["frames"] if f < frame][-3:],
                    following_frames=[f for f in visual["frames"] if f > frame][:3],
                    tested_feature_used_to_assign_context=False,
                    circularity_if_forced=circular,
                    broad_right_arm_presentation_context=(0 if frame in visual["face"] else 1),
                    source_action_match=None, educational_error=None, whole_sign_correctness=None))
                counts[str(value)] += 1
            claims.append(dict(claim_id=claim["claim_id"], clause_id=claim["clause_id"],
                source_text=claim["source_text"], source=claim["source"], query_kind=claim["query_kind"],
                local_context_definition="Later right-arm presentation relative to torso/shoulder versus earlier face presentation; finger shape is not eligibility evidence",
                source_variant_status="local_partial_context_only_full_variant_unresolved",
                context_points=points, exact_event_boundary_truth=None,
                uncertain_boundaries="Unshown transitions between sampled witnesses are not labelled",
                unresolved_dimensions=["full clause timing", "fist/thumb correctness", "whole motion completion", "linguistic permissibility"],
                whole_sign_correctness=None, educational_error=None))
        rows.append(dict(clip_id=packet["clip_id"], word_id=packet["word_id"], person=person, role="fit",
            reviewer="Codex root", source_sha256=sources[packet["clip_id"]], reviewed_sheet=packet["sheet"],
            sheet_display_count=1, prior_turn_display_reused=person == 1,
            actual_displayed_frames=visual["frames"], all_sheet_cells_visually_inspected=True, zoom_views=[],
            observed_context_sequences={k: v for k, v in visual.items() if k != "frames"},
            claims=claims, historical_awareness=True, independent_KSL_expert=False,
            shortcut_challenge=dict(manual_body_cues_available=True,
                counterfactual="Keeping the same arm/torso trajectory but opening the hand must not change this context label",
                actual_feature_changed_in_same_context_tested=False,
                same_feature_value_in_other_context_control_available_in_queue=False,
                model_shortcut_resistance_verified=False)))
    assert len(rows) == 3
    result = dict(status="DIRECT_SAMPLED_CONTEXT_REVIEW_COMPLETE_NOT_TRAINING_RELEASE", rows=rows,
        claims=sum(len(r["claims"]) for r in rows), context_queries=sum(len(c["context_points"]) for r in rows for c in r["claims"]),
        context_counts=dict(counts), reviewed_sheet_cells=sum(len(r["actual_displayed_frames"]) for r in rows),
        zoom_views=0, original_labels_read_for_assignment=False, new_model_inference=0, new_training=0,
        targeted_time_context_not_semantic_correctness=True, whole_goal_complete=False,
        readiness=dict(shape_context_both_classes_records=3,
            specific_motion_context_not_supported_independently=True,
            actual_context_feature_disentanglement_not_yet_proven=True),
        pins={str(p): sha(p) for p in [AE / "CONTEXT_REVIEW_QUEUE.json", AE / "NEXT_TIME_BINDING_REVIEW.json", U / "SCOPE.json", Path(__file__)]})
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "claims", "context_queries", "context_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
