"""Additional Codex verification from saved raw observations and full logits.

This is a separate calculation, not an external reviewer or linguistic audit.
Four service replays are fixed by source order, not chosen from the scores.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import av
import numpy as np
import torch

import probe


def main():
    plan = probe.frozen()
    result = probe.read(probe.HERE / "RESULT.json")
    entries = {r["id"]: r for r in probe.read(probe.HERE / "EXTRACTION.json")["rows"]}
    results = {r["id"]: r for r in result["rows"]}
    assert len(entries) == len(results) == len(plan["rows"]) == 36
    assert result["plan_sha256"] == probe.sha(probe.HERE / "FROZEN_PLAN.json")
    assert result["logits_sha256"] == probe.sha(probe.OUT / "logits.npz")
    with np.load(probe.OUT / "logits.npz", allow_pickle=False) as z:
        scores, ids, classes = z["logits"], z["ids"].tolist(), z["classes"].tolist()
    assert scores.shape == (len(ids), 93) and len(ids) == len(set(ids))
    assert set(ids) == {r["id"] for r in result["rows"] if r["status"] == "OK"}
    assert np.isfinite(scores).all()
    score_by_id = dict(zip(ids, scores))
    from wis_lab.core.observation import Observation
    from pilot.web_input import load_web_clip
    from wis_lab.service.neural import _load, recognise
    checked = []
    for source in plan["rows"]:
        cid = source["id"]
        entry, row = entries[cid], results[cid]
        if entry["status"] != "OK":
            assert row["status"] == "EXTRACTION_FAILED"
            continue
        obs = Observation.load(entry["observation"])
        probe.check_obs(obs, entry["observation"], source, plan)
        # Decode actual PTS independently of the extraction iterator.
        with av.open(source["path"]) as container:
            stream = container.streams.video[0]
            pts = [frame.pts * frame.time_base for frame in container.decode(stream) if frame.pts is not None]
        times = np.array([round(float(t - pts[0]) * 1e6) for t in pts], dtype=np.int64)
        np.testing.assert_array_equal(obs.timestamp_us, times)
        np.testing.assert_array_equal(obs.frame_id, np.arange(len(times)))
        assert entry["dropped_non_monotonic_pts"] == 0
        if row["status"] != "OK":
            continue
        path = probe.OUT / cid / "model_input.npz"
        assert probe.sha(path) == row["input_sha256"]
        rebuilt = load_web_clip(entry["observation"])
        with np.load(path, allow_pickle=False) as z:
            for key in ("x", "body_valid", "hand_valid"):
                np.testing.assert_array_equal(rebuilt[key], z[key])
            x, ts = z["x"], z["timestamp_us"]
            np.testing.assert_array_equal(x[:, 186:237], z["body_valid"])
            np.testing.assert_array_equal(x[:, 237:279], z["hand_valid"])
            np.testing.assert_array_equal(x[1:, 279], np.diff(ts).astype(np.float32) / 1e6)
            assert x[0, 279] == 0 and np.isfinite(x).all()
        logit = score_by_id[cid]
        order = sorted(range(93), key=lambda j: (-float(logit[j]), j))
        assert row["prediction"] == classes[order[0]]
        assert row["agreement"] == (classes[order[0]] == source["candidate_label"])
        assert row["target_rank"] == order.index(classes.index(source["candidate_label"])) + 1
        assert [r["label"] for r in row["top3"]] == [classes[j] for j in order[:3]]
        checked.append({"id": cid, "pts_exact": True, "input_exact": True, "ranking_recomputed": True})
    # Recalculate totals independently, without trusting RESULT's counters.
    for group, old in (("new34_ai_candidate", False), ("previous2_user_confirmed", True)):
        for word in ("먹다", "공부", "ALL"):
            rr = [r for r in result["rows"] if r["previous_diagnostic_exposure"] == old
                  and (word == "ALL" or r["candidate_label"] == word)]
            expected = dict(total=len(rr), scored=sum(r["status"] == "OK" for r in rr),
                            source_label_agreement=sum(r["agreement"] for r in rr))
            assert result["summaries"][group][word] == expected
    torch.set_num_threads(2)
    cache = _load(probe.MODEL_KEY)
    assert cache["status"] == "READY" and cache["classes"] == classes
    captured = []
    hook = cache["model"].register_forward_hook(lambda module, args, output: captured.append(output.detach().numpy()[0].copy()))
    service = []
    replay_ids = [f"KSL77_{prefix}_{label}" for label in ("16", "48") for prefix in ("00", "01")]
    try:
        for cid in replay_ids:
            if cid not in score_by_id:
                service.append({"id": cid, "status": "NOT_SCORED"})
                continue
            response = recognise(Observation.load(entries[cid]["observation"]), model_key=probe.MODEL_KEY)
            assert response["model_status"] == "READY"
            delta = float(np.max(np.abs(captured[-1] - score_by_id[cid])))
            assert delta <= plan["audit_service_logit_atol"]
            assert response["scopes"]["full"]["top1"] == results[cid]["prediction"]
            service.append({"id": cid, "max_abs_logit_difference": delta, "top1_equal": True})
    finally:
        hook.remove()
    probe.frozen()
    payload = torch.load(probe.read(probe.HERE / "LOCAL_PREFLIGHT.json")["baseline"]["path"],
                         map_location="cpu", weights_only=True)
    assert all(torch.equal(v, payload["state_dict"][k]) for k, v in cache["model"].state_dict().items())
    audit = dict(status="PASS_LOCAL_PIPELINE_AND_ARITHMETIC", at_kst=probe.now(), author="Codex",
                 reviewer_independence="Not a separate author or linguistic expert; separate calculation and service route",
                 checked=checked, service_replays=service, all_36_retained=True,
                 original_pts_exact_count=len(checked), source_and_model_unchanged=True,
                 pipeline_rebuild_exact_count=len(checked), training=0, new_extraction=0,
                 limits=["No expert semantic labels for34", "Signer separation unverified", "No accuracy-improvement claim"])
    probe.save_new(probe.HERE / "AUDIT.json", audit)
    print(json.dumps({k: audit[k] for k in ("status", "original_pts_exact_count", "pipeline_rebuild_exact_count", "service_replays")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
