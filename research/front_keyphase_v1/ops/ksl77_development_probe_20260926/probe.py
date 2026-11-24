"""Frozen, local-only development probe; no training, camera, or network code.

The user authorized Codex to author this analysis after the Claude CLI hit its
usage limit. Existing production extraction and model input APIs are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
WEB = HERE.parents[1]
ROOT = WEB.parents[1]
PILOT = ROOT / "research/sign_model_pilot_v1"
OUT = WEB / "runs/ksl77_development_probe_v1"
sys.path[:0] = [str(WEB / "src"), str(PILOT / "src")]
MODEL_KEY = "tcn_control_83212868"
MODEL_SHA = "83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def save_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def frozen():
    plan = read(HERE / "FROZEN_PLAN.json")
    for p, digest in plan["pins"].items():
        assert sha(p) == digest, f"Frozen input changed: {p}"
    for r in plan["rows"]:
        assert sha(r["path"]) == r["source_sha256"], r["member"]
    assert {k: importlib.metadata.version(k) for k in plan["versions"]} == plan["versions"]
    return plan


def prepare():
    from wis_lab.service.neural import MODELS
    preflight = read(HERE / "LOCAL_PREFLIGHT.json")
    rows = preflight["manifest"]
    assert len(rows) == 36 and len({r["source_sha256"] for r in rows}) == 36
    assert Counter(r["candidate_label"] for r in rows) == {"공부": 19, "먹다": 17}
    assert Counter(r["meaning_evidence"] for r in rows) == {"USER_CONFIRMED": 2, "AI_MATCH_CANDIDATE": 34}
    for r in rows:
        r["id"] = "KSL77_" + Path(r["member"]).stem
        r["cache_observation"] = (
            str(WEB / "runs/ksl77_input_prep_v1" / (r["id"] + ".npz"))
            if r["previous_diagnostic_exposure"] else None)
        assert sha(r["path"]) == r["source_sha256"]
    spec = MODELS[MODEL_KEY]
    assert spec["sha256"] == sha(spec["path"]) == MODEL_SHA
    paths = [Path(__file__), HERE / "LOCAL_PREFLIGHT.json", HERE / "ROLE_AUTHORIZATION.md",
             WEB / "config/contract.json", Path(spec["path"]),
             WEB / "ops/ksl77_form_review_20260926/FORM_REVIEW.json",
             WEB / "ops/ksl77_confirmed_pair_20260922/SEMANTIC_CONFIRMATION.json",
             WEB / "runs/union93_candidate_v1/inputs.json"]
    paths += [WEB / "src/wis_lab" / p for p in
              ("capture/landmarker.py", "capture/video_source.py", "core/observation.py", "service/neural.py")]
    paths += [PILOT / "src/pilot" / p for p in ("data.py", "web_input.py", "model_masked_v1.py")]
    paths += [Path(p) for p in preflight["assets"]]
    for r in rows:
        if r["cache_observation"]:
            p = Path(r["cache_observation"])
            paths.extend([p, p.with_suffix(".sidecar.json"), p.with_suffix(".tcn_input.npz")])
    plan = {
        "schema": "ksl77_development_probe_plan_v1", "frozen_at_kst": now(),
        "scope": "Existing 36 public clips, development diagnostic only, not training or person-separated evaluation",
        "model_key": MODEL_KEY, "model_sha256": MODEL_SHA,
        "extractor_id": preflight["extractor_id"], "versions": preflight["versions"],
        "pins": {str(p): sha(p) for p in paths}, "rows": rows,
        "ranking": "Full 93 classes only; model receives coordinates, masks and timing, never expected labels",
        "input_policy": "Existing production _channels; first-to-last usable hand span, no manual action boundaries",
        "source_label_agreement": "Report new AI-candidate 34 separately from previously exposed user-confirmed 2",
        "failure_policy": "Retain every clip, failures in denominator, no score-based selection or retry",
        "temporal_policy": "New Landmarker per new source clip; actual source PTS, no shared tracking state",
        "reuse": "Two old observations checked against current extraction identity and service input; re-extract zero",
        "lineage_limits": preflight["lineage_check"],
        "audit_service_logit_atol": 1e-5,
        "forbidden": ["training", "parameter search", "new downloads", "camera access", "user recordings",
                      "heldout arrays", "production edits", "candidate hand assignment", "synthetic filling"],
    }
    save_new(HERE / "FROZEN_PLAN.json", plan)
    frozen()
    print("FROZEN", len(rows), sha(HERE / "FROZEN_PLAN.json"), flush=True)


def check_obs(obs, path, row, plan):
    obs.validate()
    assert obs.n_frames > 0 and obs.timestamp_us[0] == 0
    assert np.all(np.diff(obs.timestamp_us) > 0)
    assert np.isfinite(obs.xy_px[obs.usable]).all()
    assert np.isnan(obs.conf[:, 33:]).all()
    assert obs.sidecar["source_sha256"] == row["source_sha256"]
    assert obs.sidecar["observation_sha256"] == sha(path)
    assert obs.sidecar["extractor_id"] == plan["extractor_id"]


def extract_all():
    from wis_lab.capture.landmarker import Landmarker
    from wis_lab.capture.video_source import extract
    from wis_lab.core.observation import Observation
    plan = frozen()
    assert not (HERE / "EXTRACTION.json").exists(), "Extraction already completed; reuse it"
    contract = read(WEB / "config/contract.json")
    entries = []
    for row in plan["rows"]:
        dest = OUT / row["id"]
        receipt = dest / "EXTRACTION.json"
        if receipt.exists():
            item = read(receipt)
            assert item["plan_sha256"] == sha(HERE / "FROZEN_PLAN.json")
            if item["status"] == "OK":
                check_obs(Observation.load(item["observation"]), item["observation"], row, plan)
            entries.append(item)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        item = {"id": row["id"], "plan_sha256": sha(HERE / "FROZEN_PLAN.json"), "started_at_kst": now(),
                "reused": row["cache_observation"] is not None}
        try:
            if item["reused"]:
                path = Path(row["cache_observation"])
                obs = Observation.load(str(path))
            else:
                path = dest / "observation.npz"
                assert not path.exists(), "Unfinished output needs explicit review; do not overwrite"
                with Landmarker(contract) as lm:
                    obs = extract(row["path"], lm, row["id"])
                obs.sidecar.update(diagnostic_only=True, training_use=False,
                                   manifest_sha256=sha(HERE / "FROZEN_PLAN.json"))
                obs.save(str(path))
            check_obs(obs, path, row, plan)
            item.update(status="OK", observation=str(path), observation_sha256=sha(path),
                        sidecar_sha256=sha(path.with_suffix(".sidecar.json")),
                        quality=obs.quality_summary(), dropped_non_monotonic_pts=obs.sidecar["dropped_non_monotonic_pts"])
        except Exception:
            item.update(status="FAILED", error=traceback.format_exc())
        item["finished_at_kst"] = now()
        save_new(receipt, item)
        entries.append(item)
        print("EXTRACT", row["id"], item["status"], "REUSED" if item["reused"] else "NEW", flush=True)
    frozen()
    save_new(HERE / "EXTRACTION.json", {"finished_at_kst": now(), "rows": entries,
             "new_attempts": sum(not r["reused"] for r in entries),
             "reused": sum(r["reused"] for r in entries), "failed": sum(r["status"] != "OK" for r in entries)})


def evaluate():
    import torch
    from pilot.data import collate
    from pilot.web_input import active_span, to_native_like
    from wis_lab.core.observation import Observation
    from wis_lab.service.neural import _channels, _load
    plan = frozen()
    assert not (HERE / "RESULT.json").exists(), "Completed evaluation must not be rerun"
    assert not (OUT / "logits.npz").exists(), "Review existing scores rather than overwrite"
    extraction = read(HERE / "EXTRACTION.json")
    by_id = {r["id"]: r for r in extraction["rows"]}
    torch.set_num_threads(2)
    cache = _load(MODEL_KEY)
    assert cache["status"] == "READY", cache["status"]
    assert cache["checkpoint_sha256"] == MODEL_SHA
    model, classes = cache["model"], cache["classes"]
    assert len(classes) == len(set(classes)) == 93
    model.eval().requires_grad_(False)
    rows, all_logits, ids = [], [], []
    for source in plan["rows"]:
        entry = by_id[source["id"]]
        row = {k: source[k] for k in ("id", "member", "candidate_label", "meaning_evidence", "previous_diagnostic_exposure")}
        if entry["status"] != "OK":
            row.update(status="EXTRACTION_FAILED", agreement=False)
            rows.append(row)
            continue
        try:
            obs = Observation.load(entry["observation"])
            check_obs(obs, entry["observation"], source, plan)
            channels, why, support = _channels(obs)
            row.update(support=support, quality=entry["quality"])
            if channels is None:
                row.update(status="INPUT_REJECTED", reason=why, agreement=False)
                rows.append(row)
                continue
            x = channels["x"]
            assert x.ndim == 2 and x.shape[1] == 280 and np.isfinite(x).all()
            # This label is collate's unused dummy target, constant for every clip.
            bx, mask, _unused_y = collate([dict(x=x, frames=len(x), label=0)])
            with torch.inference_mode():
                logit = model(torch.from_numpy(bx), torch.from_numpy(mask)).numpy()[0]
            assert logit.shape == (93,) and np.isfinite(logit).all()
            nat, present = to_native_like(obs.xy_px, obs.usable)
            lo, hi = active_span(present)
            input_path = OUT / source["id"] / "model_input.npz"
            with input_path.open("xb") as f:
                np.savez_compressed(f, x=x, body_valid=channels["body_valid"], hand_valid=channels["hand_valid"],
                                    timestamp_us=obs.timestamp_us[lo:hi+1], frame_id=obs.frame_id[lo:hi+1])
            # Ground-truth/candidate text first used AFTER the model has answered.
            order = np.argsort(-logit, kind="stable")
            target = classes.index(source["candidate_label"])
            row.update(status="OK", prediction=classes[int(order[0])], agreement=bool(order[0] == target),
                       target_rank=int(np.flatnonzero(order == target)[0]) + 1,
                       top3=[{"label": classes[int(j)], "logit": float(logit[j])} for j in order[:3]],
                       top1_top2_margin=float(logit[order[0]] - logit[order[1]]),
                       target_minus_best_other=float(logit[target] - max(v for j, v in enumerate(logit) if j != target)),
                       input_frames=len(x), input_sha256=sha(input_path), observation_sha256=entry["observation_sha256"])
            ids.append(source["id"])
            all_logits.append(logit)
        except Exception:
            row.update(status="EVALUATION_FAILED", agreement=False, error=traceback.format_exc())
        rows.append(row)
    summaries = {}
    for group, exposed in (("new34_ai_candidate", False), ("previous2_user_confirmed", True)):
        rr = [r for r in rows if r["previous_diagnostic_exposure"] == exposed]
        summaries[group] = {}
        for word in ("먹다", "공부", "ALL"):
            sub = rr if word == "ALL" else [r for r in rr if r["candidate_label"] == word]
            summaries[group][word] = {"total": len(sub), "scored": sum(r["status"] == "OK" for r in sub),
                                       "source_label_agreement": sum(r["agreement"] for r in sub)}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "logits.npz").open("xb") as f:
        np.savez_compressed(f, ids=np.array(ids), classes=np.array(classes),
                            logits=np.asarray(all_logits, dtype=np.float32).reshape(-1, 93))
    frozen()
    result = {"status": "MEASURED_PENDING_CODEX_AUDIT", "at_kst": now(), "model_sha256": MODEL_SHA,
              "plan_sha256": sha(HERE / "FROZEN_PLAN.json"), "logits_sha256": sha(OUT / "logits.npz"),
              "rows": rows, "summaries": summaries, "new_extractions": extraction["new_attempts"],
              "reused_observations": extraction["reused"], "prediction_rows": len(ids),
              "training": 0, "camera_access": 0, "downloads": 0, "user_data": 0,
              "claims": "Source-label agreement only for AI candidates; not signer-independent accuracy or improvement",
              "source_and_production_pins_unchanged": True}
    save_new(HERE / "RESULT.json", result)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "extract", "evaluate"])
    {"prepare": prepare, "extract": extract_all, "evaluate": evaluate}[parser.parse_args().phase]()
