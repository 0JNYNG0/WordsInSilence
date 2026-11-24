"""Freeze and prepare ALL36 completed source observations; no detector/classifier/training."""
import hashlib
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from input_candidate import U, O, ROOT, CONDITIONS, Observation, three_conditions, evidence

P = O / "word100_fullframe_hand_merge_20261020"


def read(p):
    return json.loads(Path(p).read_text("utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open("x", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2, allow_nan=False)


def clean_scales(x):
    return {k: float(v) if np.isfinite(v) else None for k, v in x.items()}


def freeze():
    old = read(P / "FROZEN.json")
    assert not (P / "RUNNING.json").exists()
    assert read(P / "FINAL_AUDIT.json")["training_runs"] == 0
    for path, expected in old["protected"].items():
        assert sha(path) == expected
    rows, pins = [], {}
    for r in old["rows"]:
        assert r["role"] == "train_candidate" and 1 <= r["signer_id"] <= 9
        row = {k: r[k] for k in ("clip_id", "label", "signer_id", "word_code", "box", "old_masked")}
        row["merged"] = str(P / "outputs" / r["clip_id"] / "masked.npz")
        row["merged_native"] = str(P / "outputs" / r["clip_id"] / "merged_native.npz")
        row["native"] = r["native"]
        row["record"] = str(P / "records" / (r["clip_id"] + ".json"))
        for k in ("old_masked", "merged", "merged_native", "native", "record"):
            pins[row[k]] = sha(row[k])
            if k != "record":
                sp = str(Path(row[k]).with_suffix(".sidecar.json"))
                pins[sp] = sha(sp)
        rows.append(row)
    for p in [U/"input_candidate.py", U/"prepare.py", U/"verify.py",
              P/"FINAL_AUDIT.json", P/"FROZEN.json", P/"INDEX.json", P/"VISUAL_REVIEW.json",
              P/"EXPLANATION_RECHECK.json", P/"FEATURE_EFFECT.json",
              O/"word100_fullframe_roi_pair_20261016/roi_adapter.py",
              O/"word100_image_arm_connectivity_20261018/CROSSING.json",
              O/"word100_image_arm_connectivity_20261018/ANNOTATED.json",
              ROOT/"research/sign_model_pilot_v1/src/pilot/data.py",
              ROOT/"research/sign_model_pilot_v1/src/pilot/web_input.py"]:
        pins[str(p)] = sha(p)
    (U/"before").mkdir()
    for p in [ROOT/"WordsInSilence_PROJECT_CONTEXT.md", ROOT/"NEW_CHAT_HANDOFF.md", O/"STATE.json"]:
        shutil.copyfile(p, U/"before"/p.name)
    write(U/"FROZEN.json", dict(
        at=datetime.now(timezone(timedelta(hours=9))).isoformat(), rows=rows,
        pins=pins, protected=old["protected"], conditions=CONDITIONS,
        authorization="User approved three-condition input validation only, location/shape evidence separation and normal/crossing preservation. Training after review.",
        scope="All36 completed full-original merge clips /23words /9 TRAIN signers, identical4641 frames. No new extraction. Not whole1199 training readiness.",
        gates=["Existing observed coordinates, identities, masks, timestamps preserved",
               "A and B existing features exactly equal; B/C differ only hand-local coordinates",
               "Independent direct reconstruction of all108 arrays, mask semantics and ROI exclusion",
               "Position and 15 finger-bend prerequisites separated; no shape correctness label",
               "Saved labeled normal/crossing controls retained; existing visual audit reused",
               "No normalization fallback for undefined baseline scales; no word-specific coefficients"],
        diagnostics="Rank whole/crop angle disagreements for review only. No threshold tuning, deletion or correction based on these rankings.",
        limits="Same-model scale agreement is not independent RGB anatomical verification. Baseline errors retained. No100-word/generalization/accuracy claim.",
        excluded=["training", "classifier inference", "DEV/heldout processing", "user/mentor data",
                  "detector rerun", "download", "camera", "Git", "production changes"],
    ))


def run():
    f = read(U/"FROZEN.json")
    for p, h in {**f["pins"], **f["protected"]}.items():
        assert sha(p) == h, p
    write(U/"RUNNING.json", dict(pid=os.getpid()))
    (U/"inputs").mkdir()
    (U/"evidence").mkdir()
    index, rows, top = [], [], []
    totals = Counter()
    diffs = []
    for r in f["rows"]:
        cid = r["clip_id"]
        a, c = Observation.load(r["old_masked"]), Observation.load(r["merged"])
        channels, old_scales, new_scales = three_conditions(a, c)
        folder = U/"inputs"/cid
        folder.mkdir()
        outputs = {}
        for name, arr in channels.items():
            path = folder/(name+".npz")
            assert all(v.dtype != object for v in arr.values())
            np.savez_compressed(path, **arr)
            outputs[str(path)] = sha(path)
        e = evidence(a, c, read(r["record"]))
        ep = U/"evidence"/(cid+".npz")
        np.savez_compressed(ep, **e)
        outputs[str(ep)] = sha(ep)
        added = (e["point_origin"] == 2).any(-1)
        av = e["two_scale_angle_difference_deg"]
        good = np.isfinite(av)
        diffs.extend(av[good].tolist())
        for t, s in np.argwhere(good.any(-1)):
            top.append(dict(clip_id=cid, label=r["label"], index=int(t),
                            frame_id=int(a.frame_id[t]), side=int(s),
                            max_angle_difference_deg=float(np.nanmax(av[t, s])),
                            comparable_angles=int(good[t, s].sum())))
        # Count physical observations separately from normalization-dependent input values.
        from input_candidate import angles
        old_angles, old_ok = angles(a.xy_px[:,33:].reshape(-1,2,21,2),
                                    a.usable[:,33:].reshape(-1,2,21))
        values = dict(
            frames=a.n_frames, added_hand_frames=int(added.sum()),
            added_points=int((e["point_origin"] == 2).sum()),
            old_palm_frames=int(a.usable[:,33:].reshape(-1,2,21)[...,[0,5,9,13,17]].all(-1).sum()),
            new_palm_frames=int(e["palm_available"].sum()),
            old_bend_angle_frames=int(old_ok.sum()), new_bend_angle_frames=int(e["bend_available"].sum()),
            added_without_complete_palm=int((added & ~e["palm_available"]).sum()),
            added_without_all15_angles=int((added & ~e["bend_available"].all(-1)).sum()),
            new_core_hand_frames=int((added & e["core"][:,None]).sum()),
            two_scale_comparable_angles=int(good.sum()),
        )
        totals.update(values)
        row = dict(clip_id=cid, label=r["label"], before_scales=clean_scales(old_scales),
                   after_scales=clean_scales(new_scales), **values)
        rows.append(row)
        index.append(dict(clip_id=cid, outputs=outputs))
    write(U/"INDEX.json", index)
    write(U/"INPUT_SUMMARY.json", dict(
        status="PREPARED_PENDING_INDEPENDENT_VERIFICATION", totals=dict(totals),
        clips=len(rows), input_arrays=len(rows)*3, channels=280, rows=rows,
        angle_disagreement_deg=dict(n=len(diffs), median=float(np.median(diffs)),
                                    p95=float(np.quantile(diffs,.95)), maximum=max(diffs)),
        interpretation="Finger angles computationally available, not confirmed physical finger accuracy.",
    ))
    # All ranked cases retained. Deterministic top12 are additional visual triage targets.
    top.sort(key=lambda x: (-x["max_angle_difference_deg"], x["clip_id"], x["frame_id"], x["side"]))
    write(U/"ANGLE_REVIEW_QUEUE.json", top)
    for p, h in {**f["pins"], **f["protected"]}.items():
        assert sha(p) == h, p
    (U/"RUNNING.json").unlink()
    print(json.dumps(dict(clips=len(rows), **totals), ensure_ascii=False))


if __name__ == "__main__":
    try:
        freeze() if sys.argv[1] == "freeze" else run()
    except BaseException:
        if not (U/"FIRST_ERROR.json").exists():
            write(U/"FIRST_ERROR.json", dict(error=traceback.format_exc()))
        raise
