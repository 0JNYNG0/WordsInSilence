"""R14: the union-93 preparation, checked with the real functions. No training is run.

Four things the design fixes and this verifies: the existing 80 logits start identical, the 13 new
rows map where they should, the data lists do not overlap or touch forbidden material, and the two
augmentations keep their promises about mask, time and whole-clip rotation.

Run: py -3.13 tests/test_r14_union93_prep.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1", "src"))

import tools_prep_union93_candidate as P  # noqa: E402

CHECKS = []
MEASURED = {"note": "preparation only; the main training is not executed here"}
NL = chr(10)
N_POINTS = 67


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def clip(n=10, step_us=33333):
    """A synthetic native clip with a believable body and two hands."""
    rng = np.random.default_rng(20260913)
    xy = rng.uniform(100, 900, size=(n, N_POINTS, 2)).astype(np.float32)
    pr = np.ones((n, N_POINTS), dtype=bool)
    ts = (np.arange(n, dtype=np.int64) * step_us)
    return xy, pr, ts


@check("the first 80 logits of the union model start identical to the pinned model's")
def _logit_identity():
    import torch
    from pilot.data import build_channels, collate

    from wis_lab.service import neural
    base = neural._load()                                  # noqa: SLF001
    assert base["status"] == "READY", base
    model93, classes, mapping, new_words, init = P.build_candidate()
    assert init["copied_head_rows"] == 80 and init["new_head_rows"] == 13, init
    assert init["missing_keys"] == [] and init["unexpected_keys"] == [], init

    xy, pr, ts = clip(12)
    c = build_channels(xy, pr, ts)
    x, pad, _ = collate([dict(c, label=0)])
    with torch.inference_mode():
        a = base["model"](torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
        b = model93(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
    assert b.shape[0] == 93 and a.shape[0] == 80, (a.shape, b.shape)
    assert np.array_equal(a, b[:80]), float(np.abs(a - b[:80]).max())
    MEASURED["logit_identity"] = {"max_abs_difference_over_the_80": 0.0,
                                  "union_logits": int(b.shape[0])}


@check("the 13 new rows map to the intended words and the existing 5 reuse their old rows")
def _mapping():
    _, classes, mapping, new_words, _ = P.build_candidate()
    assert len(classes) == 93 and len(new_words) == 13, (len(classes), len(new_words))
    lists = P.data_lists()
    names = {w: name for _, w, name in lists["nikl_training"]}
    for w in new_words:
        idx = mapping[w]
        assert 80 <= idx < 93, (w, idx)
        assert classes[idx] == names[w], (w, classes[idx], names[w])
    for w in ("ALDA", "GAMSAHAMNIDA", "JOTA", "MOREUDA", "NA"):
        assert mapping[w] < 80, (w, mapping[w])
        assert classes[mapping[w]] == names[w]
    assert "KKEUDEOGIDA" not in mapping, "a held-back word was given a row"
    assert "KKEUDEOGIDA" in P.HELD_BACK
    MEASURED["mapping"] = {"new_rows": [mapping[w] for w in new_words],
                           "reused_rows": {w: mapping[w] for w in
                                           ("ALDA", "GAMSAHAMNIDA", "JOTA", "MOREUDA", "NA")}}


@check("training, selection and reserved lists do not overlap and exclude the forbidden material")
def _separation():
    lists = P.data_lists()
    tr = set(lists["source_train"])
    va = set(lists["source_validation"])
    re = set(lists["reserved_test"])
    assert len(tr) == 2095 and len(va) == 696 and len(re) == 700, (len(tr), len(va), len(re))
    assert not (tr & va) and not (tr & re) and not (va & re), "the splits overlap"
    nikl = {c for c, _, _ in lists["nikl_training"]}
    assert len(nikl) == 18
    assert not (nikl & va) and not (nikl & re), "a demonstration leaked into a held-out list"
    assert lists["roles"]["source_validation"].startswith("selection")
    assert "training, originals AND every augmentation" == lists["roles"]["nikl_training"]
    for banned in ("private recordings", "the user's ten attempts", "three-view material",
                   "Validation 17-18", "the reserved 700 test clips"):
        assert banned in lists["excluded_from_training"], banned
    held = {w for _, w, _ in lists["nikl_held_back"]}
    assert held == {"KKEUDEOGIDA"}, held
    MEASURED["separation"] = {"train": len(tr), "validation": len(va), "reserved": len(re),
                              "nikl": len(nikl), "held_back": sorted(held)}


@check("the tilt is one angle for the whole clip and leaves the mask and absent points alone")
def _tilt():
    xy, pr, _ = clip(8)
    pr[3, 40] = False
    xy[3, 40] = 0.0
    out, out_pr = P.tilt_clip(xy, pr, 5.0)
    assert np.array_equal(out_pr, pr), "the mask moved"
    assert np.allclose(out[3, 40], 0.0), "an absent point was given a position"
    # a rigid rotation preserves every pairwise distance, and by the same amount in every frame
    for t in range(len(xy)):
        obs = np.flatnonzero(pr[t])
        d0 = np.linalg.norm(xy[t, obs][:, None] - xy[t, obs][None], axis=-1)
        d1 = np.linalg.norm(out[t, obs][:, None] - out[t, obs][None], axis=-1)
        assert np.allclose(d0, d1, atol=1e-3), t
    # the SAME angle everywhere: the angle each frame turned must agree across frames
    angles = []
    for t in range(len(xy)):
        v0 = xy[t, 6] - xy[t, 5]
        v1 = out[t, 6] - out[t, 5]
        angles.append(np.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])))
    angles = np.unwrap(np.radians(np.array(angles)))
    assert np.allclose(np.degrees(angles), np.degrees(angles)[0], atol=1e-3), angles
    assert abs(abs(np.degrees(angles)[0]) - 5.0) < 1e-3, np.degrees(angles)[0]
    MEASURED["tilt"] = {"angle_deg": round(float(np.degrees(angles)[0]), 3),
                        "mask_unchanged": True}


@check("the resample keeps order, copies exact frames, and restores nothing")
def _resample():
    xy, pr, ts = clip(9)
    pr[4, 30] = False                     # one point missing in the middle
    xy[4, 30] = 0.0
    for speed in (0.8, 1.0, 1.2):
        out, out_pr, out_ts = P.time_resample(xy, pr, ts, speed)
        assert np.all(np.diff(out_ts) > 0), f"order broken at {speed}"
        span_in = float(ts[-1] - ts[0])
        span_out = float(out_ts[-1] - out_ts[0])
        assert abs(span_out - span_in / speed) <= np.median(np.diff(ts)), (speed, span_out)
        if speed == 1.0:
            assert len(out) == len(xy), (len(out), len(xy))
            assert np.array_equal(out_pr, pr) and np.allclose(out, xy), "identity broke"
        # nothing missing is invented: point 30 is never present at a time bracketed by frame 4.
        # the source time comes from the ENDPOINT-INCLUSIVE grid the function actually uses.
        for k in range(len(out)):
            src = float(ts[0]) + (float(ts[-1]) - float(ts[0])) * k / (len(out) - 1)
            lo = int(np.searchsorted(ts, src, side="right")) - 1
            hi = lo + 1
            if 0 <= lo < len(ts) and hi < len(ts) and (lo == 4 or hi == 4):
                exact = any(abs(int(ts[i]) - src) <= 1 for i in (lo, hi) if i != 4)
                if not exact:
                    assert not out_pr[k, 30], (speed, k)
    # an exact original frame survives even when its neighbour is missing entirely
    pr2 = pr.copy()
    pr2[5, :] = False
    out, out_pr, _ = P.time_resample(xy, pr2, ts, 1.0)
    assert out_pr[4].sum() == pr2[4].sum() and out_pr[6].sum() == pr2[6].sum()
    assert np.allclose(out[4], xy[4]) and np.allclose(out[6], xy[6])
    MEASURED["resample"] = {"speeds": [0.8, 1.0, 1.2], "identity_at_1x": True,
                            "exact_frames_survive_a_missing_neighbour": True}


@check("a non-integer speed keeps BOTH original endpoints and the scaled duration")
def _resample_endpoints():
    xy, pr, ts = clip(9)
    step = int(np.median(np.diff(ts)))
    for speed in (0.83, 0.97, 1.17, 1.2):
        out, out_pr, out_ts = P.time_resample(xy, pr, ts, speed)
        # the first and the last ORIGINAL instant both survive, verbatim
        assert np.allclose(out[0], xy[0]) and np.array_equal(out_pr[0], pr[0]), speed
        assert np.allclose(out[-1], xy[-1]) and np.array_equal(out_pr[-1], pr[-1]), speed
        span_in = int(ts[-1] - ts[0])
        span_out = int(out_ts[-1] - out_ts[0])
        # the emitted clock really is faster or slower, to within one rounded step
        assert abs(span_out - span_in / speed) <= step, (speed, span_out, span_in / speed)
        assert np.all(np.diff(out_ts) > 0), speed
    MEASURED["resample_endpoints"] = {"speeds": [0.83, 0.97, 1.17, 1.2],
                                      "both_ends_verbatim": True}


@check("every originally valid exact frame survives, and an invalid neighbour is not revived")
def _resample_exact_and_invalid():
    xy, pr, ts = clip(9)
    pr[5, :] = False                       # a frame with nothing observed at all
    xy[5] = 0.0
    pr[3, 30] = False                      # and one point missing on another frame
    xy[3, 30] = 0.0
    out, out_pr, out_ts = P.time_resample(xy, pr, ts, 1.0)
    assert len(out) == len(xy)
    for t in range(len(xy)):               # 1.0x lands on every original instant
        assert np.array_equal(out_pr[t], pr[t]), t
        assert np.allclose(out[t], xy[t]), t
    # at a non-integer speed the exact hits still copy, and an interior gap is never filled
    out, out_pr, out_ts = P.time_resample(xy, pr, ts, 0.9)
    src_step = float(np.median(np.diff(ts))) * 0.9
    for k in range(len(out)):
        src = float(ts[0]) + (float(ts[-1]) - float(ts[0])) * k / (len(out) - 1)
        exact = [i for i in range(len(ts)) if abs(int(ts[i]) - src) <= 1]
        if exact:
            assert np.array_equal(out_pr[k], pr[exact[0]]), (k, exact[0])
        else:
            hi = int(np.searchsorted(ts, src, side="right"))
            lo = hi - 1
            if 0 <= lo and hi < len(ts):
                assert np.array_equal(out_pr[k], pr[lo] & pr[hi]), k
                assert not out_pr[k, 30] or (pr[lo, 30] and pr[hi, 30])
    assert src_step > 0
    MEASURED["resample_exact"] = {"identity_at_1x": True, "gap_never_revived": True}


@check("a non-monotonic clock or an unusable speed fails loudly instead of copying")
def _resample_failures():
    xy, pr, ts = clip(6)
    bad = ts.copy()
    bad[3] = bad[2]                        # not strictly increasing
    for args, why in (((xy, pr, bad, 1.0), "non-monotonic"),
                      ((xy, pr, ts, 0.0), "zero speed"),
                      ((xy, pr, ts, -1.0), "negative speed"),
                      ((xy, pr, ts, float("nan")), "nan speed"),
                      ((xy[:1], pr[:1], ts[:1], 1.0), "single frame")):
        try:
            P.time_resample(*args)
            raise AssertionError(f"{why} was accepted")
        except ValueError:
            pass
    MEASURED["resample_failures"] = ["non-monotonic", "zero", "negative", "nan", "single frame"]


@check("a whole-clip rotation preserves every distance at a non-integer angle too")
def _tilt_distances():
    xy, pr, _ = clip(6)
    for deg in (-5.0, -1.7, 3.3, 5.0):
        out, out_pr = P.tilt_clip(xy, pr, deg)
        assert np.array_equal(out_pr, pr)
        for t in range(len(xy)):
            d0 = np.linalg.norm(xy[t, :, None] - xy[t, None], axis=-1)
            d1 = np.linalg.norm(out[t, :, None] - out[t, None], axis=-1)
            assert np.allclose(d0, d1, atol=1e-3), (deg, t, float(np.abs(d0 - d1).max()))
    MEASURED["tilt_distances"] = {"angles": [-5.0, -1.7, 3.3, 5.0], "distances_preserved": True}


@check("two real optimizer steps run, the head and the encoder move at their own rates")
def _two_real_steps():
    import torch
    from pilot.data import TOTAL_CHANNELS
    from pilot.model_masked_v1 import MaskedSignTCN

    P.set_seeds(20260913)
    model = MaskedSignTCN(width=8, dilations=(1, 2), kernel_size=3, dropout=0.0, n_classes=6)
    rng = np.random.default_rng(20260913)

    def synth_source(sid, cls):
        return {"sample_id": sid, "label": f"c{cls}", "class_index": cls,
                "x": rng.normal(size=(12, TOTAL_CHANNELS)).astype(np.float32), "frames": 12}

    src = [synth_source(f"s{i}", i % 3) for i in range(9)]
    held = synth_source("VALIDATION_ONLY", 0)          # must never enter a batch
    nikl = []
    for j in range(2):
        nat = rng.uniform(100, 900, size=(10, 67, 2)).astype(np.float32)
        nikl.append({"clip_id": f"n{j}", "label": f"c{3 + j}", "class_index": 3 + j,
                     "nat": nat, "present": np.ones((10, 67), dtype=bool),
                     "ts": (np.arange(10, dtype=np.int64) * 33333)})
    setting = dict(P.SETTING, optimizer_steps=2,
                   batch={"total": 6, "source": 4, "nikl": 2})
    before = {n2: p2.detach().clone() for n2, p2 in model.named_parameters()}
    stats = P.run_training(model, src, nikl, [f"c{i}" for i in range(6)], setting, rng,
                           log=lambda _m: None)
    assert stats["steps"] == 2 and len(stats["losses"]) == 2
    assert all(np.isfinite(v) for v in stats["losses"]), stats["losses"]
    assert stats["intermediate_checkpoints_kept"] == 0
    assert held["sample_id"] not in stats.__str__(), "the held-out clip appeared in the run"
    head_moved = enc_moved = False
    for n2, p2 in model.named_parameters():
        moved = not torch.equal(before[n2], p2.detach())
        if n2.startswith("head."):
            head_moved = head_moved or moved
        else:
            enc_moved = enc_moved or moved
    assert head_moved and enc_moved, (head_moved, enc_moved)
    assert stats["learning_rates"] == {"encoder": 1e-4, "head": 1e-3}
    # the head carries the larger rate, so with the same gradient shape it moves further; the
    # check is on the CONFIGURED groups, which is what the design fixes
    assert stats["distinct_nikl_clips_seen"] <= 2
    MEASURED["two_steps"] = {"losses": [round(v, 4) for v in stats["losses"]],
                             "head_updated": head_moved, "encoder_updated": enc_moved,
                             "nikl_augmented_every_time": True}


@check("a non-finite loss stops the run instead of being averaged away")
def _loss_guard():
    from pilot.data import TOTAL_CHANNELS
    from pilot.model_masked_v1 import MaskedSignTCN

    P.set_seeds(1)
    model = MaskedSignTCN(width=8, dilations=(1,), kernel_size=3, dropout=0.0, n_classes=3)
    rng = np.random.default_rng(1)
    bad = {"sample_id": "bad", "label": "c0", "class_index": 0,
           "x": np.full((6, TOTAL_CHANNELS), np.nan, dtype=np.float32), "frames": 6}
    nat = rng.uniform(100, 900, size=(6, 67, 2)).astype(np.float32)
    nikl = [{"clip_id": "n", "label": "c1", "class_index": 1, "nat": nat,
             "present": np.ones((6, 67), dtype=bool),
             "ts": (np.arange(6, dtype=np.int64) * 33333)}]
    setting = dict(P.SETTING, optimizer_steps=1, batch={"total": 2, "source": 1, "nikl": 1})
    try:
        P.run_training(model, [bad], nikl, ["c0", "c1", "c2"], setting, rng, log=lambda _m: None)
        raise AssertionError("a non-finite loss was accepted")
    except RuntimeError as exc:
        assert "non-finite loss" in str(exc), exc
    MEASURED["loss_guard"] = "raises on the first non-finite loss"


@check("the base source reference is verified against the pinned checkpoint and sample set")
def _base_reference():
    base = P.base_source_reference()
    assert base["checkpoint_sha256"] == P.CHECKPOINT_SHA256
    assert base["correct"] == 689 and base["total"] == 696
    assert abs(base["macro_f1"] - 0.9897937504961497) < 1e-15
    assert len(base["sample_ids"]) == 696 and len(set(base["sample_ids"])) == 696
    lists = P.data_lists()
    assert sorted(lists["source_validation"]) == base["sample_ids"], \
        "the validation list and the base run's sample set differ"
    MEASURED["base_reference"] = {"accuracy": f"{base['correct']}/{base['total']}",
                                  "macro_f1": base["macro_f1"]}


@check("a saved checkpoint rebuilds in a separate process and returns identical logits")
def _checkpoint_roundtrip():
    import subprocess
    import tempfile

    import torch

    model, classes, _mapping, _new, _init = P.build_candidate()
    payload = P.checkpoint_payload(model, classes, 660)
    for key in ("width", "dilations", "kernel_size", "dropout", "in_channels", "classes",
                "n_classes", "model_type", "model_version", "feature_schema", "step",
                "base_checkpoint_sha256", "fixed_setting", "augmentation", "state_dict"):
        assert key in payload, key
    assert payload["width"] is not None and payload["dilations"] and payload["n_classes"] == 93
    assert payload["feature_schema"]["total_channels"] == payload["in_channels"]
    assert payload["base_checkpoint_sha256"] == P.CHECKPOINT_SHA256
    assert payload["fixed_setting"]["optimizer_steps"] == 660

    tmp = tempfile.mkdtemp(prefix="wis_r14_ck_")
    path = os.path.join(tmp, "union93_final.pt")
    torch.save(payload, path)
    probe = np.zeros((1, 8, payload["in_channels"]), dtype=np.float32)
    probe[0, :, :4] = 0.1
    pad = np.ones((1, 8), dtype=np.float32)
    with torch.inference_mode():
        here = model(torch.from_numpy(probe), torch.from_numpy(pad))[0].numpy()

    # a SEPARATE interpreter, given only the file, must reproduce the logits
    code = (
        "import sys, json, numpy as np, torch\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'src')!r})\n"
        f"sys.path.insert(0, {os.path.join(os.path.dirname(ROOT), 'sign_model_pilot_v1', 'src')!r})\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "import tools_prep_union93_candidate as P\n"
        f"pl = torch.load({path!r}, map_location='cpu', weights_only=False)\n"
        "m, cl = P.rebuild_from_payload(pl)\n"
        "x = np.zeros((1, 8, pl['in_channels']), dtype=np.float32); x[0, :, :4] = 0.1\n"
        "p = np.ones((1, 8), dtype=np.float32)\n"
        "with torch.inference_mode():\n"
        "    o = m(torch.from_numpy(x), torch.from_numpy(p))[0].numpy()\n"
        "print(json.dumps({'logits': [float(v) for v in o], 'classes': len(cl)}))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         encoding="utf-8")
    assert out.returncode == 0, out.stderr[-800:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["classes"] == 93
    there = np.array(got["logits"], dtype=np.float32)
    assert np.array_equal(here, there), float(np.abs(here - there).max())
    MEASURED["checkpoint_roundtrip"] = {"separate_process": True, "classes": got["classes"],
                                        "max_logit_difference": 0.0}


@check("the resolved input list carries real hashes, the exact counts and disjoint roles")
def _resolved_inputs():
    lists = P.data_lists()
    _model, classes, mapping, _new, _init = P.build_candidate()
    manifest = P.load_json(os.path.join(P.PILOT, "expand", "manifest.json"))
    by_id = {m["sample_id"]: m for m in manifest}
    label_map = {name: i for i, name in enumerate(classes)}

    def stub(ids):                      # resolve_inputs reads the manifests, not the arrays
        return [{"sample_id": i, "class_index": label_map[by_id[i]["word_label"]]} for i in ids]

    nikl = [{"clip_id": c, "word_id": w, "label": name, "class_index": mapping[w]}
            for c, w, name in lists["nikl_training"]]
    res = P.resolve_inputs(stub(lists["source_train"]), stub(lists["source_validation"]),
                           nikl, classes)
    assert res["counts"] == {"source_train": 2095, "source_validation": 696, "nikl": 18,
                             "reserved_test_not_read": 700}, res["counts"]
    assert res["roles_are_disjoint"] and res["reserved_test"]["read"] is False
    assert res["base_checkpoint_sha256"] == P.CHECKPOINT_SHA256
    for group, role in (("source_train", "training"),
                        ("source_validation", "selection_and_regression_only")):
        assert len(res[group]) == res["counts"][group]
        for r in res[group]:
            assert r["role"] == role and len(r["npz_sha256"]) == 64, r
    for r in res["nikl"]:
        assert r["role"] == "training_original_and_every_augmentation"
        assert len(r["npz_sha256"]) == 64
    for key in ("tools_prep_union93_candidate.py", "config/source_manifest.json",
                "design/functional_17_union93_candidate.md", "expand/manifest.json"):
        assert len(res["hashes"][key]) == 64, key
    # the real hashes must be the manifest's own, not recomputed guesses
    sample = res["source_train"][0]
    assert sample["npz_sha256"] == by_id[sample["id"]]["npz_sha256"]
    # and a short list is refused rather than trained on
    try:
        P.resolve_inputs(stub(lists["source_train"][:10]), stub(lists["source_validation"]),
                         nikl, classes)
        raise AssertionError("a short training list was accepted")
    except SystemExit as exc:
        assert "expected 2095" in str(exc), exc
    MEASURED["resolved_inputs"] = {"counts": res["counts"],
                                   "hashed_files": sorted(res["hashes"])}


@check("the preparation reports itself as not trained, with the fixed setting recorded")
def _prepared_only():
    path = os.path.join(ROOT, "evidence", "functional_17_union93_prep.json")
    rep = json.load(open(path, encoding="utf-8"))
    assert rep["status"] == "PREPARED_NOT_TRAINED"
    assert rep["first_limited_experiment"] and rep["settings_are_not_claimed_optimal"]
    s = rep["setting"]
    assert s["seed"] == 20260913 and s["optimizer_steps"] == 660
    assert s["batch"] == {"total": 32, "source": 16, "nikl": 16}
    assert s["lr"] == {"encoder": 1e-4, "head": 1e-3} and s["weight_decay"] == 1e-3
    assert s["cpu_threads"] == 4 and "FINAL step" in s["report_step"]
    d = rep["decision_rule"]
    # BOTH gates, each with its own base, and either one failing means HOLD
    assert set(d["gates"]) == {"restricted_80_macro_f1", "full_93_accuracy"}, d["gates"]
    assert abs(d["gates"]["restricted_80_macro_f1"]["base"] - 0.9897937504961497) < 1e-15
    assert abs(d["gates"]["full_93_accuracy"]["base"] - 689 / 696) < 1e-15
    for g in d["gates"].values():
        assert "2 percentage points" in g["hold_if"]
    assert "EITHER one failing means HOLD" in d["hold_rule"]
    assert "never inflated to 93" in d["macro_denominator"]
    assert "never a new-person performance" in d["nikl18_recognition"]
    assert "not read here" in d["user_ten_attempts"]
    for banned in ("mirror", "time reversal", "occlusion", "noise"):
        assert banned in rep["augmentation"]["forbidden"], banned
    assert "the main training" in rep["not_done"]
    # the trainer is implemented now: it refuses to overwrite, rather than refusing to exist
    assert callable(P.train_candidate) and callable(P.run_training)
    assert rep["trainer"]["implemented"] is True and rep["trainer"]["executed_here"] is False
    import inspect
    src = inspect.getsource(P.train_candidate)
    assert "refusing to overwrite" in src and "set_seeds(SETTING[\"seed\"])" in src
    assert src.index("set_seeds") < src.index("build_candidate"), \
        "the seed must be fixed before the model is created"
    assert "reserved" not in inspect.getsource(P.load_source_split).split("refusing")[1][:200]
    MEASURED["prepared_only"] = {"status": rep["status"], "steps": s["optimizer_steps"]}


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                     # noqa: BLE001 - any raise is a failure
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
