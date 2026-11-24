"""Synthetic tests for mechanism_pair.py: temporary fake trees and a fake backend only.

No torch, model, media, camera or device work. Run:
    python test_mechanism.py --receipt PATH
The receipt is written (exclusively) only when every test passes with nothing skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mechanism_pair as mp  # noqa: E402

CLASSES = [f"W{i:02d}" for i in range(93)]
DEADLINE = "2026-10-04T09:00:00+09:00"
NOW = dt.datetime(2026, 10, 4, 15, 0, tzinfo=mp.KST)
_DEL = object()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    Path(p).write_bytes(b"\xef\xbb\xbf" + json.dumps(obj).encode("utf-8"))      # PowerShell-style BOM


class FakeBackend:
    def __init__(self, mutate_state=False, body_scale=1.0, break_build=None, on_forward=None):
        self.classes = list(CLASSES)
        self.calls, self.builds, self.state = [], [], {"w": "s0"}
        self.mutate_state, self.body_scale = mutate_state, body_scale
        self.break_build, self.on_forward = break_build, on_forward

    def build(self, obs):
        k = len(self.builds)
        self.builds.append(sorted(obs))
        u = obs["usable"]
        t = u.shape[0]
        xy = np.where(u[..., None], obs["xy_px"], 0).astype(np.float32)
        bv, hv = u[:, :33].astype(np.float32), u[:, 33:75].astype(np.float32)
        if k == self.break_build:
            hv = hv.copy()
            hv[0, 0] = 1 - hv[0, 0]
        x = np.zeros((1, t, 280), np.float32)
        x[0, :, :150] = xy.reshape(t, -1) / 640
        x[0, :, 186:219] = bv
        x[0, :, 237:279] = hv
        x[0, 1:, 279] = np.diff(obs["timestamp_us"]).astype(np.float32) / 1e6
        return {"x": x, "pad": np.ones((1, t), np.float32), "body_valid": bv, "hand_valid": hv,
                "body_scale": self.body_scale, "hand_scales": {"left": float("nan"), "right": 2.0},
                "frames": t}

    def forward(self, x, pad):
        self.calls.append(mp.typed_sha256(x))
        if self.on_forward:
            self.on_forward()
        if self.mutate_state:
            self.state["w"] = "changed"
        s = float(x.astype(np.float64).sum())
        return [math.sin(i * 0.37 + s) for i in range(93)]

    def state_digest(self):
        return dict(self.state)

    def audit(self):
        return {"fake": True}

    def final_audit(self):
        return {"fake": True}


def make_raw(rng, identical=False, n=40):
    fid = np.arange(n, dtype=np.int64)
    ts = (1_000 + fid * 33_333).astype(np.int64)
    usable = rng.random((n, 75)) > 0.2
    usable[::4, 33:54] = False
    a = {"obs_xy_px": rng.uniform(1, 640, (n, 75, 2)).astype(np.float32), "obs_timestamp_us": ts,
         "obs_frame_id": fid, "obs_observed": usable | (rng.random((n, 75)) > 0.5), "obs_usable": usable,
         "obs_conf": rng.random((n, 75)).astype(np.float32),
         "obs_hand_score": rng.random((n, 2)).astype(np.float32),
         "decoded_frame_id": fid, "decoded_timestamp_us": ts, "decoded_pts": fid * 1001,
         "decoded_rgb_sha256": np.array([f"{i:064x}" for i in range(n)]),
         "decoded_rgb_shape": np.tile(np.array([480, 640, 3], np.int64), (n, 1)),
         "decoded_processed": np.ones(n, bool), "selected_frame_id": fid[fid % 2 == 0]}
    ev = fid % 2 == 0
    b = {k: (v[ev].copy() if k.startswith("obs_") else v.copy()) for k, v in a.items()}
    b["decoded_processed"] = ev.copy()
    if not identical:
        b["obs_xy_px"][:, :33] += 1.5
        b["obs_usable"] = rng.random((int(ev.sum()), 75)) > 0.25
        b["obs_hand_score"] = rng.random((int(ev.sum()), 2)).astype(np.float32)
    return {"A": a, "B": b}


def make_tree(base, mutate_raw=None, mutate_source=None, mutate_cfg=None, labels=None):
    ws = base / "ws"
    front = ws / "front"
    unit = front / "ops" / mp.UNIT_NAME
    srcdir, data, pilot = front / "source_result", front / "data", front / "src" / "pilot"
    for d in (unit, srcdir, data, pilot):
        d.mkdir(parents=True)
    labels = labels or [CLASSES[3], CLASSES[10], CLASSES[20], CLASSES[30]]
    inv = []

    def pin(role, p):
        inv.append({"role": role, "path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)})

    for role, name in (("pilot_data", "data.py"), ("pilot_mapping", "web_input.py"),
                       ("pilot_model", "model_masked_v1.py")):
        (pilot / name).write_text(f"# fake {role}\n")
        pin(role, pilot / name)
    for role in ("source_file_hash_manifest", "source_independent_result", "source_independent_artifact_check",
                 "source_prepared_config", "observation_schema", "frozen_plan"):
        (srcdir / f"{role}.json").write_text("{}")
        pin(role, srcdir / f"{role}.json")
    ckpt = front / "ckpt.pt"
    ckpt.write_bytes(b"fake-checkpoint")
    pin("frozen_checkpoint", ckpt)

    fb, rng = FakeBackend(), np.random.default_rng(7)
    clips, wins, sclips, raw8 = [], [], [], []
    for ci in range(4):
        raw = make_raw(rng, identical=(ci == 3))
        ts = raw["A"]["obs_timestamp_us"]
        start, end = int(ts[10]), int(ts[30])
        ids = [f for f in range(40) if f % 2 == 0 and start <= ts[f] < end]
        common = {"clip_id": f"clip{ci}", "label": labels[ci], "start_us": start, "end_us_exclusive": end,
                  "window_frame_ids": ids, "window_timestamp_us": [int(ts[f]) for f in ids]}
        arms, sw, files = {}, {}, {}
        for arm in "AB":
            d = raw[arm]
            if mutate_raw:
                mutate_raw(ci, arm, d)
            npz, meta = data / f"c{ci}_{arm}.npz", data / f"c{ci}_{arm}.meta.json"
            np.savez(npz, **d)
            write_json(meta, {"npz_sha256": sha(npz), "clip_id": f"clip{ci}", "arm": arm,
                              "tag": mp.ARM_TAG[arm], "time_base": "1/30000",
                              "counts": {"rows": int(len(d["obs_frame_id"]))}})
            pin(f"raw_{arm}_{ci}", npz)
            pin(f"meta_{arm}_{ci}", meta)
            raw8.append(str(npz))
            rows = np.isin(d["obs_frame_id"], ids)
            inp = fb.build({k: d["obs_" + k][rows] for k in mp.OBS})
            lg = fb.forward(inp["x"], inp["pad"])
            rec, score = mp.input_record(inp), mp.score_logits(lg, CLASSES, labels[ci])
            arms[arm] = {"raw_npz": str(npz), "meta_json": str(meta), "stored_rows": "fake", "source_arm": arm,
                         "baseline_input": rec, "baseline_score": score, "baseline_logits93": lg}
            sw[arm] = {"input": json.loads(json.dumps(rec)), "logits93": list(lg), "score": dict(score)}
            files[arm] = {"npz": npz.name, "meta": meta.name}
        clips.append({**common, "expected_window_rows": len(ids), "arms": arms})
        wins.append({**json.loads(json.dumps(common)), **sw})
        sclips.append({"clip_id": f"clip{ci}", "files": files})
    source = {"status": "COMPLETE", "scoring": {"classes": list(CLASSES), "windows": wins}, "clips": sclips}
    if mutate_source:
        mutate_source(source)
    src_path = srcdir / "RESULT.json"
    write_json(src_path, source)
    pin("source_result", src_path)
    cfg = {"schema": mp.CONFIG_SCHEMA, "status": "PREPARED_FROZEN", "created_at_local": "2026-10-03T22:00:00+09:00",
           "workspace_root": str(ws), "front_root": str(front), "unit_dir": str(unit),
           "output_dir": str(unit / "result"), "source_result_dir": str(srcdir),
           "source_result_sha256": sha(src_path), "deadline_kst": DEADLINE, "clips": clips,
           "pins": {e["path"]: e["sha256"] for e in inv}, "pin_inventory": inv, "variants": list(mp.VARIANTS),
           "model": {"checkpoint_path": str(ckpt), "checkpoint_sha256": sha(ckpt),
                     "model_class": "pilot.model_masked_v1.MaskedSignTCN", "n_classes": 93,
                     "classes": list(CLASSES), "cpu_threads": 2},
           "preprocessing": {"new_forward_calls": 16, "baseline_forwards": 0, "baseline_input_rechecks": 8},
           "isolation": {"allowed_raw_npz": raw8},
           "prepared_execution_counts": {"baseline_forwards": 0, "new_forward_calls": 0}}
    if mutate_cfg:
        mutate_cfg(cfg)
    cfg_path = unit / "PREPARED_CONFIG.json"
    write_json(cfg_path, cfg)
    return SimpleNamespace(ws=ws, unit=unit, cfg_path=cfg_path, cfg=cfg, inv=inv)


def write_auth(t, over=None):
    doc = {"schema": mp.RESUME_SCHEMA, "unit": mp.UNIT_NAME, "recorded_kst": "2026-10-04T14:00:00+09:00",
           "user_request": "resume this unchanged unit", "prepared_config_sha256": sha(t.cfg_path),
           "supersedes_deadline_only": DEADLINE, "allow_after_original_deadline": True,
           "scope": "same unit only", "camera": False, "training": False, "operational_change": False}
    for k, v in (over or {}).items():
        if v is _DEL:
            doc.pop(k)
        else:
            doc[k] = v
    p = t.unit / "RESUME_AUTHORIZATION.json"
    write_json(p, doc)
    return p


def write_receipt(t, **over):
    test = HERE / "test_mechanism.py"
    doc = {"schema": mp.RECEIPT_SCHEMA, "status": "PASS", "runner_sha256": sha(HERE / "mechanism_pair.py"),
           "test_sha256": sha(test), "tests_run": mp.count_tests(test), "failures": 0, "errors": 0, "skipped": 0}
    doc.update(over)
    p = t.unit / "TEST_RECEIPT.json"
    write_json(p, doc)
    return p


def small_pair(n=4):
    def arm(seed):
        r = np.random.default_rng(seed)
        return {"xy_px": r.uniform(0, 100, (n, 75, 2)).astype(np.float32),
                "timestamp_us": np.arange(n, dtype=np.int64) * 10, "frame_id": np.arange(n, dtype=np.int64) * 2,
                "observed": np.ones((n, 75), bool), "usable": r.random((n, 75)) > 0.3,
                "conf": r.random((n, 75)).astype(np.float32), "hand_score": r.random((n, 2)).astype(np.float32)}
    a, b = arm(1), arm(2)
    a["usable"][0, 33:54] = False          # row 0: left hand absent in A
    b["usable"][1, 54:75] = False          # row 1: right hand absent in B
    a["usable"][2, 33] = b["usable"][2, 33] = True
    return a, b


class VariantTests(unittest.TestCase):
    def test_originals_not_mutated(self):
        a, b = small_pair()
        a0, b0 = {k: v.copy() for k, v in a.items()}, {k: v.copy() for k, v in b.items()}
        mp.make_variants(a, b)
        for k in mp.OBS:
            self.assertTrue(mp.same(a[k], a0[k]) and mp.same(b[k], b0[k]), k)

    def test_common_retains_exact_xy_conf_and_nan_outside(self):
        a, b = small_pair()
        vs, mask = mp.make_variants(a, b)
        np.testing.assert_array_equal(mask, a["usable"] & b["usable"])
        for name, src in (("common_A", a), ("common_B", b)):
            v = vs[name]
            np.testing.assert_array_equal(v["xy_px"][mask], src["xy_px"][mask])
            np.testing.assert_array_equal(v["conf"][mask], src["conf"][mask])
            self.assertTrue(np.isnan(v["xy_px"][~mask]).all() and np.isnan(v["conf"][~mask]).all())
            np.testing.assert_array_equal(v["observed"], mask)
            np.testing.assert_array_equal(v["usable"], mask)
        mp.check_common_raw(a, b, vs, mask)

    def test_common_hand_score_nan_only_when_hand_absent(self):
        a, b = small_pair()
        vs, _ = mp.make_variants(a, b)
        for name, src in (("common_A", a), ("common_B", b)):
            hs = vs[name]["hand_score"]
            self.assertTrue(math.isnan(hs[0, 0]) and math.isnan(hs[1, 1]))
            self.assertEqual(hs[2, 0], src["hand_score"][2, 0])

    def test_swap_donor_directions(self):
        a, b = small_pair()
        vs, _ = mp.make_variants(a, b)
        for name, pose, hands in (("pose_A_hands_B", a, b), ("pose_B_hands_A", b, a)):
            v = vs[name]
            for k in ("xy_px", "observed", "usable", "conf"):
                np.testing.assert_array_equal(v[k][:, :33], pose[k][:, :33])
                np.testing.assert_array_equal(v[k][:, 33:75], hands[k][:, 33:75])
            np.testing.assert_array_equal(v["hand_score"], hands["hand_score"])
        self.assertEqual(mp.DONORS["pose_A_hands_B"], {"pose": "A", "hands": "B", "mask": None})
        self.assertEqual(mp.DONORS["pose_B_hands_A"], {"pose": "B", "hands": "A", "mask": None})

    def test_ids_timestamps_preserved(self):
        a, b = small_pair()
        vs, _ = mp.make_variants(a, b)
        for v in vs.values():
            np.testing.assert_array_equal(v["frame_id"], a["frame_id"])
            np.testing.assert_array_equal(v["timestamp_us"], a["timestamp_us"])

    def test_common_nonfinite_shared_xy_refused(self):
        a, b = small_pair()
        a["xy_px"][2, 33, 0] = np.nan
        vs, mask = mp.make_variants(a, b)
        with self.assertRaises(mp.Fail):
            mp.check_common_raw(a, b, vs, mask)

    def test_same_helper_nan_equality(self):
        n = np.array([1.0, np.nan], np.float32)
        self.assertTrue(mp.same(n, n.copy()))
        self.assertFalse(mp.same(n, n.astype(np.float64)))
        self.assertFalse(mp.same(np.array([1]), np.array([2])))
        self.assertTrue(mp.same(np.array([True]), np.array([True])))
        self.assertIsNone(mp.jnum(float("nan")))

    def test_score_full93_after_forward(self):
        lg = [0.0] * 93
        lg[5], lg[7] = 2.0, 1.0
        s = mp.score_logits(lg, CLASSES, CLASSES[7])
        self.assertEqual((s["top1"], s["correct"], s["true_rank"], s["best_other"]), (CLASSES[5], False, 2, CLASSES[5]))
        self.assertEqual(s["true_vs_best_other_margin"], -1.0)


class TreeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.base = Path(os.path.realpath(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def tree(self, name="t", **kw):
        return make_tree(self.base / name, **kw)

    def ctx(self, t, now=NOW):
        return mp.validate(t.cfg_path, write_auth(t), now)

    def run_ok(self, t, fb=None):
        ctx, fb = self.ctx(t), fb or FakeBackend()
        return fb, mp.run(ctx, write_receipt(t), backend=fb, code_dir=HERE)


class GateTests(TreeCase):
    def test_a_even_selection_and_half_open_window(self):
        t = self.tree()
        ctx = self.ctx(t)
        a = ctx["clips"][0]["A"]
        self.assertEqual(a["frame_id"].tolist(), list(range(10, 30, 2)))
        self.assertEqual(int(a["timestamp_us"][0]), t.cfg["clips"][0]["start_us"])
        self.assertNotIn(30, a["frame_id"].tolist())
        self.assertEqual(1_000 + 30 * 33_333, t.cfg["clips"][0]["end_us_exclusive"])

    def test_decoded_metadata_mismatch_refused(self):
        def bump(ci, arm, d):
            if ci == 1 and arm == "B":
                d["decoded_pts"] = d["decoded_pts"].copy()
                d["decoded_pts"][12] += 1
        t = self.tree(mutate_raw=bump)
        with self.assertRaisesRegex(mp.Fail, "decoded pts"):
            self.ctx(t)

    def test_class_order_93_required(self):
        def dup(cfg):
            cfg["model"]["classes"][1] = cfg["model"]["classes"][0]
        def rev(src):
            src["scoring"]["classes"] = src["scoring"]["classes"][::-1]
        for i, kw in enumerate(({"mutate_cfg": dup}, {"mutate_source": rev})):
            with self.subTest(i=i), self.assertRaises(mp.Fail):
                self.ctx(self.tree(f"t{i}", **kw))

    def test_default_validation_imports_nothing_heavy(self):
        t = self.tree()
        auth = write_auth(t)
        code = (f"import sys, datetime as dt; sys.path.insert(0, {str(HERE)!r}); import mechanism_pair as mp; "
                f"mp.validate({str(t.cfg_path)!r}, {str(auth)!r}, now=dt.datetime(2026,10,4,15,0,tzinfo=mp.KST)); "
                "bad=[m for m in ('torch','av','mediapipe','cv2','pilot') if m in sys.modules]; "
                "print(bad); sys.exit(1 if bad else 0)")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse((t.unit / "result").exists())

    def test_pin_tamper_refused(self):
        t = self.tree()
        with open(t.cfg["clips"][0]["arms"]["A"]["raw_npz"], "ab") as fh:
            fh.write(b"\0")
        with self.assertRaisesRegex(mp.Fail, "pins"):
            self.ctx(t)

    def test_source_disagreement_refused(self):
        def bump(src):
            src["scoring"]["windows"][0]["A"]["logits93"][0] += 1.0
        with self.assertRaisesRegex(mp.Fail, "source"):
            self.ctx(self.tree(mutate_source=bump))

    def test_invalid_paths_refused(self):
        t = self.tree()
        ws = str(t.ws)
        bad = [r"\\server\share\x", "//server/x", "http://host/x", "relative\\x", os.path.join(ws, "a", "..", "b"),
               r"\\.\COM1", r"\\?\C:\x", os.path.join(ws, "NUL"), os.path.join(ws, "f.txt:stream"), str(self.base)]
        for p in bad:
            with self.subTest(p=p), self.assertRaises(mp.Fail):
                mp.local_path(p, t.ws)
        target, link = t.ws / "real", t.ws / "link"
        target.mkdir()
        (target / "f.txt").write_text("x")
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(target, link)
        with self.assertRaisesRegex(mp.Fail, "redirection"):
            mp.local_path(str(link / "f.txt"), t.ws)

    def test_output_reuse_refused(self):
        t = self.tree()
        ctx = self.ctx(t)
        (t.unit / "result").mkdir()
        with self.assertRaisesRegex(mp.Fail, "overwrite"):
            mp.run(ctx, write_receipt(t), backend=FakeBackend(), code_dir=HERE)
        self.assertEqual(list((t.unit / "result").iterdir()), [])
        with self.assertRaisesRegex(mp.Fail, "overwrite"):
            self.ctx(t)

    def test_deadline_requires_resume(self):
        t = self.tree()
        with self.assertRaisesRegex(mp.Fail, "deadline"):
            mp.validate(t.cfg_path, None, NOW)
        mp.validate(t.cfg_path, None, dt.datetime(2026, 10, 4, 8, 0, tzinfo=mp.KST))

    def test_resume_authorization_fields(self):
        t = self.tree()
        bad = [{"schema": "x"}, {"unit": "other_unit"}, {"prepared_config_sha256": "0" * 64},
               {"allow_after_original_deadline": False}, {"allow_after_original_deadline": "true"},
               {"supersedes_deadline_only": "2026-10-05T09:00:00+09:00"}, {"camera": True}, {"training": True},
               {"operational_change": True}, {"camera": _DEL}, {"user_request": " "},
               {"recorded_kst": "2026-10-04T16:00:00+09:00"}, {"recorded_kst": "2026-10-03T23:00:00+09:00"},
               {"recorded_kst": "2026-10-04T14:00:00"}]
        for over in bad:
            with self.subTest(over=str(over)), self.assertRaisesRegex(mp.Fail, "resume"):
                mp.validate(t.cfg_path, write_auth(t, over), NOW)
        other = t.unit / "OTHER.json"
        write_json(other, json.loads(write_auth(t).read_text(encoding="utf-8-sig")))
        with self.assertRaisesRegex(mp.Fail, "resume"):
            mp.validate(t.cfg_path, other, NOW)

    def test_run_requires_matching_receipt(self):
        t = self.tree()
        ctx = self.ctx(t)
        for kw in ({"runner_sha256": "0" * 64}, {"test_sha256": "0" * 64}, {"tests_run": 1}, {"failures": 1},
                   {"status": "FAIL"}):
            with self.subTest(kw=kw), self.assertRaisesRegex(mp.Fail, "receipt"):
                mp.run(ctx, write_receipt(t, **kw), backend=FakeBackend(), code_dir=HERE)
        with self.assertRaisesRegex(mp.Fail, "receipt"):
            mp.run(ctx, None, backend=FakeBackend(), code_dir=HERE)
        self.assertFalse((t.unit / "result").exists())

    def test_receipt_count_matches_file(self):
        n = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]).countTestCases()
        self.assertEqual(n, mp.count_tests(HERE / "test_mechanism.py"))


class RunTests(TreeCase):
    def test_run_reuses8_infers16_once(self):
        t = self.tree()
        fb, res = self.run_ok(t)
        self.assertEqual(len(fb.calls), 16)                       # clip3 A==B still forwards all 4
        self.assertEqual(res["execution_counts"], {"baseline_input_rechecks": 8, "baseline_forwards": 0,
                                                   "new_input_builds": 16, "new_forward_calls": 16})
        self.assertEqual(res["score_record_count"], 24)
        for w, cc in zip(res["windows"], t.cfg["clips"]):
            self.assertEqual([r["condition"] for r in w["records"]], ["A", "B", *mp.VARIANTS])
            self.assertEqual([r["status"] for r in w["records"]], ["REUSED"] * 2 + ["INFERRED"] * 4)
            for r in w["records"][:2]:
                self.assertEqual(r["logits93"], cc["arms"][r["condition"]]["baseline_logits93"])
            self.assertIsNone(w["records"][2]["input"]["hand_scales"]["left"])
        out = t.unit / "result"
        names = {p.name for p in out.iterdir()}
        self.assertTrue({"RUN_STARTED.json", "RUNTIME.json", "RESULT.json", "SUMMARY.md", "FILE_HASHES.json"} <= names)
        self.assertEqual(len([n for n in names if n.endswith(".npz")]), 16)
        self.assertNotIn("FAILURE.json", names)
        hashes = json.loads((out / "FILE_HASHES.json").read_text(encoding="utf-8"))
        self.assertEqual(hashes["RESULT.json"]["sha256"], sha(out / "RESULT.json"))

    def test_no_label_leakage(self):
        fb1, _ = self.run_ok(self.tree("t1"))
        fb2, _ = self.run_ok(self.tree("t2", labels=[CLASSES[50], CLASSES[51], CLASSES[52], CLASSES[53]]))
        self.assertEqual(fb1.calls, fb2.calls)
        self.assertTrue(all(keys == sorted(mp.OBS) for keys in fb1.builds))

    def test_resume_recorded_in_result(self):
        t = self.tree()
        _, res = self.run_ok(t)
        auth = t.unit / "RESUME_AUTHORIZATION.json"
        self.assertEqual(res["resume_authorization"]["sha256"], sha(auth))
        self.assertTrue(res["resume_authorization"]["content"]["allow_after_original_deadline"])
        self.assertEqual(res["config"]["deadline_kst"], DEADLINE)

    def test_state_change_fails_and_preserves_failure(self):
        t = self.tree()
        with self.assertRaisesRegex(mp.Fail, "state changed"):
            self.run_ok(t, FakeBackend(mutate_state=True))
        out = t.unit / "result"
        fail = json.loads((out / "FAILURE.json").read_text(encoding="utf-8"))
        self.assertEqual((fail["status"], fail["stage"]), ("TECHNICAL_FAILURE", "forward"))
        self.assertFalse((out / "RESULT.json").exists())
        self.assertEqual(len(list(out.glob("*.npz"))), 16)

    def test_baseline_recheck_mismatch_fails_before_forward(self):
        t = self.tree()
        fb = FakeBackend(body_scale=1.0001)
        with self.assertRaisesRegex(mp.Fail, "baseline_inputs"):
            self.run_ok(t, fb)
        self.assertEqual(fb.calls, [])

    def test_common_valid_channel_mismatch_fails_before_forward(self):
        t = self.tree()
        fb = FakeBackend(break_build=9)                           # builds 0-7 baselines, 8/9 clip0 common_A/B
        with self.assertRaisesRegex(mp.Fail, "validity channels"):
            self.run_ok(t, fb)
        self.assertEqual(fb.calls, [])

    def test_post_run_tamper_detected(self):
        t = self.tree()
        plan = next(e["path"] for e in t.inv if e["role"] == "frozen_plan")

        def touch():
            with open(plan, "ab") as fh:
                fh.write(b" ")
        with self.assertRaisesRegex(mp.Fail, "pinned file changed"):
            self.run_ok(t, FakeBackend(on_forward=touch))
        self.assertTrue((t.unit / "result" / "FAILURE.json").exists())


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt")
    a = ap.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    ok = res.wasSuccessful() and not res.skipped
    if a.receipt and ok:
        doc = {"schema": mp.RECEIPT_SCHEMA, "status": "PASS",
               "runner_sha256": sha(HERE / "mechanism_pair.py"), "test_sha256": sha(Path(__file__).resolve()),
               "tests_run": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
               "skipped": len(res.skipped), "created_kst": dt.datetime.now(mp.KST).isoformat()}
        with open(a.receipt, "x", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _main()
