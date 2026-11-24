"""Tests for handgap.py and the pure parts of run_handgap.py (no model, no data, no torch).

Run: python -m pytest -q test_handgap.py     (or: python -m unittest test_handgap)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handgap as HG          # noqa: E402
import run_handgap as RH      # noqa: E402

L, R = HG.HANDS["left"], HG.HANDS["right"]
BODY = slice(0, 25)


def make(t=10, step=33333, start=0, dtype=np.float32, pdtype=bool, seed=0, diffs=None):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, (t, 67, 2)).astype(dtype)
    pr = np.ones((t, 67), dtype=pdtype)
    if diffs is None:
        ts = start + step * np.arange(t, dtype=np.int64)
    else:
        ts = start + np.concatenate([[0], np.cumsum(np.asarray(diffs, dtype=np.int64))]).astype(np.int64)
    return xy, pr, ts


def drop(xy, pr, i, side, fill=0.0):
    xy, pr = xy.copy(), pr.copy()
    xy[i, HG.HANDS[side]] = fill
    pr[i, HG.HANDS[side]] = 0
    return xy, pr


def expected_schedule(sid, t):
    d = hashlib.sha256(("WIS_HAND_GAP_V1|" + sid).encode("utf-8")).digest()
    first = d[1] % 2
    return d[0] % 6, first, [(i, ("left", "right")[(first + k) % 2])
                             for k, i in enumerate(range(2 + d[0] % 6, t - 2, 6))]


def same(a, b):
    return RH._same(a, b)


class RestoreTests(unittest.TestCase):
    def test_analytic_ramp_recovery(self):
        ts = np.array([0, 21000, 66000, 87000], dtype=np.int64)        # both spans 66000 us
        a, b = np.linspace(-3, 3, 67 * 2).reshape(67, 2), np.linspace(1, 2, 67 * 2).reshape(67, 2)
        xy = (a[None] + b[None] * (ts[:, None, None] / 1e5)).astype(np.float64)
        pr = np.ones((4, 67), bool)
        mxy, mpr = drop(xy, pr, 1, "left")
        mxy, mpr = drop(mxy, mpr, 2, "right")
        r = HG.restore(mxy, mpr, ts)
        np.testing.assert_allclose(r["xy"][1, L], xy[1, L], rtol=0, atol=1e-12)
        np.testing.assert_allclose(r["xy"][2, R], xy[2, R], rtol=0, atol=1e-12)
        self.assertTrue(r["estimated"][1, L].all() and r["estimated"][2, R].all())
        self.assertEqual(int(r["estimated"].sum()), 42)
        self.assertTrue(r["present"][1, L].all() and r["present"][2, R].all())
        self.assertFalse(r["identity"])

    def test_asymmetric_time_weights(self):
        for mid, want in ((10000, 10 / 6), (50000, 50 / 6)):
            xy, pr, _ = make(3, dtype=np.float64)
            ts = np.array([0, mid, 60000], dtype=np.int64)
            xy[0, L], xy[2, L] = 0.0, 10.0
            mxy, mpr = drop(xy, pr, 1, "left")
            r = HG.restore(mxy, mpr, ts)
            np.testing.assert_allclose(r["xy"][1, L], want, rtol=1e-12)
            self.assertFalse(np.allclose(r["xy"][1, L], 5.0))          # not a midpoint

    def test_span_boundary_66668_vs_66669(self):
        for last, filled in ((66668, True), (66669, False)):
            xy, pr, _ = make(3)
            ts = np.array([0, 33334, last], dtype=np.int64)
            mxy, mpr = drop(xy, pr, 1, "right", fill=np.nan)
            r = HG.restore(mxy, mpr, ts)
            self.assertEqual(bool(r["estimated"][1, R].all()), filled)
            self.assertEqual(bool(r["estimated"].any()), filled)
            if not filled:
                self.assertTrue(same(r["xy"], mxy) and same(r["present"], mpr) and r["identity"])

    def test_clean_identity(self):
        xy, pr, ts = make(12)
        r = HG.restore(xy, pr, ts)
        self.assertTrue(r["identity"])
        self.assertTrue(same(r["xy"], xy) and same(r["present"], pr) and same(r["ts"], ts))
        self.assertFalse(r["estimated"].any())
        self.assertTrue(np.array_equal(r["observed"], pr))

    def test_long_gap_not_filled(self):
        xy, pr, _ = make(3)
        ts = np.array([0, 50000, 100000], dtype=np.int64)
        mxy, mpr = drop(xy, pr, 1, "left")
        r = HG.restore(mxy, mpr, ts)
        self.assertTrue(r["identity"] and same(r["xy"], mxy) and same(r["present"], mpr))

    def test_two_consecutive_absences_not_filled(self):
        xy, pr, ts = make(6)
        mxy, mpr = drop(xy, pr, 2, "left")
        mxy, mpr = drop(mxy, mpr, 3, "left")
        mxy, mpr = drop(mxy, mpr, 2, "right")                         # isolated on the other side
        r = HG.restore(mxy, mpr, ts)
        self.assertFalse(r["estimated"][:, L].any())
        self.assertTrue(same(r["xy"][:, L], mxy[:, L]) and same(r["present"][:, L], mpr[:, L]))
        self.assertTrue(r["estimated"][2, R].all() and int(r["estimated"].sum()) == 21)

    def test_first_and_last_frames_not_filled(self):
        xy, pr, ts = make(5)
        mxy, mpr = drop(xy, pr, 0, "left")
        mxy, mpr = drop(mxy, mpr, 4, "right")
        r = HG.restore(mxy, mpr, ts)
        self.assertTrue(r["identity"] and same(r["xy"], mxy) and same(r["present"], mpr))

    def test_partial_hand_untouched(self):
        xy, pr, ts = make(3)
        mxy, mpr = drop(xy, pr, 1, "left")
        mpr[1, 25] = True                                              # 20 of 21 absent
        mxy[1, 25] = xy[1, 25]
        r = HG.restore(mxy, mpr, ts)
        self.assertTrue(r["identity"] and same(r["xy"], mxy) and same(r["present"], mpr))
        mxy, mpr = drop(xy, pr, 1, "left")
        mpr[0, 30] = False                                              # neighbour not whole
        r = HG.restore(mxy, mpr, ts)
        self.assertTrue(r["identity"] and same(r["xy"], mxy))

    def test_other_side_body_timestamps_preserved(self):
        xy, pr, ts = make(7, seed=3)
        mxy, mpr = drop(xy, pr, 3, "right")
        mpr[2, 4] = False
        mxy[2, 4] = np.nan
        r = HG.restore(mxy, mpr, ts)
        keep = ~r["estimated"]
        self.assertTrue(same(r["xy"][keep], mxy[keep]) and same(r["present"][keep], mpr[keep]))
        self.assertTrue(same(r["xy"][:, BODY], mxy[:, BODY]) and same(r["xy"][:, L], mxy[:, L]))
        self.assertTrue(same(r["ts"], ts))
        self.assertTrue(r["estimated"][3, R].all() and int(r["estimated"].sum()) == 21)

    def test_absent_nan_untouched(self):
        xy, pr, ts = make(6)
        mxy, mpr = drop(xy, pr, 2, "left", fill=np.nan)
        mxy, mpr = drop(mxy, mpr, 3, "left", fill=np.nan)              # 2 consecutive: not filled
        mpr[1, 7], mxy[1, 7] = False, np.nan                            # absent body joint
        for out in (HG.restore(mxy, mpr, ts), HG.mask_one_hand(mxy, mpr, ts, "nan_case")):
            self.assertTrue(np.isnan(out["xy"][2:4, L]).all() and np.isnan(out["xy"][1, 7]).all())

    def test_no_mutation_no_aliasing(self):
        xy, pr, ts = make(9)
        mxy, mpr = drop(xy, pr, 4, "left")
        snap = (mxy.copy(), mpr.copy(), ts.copy())
        r = HG.restore(mxy, mpr, ts)
        m = HG.mask_one_hand(mxy, mpr, ts, "alias")
        self.assertTrue(all(same(a, b) for a, b in zip(snap, (mxy, mpr, ts))))
        for out in (r, m):
            for k, src in (("xy", mxy), ("present", mpr), ("ts", ts)):
                self.assertFalse(np.shares_memory(out[k], src))
        r["xy"][:] = 7
        r["present"][:] = False
        r["ts"][:] = 0
        self.assertTrue(all(same(a, b) for a, b in zip(snap, (mxy, mpr, ts))))

    def test_dtypes_preserved(self):
        for xdt in (np.float32, np.float64):
            for pdt in (bool, np.uint8, np.int32, np.float32):
                for tdt in (np.int64, np.int32, np.uint64):
                    xy, pr, ts = make(5, dtype=xdt, pdtype=pdt)
                    ts = ts.astype(tdt)
                    mxy, mpr = drop(xy, pr, 2, "left")
                    r = HG.restore(mxy, mpr, ts)
                    self.assertEqual((r["xy"].dtype, r["present"].dtype, r["ts"].dtype),
                                     (np.dtype(xdt), np.dtype(pdt), np.dtype(tdt)))
                    self.assertEqual((r["observed"].dtype, r["estimated"].dtype), (np.bool_, np.bool_))
                    self.assertTrue(np.all(r["present"][2, L] == 1))
                    m = HG.mask_one_hand(xy, pr, ts, "dtype")
                    self.assertEqual((m["xy"].dtype, m["present"].dtype, m["ts"].dtype, m["injected"].dtype),
                                     (np.dtype(xdt), np.dtype(pdt), np.dtype(tdt), np.bool_))

    def test_invalid_inputs(self):
        xy, pr, ts = make(4)
        bad = [
            (xy[:, :66], pr, ts), (xy[..., :1], pr, ts), (xy.astype(np.int32), pr, ts), (xy[:0], pr[:0], ts[:0]),
            (xy, pr[:, :66], ts), (xy, pr[:3], ts), (xy, np.where(pr, 2, 0), ts), (xy, pr.astype(object), ts),
            (xy, pr.astype(np.float32) * np.nan, ts),
            (xy, pr, ts.astype(np.float64)), (xy, pr, ts.astype(bool)), (xy, pr, ts[:3]),
            (xy, pr, ts[::-1].copy()), (xy, pr, np.array([0, 1, 1, 2], dtype=np.int64)),
        ]
        inf = xy.copy()
        inf[1, 30, 0] = np.inf                                         # non-finite where present
        nan = xy.copy()
        nan[2, 3, 1] = np.nan
        bad += [(inf, pr, ts), (nan, pr, ts)]
        for args in bad:
            with self.assertRaises(ValueError):
                HG.restore(*args)
            with self.assertRaises(ValueError):
                HG.mask_one_hand(*args, "bad")
        with self.assertRaises(TypeError):
            HG.mask_one_hand(xy, pr, ts, 123)

    def test_negative_and_large_clocks(self):
        for base in (-10 ** 15, 2 ** 62, 2 ** 63 - 60001):
            xy, pr, _ = make(3, dtype=np.float64)
            ts = np.array([base, base + 10000, base + 60000], dtype=np.int64)
            xy[0, L], xy[2, L] = 0.0, 10.0
            mxy, mpr = drop(xy, pr, 1, "left")
            r = HG.restore(mxy, mpr, ts)
            np.testing.assert_allclose(r["xy"][1, L], 10 / 6, rtol=1e-12)
            self.assertTrue(same(r["ts"], ts))
        ts = np.array([2 ** 63 - 66670, 2 ** 63 - 33336, 2 ** 63 - 1], dtype=np.int64)   # span 66669
        xy, pr, _ = make(3)
        mxy, mpr = drop(xy, pr, 1, "left")
        self.assertTrue(HG.restore(mxy, mpr, ts)["identity"])

    def test_hidden_values_do_not_affect_restore(self):
        xy, pr, ts = make(5, seed=7)
        outs = [HG.restore(*drop(xy, pr, 2, "left", fill=f), ts) for f in (0.0, np.nan, 1e30, -np.inf)]
        for o in outs[1:]:
            for k in ("xy", "present", "ts", "observed", "estimated"):
                self.assertTrue(same(o[k], outs[0][k]), k)

    def test_observed_estimated_disjoint_cover(self):
        xy, pr, ts = make(20, seed=2)
        mxy, mpr = drop(xy, pr, 5, "left")
        mxy, mpr = drop(mxy, mpr, 9, "right")
        mpr[12, 3] = False
        r = HG.restore(mxy, mpr, ts)
        self.assertFalse((r["observed"] & r["estimated"]).any())
        self.assertTrue(np.array_equal(r["observed"] | r["estimated"], r["present"].astype(bool)))
        self.assertTrue(np.array_equal(r["observed"], mpr))
        self.assertEqual(int(r["estimated"].sum()), 42)

    def test_no_chaining_through_estimates(self):
        xy, pr, ts = make(5)
        mxy, mpr = drop(xy, pr, 1, "left")
        mxy, mpr = drop(mxy, mpr, 3, "left")                            # 2 is present; 1 and 3 isolated
        r = HG.restore(mxy, mpr, ts)
        self.assertTrue(r["estimated"][1, L].all() and r["estimated"][3, L].all())
        np.testing.assert_allclose(r["xy"][3, L], (mxy[2, L].astype(np.float64) + mxy[4, L]) / 2, rtol=1e-6, atol=1e-6)


class MaskTests(unittest.TestCase):
    def test_schedule_matches_frozen_rule(self):
        for sid in ("a", "row_001", "한국어_id", "x" * 50):
            xy, pr, ts = make(40, step=33334)                           # span 66668: every scheduled eligible
            offset, first, sched = expected_schedule(sid, 40)
            m = HG.mask_one_hand(xy, pr, ts, sid)
            self.assertEqual((m["offset"], m["first_side"]), (offset, ("left", "right")[first]))
            self.assertEqual([(e["frame"], e["side"]) for e in m["events"]], sched)
            exp = np.zeros((40, 67), bool)
            for i, side in sched:
                exp[i, HG.HANDS[side]] = True
            self.assertTrue(np.array_equal(m["injected"], exp))
            self.assertTrue(np.all(m["xy"][exp] == 0) and not m["present"][exp].any())
            self.assertTrue(same(m["xy"][~exp], xy[~exp]) and same(m["present"][~exp], pr[~exp]))
            self.assertTrue(same(m["ts"], ts))

    def test_deterministic(self):
        xy, pr, ts = make(30, seed=5)
        a, b = HG.mask_one_hand(xy, pr, ts, "det"), HG.mask_one_hand(xy, pr, ts, "det")
        for k in ("xy", "present", "ts", "injected"):
            self.assertTrue(same(a[k], b[k]))
        self.assertEqual(a["events"], b["events"])

    def test_skip_still_advances_alternation(self):
        sid, t = "skip_case", 40
        offset, first, sched = expected_schedule(sid, t)
        self.assertGreaterEqual(len(sched), 3)
        xy, pr, ts = make(t, step=33334)
        i0, s0 = sched[0]
        pr[i0 + 1, HG.HANDS[s0].start] = False                          # one joint missing at i+1 -> skip
        m = HG.mask_one_hand(xy, pr, ts, sid)
        self.assertFalse(m["schedule"][0]["eligible"])
        self.assertEqual([(e["frame"], e["side"]) for e in m["events"]], sched[1:])
        self.assertEqual(m["events"][0]["side"], ("left", "right")[(first + 1) % 2])

    def test_span_boundary_in_mask(self):
        sid, t = "boundary", 40
        _, _, sched = expected_schedule(sid, t)
        for step, n in ((33334, len(sched)), (33335, 0)):
            xy, pr, ts = make(t, step=step)
            self.assertEqual(len(HG.mask_one_hand(xy, pr, ts, sid)["events"]), n)
        i0 = sched[0][0]
        diffs = [33334] * (t - 1)
        diffs[i0] = 33335                                               # ts[i0+1]-ts[i0-1] = 66669
        xy, pr, ts = make(t, diffs=diffs)
        m = HG.mask_one_hand(xy, pr, ts, sid)
        self.assertEqual([(e["frame"], e["side"]) for e in m["events"]], sched[1:])

    def test_selection_ignores_geometry(self):
        xy, pr, ts = make(40, step=33334, seed=1)
        a = HG.mask_one_hand(xy, pr, ts, "geo")
        b = HG.mask_one_hand((xy * -50 + 3).astype(xy.dtype), pr, ts, "geo")
        self.assertTrue(np.array_equal(a["injected"], b["injected"]))

    def test_short_clips_identity(self):
        for t in range(1, 5):
            xy, pr, ts = make(t)
            m = HG.mask_one_hand(xy, pr, ts, "short")
            self.assertEqual((m["events"], m["schedule"]), ([], []))
            self.assertTrue(same(m["xy"], xy) and same(m["present"], pr) and not m["injected"].any())
        offset, _, _ = expected_schedule("short", 0)
        xy, pr, ts = make(4 + offset)
        self.assertEqual(HG.mask_one_hand(xy, pr, ts, "short")["events"], [])
        xy, pr, ts = make(1)
        self.assertTrue(HG.restore(xy, pr, ts)["identity"])

    def test_mask_then_restore_recovers_ramp_and_all_injected(self):
        t = 40
        ts = 33334 * np.arange(t, dtype=np.int64)
        xy = (np.linspace(0, 1, 134).reshape(67, 2)[None] + ts[:, None, None] * 1e-6).astype(np.float64)
        pr = np.ones((t, 67), bool)
        m = HG.mask_one_hand(xy, pr, ts, "ramp")
        self.assertTrue(m["events"])
        r = HG.restore(m["xy"], m["present"], m["ts"])
        self.assertTrue(np.array_equal(r["estimated"], m["injected"]))
        np.testing.assert_allclose(r["xy"], xy, rtol=0, atol=1e-12)


class RunnerPureTests(unittest.TestCase):
    def test_verify_mask_and_restore_accept_and_reject(self):
        xy, pr, ts = make(40, step=33334, seed=4)
        pr[10, R], xy[10, R] = False, np.nan                             # natural isolated gap
        m = HG.mask_one_hand(xy, pr, ts, "verify")
        vm = RH.verify_mask(xy, pr, ts, "verify", m)
        self.assertEqual(vm["events_n"], len(m["events"]))
        r = HG.restore(m["xy"], m["present"], m["ts"])
        vr = RH.verify_restore(m["xy"], m["present"], m["ts"], r)
        self.assertEqual(vr["slots"], int(r["estimated"].sum()))
        bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
        bad["xy"][0, 0, 0] += 1
        with self.assertRaises(RH.RunError):
            RH.verify_mask(xy, pr, ts, "verify", bad)
        other = next(s for s in (f"other_{k}" for k in range(100))
                     if expected_schedule(s, 40)[:2] != expected_schedule("verify", 40)[:2])
        with self.assertRaises(RH.RunError):
            RH.verify_mask(xy, pr, ts, other, m)
        i = int(np.argwhere(r["estimated"])[0][0])
        for tamper in ("value", "flag", "identity"):
            t2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            if tamper == "value":
                t2["xy"][i, :, :] += 1e-3 * t2["estimated"][i][:, None]
            elif tamper == "flag":
                t2["estimated"][0, 0] = True
            else:
                t2["identity"] = True
            with self.assertRaises(RH.RunError):
                RH.verify_restore(m["xy"], m["present"], m["ts"], t2)

    def test_event_errors_analytic(self):
        xy = np.zeros((3, 67, 2), np.float32)
        est = xy.copy()
        est[1, L] = (3.0, 4.0)                                          # |(3,4)| = 5
        xy[0, L] = (0.0, 1.0)                                           # previous copy error 1
        ev = RH.event_errors(xy, est, [{"frame": 1, "side": "left"}], 2.0)
        self.assertEqual(len(ev), 1)
        np.testing.assert_allclose(ev[0]["linear"], [2.5] * 21)
        np.testing.assert_allclose(ev[0]["previous"], [0.5] * 21)
        self.assertEqual((ev[0]["frame"], ev[0]["side"]), (1, "left"))
        for s in (0.0, -1.0, float("nan"), float("inf"), None):
            with self.assertRaises(RH.RunError):
                RH.event_errors(xy, est, [{"frame": 1, "side": "left"}], s)

    def test_coord_summary_analytic_and_gate(self):
        lin_b = [0.0] + [2.0] * 20                                      # wrist 0, others 2
        rows = [[{"frame": 1, "side": "left", "linear": [1.0] * 21, "previous": [2.5] * 21}],
                [],
                [{"frame": 2, "side": "right", "linear": lin_b, "previous": [2.5] * 21}]]
        s = RH.coord_summary(rows)
        self.assertTrue(s["available"])
        self.assertEqual((s["events"], s["clips_with_events"]), (2, 2))
        a = s["linear"]["all"]
        self.assertEqual(a["points"], 42)
        self.assertAlmostEqual(a["point_mean"], (21 + 40) / 42)
        self.assertAlmostEqual(a["point_median"], 1.0)                 # sorted: 0, 1 x21, 2 x20
        self.assertAlmostEqual(a["clip_mean_mean"], (1 + 40 / 21) / 2)
        self.assertEqual(a["clips"], 2)
        self.assertAlmostEqual(s["linear"]["wrist"]["point_mean"], 0.5)
        self.assertEqual((s["linear"]["wrist"]["points"], s["linear"]["palm"]["points"],
                          s["linear"]["fingers"]["points"]), (2, 10, 30))
        self.assertAlmostEqual(s["linear"]["palm"]["point_mean"], 1.5)
        self.assertAlmostEqual(s["previous"]["all"]["point_p95"], 2.5)
        self.assertTrue(RH.coord_gate(s))
        rows[0][0]["linear"] = [9.0] * 21
        self.assertFalse(RH.coord_gate(RH.coord_summary(rows)))
        empty = RH.coord_summary([[], []])
        self.assertFalse(empty["available"])
        self.assertIsNone(empty["linear"]["all"])
        self.assertFalse(RH.coord_gate(empty))
        json.dumps(s, allow_nan=False)

    def test_cache_plan(self):
        self.assertEqual(RH.cache_plan({"original": "a", "original_repair": "a", "masked": "b", "masked_repair": "b"}),
                         {"original": "saved_original", "original_repair": "reuse:original", "masked": "new",
                          "masked_repair": "reuse:masked"})
        self.assertEqual(RH.cache_plan({"original": "a", "original_repair": "b", "masked": "c", "masked_repair": "b"}),
                         {"original": "saved_original", "original_repair": "new", "masked": "new",
                          "masked_repair": "reuse:original_repair"})

    def _cfg(self, lab, code_dir, out):
        p = lambda *a: os.path.join(lab, *a)                          # noqa: E731
        cfg = {"lab_root": lab, "pilot_src": p("pilot", "src"), "loader_dir": p("ops", "loader"),
               "helper_dir": p("ops", "timegrid_audit"), "native_metadata": p("data", "meta.json"),
               "mp_manifest": p("data", "mp.json"), "saved_native_logits": p("ops", "saved", "n.npz"),
               "saved_mp_logits": p("ops", "saved", "m.npz"), "checkpoint": p("ckpt", "model.pt"),
               "plan_file": p("ops", "unit", "FROZEN_PLAN.json"), "output_dir": out,
               "native_roots": [p("data", "native")], "mp_roots": [p("data", "mp")], "threads": 2,
               "tolerance": 1e-5, "repro_indices": {"native": [0, 349, 698], "mediapipe": [0, 56, 112]}}
        need = [cfg[k] for k in RH.PINNED_FIELDS] + [os.path.join(cfg["helper_dir"], f) for f in RH.HELPER_FILES] \
            + [os.path.join(code_dir, "handgap.py")]
        cfg["pins"] = {q: "a" * 64 for q in need}
        return cfg

    def test_output_guard(self):
        lab = os.path.abspath(os.sep + "lab_guard")
        code = os.path.join(lab, "ops", "unit", "claude_code_only")
        ok = os.path.join(lab, "ops", "unit", "result")                # sibling result/ is valid
        RH.precheck(self._cfg(lab, code, ok), code)
        for out in (os.path.join(lab, "ops"), os.path.join(lab, "result"), code, os.path.join(code, "result"),
                    os.path.join(lab, "ops", "timegrid_audit", "x"), os.path.join(lab, "ops", "saved", "x"),
                    os.path.join(lab, "ops", "loader", "x"), "relative/out"):
            with self.assertRaises(RH.RunError, msg=out):
                RH.precheck(self._cfg(lab, code, out), code)
        for mutate in ("pin", "repro", "threads", "tolerance", "pinhex"):
            cfg = self._cfg(lab, code, ok)
            if mutate == "pin":
                del cfg["pins"][os.path.join(code, "handgap.py")]
            elif mutate == "repro":
                cfg["repro_indices"]["native"] = [0, 1, 2]
            elif mutate == "threads":
                cfg["threads"] = 4
            elif mutate == "tolerance":
                cfg["tolerance"] = 1e-4
            else:
                cfg["pins"][cfg["checkpoint"]] = "Z" * 64
            with self.assertRaises(RH.RunError, msg=mutate):
                RH.precheck(cfg, code)

    def test_main_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab")
            out = os.path.join(lab, "ops", "unit", "result")
            os.makedirs(out)
            cfg = self._cfg(lab, RH.CODE_DIR, out)
            cp = os.path.join(tmp, "cfg.json")
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            with self.assertRaises(RH.RunError):
                RH.main(["--config", cp])
            self.assertEqual(os.listdir(out), [])
            with self.assertRaises(RH.RunError):
                RH.main(["--config", "relative.json"])


if __name__ == "__main__":
    unittest.main()
