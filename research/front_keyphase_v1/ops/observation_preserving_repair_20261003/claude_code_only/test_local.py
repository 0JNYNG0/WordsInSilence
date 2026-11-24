"""Synthetic tests for local_insert.restore and the run_local guards (no data, model or prior helpers).

Run: python -m pytest -q test_local.py     (or: python -m unittest test_local)
"""
from __future__ import annotations

import copy
import os
import tempfile
import unittest
from fractions import Fraction

import numpy as np

import local_insert as LI
import run_local as RL

J = 67


def clip(ts, dtype=np.float32, seed=0, ts_dtype=np.int64):
    t = len(ts)
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, (t, J, 2)).astype(dtype)
    return xy, np.ones((t, J), dtype=bool), np.asarray(ts, dtype=ts_dtype)


def cumulative(gaps, start=0):
    out = [start]
    for g in gaps:
        out.append(out[-1] + g)
    return out


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        np.array_equal(a, b, equal_nan=bool(np.issubdtype(a.dtype, np.floating))))


class Identity(unittest.TestCase):
    def assert_identity(self, xy, pr, ts):
        out = LI.restore(xy, pr, ts)
        self.assertIs(out["identity"], True)
        self.assertTrue(same(out["xy"], xy) and same(out["present"], pr) and same(out["ts"], ts))
        self.assertTrue(np.array_equal(out["observed"], np.asarray(pr).astype(bool)))
        self.assertFalse(out["estimated"].any())
        self.assertTrue(np.array_equal(out["exact_source_index"], np.arange(len(ts))))
        for k in ("xy", "present", "ts"):
            self.assertFalse(np.shares_memory(out[k], {"xy": xy, "present": pr, "ts": ts}[k]))
        RL.verify_insertion(out, xy, pr, ts)
        return out

    def test_clean_30hz(self):
        self.assert_identity(*clip([k * 1_000_000 // 30 for k in range(40)]))

    def test_timestamp_jitter_and_just_below_window(self):
        # clock perturbation (~33 ms +/- 2 ms) and 49_999 us (< 1.5P) stay unchanged
        self.assert_identity(*clip(cumulative([31_333, 35_334, 33_333, 35_333, 31_334, 49_999])))

    def test_long_gap_unchanged(self):
        self.assert_identity(*clip([0, 100_000, 133_333]))
        self.assert_identity(*clip([0, 66_669]))

    def test_singleton(self):
        self.assert_identity(*clip([123_456_789]))

    def test_no_shared_joint_no_insert(self):
        xy, pr, ts = clip([0, 66_666])
        pr[0, 30:] = False
        pr[1, :30] = False
        self.assert_identity(xy, pr, ts)


class Insertion(unittest.TestCase):
    def test_alternating_33_67(self):
        xy, pr, ts = clip([0, 33_333, 100_000, 133_333, 200_000])
        out = LI.restore(xy, pr, ts)
        self.assertIs(out["identity"], False)
        self.assertEqual(out["ts"].tolist(), [0, 33_333, 66_666, 100_000, 133_333, 166_666, 200_000])  # .5 ties even
        self.assertEqual(out["exact_source_index"].tolist(), [0, 1, -1, 2, 3, -1, 4])
        self.assertTrue(out["estimated"][[2, 5]].all() and not out["observed"][[2, 5]].any())
        self.assertFalse(out["estimated"][[0, 1, 3, 4, 6]].any())
        RL.verify_insertion(out, xy, pr, ts)

    def test_only_short_gap_after_long_gap(self):
        out = LI.restore(*clip([0, 100_000, 166_667]))
        self.assertEqual(out["ts"].tolist(), [0, 100_000, 133_334, 166_667])     # 133_333.5 -> even 133_334
        self.assertEqual(out["exact_source_index"].tolist(), [0, 1, -1, 2])

    def test_window_boundaries(self):
        for gap, inserted in ((49_999, False), (50_000, True), (66_668, True), (66_669, False)):
            out = LI.restore(*clip([0, gap]))
            self.assertEqual(len(out["ts"]), 3 if inserted else 2, gap)
            self.assertIs(out["identity"], not inserted)

    def test_midpoint_ties_negative_and_large_clocks(self):
        top, bottom = 2 ** 63 - 1, -2 ** 63
        cases = [((0, 50_001), 25_000), ((1, 50_002), 25_002), ((-50_001, 0), -25_000), ((-50_003, 0), -25_002),
                 ((top - 60_001, top), 9_223_372_036_854_745_806),
                 ((bottom, bottom + 50_001), -9_223_372_036_854_750_808)]
        for (t0, t1), expect in cases:
            xy, pr, ts = clip([t0, t1])
            out = LI.restore(xy, pr, ts)
            self.assertEqual(out["ts"].dtype, np.int64)
            self.assertEqual(int(out["ts"][1]), expect, (t0, t1))
            self.assertEqual(int(out["ts"][1]), round(Fraction(t0 + t1, 2)))
            self.assertEqual([int(out["ts"][0]), int(out["ts"][2])], [t0, t1])
            RL.verify_insertion(out, xy, pr, ts)

    def test_analytic_ramp_and_multiple_gaps_nonrecursive(self):
        ts_list = [0, 33_333, 100_000, 150_001, 200_000, 266_668, 300_000, 433_334, 466_667]
        # gaps 33333 no, 66667 yes, 50001 yes, 49999 no, 66668 yes, 33332 no, 133334 no (long), 33333 no
        ts = np.asarray(ts_list, dtype=np.int64)
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(J, 2)), rng.normal(size=(J, 2))
        ramp = lambda t: a + b * (t * 1e-6)
        xy = np.stack([ramp(t) for t in ts_list]).astype(np.float64)
        pr = np.ones((len(ts_list), J), dtype=bool)
        out = LI.restore(xy, pr, ts)
        src = out["exact_source_index"]
        self.assertEqual(src.tolist(), [0, 1, -1, 2, -1, 3, 4, -1, 5, 6, 7, 8])
        ins = np.flatnonzero(src < 0)
        self.assertFalse(np.any(np.diff(ins) == 1))                            # never adjacent: no recursion
        for p in ins:
            mid = int(out["ts"][p])
            self.assertEqual(mid, round(Fraction(int(out["ts"][p - 1]) + int(out["ts"][p + 1]), 2)))
            np.testing.assert_allclose(out["xy"][p], ramp(mid), rtol=1e-12, atol=1e-12)
        out32 = LI.restore(xy.astype(np.float32), pr, ts)
        self.assertEqual(out32["xy"].dtype, np.float32)
        for p in ins:
            np.testing.assert_allclose(out32["xy"][p], ramp(int(out32["ts"][p])), rtol=1e-6, atol=1e-6)
        RL.verify_insertion(out, xy, pr, ts)

    def test_partial_joint_masks(self):
        xy, pr, ts = clip([0, 66_666])
        pr[0, 40:] = False
        pr[1, :20] = False
        out = LI.restore(xy, pr, ts)
        shared = np.zeros(J, dtype=bool)
        shared[20:40] = True
        self.assertTrue(np.array_equal(out["estimated"][1], shared))
        self.assertTrue(np.array_equal(out["present"][1], shared))
        self.assertFalse(out["observed"][1].any())
        self.assertTrue(np.all(out["xy"][1][~shared] == 0))
        np.testing.assert_allclose(out["xy"][1][shared], 0.5 * (xy[0][shared].astype(np.float64)
                                                               + xy[1][shared]), rtol=1e-6, atol=1e-6)
        self.assertTrue(np.array_equal(out["observed"][[0, 2]], pr))
        RL.verify_insertion(out, xy, pr, ts)

    def test_absent_nan_retained(self):
        xy, pr, ts = clip([0, 33_333, 100_000])
        pr[:, 60:] = False
        xy[:, 60:] = np.nan
        out = LI.restore(xy, pr, ts)
        pos = out["exact_source_index"] >= 0
        self.assertTrue(same(out["xy"][pos], xy))
        self.assertTrue(np.all(out["xy"][~pos][:, 60:] == 0) and np.isfinite(out["xy"][~pos]).all())
        RL.verify_insertion(out, xy, pr, ts)

    def test_dtypes_preserved(self):
        for pdt in (np.int8, np.float32, np.bool_):
            xy, pr, ts = clip([0, 66_666, 99_999], dtype=np.float64, ts_dtype=np.int32)
            pr = pr.astype(pdt)
            out = LI.restore(xy, pr, ts)
            self.assertEqual((out["xy"].dtype, out["present"].dtype, out["ts"].dtype),
                             (np.dtype(np.float64), np.dtype(pdt), np.dtype(np.int32)))
            self.assertTrue(np.all(out["present"][1] == 1))
            RL.verify_insertion(out, xy, pr, ts)

    def test_inputs_unchanged_and_not_aliased(self):
        for ts_list in ([0, 33_333, 66_667], [0, 66_666, 100_000]):
            xy, pr, ts = clip(ts_list)
            keep = (xy.copy(), pr.copy(), ts.copy())
            out = LI.restore(xy, pr, ts)
            for o in out.values():
                if isinstance(o, np.ndarray):
                    self.assertFalse(any(np.shares_memory(o, s) for s in (xy, pr, ts)))
            out["xy"][...] = 7
            out["present"][...] = False
            out["ts"][...] = 0
            self.assertTrue(all(same(a, b) for a, b in zip(keep, (xy, pr, ts))))


class Invalid(unittest.TestCase):
    def test_rejected(self):
        xy, pr, ts = clip([0, 33_333, 66_667])
        bad_nan = xy.copy()
        bad_nan[1, 5, 0] = np.nan
        bad_inf = xy.copy()
        bad_inf[2, 0, 1] = np.inf
        pr2, prn = pr.astype(np.int8), pr.astype(np.float32)
        pr2[0, 0] = 2
        prn[0, 0] = np.nan
        for args in ((xy[:, :66], pr[:, :66], ts), (xy.astype(np.int32), pr, ts), (xy[:0], pr[:0], ts[:0]),
                     (xy, pr[:2], ts), (xy, pr[:, :66], ts), (xy, pr2, ts), (xy, prn, ts),
                     (xy, pr.astype(object), ts), (xy, pr, ts.astype(np.float64)), (xy, pr, ts.astype(bool)),
                     (xy, pr, np.array([0, 33_333, 33_333])), (xy, pr, np.array([0, 66_667, 33_333])),
                     (xy, pr, ts[:2]), (bad_nan, pr, ts), (bad_inf, pr, ts)):
            with self.assertRaises(ValueError):
                LI.restore(*args)

    def test_nonfinite_where_absent_is_allowed(self):
        xy, pr, ts = clip([0, 33_333])
        xy[0, 5] = np.nan
        pr[0, 5] = False
        LI.restore(xy, pr, ts)


class RunnerGuards(unittest.TestCase):
    LAB = os.path.join(tempfile.gettempdir(), "wis_guard_lab")

    def cfg(self, out, lab=None):
        lab = lab or self.LAB
        p = lambda *a: os.path.join(lab, "ops", *a)
        c = {"helper_dir": p("old_helper"), "irregular_dir": p("old_irregular"),
             "prior_result": p("prior_out", "RESULT.json"), "regular_result": p("regular_out", "RESULT.json"),
             "prior_scores": {g: p("prior_out", f"SCORES_{g}.npz") for g in RL.GROUPS},
             "regular_scores": {g: p("regular_out", f"SCORES_{g}.npz") for g in RL.GROUPS},
             "plan_file": p("plan", "PLAN.md"), "lab_root": lab, "output_dir": out,
             "native_roots": [p("native_data")], "mp_roots": [os.path.join(lab, "mp_data")],
             "repro_indices": copy.deepcopy(RL.REPRO_INDICES)}
        need = [os.path.join(c["helper_dir"], f) for f in RL.HELPER_FILES] + \
               [os.path.join(c["irregular_dir"], f) for f in RL.IRREGULAR_FILES] + \
               [c["prior_result"], c["regular_result"], c["plan_file"], *c["prior_scores"].values(),
                *c["regular_scores"].values(), os.path.join(RL.CODE_DIR, "local_insert.py")]
        c["pins"] = {x: "0" * 64 for x in need}
        return c

    def test_fresh_ops_subdirectory_accepted(self):
        RL.precheck(self.cfg(os.path.join(self.LAB, "ops", "local_insert_run_1")))

    def test_protected_output_paths_rejected(self):
        p = lambda *a: os.path.join(self.LAB, *a)
        for out in (p("ops"), p("results", "x"), p("src", "x"), p("ops", "old_helper", "x"),
                    p("ops", "old_irregular", "x"), p("ops", "prior_out", "x"), p("ops", "regular_out"),
                    p("ops", "native_data", "x")):
            with self.assertRaises(RL.RunError, msg=out):
                RL.precheck(self.cfg(out))

    def test_output_inside_code_dir_rejected(self):
        lab = os.path.dirname(os.path.dirname(os.path.dirname(RL.CODE_DIR)))
        RL.precheck(self.cfg(os.path.join(os.path.dirname(RL.CODE_DIR), "run_x"), lab))
        with self.assertRaises(RL.RunError):
            RL.precheck(self.cfg(os.path.join(RL.CODE_DIR, "run_x"), lab))

    def test_unpinned_candidate_or_scores_rejected(self):
        out = os.path.join(self.LAB, "ops", "run_2")
        for drop in (os.path.join(RL.CODE_DIR, "local_insert.py"),
                     os.path.join(self.LAB, "ops", "regular_out", "SCORES_mediapipe.npz")):
            c = self.cfg(out)
            del c["pins"][drop]
            with self.assertRaises(RL.RunError):
                RL.precheck(c)
        c = self.cfg(out)
        c["repro_indices"]["native"] = [0, 1, 2]
        with self.assertRaises(RL.RunError):
            RL.precheck(c)

    def test_cache_only_on_identical_input_with_fixed_order(self):
        h = {"original": "A", "unrestored": "U", "global_linear": "G", "local_insert": "X"}
        self.assertEqual(RL.choose_cache("U", h), "unrestored")
        self.assertEqual(RL.choose_cache("G", h), "global_linear")
        self.assertEqual(RL.choose_cache("A", h), "original")
        self.assertIsNone(RL.choose_cache("X", h))                         # never its own entry
        self.assertIsNone(RL.choose_cache("Z", h))
        self.assertEqual(RL.choose_cache("S", {"original": "S", "unrestored": "S", "global_linear": "S"}),
                         "unrestored")
        self.assertEqual(RL.choose_cache("S", {"original": "S", "unrestored": "U", "global_linear": "S"}),
                         "global_linear")

    def test_verify_rejects_tampering(self):
        xy, pr, ts = clip([0, 33_333, 100_000])
        good = LI.restore(xy, pr, ts)
        tampers = []
        t = copy.deepcopy(good); t["xy"][0, 0, 0] += 1; tampers.append((t, ts))           # source changed
        t = copy.deepcopy(good); t["ts"][2] += 1; tampers.append((t, ts))                 # midpoint moved
        t = copy.deepcopy(good); t["observed"][2] = t["estimated"][2]; tampers.append((t, ts))  # estimate as observed
        t = copy.deepcopy(good); t["identity"] = True; tampers.append((t, ts))
        missing = {"xy": xy.copy(), "present": pr.copy(), "ts": ts.copy(), "observed": pr.copy(),
                   "estimated": np.zeros_like(pr), "exact_source_index": np.arange(3, dtype=np.int64),
                   "identity": True}
        tampers.append((missing, ts))                                                    # eligible gap skipped
        long_ = LI.restore(xy[:2], pr[:2], np.array([0, 66_666], dtype=np.int64))
        long_["ts"] = np.array([0, 50_000, 100_000], dtype=np.int64)
        tampers.append((long_, np.array([0, 100_000], dtype=np.int64)))                  # insertion in long gap
        for k, (li, tsk) in enumerate(tampers):
            n = len(tsk)
            with self.assertRaises(RL.RunError, msg=k):
                RL.verify_insertion(li, xy[:n], pr[:n], tsk)


if __name__ == "__main__":
    unittest.main()
