"""Synthetic unittest for timegrid.restore and the runner's comparison arithmetic. No real data.

Run: python -m unittest test_timegrid      (or: python test_timegrid.py)
"""
from __future__ import annotations

import inspect
import os
import sys
import unittest
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timegrid as TG  # noqa: E402
import run_audit as RA  # noqa: E402

P = 1e6 / 30.0
J = TG.N_JOINTS


def nominal_ts(n, start=0):
    return np.array([start + int(round(k * P)) for k in range(n)], dtype=np.int64)


def clip(ts, seed=0, velocity=None):
    """[T,67,2] float32; constant velocity in px/s when given, else random."""
    t = len(ts)
    rng = np.random.default_rng(seed)
    if velocity is None:
        xy = rng.uniform(10, 500, size=(t, J, 2))
    else:
        base = rng.uniform(50, 150, size=(J, 2))
        xy = base[None] + velocity * (np.asarray(ts, np.float64)[:, None, None] / 1e6)
    return xy.astype(np.float32), np.ones((t, J), dtype=bool)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


class IdentityTests(unittest.TestCase):
    def test_signature_has_no_original_argument(self):
        self.assertEqual(list(inspect.signature(TG.restore).parameters), ["xy", "present", "ts", "method"])

    def test_30hz_identity_copies(self):
        ts = nominal_ts(10)
        xy, pr = clip(ts)
        pr[3, 7] = False
        xy[3, 7] = np.nan
        for m in TG.METHODS:
            r = TG.restore(xy, pr, ts, m)
            self.assertTrue(r["identity"])
            self.assertTrue(same(r["xy"], xy) and same(r["present"], pr) and same(r["ts"], ts))
            self.assertTrue(np.array_equal(r["observed"], pr))
            self.assertFalse(r["estimated"].any())
            self.assertTrue(np.array_equal(r["exact_source_index"], np.arange(10)))
            for k in ("xy", "present", "ts"):
                self.assertFalse(np.shares_memory(r[k], {"xy": xy, "present": pr, "ts": ts}[k]))

    def test_clock_tolerance_2us(self):
        ts = np.arange(10, dtype=np.int64) * 33335                 # |d-P| = 1.67 us
        xy, pr = clip(ts)
        self.assertTrue(TG.restore(xy, pr, ts)["identity"])
        ts = np.arange(10, dtype=np.int64) * 33336                 # |d-P| = 2.67 us
        r = TG.restore(xy, pr, ts)
        self.assertFalse(r["identity"])
        self.assertTrue(np.array_equal(r["exact_source_index"], np.arange(10)))   # all exact copies
        self.assertFalse(r["estimated"].any())
        self.assertTrue(np.array_equal(r["ts"], ts))

    def test_one_frame_identity(self):
        ts = np.array([123], dtype=np.int64)
        xy, pr = clip(ts)
        r = TG.restore(xy, pr, ts, "linear")
        self.assertTrue(r["identity"] and same(r["xy"], xy) and same(r["ts"], ts))


class GridTests(unittest.TestCase):
    def test_odd_original_decimated(self):
        ts = nominal_ts(9)
        xy, pr = clip(ts)
        r = TG.restore(xy[::2], pr[::2], ts[::2], "linear")
        self.assertFalse(r["identity"])
        self.assertEqual(len(r["ts"]), 9)
        self.assertTrue(np.all(np.abs(r["ts"] - ts) <= 1))
        self.assertTrue(np.array_equal(r["exact_source_index"], [0, -1, 1, -1, 2, -1, 3, -1, 4]))
        self.assertEqual(int(r["ts"][0]), int(ts[0]))
        self.assertEqual(int(r["ts"][-1]), int(ts[8]))

    def test_even_original_ends_at_last_retained(self):
        ts = nominal_ts(10)
        xy, pr = clip(ts)
        ts2 = ts[::2]
        r = TG.restore(xy[::2], pr[::2], ts2, "previous")
        self.assertEqual(len(r["ts"]), 9)                         # dropped final frame NOT restored
        self.assertEqual(int(r["ts"][-1]), int(ts2[-1]))
        self.assertLess(int(r["ts"][-1]), int(ts[-1]))

    def test_15hz_frame_count(self):
        ts = nominal_ts(61)
        xy, pr = clip(ts)
        self.assertEqual(len(TG.restore(xy[::2], pr[::2], ts[::2])["ts"]), 61)

    def test_endpoints_exact_for_odd_span(self):
        ts = np.array([5, 70001, 133338, 200003], dtype=np.int64)
        xy, pr = clip(ts)
        r = TG.restore(xy, pr, ts)
        self.assertEqual(int(r["ts"][0]), 5)
        self.assertEqual(int(r["ts"][-1]), 200003)
        self.assertTrue(np.all(np.diff(r["ts"]) > 0))

    def test_grid_times_formula(self):
        self.assertEqual(TG.grid_times(0, 200000), [0, 33333, 66667, 100000, 133333, 166667, 200000])
        self.assertEqual(TG.grid_times(0, 1), [0, 1])

    def test_round_half_even_matches_python_round(self):
        for den in (1, 2, 3, 4, 8, 1_000_000):
            for num in range(-3 * den - 2, 3 * den + 3):
                self.assertEqual(TG._round_half_even(num, den), round(Fraction(num, den)), (num, den))

    def test_nout_half_ties_to_even(self):
        self.assertEqual(TG.grid_times(0, 150000), [0, 37500, 75000, 112500, 150000])   # 4.5 -> 4
        self.assertEqual(len(TG.grid_times(0, 250000)), 9)                              # 7.5 -> 8
        self.assertEqual(len(TG.grid_times(7, 150007)), 5)                              # origin-independent

    def test_absolute_grid_half_ties(self):
        self.assertEqual(TG.grid_times(1, 66668), [1, 33334, 66668])     # 33334.5 -> 33334 (not first+33334)
        self.assertEqual(TG.grid_times(0, 66667), [0, 33334, 66667])     # 33333.5 -> 33334
        self.assertEqual(TG.grid_times(-66667, 0), [-66667, -33334, 0])  # -33333.5 -> -33334
        self.assertEqual(TG.grid_times(-66668, -1), [-66668, -33334, -1])  # -33334.5 -> -33334

    def test_grid_times_matches_fraction_reference(self):
        for first in (-66668, -1, 0, 1, 5, 33333):
            for span in (1, 66667, 66669, 133333, 150000, 200001, 266667):
                last = first + span
                nout = max(2, round(Fraction(span * 30, 1_000_000)) + 1)
                ref = [round(first + Fraction(span * k, nout - 1)) for k in range(nout)]
                self.assertEqual(TG.grid_times(first, last), ref, (first, last))

    def test_restore_uses_absolute_tie_rounding(self):
        ts = np.array([1, 66668], dtype=np.int64)
        xy, pr = clip(ts, seed=7)
        r = TG.restore(xy, pr, ts, "linear")
        self.assertTrue(np.array_equal(r["ts"], [1, 33334, 66668]))
        self.assertTrue(np.array_equal(r["exact_source_index"], [0, -1, 1]))
        w = (33334 - 1) / 66667
        exp = ((1 - w) * xy[0].astype(np.float64) + w * xy[1].astype(np.float64)).astype(np.float32)
        self.assertTrue(np.array_equal(r["xy"][1], exp))


class ValueTests(unittest.TestCase):
    def test_constant_velocity_linear_vs_hold(self):
        ts = nominal_ts(9)
        xy, pr = clip(ts, velocity=300.0)
        xy2, pr2, ts2 = xy[::2], pr[::2], ts[::2]
        lin = TG.restore(xy2, pr2, ts2, "linear")
        prev = TG.restore(xy2, pr2, ts2, "previous")
        odd = np.arange(1, 9, 2)
        np.testing.assert_allclose(lin["xy"][odd], xy[odd], atol=1e-3)     # recovers the line
        for k in odd:
            self.assertTrue(np.array_equal(prev["xy"][k], xy2[(k - 1) // 2]))   # holds left exactly
        self.assertTrue(np.all(np.abs(lin["xy"][odd] - prev["xy"][odd]) > 5.0))
        for name in ("ts", "present", "observed", "estimated", "exact_source_index"):
            self.assertTrue(np.array_equal(lin[name], prev[name]), name)
        even = np.arange(0, 9, 2)
        self.assertTrue(np.array_equal(lin["xy"][even], prev["xy"][even]))
        self.assertEqual(lin["xy"].dtype, np.float32)

    def test_exact_value_preservation(self):
        ts = nominal_ts(7)
        xy, pr = clip(ts, seed=3)
        pr[2, 10] = False
        xy[2, 10] = np.nan                                         # absent joint on a retained frame
        r = TG.restore(xy[::2], pr[::2], ts[::2], "linear")
        for k, s in ((0, 0), (2, 1), (4, 2), (6, 3)):
            self.assertTrue(same(r["xy"][k], xy[::2][s]))
            self.assertTrue(np.array_equal(r["present"][k], pr[::2][s]))
        self.assertFalse(r["present"][2, 10])

    def test_irregular_ts(self):
        ts = np.array([0, 50000, 100000, 166667, 200000], dtype=np.int64)
        xy, pr = clip(ts, seed=1)
        r = TG.restore(xy, pr, ts, "linear")
        self.assertTrue(np.array_equal(r["ts"], [0, 33333, 66667, 100000, 133333, 166667, 200000]))
        self.assertTrue(np.array_equal(r["exact_source_index"], [0, -1, -1, 2, -1, 3, 4]))
        w = 33333 / 50000
        exp = ((1 - w) * xy[0].astype(np.float64) + w * xy[1].astype(np.float64)).astype(np.float32)
        self.assertTrue(np.array_equal(r["xy"][1], exp))
        self.assertTrue(r["estimated"][4].all())                  # gap 66667 <= 2P+2

    def test_long_gap_unsupported(self):
        ts = np.array([0, 100000, 200000], dtype=np.int64)
        xy, pr = clip(ts)
        for m in TG.METHODS:
            r = TG.restore(xy, pr, ts, m)
            interior = r["exact_source_index"] < 0
            self.assertEqual(int(interior.sum()), 4)
            self.assertFalse(r["present"][interior].any())
            self.assertFalse(r["estimated"].any())
            self.assertFalse(r["xy"][interior].any())

    def test_gap_boundary(self):
        ok = np.array([0, 66668, 133336], dtype=np.int64)         # 66668 <= 66668.67
        bad = np.array([0, 66669, 133338], dtype=np.int64)
        xy, pr = clip(ok)
        self.assertTrue(TG.restore(xy, pr, ok)["estimated"][1].all())
        self.assertFalse(TG.restore(xy, pr, bad)["estimated"].any())

    def test_missing_bracket_and_endpoint(self):
        ts = nominal_ts(7)
        xy, pr = clip(ts, seed=2)
        xy2, pr2, ts2 = xy[::2].copy(), pr[::2].copy(), ts[::2]
        pr2[1, 5] = False                                          # missing at an interior source
        xy2[1, 5] = np.nan
        pr2[0, 0] = False                                          # missing at the first endpoint
        xy2[0, 0] = np.nan
        for m in TG.METHODS:
            r = TG.restore(xy2, pr2, ts2, m)
            self.assertFalse(r["present"][1, 5] or r["present"][3, 5])   # both sides of source 1
            self.assertTrue(r["present"][5, 5])
            self.assertFalse(r["present"][0, 0] or r["present"][1, 0])
            self.assertTrue(np.isnan(r["xy"][0, 0]).all())         # verbatim copy of the endpoint
            self.assertTrue(np.all(r["xy"][1, 0] == 0) and np.all(r["xy"][1, 5] == 0))

    def test_provenance(self):
        ts = nominal_ts(9)
        xy, pr = clip(ts, seed=4)
        pr[2, 3] = False
        r = TG.restore(xy[::2], pr[::2], ts[::2], "linear")
        obs, est, pb = r["observed"], r["estimated"], r["present"].astype(bool)
        exact = r["exact_source_index"] >= 0
        self.assertFalse((obs & est).any())
        self.assertTrue(np.array_equal(obs | est, pb))
        self.assertFalse(obs[~exact].any())
        self.assertFalse(est[exact].any())


class ContractTests(unittest.TestCase):
    def test_no_original_access(self):
        ts = nominal_ts(9)
        a, pr = clip(ts, seed=5)
        b = a.copy()
        b[1::2] += 1000.0                                          # differs only on dropped frames
        for m in TG.METHODS:
            ra, rb = TG.restore(a[::2], pr[::2], ts[::2], m), TG.restore(b[::2], pr[::2], ts[::2], m)
            for k in ("xy", "present", "ts", "observed", "estimated", "exact_source_index"):
                self.assertTrue(np.array_equal(ra[k], rb[k]), k)

    def test_nonmutating(self):
        ts = nominal_ts(9)[::2].copy()
        xy, pr = clip(ts, seed=6)
        pr[1, 2] = False
        snap = (xy.copy(), pr.copy(), ts.copy())
        for m in TG.METHODS:
            r = TG.restore(xy, pr, ts, m)
            self.assertTrue(same(xy, snap[0]) and same(pr, snap[1]) and same(ts, snap[2]))
            for k in ("xy", "present", "ts"):
                self.assertFalse(np.shares_memory(r[k], {"xy": xy, "present": pr, "ts": ts}[k]))

    def test_malformed_masks(self):
        ts = nominal_ts(5)[::2].copy()
        xy, pr = clip(ts)
        for bad in (np.full((3, J), 2), np.full((3, J), np.nan), np.ones((3, J - 1), bool),
                    np.full((3, J), 0.5)):
            with self.assertRaises(ValueError):
                TG.restore(xy, bad, ts)
        r = TG.restore(xy, pr.astype(np.float32), ts)              # exact 0/1 accepted
        self.assertEqual(r["present"].dtype, np.float32)

    def test_malformed_timestamps(self):
        xy, pr = clip(np.arange(3))
        for bad in (np.array([0.0, 66667.0, 133333.0]), np.array([0, 66667, 66667]),
                    np.array([0, 66667, 10]), np.array([0, 66667]), np.array([True, False, True])):
            with self.assertRaises(ValueError):
                TG.restore(xy, pr, bad)

    def test_malformed_coordinates_and_method(self):
        ts = np.array([0, 66667, 133333], dtype=np.int64)
        xy, pr = clip(ts)
        with self.assertRaises(ValueError):
            TG.restore(xy[:, :66], pr[:, :66], ts)
        with self.assertRaises(ValueError):
            TG.restore(xy.astype(np.int32), pr, ts)
        bad = xy.copy()
        bad[1, 4, 0] = np.inf
        with self.assertRaises(ValueError):
            TG.restore(bad, pr, ts)
        pr2 = pr.copy()
        pr2[1, 4] = False
        TG.restore(bad, pr2, ts)                                   # non-finite where absent is fine
        with self.assertRaises(ValueError):
            TG.restore(xy, pr, ts, "cubic")


def arm_row(i, signer, preds, true="A"):
    return {"id": f"r{i}", "signer_id": signer, "true": true,
            "arms": {a: {"pred": preds.get(a, "B"), "correct": preds.get(a, "B") == true,
                         "rank": 1 if preds.get(a, "B") == true else 2} for a in RA.ARMS}}


class ArithmeticTests(unittest.TestCase):
    def test_compare_partition(self):
        rows = [arm_row(0, 1, {"sparse": "A", "linear": "A"}),     # correct -> correct
                arm_row(1, 1, {"sparse": "B", "linear": "A"}),     # gain
                arm_row(2, 2, {"sparse": "A", "linear": "C"}),     # loss
                arm_row(3, 2, {"sparse": "B", "linear": "C"}),     # wrong -> wrong changed
                arm_row(4, 2, {"sparse": "B", "linear": "B"})]     # wrong -> wrong same
        c = RA.compare_arms(rows, "sparse", "linear")
        self.assertEqual((c["net"], c["gains_n"], c["losses_n"], c["wrong_to_wrong_changed_n"],
                          c["wrong_to_wrong_same"], c["correct_to_correct"], c["top1_changed"]),
                         (0, 1, 1, 1, 1, 1, 3))
        self.assertEqual([g["id"] for g in c["gains"]], ["r1"])
        self.assertEqual([g["id"] for g in c["losses"]], ["r2"])

    def test_native_gate(self):
        rows = [arm_row(i, 11 + i % 3, {"original": "A", "linear": "A"}) for i in range(7)]
        rows.append(arm_row(7, 11, {"original": "A", "sparse": "A"}))         # one newly wrong row
        rows += [arm_row(8 + i, 11 + i % 3, {"original": "A", "sparse": "A", "linear": "A"}) for i in range(3)]
        comps = {f"{s}->{t}": RA.compare_arms(rows, s, t) for s, t in RA.COMPARISONS + RA.CONTEXT_COMPARISONS}
        oc = RA.outcome_checks("native", rows, comps, len(rows), 0)
        self.assertEqual(oc["linear_net_vs_sparse"], 6)
        self.assertFalse(oc["gate_net_gain_ge_7"])
        self.assertFalse(oc["primary_pass"])
        self.assertEqual(oc["newly_wrong_sparse_correct_linear_wrong"], ["r7"])
        rows.append(arm_row(11, 12, {"linear": "A"}))
        comps = {f"{s}->{t}": RA.compare_arms(rows, s, t) for s, t in RA.COMPARISONS + RA.CONTEXT_COMPARISONS}
        oc = RA.outcome_checks("native", rows, comps, len(rows), 0)
        self.assertTrue(oc["gate_net_gain_ge_7"] and oc["all_signers_nondecrease"] and oc["primary_pass"])
        self.assertFalse(RA.outcome_checks("native", rows, comps, len(rows) - 1, 0)["primary_pass"])

    def test_recoveries_and_recording_id(self):
        rows = [arm_row(0, 1, {"original": "A", "linear": "A"}), arm_row(1, 1, {"original": "A"})]
        rec = RA.recoveries(rows)
        self.assertEqual((rec["n"], rec["correct_in_linear"], rec["correct_in_previous"]), (2, 1, 0))
        self.assertEqual(RA.recording_id("S1_word_a03_012"), "S1_word")
        self.assertEqual(RA.recording_id("S1_word"), "S1_word")

    def test_score_row(self):
        s = RA.score_row(np.array([1.0, 3.0, 2.0]), ["A", "B", "C"], "C")
        self.assertEqual((s["pred"], s["correct"], s["rank"], s["margin"]), ("B", False, 2, -1.0))


if __name__ == "__main__":
    unittest.main()
