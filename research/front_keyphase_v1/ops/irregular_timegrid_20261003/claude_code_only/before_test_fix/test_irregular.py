"""Synthetic checks only: no real data, model, helper, torch or camera import.

Run from this directory: python -m unittest -v test_irregular
"""
from __future__ import annotations

import hashlib
import inspect
import sys
import unittest

import numpy as np

import corruptions as C
import irregular_audit as A
import timegrid_reference as TGR      # pure NumPy copy of the restorer API, smoke tests only

DROPS = ("isolated_drop", "burst_drop")


def D(sid, case, i):
    return hashlib.sha256(f"WIS_TIMEGRID_IRREG_V1|{sid}|{case}|{i}".encode("utf-8")).digest()


def expected_indices(t, sid, case):
    if case == "clock_noise":
        return list(range(t))
    out, cur = [0], 0
    while cur < t - 1:
        b = D(sid, case, cur)[0]
        cur = min(cur + ((1 + b % 2) if case == "isolated_drop" else (3 if b % 4 == 0 else 1)), t - 1)
        out.append(cur)
    return out


def nominal_ts(t, t0=1_000_000):
    return np.array([t0 + round(i * 1e6 / 30) for i in range(t)], dtype=np.int64)


def clip(t, seed=0, p_present=0.8):
    rng = np.random.default_rng(seed)
    xy = rng.normal(size=(t, 67, 2)).astype(np.float32)
    pr = rng.random((t, 67)) < p_present
    return xy, pr, nominal_ts(t)


class ImportHygiene(unittest.TestCase):
    def test_no_model_helper_or_camera_imports(self):
        for name in ("torch", "run_audit", "timegrid", "cv2", "mediapipe", "framerate_impl",
                     "tools_prep_union93_candidate", "pilot"):
            self.assertNotIn(name, sys.modules, name)


class Selection(unittest.TestCase):
    def test_signature_has_no_label(self):
        self.assertEqual(list(inspect.signature(C.corrupt).parameters), ["xy", "present", "ts", "sample_id", "case"])
        self.assertEqual(C.CASES, ["isolated_drop", "burst_drop", "clock_noise"])
        xy, pr, ts = clip(5)
        with self.assertRaises(TypeError):
            C.corrupt(xy, pr, ts, "s", "isolated_drop", label="NA")

    def test_independent_digest_selection(self):
        for case in C.CASES:
            for sid in ("row_a", "row_b", "한글_id"):
                for t in (1, 2, 3, 4, 5, 10, 37, 120):
                    want = expected_indices(t, sid, case)
                    self.assertEqual(C.select_indices(t, sid, case), want)
                    xy, pr, ts = clip(t)
                    self.assertEqual(C.corrupt(xy, pr, ts, sid, case)["indices"].tolist(), want)

    def test_deterministic(self):
        xy, pr, ts = clip(80, seed=3)
        for case in C.CASES:
            a, b = C.corrupt(xy, pr, ts, "s1", case), C.corrupt(xy, pr, ts, "s1", case)
            for k in ("xy", "present", "ts", "indices", "clock_offset_us"):
                self.assertTrue(A.same(a[k], b[k]), (case, k))

    def test_different_ids_differ(self):
        for case in DROPS:
            self.assertNotEqual(C.select_indices(200, "row_A", case), C.select_indices(200, "row_B", case))
        xy, pr, ts = clip(50)
        self.assertFalse(np.array_equal(C.corrupt(xy, pr, ts, "row_A", "clock_noise")["ts"],
                                        C.corrupt(xy, pr, ts, "row_B", "clock_noise")["ts"]))

    def test_gaps_and_endpoints(self):
        for case in C.CASES:
            seen = set()
            for sid in (f"id{k}" for k in range(20)):
                for t in (1, 2, 3, 6, 7, 61, 200):
                    idx = C.select_indices(t, sid, case)
                    self.assertEqual((idx[0], idx[-1]), (0, t - 1))
                    g = np.diff(idx)
                    self.assertTrue(((g >= 1) & (g <= C.MAX_SOURCE_GAP[case])).all())
                    seen.update(g.tolist())
            want = {"isolated_drop": {1, 2}, "burst_drop": {1, 3}, "clock_noise": {1}}[case]
            self.assertEqual(seen, want, case)

    def test_single_odd_even(self):
        for case in C.CASES:
            for t in (1, 2):
                xy, pr, ts = clip(t)
                c = C.corrupt(xy, pr, ts, "s", case)
                self.assertEqual(c["indices"].tolist(), list(range(t)))
                self.assertEqual(c["clock_offset_us"].tolist(), [0] * t)
                self.assertTrue(A.same(c["ts"], ts))
        for t in (3, 4, 9, 10):
            for case in DROPS:
                self.assertEqual(C.select_indices(t, "odd_even", case)[-1], t - 1)

    def test_coordinates_not_interpolated(self):
        xy, pr, ts = clip(90, seed=7)
        for case in C.CASES:
            c = C.corrupt(xy, pr, ts, "coord", case)
            idx = c["indices"]
            self.assertTrue(A.same(c["xy"], xy[idx]) and A.same(c["present"], pr[idx]))
            src_rows = {xy[i].tobytes() for i in range(len(xy))}
            self.assertTrue(all(r.tobytes() in src_rows for r in c["xy"]))
            if case != "clock_noise":
                self.assertTrue(A.same(c["ts"], ts[idx]))
                self.assertFalse(c["clock_offset_us"].any())

    def test_clock_offsets_examples(self):
        sid, t = "clk_example", 12
        xy, pr, ts = clip(t)
        c = C.corrupt(xy, pr, ts, sid, "clock_noise")
        for i in range(t):
            want = 0 if i in (0, t - 1) else int.from_bytes(D(sid, "clock_noise", i)[:2], "big") % 2001 - 1000
            self.assertEqual(int(c["ts"][i]) - int(ts[i]), want)
            self.assertEqual(int(c["clock_offset_us"][i]), want)
            self.assertLessEqual(abs(want), 1000)
        self.assertTrue(A.same(c["xy"], xy) and A.same(c["present"], pr))
        self.assertEqual(c["ts"].dtype, ts.dtype)

    def test_monotonic_failure_refused(self):
        xy, pr, _ = clip(50)
        tight = np.arange(50, dtype=np.int64) * 5           # 5 us steps: +-1000 us noise must reorder
        with self.assertRaises(C.CorruptionError):
            C.corrupt(xy, pr, tight, "short", "clock_noise")
        for case in DROPS:                                   # drops keep source times, still valid
            C.corrupt(xy, pr, tight, "short", case)

    def test_input_validation(self):
        xy, pr, ts = clip(6)
        bad = [(xy, pr, ts[::-1].copy(), "s", "isolated_drop"), (xy, pr, ts.astype(np.float64), "s", "burst_drop"),
               (xy, pr, ts, "s", "sparse"), (xy, pr, ts, "", "clock_noise"), (xy[:, :66], pr, ts, "s", "clock_noise"),
               (xy[:0], pr[:0], ts[:0], "s", "isolated_drop"), (xy, pr[:5], ts, "s", "isolated_drop"),
               (xy, pr, np.array([0, 1, 1, 2, 3, 4], np.int64), "s", "isolated_drop")]
        for args in bad:
            with self.assertRaises(C.CorruptionError):
                C.corrupt(*args)

    def test_nonmutation(self):
        xy, pr, ts = clip(40, seed=11)
        keep = (xy.copy(), pr.copy(), ts.copy())
        for a in (xy, pr, ts):
            a.flags.writeable = False
        for case in C.CASES:
            c = C.corrupt(xy, pr, ts, "nm", case)
            c["xy"][0, 0, 0] += 1.0                          # output is a copy, never a view
        self.assertTrue(all(A.same(a, b) for a, b in zip(keep, (xy, pr, ts))))


class ContractCheck(unittest.TestCase):
    def test_valid_and_tampered(self):
        xy, pr, ts = clip(70, seed=5)
        for case in C.CASES:
            c = C.corrupt(xy, pr, ts, "chk", case)
            info = A.check_corruption(xy, pr, ts, c, case, "chk")
            self.assertLessEqual(info["max_source_index_gap"], C.MAX_SOURCE_GAP[case])
            tampers = []
            t1 = dict(c, xy=c["xy"].copy()); t1["xy"][1, 0, 0] += 0.5; tampers.append(t1)
            t2 = dict(c, ts=c["ts"].copy(), clock_offset_us=c["clock_offset_us"].copy())
            t2["ts"][-1] += 7; t2["clock_offset_us"][-1] += 7; tampers.append(t2)
            tampers.append(dict(c, indices=c["indices"][1:]))
            tampers.append(dict(c, case="other"))
            t5 = dict(c, present=c["present"].copy()); t5["present"][0] = ~t5["present"][0]; tampers.append(t5)
            for bad in tampers:
                with self.assertRaises(A.IrregularError):
                    A.check_corruption(xy, pr, ts, bad, case, "chk")


class TimegridSmoke(unittest.TestCase):
    def restore(self, c):
        res = {m: TGR.restore(c["xy"], c["present"], c["ts"], m) for m in ("previous", "linear")}
        for k in ("ts", "present", "observed", "estimated", "exact_source_index"):
            self.assertTrue(A.same(res["previous"][k], res["linear"][k]))
        return res["linear"]

    def test_isolated_drop_estimates_only_dropped(self):
        xy, pr, ts = clip(60, p_present=1.0)
        c = C.corrupt(xy, pr, ts, "iso", "isolated_drop")
        lin = self.restore(c)
        b = A.bracket_check(c["ts"], c["present"], lin["ts"], lin["exact_source_index"], lin["estimated"])
        self.assertEqual(b["grid_frames"], 60)
        self.assertEqual(b["exact_frames"], len(c["indices"]))
        self.assertEqual(b["estimated_frames"], 60 - len(c["indices"]))
        self.assertEqual(b["frames_with_long_bracket"], 0)

    def test_burst_long_gap_unsupported(self):
        xy, pr, ts = clip(200, p_present=1.0)
        c = C.corrupt(xy, pr, ts, "burst", "burst_drop")
        lin = self.restore(c)
        b = A.bracket_check(c["ts"], c["present"], lin["ts"], lin["exact_source_index"], lin["estimated"])
        g = np.diff(c["indices"])
        self.assertGreater(int((g == 3).sum()), 0)
        self.assertEqual(b["frames_with_long_bracket"], 2 * int((g == 3).sum()))
        self.assertEqual(b["estimated_frames"], int((g == 2).sum()))
        k = int(np.flatnonzero(lin["exact_source_index"] < 0)[0])
        est = lin["estimated"].copy(); est[k] = True              # forged estimate across a 100 ms span
        with self.assertRaises(A.IrregularError):
            A.bracket_check(c["ts"], c["present"], lin["ts"], lin["exact_source_index"], est)

    def test_absent_bracket_joint_not_estimated(self):
        xy, pr, ts = clip(60, p_present=1.0)
        c = C.corrupt(xy, pr, ts, "iso2", "isolated_drop")
        k = int(np.flatnonzero(np.diff(c["indices"]) == 2)[0])   # retained position before a drop
        c["present"][k, 30] = False
        lin = self.restore(c)
        grid_k = int(np.flatnonzero(lin["exact_source_index"] == k)[0]) + 1
        self.assertFalse(lin["estimated"][grid_k, 30])
        self.assertTrue(lin["estimated"][grid_k, 31])
        A.bracket_check(c["ts"], c["present"], lin["ts"], lin["exact_source_index"], lin["estimated"])
        est = lin["estimated"].copy(); est[grid_k, 30] = True
        with self.assertRaises(A.IrregularError):
            A.bracket_check(c["ts"], c["present"], lin["ts"], lin["exact_source_index"], est)

    def test_clock_noise_estimates_without_drops(self):
        xy, pr, ts = clip(40, p_present=1.0)
        c = C.corrupt(xy, pr, ts, "noise", "clock_noise")
        lin = self.restore(c)
        self.assertFalse(lin["identity"])
        b = A.bracket_check(c["ts"], c["present"], lin["ts"], lin["exact_source_index"], lin["estimated"])
        self.assertGreater(b["estimated_frames"], 0)
        self.assertEqual(b["frames_with_long_bracket"], 0)
        self.assertEqual([int(lin["ts"][0]), int(lin["ts"][-1])], [int(ts[0]), int(ts[-1])])


def cell(corr, lin, per_signer, newly_wrong=0, prev=0):
    return {"correct": {"original": 0, "corrupted": corr, "previous": prev, "linear": lin},
            "per_signer": {s: {"corrupted": a, "linear": b} for s, (a, b) in per_signer.items()},
            "newly_wrong": newly_wrong}


def passing():
    st = {g: {c: cell(10, 10, {"1": (5, 5), "2": (5, 5)}) for c in A.CASES} for g in A.GROUPS}
    st["native"]["isolated_drop"] = cell(100, 107, {"11": (30, 32), "12": (30, 33), "13": (40, 42)})
    return st


class Gates(unittest.TestCase):
    ok = {"contracts": True}

    def test_pass(self):
        r = A.admission(passing(), self.ok)
        self.assertTrue(r["all_conditions_hold"])
        self.assertEqual(r["decision"], "ADMISSION_CONDITIONS_MET_PENDING_CODEX_VERIFICATION")
        self.assertFalse(r["production_authorised"])
        self.assertFalse(r["web_integration_halted"])

    def expect_hold(self, st, cond, contracts=None):
        r = A.admission(st, self.ok if contracts is None else contracts)
        self.assertFalse(r[cond]["pass"], cond)
        self.assertFalse(r["all_conditions_hold"])
        self.assertEqual(r["decision"], "HOLD")
        self.assertTrue(r["web_integration_halted"])
        return r

    def test_native_net_six(self):
        st = passing(); st["native"]["isolated_drop"] = cell(100, 106, {"11": (30, 32), "12": (30, 32), "13": (40, 42)})
        r = self.expect_hold(st, "condition_1_native_isolated_drop")
        self.assertEqual(r["condition_1_native_isolated_drop"]["net_linear_vs_corrupted"], 6)

    def test_native_signer_decrease(self):
        st = passing(); st["native"]["isolated_drop"] = cell(100, 108, {"11": (30, 29), "12": (30, 36), "13": (40, 43)}, 1)
        self.expect_hold(st, "condition_1_native_isolated_drop")
        self.assertIn("native/isolated_drop", A.admission(st, self.ok)["condition_3_all_cases_no_newly_wrong"]["failing"])

    def test_mp_negative(self):
        st = passing(); st["mediapipe"]["isolated_drop"] = cell(10, 9, {"1": (5, 4), "2": (5, 5)}, 1)
        self.expect_hold(st, "condition_2_mediapipe_isolated_drop")

    def test_burst_guard_newly_wrong_halts(self):
        st = passing(); st["native"]["burst_drop"] = cell(10, 11, {"1": (5, 6), "2": (5, 5)}, newly_wrong=1)
        r = self.expect_hold(st, "condition_3_all_cases_no_newly_wrong")
        self.assertTrue(r["condition_1_native_isolated_drop"]["pass"] and r["condition_2_mediapipe_isolated_drop"]["pass"])
        self.assertEqual(r["condition_3_all_cases_no_newly_wrong"]["failing"], ["native/burst_drop"])

    def test_clock_noise_signer_guard(self):
        st = passing(); st["mediapipe"]["clock_noise"] = cell(10, 10, {"1": (5, 4), "2": (5, 6)})
        self.expect_hold(st, "condition_3_all_cases_no_newly_wrong")

    def test_contracts(self):
        self.expect_hold(passing(), "condition_4_contracts", {"a": True, "b": False})
        self.expect_hold(passing(), "condition_4_contracts", {})

    def test_previous_never_promoted(self):
        st = passing(); st["native"]["isolated_drop"] = cell(100, 100, {"11": (30, 30), "12": (30, 30), "13": (40, 40)}, prev=150)
        self.expect_hold(st, "condition_1_native_isolated_drop")


class Tables(unittest.TestCase):
    def test_breakdown_and_recoveries(self):
        def arm(ok, pred="X"):
            return {"correct": ok, "pred": pred, "rank": 1 if ok else 2}
        rows = [{"id": "a", "signer_id": 1, "true": "X",
                 "arms": {"original": arm(True), "corrupted": arm(False, "Y"), "previous": arm(False, "Y"), "linear": arm(True)}},
                {"id": "b", "signer_id": 2, "true": "X",
                 "arms": {"original": arm(True), "corrupted": arm(True), "previous": arm(True), "linear": arm(True)}}]
        b = A.breakdown(rows, "signer_id")
        self.assertEqual(b["1"], {"n": 1, "original": 1, "corrupted": 0, "previous": 0, "linear": 1})
        r = A.recoveries(rows)
        self.assertEqual((r["n"], r["correct_in_linear"], r["correct_in_previous"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
