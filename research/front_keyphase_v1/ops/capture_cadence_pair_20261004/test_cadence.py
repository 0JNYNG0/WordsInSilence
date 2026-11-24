"""Synthetic tests for cadence_pair.py.

Fake decoder / landmarker / model only. No media file is decoded, no MediaPipe or torch model is loaded,
no camera is opened. Fake trees live in a system temporary directory and are removed afterwards.

Run:  python test_cadence.py
On full success this writes TEST_RECEIPT.json next to this file (exclusive create; delete it by hand
before a deliberate re-run). Running through pytest/unittest discovery never writes a receipt.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_MODULES_BEFORE = set(sys.modules)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import cadence_pair as cp  # noqa: E402

HEAVY = ("av", "mediapipe", "torch", "cv2")
HEAVY_IMPORTED_BY_CP = [m for m in HEAVY if m in sys.modules and m not in _MODULES_BEFORE]
CLASSES = ["나", "죄송하다"] + [f"w{i:02d}" for i in range(91)]
BEFORE_DEADLINE = datetime(2026, 10, 4, 6, 0, 0, tzinfo=cp.KST)


# ---------------------------------------------------------------- fakes
def fake_empty_frames(t):
    return {"xy_px": np.full((t, 75, 2), np.nan, np.float32), "timestamp_us": np.zeros(t, np.int64),
            "frame_id": np.zeros(t, np.int64), "observed": np.zeros((t, 75), np.bool_),
            "usable": np.zeros((t, 75), np.bool_), "conf": np.full((t, 75), np.nan, np.float32),
            "hand_score": np.full((t, 2), np.nan, np.float32)}


class FakeObservation:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def validate(self):
        t = self.xy_px.shape[0]
        assert self.xy_px.shape == (t, 75, 2) and self.xy_px.dtype == np.float32
        assert self.timestamp_us.dtype == np.int64 and self.frame_id.dtype == np.int64
        assert not np.isfinite(self.xy_px[~self.observed]).any()
        assert not (self.usable & ~self.observed).any()
        if t > 1:
            assert (np.diff(self.timestamp_us) > 0).all()


def rgb_for(i, shape=(4, 6, 3)):
    a = np.full(shape, i % 251, np.uint8)
    a[0, 0, 1] = (i // 251) % 256
    return a


def nonuniform_pts(n, start=1000):
    pts, p = [], start
    for i in range(n):
        pts.append(p)
        p += 1024 if i % 7 == 6 else 512           # tb 1/15360: mostly 33.3 ms, sometimes 66.7 ms
    return pts


class FakeDecoder:
    def __init__(self, pts, tb=Fraction(1, 15360), rgb=rgb_for, log=None):
        self.pts, self.time_base, self._rgb, self.log = list(pts), tb, rgb, log
        self.meta = {"time_base": f"{tb.numerator}/{tb.denominator}", "width": 6, "height": 4,
                     "average_rate_hz": 30.0}

    def __enter__(self):
        if self.log is not None:
            self.log.append("open")
        return self

    def raw(self):
        for i, p in enumerate(self.pts):
            yield p, self._rgb(i)

    def __exit__(self, *exc):
        if self.log is not None:
            self.log.append("close")
        return False


class FakeLandmarker:
    """Deterministic, but output depends on how many frames this instance has seen (tracking history)."""

    def __init__(self, noise=0.0):
        self._last_ms = -1
        self.handedness_disagreements = self.handedness_comparisons = self.unassigned_hands = 0
        self.calls, self.closed, self.noise = [], False, noise

    def process(self, rgb, ts, out, row):
        assert not self.closed
        k = len(self.calls)
        self.calls.append((int(ts), int(row), int(rgb[0, 0, 0])))
        v = float(rgb[0, 0, 0])
        out["xy_px"][row, 0:33] = [v + 1.0, v + 2.0]
        out["observed"][row, 0:33] = True
        out["usable"][row, 0:33] = True
        out["conf"][row, 0:33] = 0.9
        out["xy_px"][row, 33:54] = [v + 0.25 * k + self.noise, v]
        out["observed"][row, 33:54] = True
        out["usable"][row, 33:54] = k % 4 != 3
        out["hand_score"][row, 0] = 0.8
        if k % 3 == 0:
            out["xy_px"][row, 54:75] = [v, v + 3.0]
            out["observed"][row, 54:75] = True
            out["usable"][row, 54:75] = True
            out["hand_score"][row, 1] = 0.7
        self._last_ms = max(int(ts) // 1000, self._last_ms + 1)

    def extractor_id(self):
        return "fake_extractor"

    def close(self):
        self.closed = True


class Factory:
    def __init__(self, noise_on=(), reuse=False):
        self.instances, self.noise_on, self.reuse = [], set(noise_on), reuse

    def __call__(self):
        if self.reuse and self.instances:
            return self.instances[0]
        lm = FakeLandmarker(noise=1e-3 if len(self.instances) in self.noise_on else 0.0)
        self.instances.append(lm)
        return lm


def fake_to_native_like(xy, usable):
    return np.where(usable[..., None], xy, 0).astype(np.float32), usable.copy()


def fake_build_channels(xy, present, ts):
    t = xy.shape[0]
    dt = np.zeros((t, 1), np.float32)
    dt[1:, 0] = np.diff(ts).astype(np.float32) / 1e6
    x = np.concatenate([xy[:, 0, :], present[:, :1].astype(np.float32), dt], axis=1).astype(np.float32)
    return {"x": x, "frames": t, "body_scale": 1.0, "hand_scales": {"left": 2.0, "right": float("nan")},
            "body_valid": present[:, :1].astype(np.float32), "hand_valid": np.zeros((t, 1), np.float32)}


def fake_collate(clips):
    c = clips[0]
    assert c["label"] == 0
    return c["x"][None], np.ones((1, c["frames"]), np.float32), np.array([c["label"]], np.int64)


class FakeModel:
    training = False
    state = "s0"


def make_deps(pts_by_path, factory=None, decoder_log=None, tb=Fraction(1, 15360), rgb=rgb_for,
              state_mutates=False, build_channels=fake_build_channels):
    factory = factory or Factory()
    model = FakeModel()
    record = {"forwards": 0}

    def forward(m, x, pad):
        record["forwards"] += 1
        lg = np.zeros(93)
        lg[0] = float(x[0, :, 0].mean()) % 5.0
        lg[1] = 2.0
        if state_mutates:
            m.state = "s1"
        return lg

    def open_decoder(p):
        rgbf = rgb[p] if isinstance(rgb, dict) else rgb
        return FakeDecoder(pts_by_path[p], tb, rgb=rgbf, log=decoder_log)

    deps = cp.Deps(open_decoder=open_decoder, landmarker_factory=factory, empty_frames=fake_empty_frames,
                   observation_cls=FakeObservation, to_native_like=fake_to_native_like,
                   build_channels=build_channels, collate=fake_collate,
                   load_model=lambda: (model, list(CLASSES), {"fake": True}), forward=forward,
                   state_hash=lambda m: m.state, n_channels=4, versions={"fake": True},
                   import_audit={"fake": True})
    return deps, factory, record


def small_plan(sizes, scores=None):
    clips, pts = [], {}
    for i, n in enumerate(sizes):
        p = Path(f"/fake/c{i}.mp4")
        clips.append({"clip_id": f"c{i}", "path": p, "sha256": "0" * 64, "n_frames": n,
                      "n_selected": (n + 1) // 2, "score": (scores or {}).get(i)})
        pts[str(p)] = nonuniform_pts(n)
    return {"clips": clips}, pts


def extract(deps, pts_key, n, arm, counters=None, registry=None, clip_id="c0"):
    return cp.extract_arm(path=pts_key, clip_id=clip_id, arm=arm, n_frames=n, n_selected=(n + 1) // 2,
                          deps=deps, counters=counters if counters is not None else cp.new_counters(),
                          registry=registry if registry is not None else [])


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="cadence_test_")
        self.tmp = Path(self._td.name).resolve()

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------- decode / selection
class TestDecodeSelection(TmpCase):
    def test_import_has_no_heavy_modules(self):
        self.assertEqual(HEAVY_IMPORTED_BY_CP, [])

    def test_even_selection_starts_at_0_and_never_appends_last(self):
        for n, want in ((5, [0, 2, 4]), (6, [0, 2, 4]), (1, [0])):
            deps, _, _ = make_deps({"p": nonuniform_pts(n)})
            a = extract(deps, "p", n, cp.ARM_A)
            b = extract(deps, "p", n, cp.ARM_B)
            self.assertEqual(a["selected_meta"]["frame_id"].tolist(), want)
            self.assertEqual(b["selected_obs"]["frame_id"].tolist(), want)
            self.assertEqual(a["counts"]["process_calls"], n)
            self.assertEqual(b["counts"]["process_calls"], len(want))
            self.assertEqual(b["counts"]["decoded_frames"], n)
            self.assertTrue(cp.check_selection(a, b)["match"])

    def test_true_nonuniform_pts_preserved(self):
        tb = Fraction(1, 90000)
        pts = [1000, 4003, 7006, 13012, 16015]
        want = [int(round(float(Fraction(p - 1000) * tb) * 1_000_000)) for p in pts]
        got = list(cp.strict_frames(((p, rgb_for(i)) for i, p in enumerate(pts)), tb))
        self.assertEqual([g[0] for g in got], [0, 1, 2, 3, 4])
        self.assertEqual([g[2] for g in got], want)
        self.assertEqual(want[0], 0)
        deps, _, _ = make_deps({"p": pts}, tb=tb)
        b = extract(deps, "p", 5, cp.ARM_B)
        self.assertEqual(b["selected_obs"]["timestamp_us"].tolist(), want[::2])
        self.assertEqual(b["selected_obs"]["timestamp_us"].dtype, np.int64)
        self.assertGreater(b["intervals"]["selected"]["n_unique_gaps"], 1)

    def test_missing_pts_refused_and_landmarker_closed(self):
        with self.assertRaises(cp.DecodeContractError):
            list(cp.strict_frames(iter([(0, None), (None, None)]), Fraction(1, 1000)))
        factory = Factory()
        deps, _, _ = make_deps({"p": [0, None, 1024]}, factory=factory)
        counters = cp.new_counters()
        with self.assertRaises(cp.DecodeContractError):
            extract(deps, "p", 3, cp.ARM_A, counters=counters)
        self.assertTrue(factory.instances[0].closed)
        self.assertEqual(counters["landmarker_closed"], 1)

    def test_nonincreasing_pts_refused(self):
        for pts, tb in (([0, 512, 512], Fraction(1, 15360)), ([0, 1024, 512], Fraction(1, 15360)),
                        ([0, 1], Fraction(1, 10 ** 9))):        # last: distinct PTS, same microsecond
            with self.assertRaises(cp.DecodeContractError):
                list(cp.strict_frames(((p, None) for p in pts), tb))

    def test_unexpected_frame_count_refused(self):
        deps, _, _ = make_deps({"p": nonuniform_pts(5)})
        for n in (4, 6):
            with self.assertRaises(cp.DecodeContractError):
                extract(deps, "p", n, cp.ARM_A)

    def test_rgb_hash_covers_dtype_and_shape(self):
        a = np.zeros((2, 6, 3), np.uint8)
        b = np.zeros((4, 3, 3), np.uint8)
        self.assertNotEqual(cp.rgb_digest(a), cp.rgb_digest(b))
        with self.assertRaises(cp.DecodeContractError):
            cp.rgb_digest(np.zeros((2, 6, 3), np.float32))
        with self.assertRaises(cp.DecodeContractError):
            cp.rgb_digest(np.zeros((2, 6, 4), np.uint8))

    def _pair(self, b_pts=None, b_rgb=rgb_for, n=7):
        pts = nonuniform_pts(n)
        deps, _, _ = make_deps({"a": pts, "b": b_pts or pts}, rgb={"a": rgb_for, "b": b_rgb})
        return extract(deps, "a", n, cp.ARM_A), extract(deps, "b", n, cp.ARM_B)

    def test_selected_rgb_hash_mismatch_rejected(self):
        def tweak(i):
            r = rgb_for(i)
            if i == 2:
                r[1, 1, 2] ^= 1
            return r
        a, b = self._pair(b_rgb=tweak)
        with self.assertRaises(cp.SelectionMismatch):
            cp.check_selection(a, b)

    def test_selected_pts_mismatch_rejected(self):
        pts = nonuniform_pts(7)
        bad = list(pts)
        bad[2] += 1
        a, b = self._pair(b_pts=bad)
        with self.assertRaises(cp.SelectionMismatch):
            cp.check_selection(a, b)

    def test_odd_frame_difference_rejected_by_full_audit(self):
        def tweak(i):
            r = rgb_for(i)
            if i == 3:
                r[0, 0, 2] ^= 1
            return r
        a, b = self._pair(b_rgb=tweak)
        with self.assertRaises(cp.SelectionMismatch):
            cp.check_selection(a, b)

    def test_b_skips_odd_process_calls(self):
        n = 9
        factory = Factory()
        log = []
        deps, _, _ = make_deps({"p": nonuniform_pts(n)}, factory=factory, decoder_log=log)
        counters = cp.new_counters()
        b = extract(deps, "p", n, cp.ARM_B, counters=counters)
        calls = factory.instances[0].calls
        even = list(range(0, n, 2))
        self.assertEqual([c[1] for c in calls], list(range(len(even))))
        self.assertEqual([c[2] for c in calls], [rgb_for(i)[0, 0, 0] for i in even])
        self.assertEqual([c[0] for c in calls], b["decoded_meta"]["timestamp_us"][even].tolist())
        self.assertEqual(counters["decoded_frames"], n)
        self.assertEqual(counters["landmarker_process_calls"], len(even))
        self.assertEqual(b["decoded_meta"]["processed"].tolist(), [i % 2 == 0 for i in range(n)])
        self.assertEqual(log, ["open", "close"])


# ---------------------------------------------------------------- determinism / freshness
class TestDeterminismFreshness(TmpCase):
    def test_exact_equal_nan_dtype_shape(self):
        x = np.array([1.0, np.nan, 3.0], np.float32)
        self.assertTrue(cp.exact_equal(x, x.copy())["equal"])
        self.assertFalse(cp.exact_equal(x, np.array([1.0, 2.0, np.nan], np.float32))["equal"])
        self.assertFalse(cp.exact_equal(x, np.array([1.0, np.nan, 3.0000002], np.float32))["equal"])
        self.assertFalse(cp.exact_equal(x, x.astype(np.float64))["equal"])
        self.assertFalse(cp.exact_equal(x, x[:2])["equal"])
        self.assertFalse(cp.exact_equal(np.array([True, False]), np.array([True, True]))["equal"])
        r = cp.exact_equal(x, np.array([1.0, 2.0, 3.0], np.float32))
        self.assertFalse(r["nan_pattern_equal"])

    def test_nondeterminism_stops_before_b_and_other_clips(self):
        plan, pts = small_plan([7, 5])
        factory = Factory(noise_on={1})
        deps, _, _ = make_deps(pts, factory=factory)
        writer = cp.ResultWriter(self.tmp)
        counters, log = cp.new_counters(), []
        with self.assertRaises(cp.NondeterminismError):
            cp.run_extractions(plan, deps, counters, writer, log)
        self.assertEqual(len(factory.instances), 2)
        self.assertEqual([e["tag"] for e in log], ["A", "A_repeat"])
        det = json.loads((self.tmp / "DETERMINISM.json").read_text(encoding="utf-8"))
        self.assertFalse(det["all_equal"])
        self.assertFalse(det["observation_fields"]["xy_px"]["equal"])
        self.assertEqual(len(list(self.tmp.glob("*.npz"))), 2)

    def test_fresh_landmarker_every_clip_arm_in_fixed_order(self):
        plan, pts = small_plan([7, 5, 6])
        factory = Factory()
        deps, _, _ = make_deps(pts, factory=factory)
        counters, log = cp.new_counters(), []
        results, det, _files = cp.run_extractions(plan, deps, counters, cp.ResultWriter(self.tmp), log)
        self.assertTrue(det["all_equal"])
        self.assertEqual([(e["clip_id"], e["tag"]) for e in log],
                         [("c0", "A"), ("c0", "A_repeat"), ("c0", "B"), ("c1", "A"), ("c1", "B"),
                          ("c2", "A"), ("c2", "B")])
        self.assertEqual(len(factory.instances), 7)
        self.assertEqual(len({id(x) for x in factory.instances}), 7)
        self.assertTrue(all(x.closed for x in factory.instances))
        self.assertEqual(counters["landmarker_process_calls"], 7 + 7 + 4 + 5 + 3 + 6 + 3)
        self.assertEqual(counters["decoded_frames"], 7 * 3 + 5 * 2 + 6 * 2)
        # history differs between arms, so B observations differ while ids/PTS/RGB match
        r = results["c0"]
        self.assertFalse(np.array_equal(r["A"]["selected_obs"]["xy_px"], r["B"]["selected_obs"]["xy_px"],
                                        equal_nan=True))
        self.assertTrue(r["selection"]["match"])

    def test_reused_or_used_landmarker_refused(self):
        plan, pts = small_plan([5])
        deps, _, _ = make_deps(pts, factory=Factory(reuse=True))
        with self.assertRaises(cp.LandmarkerNotFresh):
            cp.run_extractions(plan, deps, cp.new_counters(), cp.ResultWriter(self.tmp), [])
        used = FakeLandmarker()
        used._last_ms = 5
        deps2, _, _ = make_deps(pts, factory=lambda: used)
        with self.assertRaises(cp.LandmarkerNotFresh):
            extract(deps2, str(plan["clips"][0]["path"]), 5, cp.ARM_A)
        self.assertTrue(used.closed)


# ---------------------------------------------------------------- metrics
class TestMetrics(unittest.TestCase):
    def _sel(self, t=2):
        return {"xy_px": np.full((t, 75, 2), np.nan, np.float32), "usable": np.zeros((t, 75), bool),
                "observed": np.zeros((t, 75), bool), "frame_id": np.arange(0, 2 * t, 2, dtype=np.int64)}

    def test_distance_uses_common_usable_finite_only(self):
        a, b = self._sel(), self._sel()
        a["xy_px"][0, 0], b["xy_px"][0, 0] = (0, 0), (3, 4)
        a["usable"][0, 0] = b["usable"][0, 0] = True
        a["xy_px"][0, 33], b["xy_px"][0, 33] = (0, 0), (100, 0)
        a["usable"][0, 33] = True                               # B not usable: excluded
        a["xy_px"][1, 54] = (0, 0)
        a["usable"][1, 54] = b["usable"][1, 54] = True          # B NaN: excluded
        d = cp.coord_distances(a, b)
        self.assertEqual(d["pose"]["count"], 1)
        self.assertAlmostEqual(d["pose"]["mean"], 5.0)
        self.assertEqual(d["all"]["max"], 5.0)
        for g in ("left_hand", "right_hand"):
            self.assertEqual(d[g], {"count": 0, "mean": None, "median": None, "p95": None, "max": None})

    def test_no_common_point_is_none_not_zero(self):
        d = cp.coord_distances(self._sel(), self._sel())
        self.assertIsNone(d["all"]["mean"])
        self.assertEqual(d["all"]["count"], 0)

    def test_mask_diff_directional(self):
        ma = np.zeros((2, 75), bool)
        mb = np.zeros((2, 75), bool)
        ma[0, 40] = True
        mb[1, 60] = True
        ma[1, 5] = mb[1, 5] = True
        m = cp.mask_diff(ma, mb)
        self.assertEqual(m["left_hand"]["A_true_B_false"], 1)
        self.assertEqual(m["right_hand"]["A_false_B_true"], 1)
        self.assertEqual(m["pose"]["differing"], 0)
        self.assertEqual(m["all"]["differing"], 2)

    def test_hand_rates_and_absent_runs(self):
        u = np.zeros((6, 75), bool)
        u[[0, 1, 5], 33:54] = True
        u[0, 54] = True
        h = cp.hand_rates(u)
        self.assertEqual(h["left_hand"]["all21_absent_runs"], [3])
        self.assertAlmostEqual(h["left_hand"]["all21_usable_fraction"], 0.5)
        self.assertEqual(h["right_hand"]["all21_absent_runs"], [5])
        self.assertAlmostEqual(h["right_hand"]["any_usable_fraction"], 1 / 6)
        self.assertEqual(h["right_hand"]["all21_usable_fraction"], 0.0)


# ---------------------------------------------------------------- scoring
class TestScoring(TmpCase):
    def test_half_open_window(self):
        ts = np.array([0, 100, 200, 300, 400], np.int64)
        self.assertEqual(cp.window_rows(ts, 100, 300).tolist(), [1, 2])
        self.assertEqual(cp.window_rows(ts, 101, 301).tolist(), [2, 3])

    def test_all_tie_ranking_deterministic(self):
        r = cp.rank_against_label(np.zeros(93), CLASSES, CLASSES[5])
        self.assertEqual(r["top1"], CLASSES[0])
        self.assertEqual(r["true_rank"], 6)
        self.assertEqual(r["best_other"], CLASSES[0])
        self.assertEqual(r["true_vs_best_other_margin"], 0.0)
        self.assertFalse(r["correct"])

    def test_label_does_not_filter_classes(self):
        rng = np.random.default_rng(0)
        lg = rng.normal(size=93)
        label = CLASSES[1]
        r = cp.rank_against_label(lg, CLASSES, label)
        order = sorted(range(93), key=lambda j: (-lg[j], j))
        self.assertEqual(r["true_rank"], order.index(1) + 1)
        self.assertEqual(r["top1"], CLASSES[order[0]])
        with self.assertRaises(cp.ScoringError):
            cp.rank_against_label(lg, CLASSES, "없는단어")
        with self.assertRaises(cp.ScoringError):
            cp.rank_against_label(lg[:18], CLASSES, label)
        bad = lg.copy()
        bad[3] = np.nan
        with self.assertRaises(cp.ScoringError):
            cp.rank_against_label(bad, CLASSES, label)

    def _results(self, state_mutates=False, build_channels=fake_build_channels):
        plan, pts = small_plan([9, 5], scores={0: {"label": "죄송하다", "start_us": 0, "end_us_exclusive": 200_000}})
        deps, _, rec = make_deps(pts, state_mutates=state_mutates, build_channels=build_channels)
        counters, log = cp.new_counters(), []
        results, _, _ = cp.run_extractions(plan, deps, counters, cp.ResultWriter(self.tmp), log)
        return plan, deps, results, counters, log, rec

    def test_scoring_two_forwards_per_window_state_protected(self):
        plan, deps, results, counters, log, rec = self._results()
        s = cp.run_scoring(plan, results, deps, counters, log)
        self.assertEqual(counters["tcn_forwards"], 2)
        self.assertEqual(rec["forwards"], 2)
        self.assertEqual(counters["model_loads"], 1)
        w = s["windows"][0]
        self.assertEqual(len(w["A"]["logits93"]), 93)
        self.assertEqual(w["window_frame_ids"].tolist(), results["c0"]["A"]["selected_obs"]["frame_id"][
            cp.window_rows(results["c0"]["A"]["selected_obs"]["timestamp_us"], 0, 200_000)].tolist())
        self.assertIn(w["direction"], ("both_correct", "both_wrong", "B_regression_vs_A", "B_improvement_vs_A"))
        self.assertTrue(s["state_unchanged"])

    def test_model_state_change_detected(self):
        plan, deps, results, counters, log, _ = self._results(state_mutates=True)
        with self.assertRaises(cp.ModelStateChanged):
            cp.run_scoring(plan, results, deps, counters, log)

    def test_preprocessing_failure_is_technical(self):
        def boom(*a):
            raise ValueError("no frame has both shoulders present")
        plan, deps, results, counters, log, _ = self._results(build_channels=boom)
        with self.assertRaises(cp.PreprocessingError):
            cp.run_scoring(plan, results, deps, counters, log)

    def test_interpretation_directions(self):
        def w(d, same=True):
            return {"direction": d, "same_top1": same}
        self.assertEqual(cp.interpret([w("both_correct")] * 4)["code"], "SAME_CORRECTNESS_ALL_WINDOWS")
        self.assertIn("not proof", cp.interpret([w("both_correct")] * 4)["text"])
        self.assertEqual(cp.interpret([w("B_regression_vs_A", False)] + [w("both_correct")] * 3)["code"],
                         "B_WORSE_IN_SOME_WINDOW")
        self.assertEqual(cp.interpret([w("B_improvement_vs_A", False)] + [w("both_wrong")] * 3)["code"],
                         "B_BETTER_IN_SOME_WINDOW")
        self.assertEqual(cp.interpret([w("B_improvement_vs_A", False), w("B_regression_vs_A", False)])["code"],
                         "MIXED_DIRECTIONS")


# ---------------------------------------------------------------- config / CLI with a fake tree
def make_cfg(roots, entries, hashes):
    rp = cp.role_paths(roots)
    clips = []
    for cid, rel, n, win in cp.FROZEN_CLIPS:
        p = str(roots.front / rel)
        clips.append({"clip_id": cid, "path": p, "sha256": hashes[p], "expected_frames_from_prior": n,
                      "expected_selected_even_frames": (n + 1) // 2,
                      "score": None if win is None else {"label": win[0], "start_us": win[1],
                                                         "end_us_exclusive": win[2]},
                      "source_scope": "synthetic test"})
    return {
        "schema": cp.CONFIG_SCHEMA, "status": "PREPARED_FOR_TEST",
        "unit_dir": str(roots.unit), "output_dir": str(roots.result),
        "workspace_root": str(roots.workspace), "front_root": str(roots.front), "clips": clips,
        "pins": {str(p): hashes[str(p)] for p, _ in entries},
        "pin_inventory": [{"path": str(p), "role": role, "sha256": hashes[str(p)]} for p, role in entries],
        "extraction": {"arms": [cp.ARM_A, cp.ARM_B], "stride": 2, "phase": 0, "running_mode": "VIDEO",
                       "reject_missing_pts": True, "reject_nonincreasing_pts": True,
                       "expected_counts_required": True,
                       "hand_model_path": str(rp["mediapipe_hand_model"]),
                       "pose_model_path": str(rp["mediapipe_pose_model"]),
                       "contract_path": str(rp["observation_extraction_contract"]),
                       "estimated_total_process_calls": 1450},
        "determinism": {"clip_id": cp.FROZEN_CLIPS[0][0], "repeat_arm": cp.ARM_A, "extra_runs": 1,
                        "arrays": list(cp.OBS_FIELDS)},
        "scoring": {"checkpoint_path": str(rp["frozen_tcn_checkpoint"]), "checkpoint_sha256": cp.TCN_SHA256,
                    "model_class": "pilot.model_masked_v1.MaskedSignTCN", "state_dict_load": "strict",
                    "n_classes": 93, "cpu_threads": 2, "scored_clips": 4, "forward_calls": 8,
                    "active_span": False, "handgap": False, "timegrid": False, "interpolation": False},
        "decision": {"limits": ["synthetic limit"]},
        "isolation": {"allowed_video_paths": [c["path"] for c in clips], "user_data": False, "camera": False,
                      "device_enumeration": False, "network": False, "downloads": False, "training": False,
                      "operational_change": False, "other_experiment_changes": False, "source_mutation": False,
                      "output_overwrite": False, "execute_default": False, "explicit_run_flag_required": True},
    }


class FakeTree:
    def __init__(self, base: Path):
        ws = base / "ws"
        front = ws / "research" / "front_keyphase_v1"
        unit = front / "ops" / "capture_cadence_pair_20261004"
        self.roots = cp.Roots(ws, front, unit)
        rp = cp.role_paths(self.roots)
        self.entries = [(front / rel, "public_source_video") for _, rel, _, _ in cp.FROZEN_CLIPS]
        self.entries += [(p, role) for role, p in rp.items()]
        self.entries.append((front / "config" / "source_manifest.json", "source_manifest"))
        for p, role in self.entries:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"synthetic {role} {p.name}".encode("utf-8"))
        self.override = {cp._norm(rp["frozen_tcn_checkpoint"]): cp.TCN_SHA256}
        self.hashes = {str(p): self.hash(p) for p, _ in self.entries}
        self.cfg = make_cfg(self.roots, self.entries, self.hashes)
        self.cfg_path = unit / cp.CONFIG_NAME
        self.write_cfg()
        self.closeout = front / "ops" / "teacher_unit_fake" / cp.CLOSEOUT_NAME
        self.closeout.parent.mkdir(parents=True)
        self.set_closeout("COMPLETE")
        self.pts_by_path = {str(front / rel): nonuniform_pts(n) for _, rel, n, _ in cp.FROZEN_CLIPS}

    def hash(self, p):
        return self.override.get(cp._norm(p)) or hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def write_cfg(self):
        self.cfg_path.write_text(json.dumps(self.cfg, ensure_ascii=False), encoding="utf-8")

    def set_closeout(self, status):
        self.closeout.write_text(json.dumps({"status": status, "verdict": "REJECT"}), encoding="utf-8")

    def write_receipt(self, code=None):
        (self.roots.unit / cp.RECEIPT_NAME).write_text(json.dumps({
            "schema": cp.RECEIPT_SCHEMA, "status": "PASS", "num_tests": 1, "failures": 0, "errors": 0,
            "code_sha256": code or cp.own_code_hashes()}), encoding="utf-8")


def _boom_loader(*a, **k):
    raise AssertionError("runtime dependencies must not be loaded")


class TestConfigCli(TmpCase):
    def setUp(self):
        super().setUp()
        self.tree = FakeTree(self.tmp)
        self.buf = []

    def main(self, argv, **kw):
        kw.setdefault("roots", self.tree.roots)
        kw.setdefault("deps_loader", _boom_loader)
        kw.setdefault("hash_file", self.tree.hash)
        kw.setdefault("now", lambda: BEFORE_DEADLINE)
        return cp.main(argv, out=self.buf.append, **kw)

    def last(self):
        return json.loads(self.buf[-1])

    def run_args(self):
        return ["--run", "--config", str(self.tree.cfg_path), "--teacher-closeout", str(self.tree.closeout)]

    def test_check_local_path_rejections(self):
        root = self.tree.roots.workspace
        good = str(self.tree.roots.front / cp.FROZEN_CLIPS[0][1])
        self.assertEqual(cp.check_local_path(good, root, ".mp4"), Path(good))
        outside = self.tmp / "outside.mp4"
        outside.write_bytes(b"x")
        bad = ["http://example.com/a.mp4", "rtsp://cam/stream", "0", "\\\\.\\pipe\\x", "//server/share/a.mp4",
               "relative/a.mp4", str(outside), good[:-4] + ".avi", good + " ",
               str(self.tree.roots.front / "ops" / ".." / "ops" / "x.mp4"), str(root / "missing.mp4"), None, ""]
        for p in bad:
            with self.assertRaises(cp.ConfigError, msg=repr(p)):
                cp.check_local_path(p, root, ".mp4")

    def test_valid_fake_config(self):
        plan = cp.validate_config(self.tree.cfg, self.tree.roots)
        self.assertEqual(len(plan["clips"]), 6)
        self.assertEqual(sum(c["score"] is not None for c in plan["clips"]), 4)

    def test_placeholder_and_example_config_refused(self):
        for mutate in (lambda c: c.update(status="SCHEMA_EXAMPLE_NOT_REAL_CONFIG"),
                       lambda c: c.update(unit_dir="<WORKSPACE>\\x")):
            cfg = json.loads(json.dumps(self.tree.cfg))
            mutate(cfg)
            with self.assertRaises(cp.ConfigError):
                cp.validate_config(cfg, self.tree.roots)

    def test_frozen_plan_tamper_refused(self):
        mutations = [
            lambda c: c["clips"][0]["score"].update(start_us=3_300_000),
            lambda c: c["clips"][4].update(score={"label": "안녕하세요", "start_us": 0, "end_us_exclusive": 1}),
            lambda c: c["clips"].reverse(),
            lambda c: c["extraction"].update(phase=1),
            lambda c: c["scoring"].update(checkpoint_sha256="1" * 64),
            lambda c: c.update(output_dir=str(self.tree.roots.unit / "result2")),
            lambda c: c["isolation"]["allowed_video_paths"].pop(),
            lambda c: c["pin_inventory"].pop(),
        ]
        for m in mutations:
            cfg = json.loads(json.dumps(self.tree.cfg))
            m(cfg)
            with self.assertRaises(cp.ConfigError):
                cp.validate_config(cfg, self.tree.roots)

    def test_default_preflight_constructs_nothing(self):
        heavy_before = {m for m in HEAVY if m in sys.modules}
        self.assertEqual(self.main(["--config", str(self.tree.cfg_path)]), 0)
        rep = self.last()
        self.assertEqual(rep["status"], "PREFLIGHT_OK")
        self.assertFalse(rep["ready_for_run"])                   # no receipt, no closeout given
        self.assertEqual(rep["constructed"], {"decoder": 0, "landmarker": 0, "model": 0})
        self.assertEqual({m for m in HEAVY if m in sys.modules}, heavy_before)
        self.assertFalse(self.tree.roots.result.exists())
        self.assertEqual(self.main([]), 0)                       # default config path = <unit>/PREPARED_CONFIG.json

    def test_bad_pin_fails_preflight(self):
        self.tree.override[cp._norm(self.tree.roots.front / cp.FROZEN_CLIPS[2][1])] = "f" * 64
        self.assertEqual(self.main(["--config", str(self.tree.cfg_path)]), 2)
        self.assertEqual(self.last()["kind"], "PIN_MISMATCH")

    def test_config_outside_unit_refused(self):
        other = self.tmp / cp.CONFIG_NAME
        other.write_text(json.dumps(self.tree.cfg), encoding="utf-8")
        self.assertEqual(self.main(["--config", str(other)]), 2)

    def test_run_gates_refuse_without_side_effects(self):
        self.assertEqual(self.main(["--run"]), 2)                                     # no --config
        self.assertEqual(self.main(self.run_args()), 2)                               # no receipt
        self.tree.write_receipt(code={"cadence_pair.py": "0" * 64, "test_cadence.py": "0" * 64})
        self.assertEqual(self.main(self.run_args()), 2)                               # stale receipt
        self.tree.write_receipt()
        late = datetime(2026, 10, 3, 23, 0, 0, tzinfo=timezone.utc)                    # 08:00 KST
        self.assertEqual(self.main(self.run_args(), now=lambda: late), 2)
        self.tree.set_closeout("RUNNING")
        self.assertEqual(self.main(self.run_args()), 2)
        self.tree.set_closeout("COMPLETE")
        self.assertEqual(self.main(["--run", "--config", str(self.tree.cfg_path)]), 2)  # no closeout arg
        self.assertFalse(self.tree.roots.result.exists())
        self.tree.roots.result.mkdir()
        marker = self.tree.roots.result / "keep.txt"
        marker.write_text("previous", encoding="utf-8")
        self.assertEqual(self.main(self.run_args()), 2)                               # output not fresh
        self.assertEqual(marker.read_text(encoding="utf-8"), "previous")
        self.assertEqual(sorted(p.name for p in self.tree.roots.result.iterdir()), ["keep.txt"])

    def test_closeout_status_tokens(self):
        accepted = ("COMPLETE", "COMPLETED", " completed ",
                    "COMPLETE_TECHNICAL_PREFLIGHT_STOP_NO_PERFORMANCE_RESULT",
                    "COMPLETE_INDEPENDENTLY_VERIFIED_REJECTED_NO_PROMOTION", "COMPLETED_REJECTED")
        refused = ("INCOMPLETE", "NOT_COMPLETE", "COMPLETE_PENDING", "COMPLETE_PENDING_REVIEW",
                   "COMPLETED_RUNNING", "COMPLETE_WAITING_FOR_CODEX", "COMPLETE_IN_PROGRESS", "COMPLETEX",
                   "COMPLETE-STOP", "RUNNING", "PENDING_COMPLETE", "", None, 1)
        for s in accepted:
            self.assertTrue(cp.closeout_status_closed(s), repr(s))
        for s in refused:
            self.assertFalse(cp.closeout_status_closed(s), repr(s))

    def test_closeout_preserves_original_details(self):
        status = "COMPLETE_TECHNICAL_PREFLIGHT_STOP_NO_PERFORMANCE_RESULT"
        body = {"status": status, "verdict": "NUMERICAL_PREFLIGHT_STOP", "error": "nonfinite loss at step 0",
                "main_training_steps": 0}
        self.tree.closeout.write_text(json.dumps(body), encoding="utf-8")
        got = cp.check_closeout(str(self.tree.closeout), self.tree.roots, self.tree.hash)
        self.assertEqual(got["status"], status)
        self.assertEqual(got["verdict"], "NUMERICAL_PREFLIGHT_STOP")
        self.assertEqual(got["details"], {"verdict": "NUMERICAL_PREFLIGHT_STOP", "error": "nonfinite loss at step 0"})
        self.assertEqual(got["sha256"], self.tree.hash(self.tree.closeout))
        self.tree.set_closeout("COMPLETE_PENDING")
        with self.assertRaises(cp.GateError):
            cp.check_closeout(str(self.tree.closeout), self.tree.roots, self.tree.hash)
        self.tree.write_receipt()                                 # closeout is the only refusing gate
        self.tree.set_closeout("INCOMPLETE")
        self.assertEqual(self.main(self.run_args()), 2)
        self.assertEqual(self.last()["kind"], "RUN_GATE")
        self.assertIn("INCOMPLETE", self.last()["error"])
        self.assertFalse(self.tree.roots.result.exists())

    def test_run_end_to_end_with_fakes(self):
        self.tree.write_receipt()
        factory = Factory()
        holder = {}

        def loader(plan, roots, hash_file):
            deps, _, rec = make_deps(self.tree.pts_by_path, factory=factory)
            holder["rec"] = rec
            return deps

        self.assertEqual(self.main(self.run_args(), deps_loader=loader), 0, self.buf)
        rd = self.tree.roots.result
        res = json.loads((rd / "RESULT.json").read_text(encoding="utf-8"))
        self.assertEqual(res["status"], "COMPLETE")
        self.assertEqual(res["execution_counts"]["actual"], cp.EXPECTED_COUNTS)
        self.assertEqual(res["execution_counts"]["actual"]["landmarker_process_calls"], 1450)
        self.assertEqual(holder["rec"]["forwards"], 8)
        self.assertEqual(len(factory.instances), 13)
        self.assertEqual(len(res["scoring"]["windows"]), 4)
        self.assertEqual({w["label"] for w in res["scoring"]["windows"]}, {"나", "죄송하다"})
        self.assertTrue(all(len(w["A"]["logits93"]) == 93 for w in res["scoring"]["windows"]))
        self.assertEqual([c["frames"]["selected"] for c in res["clips"]], [85, 83, 66, 59, 69, 66])
        self.assertTrue(res["pins_unchanged"])
        self.assertEqual(len(list(rd.glob("*.npz"))), 13)
        self.assertFalse((rd / "FAILURE.json").exists())
        self.assertTrue((rd / "SUMMARY.md").is_file())
        hashes = json.loads((rd / "FILE_HASHES.json").read_text(encoding="utf-8"))
        self.assertEqual(hashes["RESULT.json"], cp.sha256_file(rd / "RESULT.json"))
        with np.load(rd / f"02_{cp.FROZEN_CLIPS[0][0]}_B.npz", allow_pickle=False) as z:
            self.assertEqual(z["obs_frame_id"].tolist(), list(range(0, 169, 2)))
            self.assertEqual(int(z["decoded_processed"].sum()), 85)
            self.assertNotIn("rgb", " ".join(k for k in z.files if "sha" not in k and "shape" not in k))
        # rerun must not touch the finished result
        snapshot = {p.name: cp.sha256_file(p) for p in rd.iterdir()}
        self.assertEqual(self.main(self.run_args(), deps_loader=loader), 2)
        self.assertEqual({p.name: cp.sha256_file(p) for p in rd.iterdir()}, snapshot)

    def test_run_failure_preserved(self):
        self.tree.write_receipt()
        factory = Factory(noise_on={1})

        def loader(plan, roots, hash_file):
            return make_deps(self.tree.pts_by_path, factory=factory)[0]

        self.assertEqual(self.main(self.run_args(), deps_loader=loader), 3)
        rd = self.tree.roots.result
        fail = json.loads((rd / "FAILURE.json").read_text(encoding="utf-8"))
        self.assertEqual(fail["kind"], "NONDETERMINISTIC")
        self.assertTrue(fail["no_accuracy_verdict"])
        self.assertEqual(fail["counters"]["landmarker_process_calls"], 169 * 2)
        self.assertEqual(fail["counters"]["tcn_forwards"], 0)
        self.assertEqual(fail["pin_mismatches_after"], [])
        self.assertFalse((rd / "RESULT.json").exists())
        self.assertTrue((rd / "DETERMINISM.json").is_file())
        self.assertEqual(len(list(rd.glob("*.npz"))), 2)


# ---------------------------------------------------------------- receipt
def _write_receipt(result) -> Path:
    p = HERE / cp.RECEIPT_NAME
    receipt = {"schema": cp.RECEIPT_SCHEMA, "status": "PASS", "num_tests": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "code_sha256": cp.own_code_hashes(),
               "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "scope": "synthetic fakes only: no media decoded, no MediaPipe/torch model, no camera"}
    with open(p, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(receipt, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return p


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful() or res.testsRun == 0:
        print("tests failed; no receipt written")
        sys.exit(1)
    try:
        print("receipt:", _write_receipt(res))
    except FileExistsError:
        print(f"{cp.RECEIPT_NAME} already exists; not overwritten (remove it deliberately to re-issue)")
        sys.exit(1)
    sys.exit(0)
