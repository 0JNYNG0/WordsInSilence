"""Focused synthetic checks for span_factorial.py. No MediaPipe, model, or private data.
official_prep/input_record come verbatim from API_REFERENCE.txt (the pinned helper source)."""
import hashlib
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

import span_factorial as sf

HERE = Path(__file__).resolve().parent
N_LM, FRAMES = 75, 78


def reference_helpers():
    text = (HERE / "API_REFERENCE.txt").read_text("utf-8")
    src = text[text.index("def official_prep"):text.index("Model construction used locally")]

    class TechnicalFailure(Exception):
        pass

    def _req(c, m):
        if not c:
            raise TechnicalFailure(m)
    ns = {"math": math, "_req": _req, "TechnicalFailure": TechnicalFailure}
    exec(src, ns)
    return types.SimpleNamespace(**ns)


REF = reference_helpers()


def to_native_like(xy, usable):
    present = usable[:, 33:].reshape(len(usable), 2, 21).all(-1)
    return np.where(np.isfinite(xy), xy, 0).astype(np.float32).reshape(len(xy), -1), present


def build_channels(nat, present, ts):
    x = np.concatenate([nat, present.astype(np.float32)], 1)
    return {"x": x, "body_scale": float(nat[:, 0].std() + 1.0), "hand_scales": {"L": None, "R": 1.0}}


def collate(items):
    x = np.stack([it["x"] for it in items]).astype(np.float32)
    return x, np.ones(x.shape[:2], np.float32), [it["label"] for it in items]


FNS = {"to_native_like": to_native_like, "build_channels": build_channels, "collate": collate,
       "n_channels": N_LM * 2 + 2}


def synthetic(hand_missing=()):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 640, (FRAMES, N_LM, 2)).astype(np.float32)
    usable = np.ones((FRAMES, N_LM), bool)
    for i in hand_missing:
        usable[i, 33:] = False
        xy[i, 33:] = np.nan
    return {"obs_xy_px": xy, "obs_usable": usable,
            "obs_timestamp_us": np.arange(FRAMES, dtype=np.int64) * 33333,
            "obs_frame_id": np.arange(FRAMES, dtype=np.int64) + 100,
            "obs_observed": usable.copy(), "obs_conf": np.ones((FRAMES, N_LM), np.float32),
            "obs_hand_score": np.ones((FRAMES, 2), np.float32)}


def readonly(obs):
    for a in obs.values():
        a.setflags(write=False)
    return obs


C_MISSING = (20, *range(70, FRAMES))


class SpanFactorialTests(unittest.TestCase):
    def test_array_sha256_format(self):
        a = np.arange(6, dtype="<i4").reshape(2, 3)
        self.assertEqual(sf.array_sha256(a), hashlib.sha256(b"<i4|2x3|" + a.tobytes()).hexdigest())

    def test_load_obs_seven_readonly_never_selected(self):
        obs = synthetic()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.npz"
            np.savez(p, selected_frame_id=np.arange(112, 170), decoded_xy=np.zeros(3), **obs)
            got = sf.load_obs(p)
        self.assertEqual(tuple(got), sf.OBS_KEYS)
        self.assertTrue(all(not a.flags.writeable for a in got.values()))
        with self.assertRaises(ValueError):
            got["obs_xy_px"][0, 0, 0] = 1

    def test_explicit_inclusive_spans_use_all_stored_rows(self):
        obs = readonly(synthetic(hand_missing=(13, 40)))
        for lo, hi in ([12, 74], [12, 69]):
            p = sf.prepare(REF, obs, [lo, hi], FNS, FRAMES)
            self.assertEqual(p["x"].shape, (1, hi - lo + 1, FNS["n_channels"]))
            self.assertEqual(p["frame_id"], list(range(100 + lo, 100 + hi + 1)))
            self.assertEqual(p["present_count"], 2 * (hi - lo + 1) - 4)  # missing-hand rows kept, not dropped
            rec = sf.record(REF, p)
            self.assertEqual(rec, sf.record(REF, sf.prepare(REF, obs, [lo, hi], FNS, FRAMES)))
            self.assertIn("pad_sha256", rec)
            self.assertEqual(rec["hand_scales"]["L"], {"value": None, "status": "absent"})
        with self.assertRaises(sf.TechnicalFailure):
            sf.prepare(REF, obs, [12, 78], FNS, FRAMES)

    def test_c_long_not_retrimmed_pad1_source_unchanged(self):
        obs = readonly(synthetic(hand_missing=C_MISSING))
        before = sf.obs_hashes(obs)
        p = sf.prepare(REF, obs, [12, 74], FNS, FRAMES)
        self.assertEqual(p["x"].shape[1], 63)
        self.assertTrue(np.all(p["pad"] == 1))
        self.assertTrue(np.all(p["x"][0, 58:, -2:] == 0))  # rows 70..74 present-channels empty, real PTS kept
        self.assertEqual(p["timestamp_us"][-5:], [i * 33333 for i in range(70, 75)])
        self.assertEqual(sf.obs_hashes(obs), before)

        def mutating(xy, usable):
            xy[0] = 0
            return to_native_like(xy, usable)
        with self.assertRaises(ValueError):
            sf.prepare(REF, obs, [12, 74], dict(FNS, to_native_like=mutating), FRAMES)

    def test_pair_requires_identical_pose_ids_pts(self):
        b, c = synthetic(), synthetic(hand_missing=C_MISSING)
        sf.check_pair(b, c, FRAMES)
        c["obs_xy_px"][5, 0, 0] += 1
        with self.assertRaises(sf.TechnicalFailure):
            sf.check_pair(b, c, FRAMES)

    def test_diagonals_reused_only_two_cross_forwards(self):
        self.assertEqual([c["id"] for c in sf.CROSS], ["BC", "CB"])
        calls = []
        stored = {"BB": [0.0] * 93, "CC": [1.0] * 93}
        logits, reused = sf.collect_logits(["BB", "CC", "BC", "CB"], stored,
                                           lambda cid: calls.append(cid) or [2.0] * 93)
        self.assertEqual(calls, ["BC", "CB"])
        self.assertEqual(reused, {"BB": True, "CC": True, "BC": False, "CB": False})
        self.assertIs(logits["BB"], stored["BB"])
        with self.assertRaises(sf.TechnicalFailure):
            sf.collect_logits(["BB", "CC", "BC", "CB"], {"BB": stored["BB"]}, lambda cid: [0.0] * 93)

    def test_four_patterns(self):
        got = {k: sf.interpret(*k).split(":")[0] for k in sf.PATTERNS}
        self.assertEqual(got, {(True, False): "observation_bundle", (False, True): "span_renormalization_bundle",
                               (True, True): "interaction", (False, False): "B_long_only"})

    def test_full93_ranking_and_truth_scoring(self):
        classes = [f"c{i}" for i in range(93)]
        l = [float(i % 10) for i in range(93)]
        l[92], l[5], l[7] = 50.0, 9.5, 3.0
        self.assertEqual(sf.top1(l, classes), "c92")  # top beyond any truncated subset
        s = sf.score(l, classes, "c5", "c7")
        self.assertEqual((s["top1"], s["correct"], s["best_rival"]), ("c92", False, "c92"))
        self.assertEqual(s["true_rank"], 2)
        self.assertEqual((s["true_margin"], s["fixed_rival_gap"]), (9.5 - 50.0, 6.5))
        with self.assertRaises(sf.TechnicalFailure):
            sf.top1(l[:92], classes[:92])
        with self.assertRaises(sf.TechnicalFailure):
            sf.top1(l[:92] + [float("nan")], classes)

    def test_contrasts_interaction(self):
        s = {k: {"true_margin": v, "fixed_rival_gap": -v} for k, v in {"BB": 1.0, "CC": -2.0, "BC": 0.5, "CB": -0.5}.items()}
        c = sf.contrasts(s)["true_margin"]
        self.assertEqual(c["span_on_B_obs_BC_minus_BB"], -0.5)
        self.assertEqual(c["obs_on_long_span_CB_minus_BB"], -1.5)
        self.assertEqual(c["interaction_CC_minus_BC_minus_CB_plus_BB"], -1.0)

    def test_example_config_rejected(self):
        cfg = json.loads((HERE / "CONFIG_EXAMPLE.json").read_text("utf-8"))
        self.assertEqual(set(cfg), sf.KEYS)
        for c in (cfg, dict(cfg, is_example=False)):
            with self.assertRaises(sf.TechnicalFailure):
                sf.validate_config(c)


if __name__ == "__main__":
    unittest.main()
