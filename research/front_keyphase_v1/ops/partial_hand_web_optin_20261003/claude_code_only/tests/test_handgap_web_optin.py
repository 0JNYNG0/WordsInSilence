"""2026-10-03: the opt-in hand-gap comparison entry, exercised with synthetic inputs only.

No camera, no device, no media, no checkpoint, no torch, no pilot package: the pilot adapter
functions, torch and the model are replaced by small deterministic fakes for the length of each
test, and every model cache is stubbed READY. The real hand_gap.restore is used (it is pure NumPy).
Real-model parity is checked separately by the verifier, not here.

Run: py -3.13 tests/test_handgap_web_optin.py
  or py -3.13 -m unittest discover -s tests -p test_handgap_web_optin.py
"""
import contextlib
import hashlib
import inspect
import json
import os
import sys
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wis_lab.core.observation import HAND_SLICES, N_POINTS, Observation, POSE_SLICE  # noqa: E402
from wis_lab.service import neural as N  # noqa: E402
from wis_lab.service import review_log  # noqa: E402

RAW = "tcn_control_83212868"
GAP = "tcn_control_83212868_handgap"
SHA832 = "83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91"
FROZEN_HANDGAP_SHA = "6ae29cea6ae030d8567a9c79749cf8717738a1b2c2b73eaa93d4b38fd539ac07"
HANDGAP_PATH = os.path.join(ROOT, "src", "wis_lab", "service", "hand_gap.py")
LABELS = sorted(set(N.PROJECT_18.values())) + [f"cls{i:02d}" for i in range(93 - 18)]
BASE_LABELS = sorted(set(N.PROJECT_OVERLAP.values())) + [f"b{i:02d}" for i in range(80 - 5)]
FILL_KEYS = {"side", "source_row", "span_row", "source_frame_id", "timestamp_us",
             "neighbour_source_rows", "neighbour_timestamps_us", "estimated_points", "kind"}

CALLS = []          # every build_channels call, in order: one per arm that reached it


# ------------------------------------------------------------------ fakes
def _to_native_like(xy, usable):
    """75 -> 67: first 25 pose points, then left hand (25:46) and right hand (46:67)."""
    nat = np.concatenate([xy[:, :25], xy[:, 33:75]], axis=1).astype(np.float32)
    present = np.concatenate([usable[:, :25], usable[:, 33:75]], axis=1).astype(bool)
    return nat, present


def _active_span(present):
    idx = np.flatnonzero(present[:, 25:67].any(axis=1))
    return None if idx.size == 0 else (int(idx[0]), int(idx[-1]))


def _build_channels(xy, present, ts):
    CALLS.append({"xy": np.array(xy, copy=True), "present": np.array(present, copy=True),
                  "ts": np.array(ts, copy=True)})
    return {"frames": int(len(ts)), "_xy": np.array(xy, copy=True),
            "_present": np.array(present, copy=True)}


def _collate(batch):
    return batch[0], batch[0], None


class _T:
    def __init__(self, a):
        self.a = a

    def numpy(self):
        return self.a


def _model(x, _pad):
    logit = -np.arange(93, dtype=np.float32) * np.float32(0.01)
    logit[3] += np.float32(x["_present"].sum() * 1e-3)
    return [_T(logit)]


def _base_model(x, _pad):
    return [_T(-np.arange(80, dtype=np.float32) * np.float32(0.01))]


def _fake_modules():
    pilot = types.ModuleType("pilot")
    data = types.ModuleType("pilot.data")
    data.build_channels, data.collate = _build_channels, _collate
    web = types.ModuleType("pilot.web_input")
    web.to_native_like, web.active_span = _to_native_like, _active_span
    pilot.data, pilot.web_input = data, web
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda a: a
    torch.inference_mode = contextlib.nullcontext
    return {"pilot": pilot, "pilot.data": data, "pilot.web_input": web, "torch": torch}


def _ready(model, labels):
    return {"model": model, "classes": list(labels), "status": "READY", "detail": None,
            "checkpoint_step": 1, "model_version": "synthetic", "checkpoint_sha256": "synthetic",
            "feature_schema": "pilot.data.build_channels", "in_channels": 280}


def _file_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------------------------ synthetic observation
def make_obs(t=16, hands=range(2, 14), drop=(), step=33_000, ts=None, frame_id=None):
    """Pose always seen; both hands seen on `hands`; drop = (frame, side 0/1, joint index or slice)."""
    xy = np.full((t, N_POINTS, 2), np.nan, dtype=np.float32)
    observed = np.zeros((t, N_POINTS), dtype=bool)
    xy[:, POSE_SLICE, 0] = 50.0
    xy[:, POSE_SLICE, 1] = 60.0
    observed[:, POSE_SLICE] = True
    j = np.arange(21, dtype=np.float32)
    for f in hands:
        for k, sl in enumerate(HAND_SLICES):
            xy[f, sl, 0] = 100.0 + 50.0 * k + j + 2.0 * f
            xy[f, sl, 1] = 300.0 + j - 1.0 * f
            observed[f, sl] = True
    for f, side, joints in drop:
        sl = HAND_SLICES[side]
        idx = np.arange(sl.start, sl.stop)[joints]
        xy[f, idx] = np.nan
        observed[f, idx] = False
    ts = (np.asarray(ts, dtype=np.int64) if ts is not None
          else 1_000_000 + np.arange(t, dtype=np.int64) * step)
    fid = (np.asarray(frame_id, dtype=np.int64) if frame_id is not None
           else 500 + np.arange(t, dtype=np.int64))
    obs = Observation(xy_px=xy, timestamp_us=ts, frame_id=fid, observed=observed,
                      usable=observed.copy(), conf=np.full((t, N_POINTS), 0.9, np.float32),
                      hand_score=np.full((t, 2), 0.8, np.float32), sidecar={})
    for a in (obs.xy_px, obs.timestamp_us, obs.frame_id, obs.observed, obs.usable, obs.conf,
              obs.hand_score):
        a.setflags(write=False)      # any in-place write anywhere in the adapter raises
    return obs


ALL = slice(None)


class HandGapOptIn(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        self._mods = {k: sys.modules.get(k) for k in ("pilot", "pilot.data", "pilot.web_input",
                                                      "torch")}
        sys.modules.update(_fake_modules())
        self._caches = dict(N._CACHES)
        N._CACHES[RAW] = _ready(_model, LABELS)
        N._CACHES[GAP] = _ready(_model, LABELS)
        N._CACHES["base"] = _ready(_base_model, BASE_LABELS)
        self._hg = (N.HANDGAP_MODULE, N.HANDGAP_SOURCE_SHA256, dict(N._HANDGAP))
        # the functional tests use the supplied file whatever its bytes; the pin itself is tested
        # on its own in test_pinned_source_is_the_frozen_candidate
        N.HANDGAP_SOURCE_SHA256 = _file_sha(HANDGAP_PATH)
        N._HANDGAP.update(module=None, status=None, detail=None, source_sha256=None)

    def tearDown(self):
        for k, v in self._mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        N._CACHES.clear()
        N._CACHES.update(self._caches)
        N.HANDGAP_MODULE, N.HANDGAP_SOURCE_SHA256 = self._hg[0], self._hg[1]
        N._HANDGAP.clear()
        N._HANDGAP.update(self._hg[2])

    def run_gap(self, obs):
        out = N.recognise(obs, GAP)
        json.dumps(out, ensure_ascii=False, allow_nan=False)        # strict JSON, always
        return out, out["hand_gap_comparison"]["original"], out["input_transform"]["provenance"]

    # ---------------------------------------------------------- registry and identity
    def test_pinned_source_is_the_frozen_candidate(self):
        self.assertEqual(self._hg[1], FROZEN_HANDGAP_SHA)   # the module's own pin, pre-setUp
        self.assertEqual(_file_sha(HANDGAP_PATH), FROZEN_HANDGAP_SHA,
                         "hand_gap.py must be byte-identical to the frozen candidate")

    def test_registry_entry(self):
        self.assertEqual(N.DEFAULT_MODEL, "base")
        spec, raw = N.MODELS[GAP], N.MODELS[RAW]
        self.assertEqual((spec["path"], spec["sha256"]), (raw["path"], raw["sha256"]))
        self.assertEqual(spec["sha256"], SHA832)
        self.assertEqual(spec["model_id"], GAP)
        self.assertEqual(sum(1 for s in N.MODELS.values() if s["model_id"] == GAP), 1)
        self.assertEqual(spec["status_label"], "시험용")
        self.assertEqual(spec["input_transform"], N.HANDGAP_TRANSFORM_ID)
        self.assertEqual([k for k, s in N.MODELS.items() if s.get("input_transform")], [GAP])
        # the existing entries are exactly as they were
        self.assertEqual({k: (s["model_id"], s["sha256"], s["status_label"])
                          for k, s in N.MODELS.items() if k != GAP}, {
            "base": ("masked_sign_tcn_bn_pair_best", N.CHECKPOINT_SHA256, "기본"),
            "union93_candidate_v1": ("union93_candidate_v1", "e7c9c31a9c747e2c12a9a56695d2e4e9dbec9c372c8e2a3f209d00fa4c25dced", "검증 중"),
            "union93_visibility_v1": ("union93_visibility_v1", "258f23e1ad515a25b50e6affac8ac4e284ba8f6ae6fac3372c1e7a31ea6848a8", "검증 중"),
            RAW: (RAW, SHA832, "시험용")})
        self.assertIn("NOT a new trained model", spec["role"])
        self.assertIn("not evidence of better accuracy", spec["role"])

    def test_identity_is_distinct_and_carries_the_transform(self):
        raw, gap = N.model_identity(RAW), N.model_identity(GAP)
        self.assertNotIn("input_transform", raw)
        self.assertEqual(raw["checkpoint_sha256"], gap["checkpoint_sha256"])
        self.assertNotEqual(raw["model_id"], gap["model_id"])
        tx = gap["input_transform"]
        self.assertEqual(tx["transform_id"], N.HANDGAP_TRANSFORM_ID)
        self.assertEqual(tx["status"], "READY")
        self.assertEqual(tx["loaded_source_sha256"], tx["pinned_source_sha256"])
        self.assertFalse(tx["is_a_trained_model"])
        self.assertEqual(gap["trained_class_set"]["model_id"], GAP)
        self.assertEqual(raw["trained_class_set"]["model_id"], RAW)

    def test_review_log_never_pools_raw_and_transformed(self):
        rows = []
        for i, key in enumerate((RAW, GAP)):
            creation = dict(N.model_identity(key), opted_in=True)
            rows.append({"trial_id": f"t{i}", "actual_word_id": "NA", "result": None,
                         "result_reason": "NO_RESULT_FILE", "creation_research_model": creation})
        groups = review_log.cohorts(rows)["groups"]
        self.assertEqual(len(groups), 4)                  # two identities x two scopes
        self.assertEqual({g["identity"]["model_id"] for g in groups.values()}, {RAW, GAP})
        self.assertTrue(all(g["n_supported"] == 1 for g in groups.values()))

    def test_app_identity_field_is_additive_only(self):
        with open(os.path.join(ROOT, "src", "wis_lab", "service", "app.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('if choice.get("input_transform") is not None:', src)
        self.assertNotIn("hand_gap", src)                 # feedback/DTW paths never see it
        self.assertIn("result = predict(prep, st.refs, st.contract)", src)

    # ---------------------------------------------------------- blind, opt-out, default
    def test_signature_is_blind(self):
        self.assertEqual(list(inspect.signature(N.recognise).parameters), ["obs", "model_key"])
        out, orig, _ = self.run_gap(make_obs())
        for block in (out, orig):
            self.assertIsNone(block["accepted"])
            self.assertFalse(block["is_a_verdict"])
            self.assertEqual(block["execution_assessment"], "UNCONFIGURED")
            self.assertNotIn("requested_word_coverage", block)

    def test_opt_out_raw_entry_unchanged(self):
        obs = make_obs(drop=[(7, 1, ALL)])
        out = N.recognise(obs, RAW)
        for k in ("hand_gap_comparison", "input_transform", "arm"):
            self.assertNotIn(k, out)
        self.assertEqual(out["model_id"], RAW)
        _, orig, _ = self.run_gap(obs)
        self.assertEqual(orig["scopes"], out["scopes"])
        self.assertEqual(orig["support"], out["support"])

    def test_default_is_base(self):
        out = N.recognise(make_obs())
        self.assertEqual(out["model_key"], "base")
        self.assertNotIn("hand_gap_comparison", out)
        self.assertEqual(out["scopes"]["full"]["candidate_count"], 80)

    # ---------------------------------------------------------- observation and span
    def test_observation_not_mutated(self):
        obs = make_obs(drop=[(7, 1, ALL), (9, 0, slice(0, 5))])
        before = {k: np.array(getattr(obs, k), copy=True) for k in
                  ("xy_px", "timestamp_us", "frame_id", "observed", "usable")}
        nan_before = np.isnan(obs.xy_px).copy()
        out, orig, prov = self.run_gap(obs)
        self.assertEqual((out["model_status"], orig["model_status"]), ("READY", "READY"))
        self.assertEqual(prov["status"], "FILLED")
        for k, v in before.items():
            self.assertTrue(np.array_equal(getattr(obs, k), v, equal_nan=(k == "xy_px")), k)
        self.assertTrue(np.array_equal(np.isnan(obs.xy_px), nan_before))
        self.assertTrue(np.isnan(obs.xy_px[7, HAND_SLICES[1]]).all())   # still absent

    def test_active_span_preserved(self):
        obs = make_obs(t=18, hands=range(3, 15), drop=[(8, 0, ALL)])
        out, orig, prov = self.run_gap(obs)
        self.assertEqual(len(CALLS), 2)
        self.assertTrue(np.array_equal(CALLS[0]["ts"], obs.timestamp_us[3:15]))
        self.assertTrue(np.array_equal(CALLS[1]["ts"], obs.timestamp_us[3:15]))
        self.assertEqual(CALLS[0]["xy"].shape, CALLS[1]["xy"].shape)
        self.assertEqual(orig["support"], out["support"])
        span = prov["source_active_span"]
        self.assertEqual((span["source_row_first"], span["source_row_last"]), (3, 14))
        self.assertEqual((span["source_frame_id_first"], span["source_frame_id_last"]), (503, 514))
        self.assertEqual(span["timestamp_us_first"], int(obs.timestamp_us[3]))

    def test_insufficient_support_is_not_repaired_into_input(self):
        # hands on 2, 4, 6 only: 3 observed frames < 4. Frames 3 and 5 are each a one-frame gap
        # restore WOULD fill - the gate must refuse before it ever runs.
        out, orig, prov = self.run_gap(make_obs(hands=(2, 4, 6)))
        for block in (out, orig):
            self.assertEqual(block["model_status"], "INSUFFICIENT_OBSERVATION")
            self.assertEqual(block["detail"], "TOO_FEW_OBSERVED_HAND_FRAMES")
            self.assertEqual(block["support"]["observed_hand_frames"], 3)
            self.assertNotIn("scopes", block)
        self.assertEqual(prov["status"], "NOT_ATTEMPTED")
        self.assertEqual(CALLS, [])

    def test_malformed_timestamps_fail_safely(self):
        ts = 1_000_000 + np.arange(16, dtype=np.int64) * 33_000
        ts[8] = ts[6]
        out, orig, prov = self.run_gap(make_obs(ts=ts, drop=[(7, 1, ALL)]))
        for block in (out, orig):
            self.assertEqual(block["detail"], "NON_MONOTONIC_TIMESTAMPS")
        self.assertEqual(prov["status"], "NOT_ATTEMPTED")
        self.assertEqual(CALLS, [])

    # ---------------------------------------------------------- the repair itself
    def test_zero_fill_identity(self):
        out, orig, prov = self.run_gap(make_obs())
        self.assertEqual(prov["status"], "IDENTITY_NO_FILL")
        self.assertTrue(prov["identity"])
        self.assertEqual(prov["fills"], [])
        self.assertEqual(prov["filled_hand_frames"], {"left": 0, "right": 0, "total": 0})
        self.assertEqual(prov["hand_points"]["estimated_total"], 0)
        a, b = CALLS
        self.assertTrue(np.array_equal(a["xy"], b["xy"], equal_nan=True))
        self.assertTrue(np.array_equal(a["present"], b["present"]))
        audit = out["hand_gap_comparison"]["logit_audit"]
        self.assertEqual(audit["original_coordinates"], audit["short_gap_restored"])
        self.assertEqual(audit["max_abs_difference"], 0.0)
        self.assertEqual(out["scopes"], orig["scopes"])

    def test_one_frame_fill(self):
        obs = make_obs(drop=[(7, 1, ALL)])
        out, orig, prov = self.run_gap(obs)
        self.assertEqual(prov["status"], "FILLED")
        self.assertFalse(prov["identity"])
        self.assertEqual(prov["source_index_kind"], "frame_id")
        self.assertEqual(prov["filled_hand_frames"], {"left": 0, "right": 1, "total": 1})
        self.assertEqual(prov["hand_points"]["estimated"], {"left": 0, "right": 21})
        self.assertEqual(prov["hand_points"]["observed"], {"left": 12 * 21, "right": 11 * 21})
        (fill,) = prov["fills"]
        self.assertEqual(set(fill), FILL_KEYS)               # rows/times/sides, never coordinates
        self.assertEqual((fill["side"], fill["source_row"], fill["span_row"]), ("right", 7, 5))
        self.assertEqual(fill["source_frame_id"], 507)
        self.assertEqual(fill["timestamp_us"], int(obs.timestamp_us[7]))
        self.assertEqual(fill["neighbour_source_rows"], [6, 8])
        self.assertEqual(fill["kind"], "estimated_not_observed")
        self.assertFalse(prov["coordinates_included"])
        # the original gate counted only real observation, in both arms
        self.assertEqual(out["support"], orig["support"])
        self.assertEqual(out["support"]["observed_hand_frames"], 12)
        # what reached build_channels: original absent, repaired estimated at the midpoint
        raw, rep = CALLS
        self.assertFalse(raw["present"][5, 46:67].any())
        self.assertTrue(rep["present"][5, 46:67].all())
        j = np.arange(21, dtype=np.float32)
        self.assertTrue(np.allclose(rep["xy"][5, 46:67, 0], 150.0 + j + 2.0 * 7))
        self.assertTrue(np.array_equal(np.delete(raw["present"], 5, axis=0),
                                       np.delete(rep["present"], 5, axis=0)))
        audit = out["hand_gap_comparison"]["logit_audit"]
        self.assertEqual(audit["label_order"], LABELS)
        self.assertEqual(len(audit["original_coordinates"]), 93)
        self.assertEqual(len(audit["short_gap_restored"]), 93)
        self.assertNotEqual(audit["original_coordinates"], audit["short_gap_restored"])
        self.assertEqual(out["model_id"], GAP)
        self.assertEqual(out["arm"], "short_gap_restored")
        self.assertEqual(orig["arm"], "original_coordinates")
        self.assertNotIn("hand_gap_comparison", orig)       # no recursion

    def test_frame_id_fallback_to_source_row(self):
        fid = 500 + np.arange(16, dtype=np.int64)
        fid[10] = fid[9]
        _, _, prov = self.run_gap(make_obs(drop=[(7, 1, ALL)], frame_id=fid))
        self.assertEqual(prov["source_index_kind"], "source_row")
        self.assertEqual(prov["source_frame_id_unused_reason"], "FRAME_ID_NOT_INCREASING")
        self.assertIsNone(prov["fills"][0]["source_frame_id"])
        self.assertEqual(prov["fills"][0]["source_row"], 7)

    def test_no_fill_cases(self):
        cases = {
            "two_frame": dict(drop=[(7, 1, ALL), (8, 1, ALL)]),
            "partial_hand": dict(drop=[(7, 1, slice(0, 10))]),
            "long_gap": dict(drop=[(7, 1, ALL)], step=40_000),
            "span_boundary": dict(drop=[(2, 0, ALL), (13, 1, ALL)]),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                CALLS.clear()
                out, orig, prov = self.run_gap(make_obs(**kw))
                self.assertEqual(out["model_status"], "READY")
                self.assertEqual(prov["status"], "IDENTITY_NO_FILL")
                self.assertEqual(prov["fills"], [])
                audit = out["hand_gap_comparison"]["logit_audit"]
                self.assertEqual(audit["original_coordinates"], audit["short_gap_restored"])
                self.assertTrue(np.array_equal(CALLS[0]["present"], CALLS[1]["present"]))

    # ---------------------------------------------------------- failure, no fallback
    def _assert_unavailable(self, out, orig, status, needle):
        self.assertEqual(out["model_status"], status)
        self.assertIn(needle, out["detail"])
        self.assertIsNone(out["ranking"])
        self.assertNotIn("scopes", out)                     # never borrowed from the original
        self.assertEqual(out["model_id"], GAP)
        self.assertEqual(orig["model_status"], "READY")
        self.assertIn("scopes", orig)
        audit = out["hand_gap_comparison"]["logit_audit"]
        self.assertIsNone(audit["short_gap_restored"])
        self.assertEqual(len(audit["original_coordinates"]), 93)

    def test_helper_import_failure(self):
        N.HANDGAP_MODULE = "wis_lab.service._no_such_hand_gap_module"
        out, orig, prov = self.run_gap(make_obs(drop=[(7, 1, ALL)]))
        self._assert_unavailable(out, orig, "INPUT_TRANSFORM_UNAVAILABLE", "TRANSFORM_IMPORT_FAILED")
        self.assertEqual(prov["status"], "UNAVAILABLE")
        self.assertEqual(out["input_transform"]["status"], "TRANSFORM_IMPORT_FAILED")
        self.assertEqual(len(CALLS), 1)                     # only the original reached the model

    def test_helper_hash_mismatch(self):
        N.HANDGAP_SOURCE_SHA256 = "0" * 64
        out, orig, _ = self.run_gap(make_obs(drop=[(7, 1, ALL)]))
        self._assert_unavailable(out, orig, "INPUT_TRANSFORM_UNAVAILABLE",
                                 "TRANSFORM_SOURCE_HASH_MISMATCH")

    def test_helper_raises(self):
        def bad_restore(*_a):
            raise ValueError("ts must be strictly increasing")
        N._HANDGAP.update(module=types.SimpleNamespace(restore=bad_restore), status="READY",
                          detail=None, source_sha256=N.HANDGAP_SOURCE_SHA256)
        out, orig, prov = self.run_gap(make_obs(drop=[(7, 1, ALL)]))
        self._assert_unavailable(out, orig, "INPUT_TRANSFORM_FAILED", "ValueError")
        self.assertEqual(prov["status"], "UNAVAILABLE")

    def test_model_not_ready_is_not_answered_by_the_raw_entry(self):
        N._CACHES[GAP] = {"model": None, "classes": None, "status": "CHECKPOINT_MISSING",
                          "detail": "the pinned checkpoint is not present"}
        out, orig, prov = self.run_gap(make_obs(drop=[(7, 1, ALL)]))
        self.assertEqual((out["model_status"], orig["model_status"]),
                         ("CHECKPOINT_MISSING", "CHECKPOINT_MISSING"))
        self.assertEqual(prov["status"], "NOT_ATTEMPTED")
        self.assertIsNone(out["hand_gap_comparison"]["logit_audit"]["label_order"])
        self.assertEqual(CALLS, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
