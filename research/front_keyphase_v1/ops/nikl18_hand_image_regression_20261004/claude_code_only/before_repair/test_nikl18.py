"""Synthetic tests for nikl18_pair.py (no MediaPipe/project/model/video). Run: python -m unittest test_nikl18 -v"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
_before = set(sys.modules)
sys.path.insert(0, str(HERE))
import nikl18_pair as m  # noqa: E402

_ADAPTER_IMPORTS = set(sys.modules) - _before
import numpy as np  # noqa: E402

DOC = json.loads((HERE / "SCHEMA_EXAMPLE.json").read_text(encoding="utf-8"))
SCHEMA, EXAMPLE = DOC["json_schema"], DOC["example_config"]


def ref_bitexact(a, b):  # stand-in with the documented helper semantics (dtype+shape+raw bytes)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def fake_fns(calls):
    def to_native_like(xy, usable):
        calls.append(("native", xy.dtype, usable.dtype))
        return xy.copy(), usable.copy()

    def active_span(present):
        idx = np.flatnonzero(present.any(axis=1))
        return None if idx.size == 0 else (int(idx[0]), int(idx[-1]))

    def build_channels(nat, present, ts):
        calls.append(("channels", len(nat), ts.dtype))
        return {"frames": len(ts), "body_scale": 1.5, "hand_scales": np.array([np.nan, 2.0])}

    def collate(items):
        calls.append(("label", items[0]["label"]))
        t = items[0]["frames"]
        return np.zeros((1, t, 4), np.float32), np.ones((1, t), np.float32), [items[0]["label"]]

    return {"to_native_like": to_native_like, "active_span": active_span, "build_channels": build_channels,
            "collate": collate, "n_channels": 4}


def obs(n, usable_rows):
    usable = np.zeros((n, 75), bool)
    usable[usable_rows, 40] = True
    return {"xy_px": np.arange(n * 75 * 2, dtype=np.float64).reshape(n, 75, 2), "usable": usable,
            "frame_id": np.arange(n), "timestamp_us": np.arange(n, dtype=np.int64) * 33333}


class AdapterTests(unittest.TestCase):
    def test_import_has_no_runtime_dependencies(self):
        heavy = {x for x in _ADAPTER_IMPORTS if x.split(".")[0] in ("numpy", "torch", "mediapipe", "av", "wis_lab", "pilot")}
        self.assertEqual(heavy, set())

    def test_schema_exact_keys_and_types(self):
        m.validate_schema(EXAMPLE, SCHEMA)
        for mutate in (lambda c: c.update(extra=1),
                       lambda c: c["clips"][0].pop("time_base"),
                       lambda c: c["budgets"].update(forward_calls=36),            # old key rejected
                       lambda c: c["scoring"].update(cpu_threads=True),            # bool is not integer
                       lambda c: c["clips"][0].update(raw_sha256="A" * 64),        # pattern
                       lambda c: c["clips"].pop()):                                # 17 clips
            bad = copy.deepcopy(EXAMPLE)
            mutate(bad)
            with self.assertRaises(m.TechnicalFailure):
                m.validate_schema(bad, SCHEMA)

    def test_frozen_values_match_example_contract(self):
        cfg = dict(copy.deepcopy(EXAMPLE), is_schema_example=False)
        self.assertEqual(m.check_frozen_values(cfg), [])
        cfg["budgets"]["forward_calls_max"] = 37
        cfg["clips"][0]["expected_frames"] += 1
        self.assertEqual(len(m.check_frozen_values(cfg)), 2)

    def test_schema_example_rejected_before_file_access(self):
        def no_hash(*_a):
            raise AssertionError("file hashed")
        orig, m.sha256_file = m.sha256_file, no_hash
        try:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "cfg.json"
                p.write_text(json.dumps(EXAMPLE), encoding="utf-8")
                with self.assertRaisesRegex(m.TechnicalFailure, "schema example"):
                    m.preflight(p, HERE / "SCHEMA_EXAMPLE.json")
        finally:
            m.sha256_file = orig

    def test_full_view_uses_all_stored_obs(self):
        stored = {"timestamp_us": np.arange(5) * 10}
        r = {"stored_obs": stored, "decoded_meta": {"frame_id": np.arange(5)},
             "selected_obs": {"timestamp_us": np.arange(0, 5, 2) * 10}, "selected_meta": {"frame_id": np.arange(0, 5, 2)}}
        v = m.full_view(r)
        self.assertIs(v["selected_obs"], stored)
        self.assertEqual(len(v["selected_meta"]["frame_id"]), 5)
        self.assertEqual(len(r["selected_obs"]["timestamp_us"]), 3)  # original untouched

    def test_cache_bit_gate_including_nan_payload(self):
        n = 4
        cache = {"schema": np.array("s"), **{f: np.zeros((n, 2), np.float32) for f in m.FIELDS7}}
        cache["frame_id"], cache["timestamp_us"] = np.arange(n), np.array([0, 33300, 66633, 100000], np.int64)
        cache["conf"][1, 0] = np.nan
        m.check_cache_arrays(cache, m.FIELDS7, n)
        same = {k: v.copy() for k, v in cache.items()}
        self.assertEqual(m.cache_gate(same, cache, m.FIELDS7, ref_bitexact), [])
        other_nan = {k: v.copy() for k, v in cache.items()}
        other_nan["conf"].view(np.uint32)[1, 0] = 0x7FC00001  # NaN with different payload
        self.assertEqual(len(m.cache_gate(other_nan, cache, m.FIELDS7, ref_bitexact)), 1)
        widened = dict(same, hand_score=same["hand_score"].astype(np.float64))
        self.assertEqual(len(m.cache_gate(widened, cache, m.FIELDS7, ref_bitexact)), 1)
        for bad in ({("obs_" + k if k != "schema" else k): v for k, v in cache.items()},
                    dict(cache, frame_id=np.array([0, 1, 3, 4])),
                    dict(cache, timestamp_us=np.array([1, 2, 3, 4], np.int64))):
            with self.assertRaises(m.TechnicalFailure):
                m.check_cache_arrays(bad, m.FIELDS7, n)

    def test_inclusive_independent_spans(self):
        calls = []
        fns = fake_fns(calls)
        pb = m.prep_arm(obs(8, [2, 5]), fns, "B_video")
        pc = m.prep_arm(obs(8, [1, 3]), fns, "C_hand_image")
        self.assertEqual((pb["span"], pb["frame_id"], pb["x"].shape), ([2, 5], [2, 3, 4, 5], (1, 4, 4)))
        self.assertEqual(pb["timestamp_us"], [66666, 99999, 133332, 166665])
        self.assertEqual((pc["span"], pc["x"].shape), ([1, 3], (1, 3, 4)))
        self.assertIn(("native", np.float32, np.bool_), calls)
        self.assertIn(("channels", 4, np.int64), calls)
        self.assertEqual({c[1] for c in calls if c[0] == "label"}, {0})
        self.assertEqual(pb["hand_scales"]["value"], [None, 2.0])  # NaN scale -> null, not unavailable
        json.dumps(m.input_record(pb, lambda a: "h"), allow_nan=False)

    def test_no_span_c_unavailable_b_technical_and_errors_technical(self):
        fns = fake_fns([])
        self.assertIsNone(m.prep_arm(obs(5, []), fns, "C_hand_image"))
        with self.assertRaisesRegex(m.TechnicalFailure, "baseline reproduction"):
            m.prep_arm(obs(5, []), fns, "B_video")
        broken = dict(fns, build_channels=lambda *a: (_ for _ in ()).throw(ValueError("shape")))
        with self.assertRaisesRegex(m.TechnicalFailure, "ValueError"):
            m.prep_arm(obs(5, [1]), broken, "C_hand_image")
        nonfinite = dict(fns, collate=lambda it: (np.full((1, it[0]["frames"], 4), np.nan, np.float32),
                                                  np.ones((1, it[0]["frames"])), [0]))
        with self.assertRaisesRegex(m.TechnicalFailure, "non-finite"):
            m.prep_arm(obs(5, [1]), nonfinite, "C_hand_image")

    def test_decision_fixed_denominator(self):
        b = [{"clip_id": f"c{i}", "correct": True} for i in range(18)]
        c_ok = [{"clip_id": f"c{i}", "status": "AVAILABLE", "correct": True} for i in range(18)]
        d = m.decide(b, c_ok)
        self.assertEqual((d["code"], d["expected_forwards"]), ("PASS_FROZEN_MODE_REGRESSION", 36))
        c_un = copy.deepcopy(c_ok)
        c_un[3] = {"clip_id": "c3", "status": m.UNAVAILABLE, "correct": None}
        d = m.decide(b, c_un)
        self.assertEqual((d["code"], d["denominator"], d["C_unavailable"], d["expected_forwards"], d["C_correct"]),
                         ("FAIL_FROZEN_MODE_REGRESSION", 18, 1, 35, 17))
        c_wrong = copy.deepcopy(c_ok)
        c_wrong[0]["correct"] = False
        self.assertEqual(m.decide(b, c_wrong)["C_regressions"], ["c0"])

    def test_extraction_count_view(self):
        actual = {k: v for k, v in m.BUDGETS.items() if k not in m.NON_EXTRACTION_BUDGETS}
        counters = {"tcn_forwards": 0, "model_loads": 0, "camera_calls": 0, "training_runs": 0, "downloads": 0,
                    "landmarker_constructed": 36, "landmarker_closed": 36}
        self.assertEqual(m.check_extraction_counts(actual, m.BUDGETS, counters), [])
        self.assertEqual(len(m.check_extraction_counts(dict(actual, landmarker_process_calls=2673), m.BUDGETS, counters)), 1)
        missing = {k: v for k, v in actual.items() if k != "hand_detect_image"}
        self.assertEqual(len(m.check_extraction_counts(missing, m.BUDGETS, counters)), 1)
        self.assertEqual(len(m.check_extraction_counts(actual, m.BUDGETS, dict(counters, landmarker_closed=35))), 1)

    def test_receipt_guard(self):
        good = {"status": "PASS", "runner_sha256": "a" * 64, "test_sha256": "b" * 64, "tests_run": 11}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            p.write_text(json.dumps(good), encoding="utf-8")
            self.assertEqual(m.validate_receipt(p, "a" * 64, "b" * 64)["tests_run"], 11)
            with self.assertRaises(m.TechnicalFailure):
                m.validate_receipt(p, "c" * 64, "b" * 64)
            for bad in (dict(good, status="FAIL"), dict(good, tests_run=0), dict(good, tests_run=True),
                        {k: v for k, v in good.items() if k != "test_sha256"}):
                p.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(m.TechnicalFailure):
                    m.validate_receipt(p, "a" * 64, "b" * 64)


if __name__ == "__main__":
    unittest.main()
