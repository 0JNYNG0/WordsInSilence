"""Synthetic unit tests for temporal_factor_audit. No real data, model or torch required.

Run: python -m unittest test_temporal_factor_audit -v
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import temporal_factor_audit as T  # noqa: E402

try:                                    # supplied numpy-only API reference, if present alongside
    import data_api
except Exception:                       # pragma: no cover
    data_api = None

TS_IRREGULAR = np.array([0, 33000, 70001, 100000, 140123, 171000, 205555, 233334, 270000], dtype=np.int64)


def synthetic_x(ts, seed=0):
    rng = np.random.default_rng(seed)
    t = len(ts)
    x = rng.normal(size=(t, T.N_CH)).astype(np.float32)
    x[:, T.MASKS] = (rng.random((t, 93)) > 0.3).astype(np.float32)
    x[:, T.DT_COL] = T.dt_column(ts)[:, 0]
    return x


def synthetic_native(ts, seed=1):
    """Native [T,67] arrays; right hand MCPs only on odd frames, so stride-2 loses its scale."""
    rng = np.random.default_rng(seed)
    t = len(ts)
    xy = (rng.random((t, 67, 2)) * 100 + 200).astype(np.float32)
    xy[:, 5] = [300.0, 200.0]                                       # LShoulder
    # RShoulder: separation 100+k on even frames, 150 on odd -> median 108 (all) vs 104 (::2)
    xy[:, 2] = [[200.0 - (k if k % 2 == 0 else 50), 200.0] for k in range(t)]
    present = np.ones((t, 67), dtype=bool)
    rh = 46
    present[:, rh:rh + 21] = False
    present[:, rh] = True                                           # wrist always
    present[:, rh + 4] = True                                       # thumb tip always
    for m in (5, 9, 13, 17):
        present[1::2, rh + m] = True                                # MCPs on odd frames only
    return xy, present, ts.copy()


class TestConstruction(unittest.TestCase):
    def test_dt_column_irregular_matches_build_channels_ops(self):
        dec = TS_IRREGULAR[::2]
        dt = T.dt_column(dec)
        expect = np.zeros((len(dec), 1), np.float32)
        expect[1:, 0] = np.diff(dec).astype(np.float32) / 1e6
        self.assertEqual(dt.dtype, np.float32)
        self.assertTrue(np.array_equal(dt, expect))
        self.assertEqual(dt[0, 0], 0.0)
        self.assertEqual(len(set(np.diff(dec).tolist())), len(dec) - 1)   # really irregular
        self.assertEqual(T.dt_column(np.array([5], np.int64)).tolist(), [[0.0]])

    def test_c_d_construction_preserves_channels_and_input(self):
        x = synthetic_x(TS_IRREGULAR)
        before = x.copy()
        c, d, ts_dec = T.make_c_d(x, TS_IRREGULAR)
        self.assertTrue(np.array_equal(x, before))
        self.assertFalse(np.shares_memory(c, x) or np.shares_memory(d, x) or np.shares_memory(c, d))
        self.assertEqual(c.shape, (5, T.N_CH))
        self.assertTrue(np.array_equal(ts_dec, TS_IRREGULAR[::2]))
        self.assertTrue(np.array_equal(d, x[::2]))
        self.assertTrue(np.array_equal(c[:, :T.DT_COL], d[:, :T.DT_COL]))
        self.assertTrue(np.array_equal(c[:, T.MASKS], x[::2, T.MASKS]))
        self.assertTrue(np.array_equal(c[:, T.DT_COL], T.dt_column(TS_IRREGULAR[::2])[:, 0]))
        self.assertTrue(np.array_equal(d[:, T.DT_COL], x[::2, T.DT_COL]))
        self.assertFalse(np.array_equal(c[1:, T.DT_COL], d[1:, T.DT_COL]))
        T.check_cd(c, d, x)

    def test_odd_and_single_frame(self):
        ts = np.array([0, 40000, 70000], np.int64)
        c, d, _ = T.make_c_d(synthetic_x(ts), ts)
        self.assertEqual(len(c), 2)
        ts1 = np.array([7], np.int64)
        c1, d1, _ = T.make_c_d(synthetic_x(ts1), ts1)
        self.assertEqual(c1[0, T.DT_COL], 0.0)

    def test_check_cd_detects_tampering(self):
        x = synthetic_x(TS_IRREGULAR)
        c, d, _ = T.make_c_d(x, TS_IRREGULAR)
        c2 = c.copy()
        c2[0, 3] += 1
        with self.assertRaises(T.AuditError):
            T.check_cd(c2, d, x)
        b = c.copy()
        b[2, T.DT_COL] += 1e-3
        with self.assertRaises(T.AuditError):
            T.check_cd(c, d, x, b)

    def test_finite_and_shape_rejection(self):
        x = synthetic_x(TS_IRREGULAR)
        bad = x.copy()
        bad[1, 7] = np.nan
        with self.assertRaises(T.AuditError):
            T.make_c_d(bad, TS_IRREGULAR)
        bad = x.copy()
        bad[0, 0] = np.inf
        with self.assertRaises(T.AuditError):
            T.check_input(bad)
        with self.assertRaises(T.AuditError):
            T.make_c_d(x[:, :279].copy(), TS_IRREGULAR)
        with self.assertRaises(T.AuditError):
            T.make_c_d(x, TS_IRREGULAR[:-1])
        with self.assertRaises(T.AuditError):
            T.check_input(x.astype(np.float64))
        with self.assertRaises(T.AuditError):
            T.check_input(x[0])

    def test_mask_diff_counts(self):
        x = synthetic_x(TS_IRREGULAR)
        b = x.copy()
        c = x.copy()
        b[0, 240] = 0.0
        c[0, 240] = 1.0
        b[1, 190] = 1.0
        c[1, 190] = 0.0
        c[2, 10] = b[2, 10] + 0.5
        m = T.mask_diff(b, c)
        self.assertEqual(m["all_valid_186_279"]["entries_differ"], 2)
        self.assertEqual(m["left_hand_valid"]["b0_c1"], 1)
        self.assertEqual(m["body_valid"]["b1_c0"], 1)
        self.assertEqual(m["right_hand_valid"]["entries_differ"], 0)
        self.assertAlmostEqual(m["max_abs_diff_coords"], 0.5, places=5)
        self.assertEqual(m["max_abs_diff_dt"], 0.0)

    def test_num_or_null(self):
        self.assertIsNone(T.num_or_null(float("nan")))
        self.assertIsNone(T.num_or_null(None))
        self.assertEqual(T.num_or_null(np.float32(2.0)), 2.0)
        json.dumps({"s": T.num_or_null(float("inf"))}, allow_nan=False)


@unittest.skipIf(data_api is None, "data_api.py reference not importable")
class TestWithReferenceBuildChannels(unittest.TestCase):
    def test_c_dt_equals_b_dt_and_scale_effects(self):
        xy, pr, ts = synthetic_native(TS_IRREGULAR)
        xy0, pr0, ts0 = xy.copy(), pr.copy(), ts.copy()
        a = data_api.build_channels(xy, pr, ts)
        b = data_api.build_channels(xy[::2].copy(), pr[::2].copy(), ts[::2].copy())
        c, d, _ = T.make_c_d(a["x"], ts)
        T.check_cd(c, d, a["x"], b["x"])
        self.assertTrue(np.array_equal(xy, xy0) and np.array_equal(pr, pr0) and np.array_equal(ts, ts0))
        sa, sb = T.scales_of(a), T.scales_of(b)
        self.assertIsNotNone(sa["right_hand"])
        self.assertIsNone(sb["right_hand"])                         # NaN -> null
        self.assertNotEqual(sa["body"], sb["body"])                 # parity-dependent shoulders
        m = T.mask_diff(b["x"], c)
        # B loses the whole right hand; C keeps wrist + thumb tip on the 5 retained frames
        self.assertEqual(m["right_hand_valid"]["b0_c1"], 10)
        self.assertEqual(m["right_hand_valid"]["b1_c0"], 0)
        self.assertEqual(m["body_valid"]["entries_differ"], 0)
        self.assertGreater(m["max_abs_diff_coords"], 0)
        self.assertEqual(m["max_abs_diff_dt"], 0.0)


def _row(i, true, a, b):
    return {"id": f"r{i}", "true": true, "signer_id": 11 + i % 3,
            "arms": {"original": {"pred": a, "correct": a == true},
                     "raw_decimated": {"pred": b, "correct": b == true},
                     "fixed_scale_real_dt": {"pred": b, "correct": b == true, "rank": 1},
                     "fixed_scale_old_dt": {"pred": a, "correct": a == true, "rank": 1}}}


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.rows = [_row(0, "X", "X", "X"),      # correct -> correct
                     _row(1, "X", "Y", "X"),      # gain
                     _row(2, "X", "X", "Z"),      # loss
                     _row(3, "X", "Y", "Z"),      # wrong -> wrong, changed
                     _row(4, "X", "Y", "Y")]      # wrong -> wrong, same

    def test_compare_arithmetic(self):
        c = T.compare_arms(self.rows, "original", "raw_decimated")
        self.assertEqual((c["src_correct"], c["dst_correct"], c["net"]), (2, 2, 0))
        self.assertEqual([g["id"] for g in c["gains"]], ["r1"])
        self.assertEqual([g["id"] for g in c["losses"]], ["r2"])
        self.assertEqual([g["id"] for g in c["wrong_to_wrong_changed"]], ["r3"])
        self.assertEqual(c["wrong_to_wrong_same"], 1)
        self.assertEqual(c["correct_to_correct"], 1)
        self.assertEqual(c["top1_changed"], 3)

    def test_a_correct_b_wrong_subset(self):
        s = T.a_correct_b_wrong(self.rows)
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["rows"][0]["id"], "r2")
        self.assertEqual((s["correct_in_c"], s["correct_in_d"], s["correct_in_d_only"]), (0, 1, 1))

    def test_per_signer(self):
        p = T.per_signer(self.rows)
        self.assertEqual(sum(g["n"] for g in p.values()), 5)
        self.assertEqual(sum(g["original"] for g in p.values()), 2)

    def test_score_row(self):
        classes = ["A", "B", "C"]
        s = T.score_row(np.array([1.0, 3.0, 2.0]), classes, "C")
        self.assertEqual((s["pred"], s["correct"], s["rank"]), ("B", False, 2))
        self.assertAlmostEqual(s["margin"], -1.0)
        s = T.score_row(np.array([5.0, 3.0, 2.0]), classes, "A")
        self.assertEqual((s["correct"], s["rank"]), (True, 1))
        self.assertAlmostEqual(s["margin"], 2.0)
        with self.assertRaises(T.AuditError):
            T.score_row(np.array([np.nan, 1.0, 2.0]), classes, "A")
        with self.assertRaises(T.AuditError):
            T.score_row(np.array([1.0, 2.0]), classes, "A")


class TestSavedValidation(unittest.TestCase):
    def setUp(self):
        self.classes = ["A", "B", "C"]
        self.meta = [{"id": f"r{i}", "label": self.classes[i % 3], "signer_id": 11 + i % 3} for i in range(4)]
        self.res = [{"id": m["id"], "true": m["label"], "signer_id": m["signer_id"]} for m in self.meta]
        rng = np.random.default_rng(0)
        self.a = rng.normal(size=(4, 3))
        self.b = rng.normal(size=(4, 3))

    def call(self, **kw):
        args = dict(saved_ids=[m["id"] for m in self.meta], saved_classes=self.classes, orig=self.a,
                    dec=self.b, meta_rows=self.meta, result_rows=self.res, n_rows=4, n_classes=3)
        args.update(kw)
        return T.validate_saved(**args)

    def test_ok(self):
        self.assertEqual(self.call(), self.classes)

    def test_rejections(self):
        nan = self.b.copy()
        nan[1, 1] = np.nan
        bad_res = [dict(r) for r in self.res]
        bad_res[2]["signer_id"] = 99
        bad_true = [dict(r) for r in self.res]
        bad_true[0]["true"] = "B"
        cases = [dict(dec=nan), dict(orig=self.a[:3]), dict(orig=self.a[:, :2]),
                 dict(saved_ids=["r1", "r0", "r2", "r3"]), dict(saved_classes=["A", "B"]),
                 dict(saved_classes=["A", "A", "B"]), dict(result_rows=bad_res),
                 dict(result_rows=bad_true), dict(n_rows=5)]
        for kw in cases:
            with self.subTest(kw=list(kw)), self.assertRaises(T.AuditError):
                self.call(**kw)

    def test_stored_pred_consistency(self):
        res = [dict(r) for r in self.res]
        top = self.classes[int(np.argmax(self.a[0]))]
        wrong = next(c for c in self.classes if c != top)
        res[0]["original"] = {"pred": top}
        self.call(result_rows=res)
        res[0]["original"] = {"pred": wrong}
        with self.assertRaises(T.AuditError):
            self.call(result_rows=res)

    def test_rows_allowed(self):
        rows = [{"id": "r", "label": "A", "path": "p", "sha256": "0" * 64, "role": "validation", "signer_id": 11}]
        T.check_rows_allowed(rows, ["validation"], [11])
        with self.assertRaises(T.AuditError):
            T.check_rows_allowed(rows, ["train"], [11])
        with self.assertRaises(T.AuditError):
            T.check_rows_allowed(rows, ["validation"], [12])


class TestOutputGuard(unittest.TestCase):
    def _config(self, tmp, out_dir):
        p = lambda n: os.path.join(tmp, "missing", n)
        cfg = {"lab_root": p("lab"), "pilot_src": p("src"), "loader_dir": p("loader"),
               "metadata": p("meta.json"), "checkpoint": p("ck.pt"), "saved_logits": p("l.npz"),
               "saved_result": p("r.json"), "plan_file": p("plan.md"), "output_dir": out_dir,
               "expected": {"rows": 3, "classes": 93, "roles": ["validation"], "signers": [11, 12, 13],
                            "allowed_roots": [p("roots")]},
               "repro_indices": [0, 2], "repro_atol": 1e-5, "threads": 2}
        cfg["pins"] = {cfg[k]: "a" * 64 for k in T.REQUIRED_PIN_KEYS}
        path = os.path.join(tmp, "cfg.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        return path

    def test_existing_output_fails_before_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            os.mkdir(out)
            with self.assertRaises(T.AuditError) as cm:
                T.main(["--config", self._config(tmp, out)])
            self.assertIn("exists", str(cm.exception))
            self.assertEqual(os.listdir(out), [])                   # no FAILURE.json, nothing touched

    def test_failure_receipt_when_own_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out_new")
            with self.assertRaises(FileNotFoundError):
                T.main(["--config", self._config(tmp, out)])    # pins point to missing files
            with open(os.path.join(out, "FAILURE.json"), encoding="utf-8") as fh:
                f = json.load(fh)
            self.assertEqual((f["status"], f["stage"]), ("FAILED", "pins"))
            self.assertEqual(f["execution_counts"]["forwards_new"], 0)
            self.assertEqual(sorted(os.listdir(out)), ["FAILURE.json"])

    def test_config_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "o")
            with open(self._config(tmp, out), encoding="utf-8") as fh:
                base = json.load(fh)
            bad_pin = dict(base, pins={base["metadata"]: "a" * 64})
            bad_rel = dict(base, metadata="relative.json")
            bad_idx = dict(base, repro_indices=[0, 3])
            for cfg in (bad_pin, bad_rel, bad_idx):
                with self.assertRaises(T.AuditError):
                    T.validate_config(cfg)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
