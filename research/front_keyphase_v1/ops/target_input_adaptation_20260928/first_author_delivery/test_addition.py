"""Synthetic-only tests for train_addition.py (no real data, no camera, no network).

    python test_addition.py                   # run
    python test_addition.py --write-evidence  # run; write TEST_RECEIPT.json (or TEST_FAILED_*.json)

experiment.py train refuses to run without a PASSED receipt whose script / generated_v1 hashes
match the current files.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_addition as T  # noqa: E402

G = T.G
SCRIPTS = ("experiment.py", "train_addition.py", "test_addition.py")


class Blk(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv1d(w, w, 3, padding=1)
        self.norm = nn.BatchNorm1d(w)

    def forward(self, h, m):
        return (h + torch.tanh(self.norm(self.conv(h * m)))) * m


class Net(nn.Module):
    def __init__(self, c=6, w=8, k=5):
        super().__init__()
        self.proj = nn.Conv1d(c, w, 1)
        self.blocks = nn.ModuleList([Blk(w), Blk(w)])
        self.head = nn.Linear(w, k)
        with torch.no_grad():
            for b in self.blocks:
                b.norm.running_mean.normal_()
                b.norm.running_var.uniform_(0.5, 2.0)


def synth(steps=12, nb=20, nc=10, ne=9, w=8, k=5, seed=0):
    rng = np.random.default_rng(seed)
    mk = lambda n: [rng.normal(size=(int(rng.integers(3, 10)), w)).astype(np.float32) for _ in range(n)]
    d = dict(base=mk(nb), by=rng.integers(0, k, nb).astype(np.int64),
             ctx=mk(nc), cy=rng.integers(0, k, nc).astype(np.int64),
             teacher=rng.normal(size=(nb, k)),
             bidx=rng.integers(0, nb, (steps, 4)).astype(np.int64),
             cidx=rng.integers(0, nc, (steps, 4)).astype(np.int64),
             ext=mk(ne), ey=np.array([1] * 4 + [3] * 5, dtype=np.int64))
    d["eidx"] = T.cyclic_schedule([4, 5], steps, 2)
    return d


def model(seed=0):
    torch.manual_seed(seed)
    return Net().eval()


def args(d):
    return (d["base"], d["by"], d["ctx"], d["cy"], d["teacher"], d["bidx"], d["cidx"])


def same_state(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


MEMBERS = [f"{w}/{p:02d}_{w}.MP4" for p in range(12) for w in (16, 48)]


class LearnerTests(unittest.TestCase):
    def test_control_matches_prior_run_pair_control(self):
        d, net = synth(), model()
        prior = G.run_pair(net, *args(d))       # also trains the block arm: test-only
        mine = T.train_arm(net, *args(d), **T.TRAIN_KW)
        self.assertTrue(same_state(mine["state_dict"], prior["control"]["state_dict"]))
        self.assertTrue(np.array_equal(mine["losses"][:, [0, 3, 4, 5]], prior["control"]["losses"]))
        self.assertTrue(np.isnan(mine["losses"][:, 1:3]).all())
        self.assertEqual(mine["trainable_names"], prior["control"]["trainable_names"])

    def test_extra_weight_zero_equals_control(self):
        d, net = synth(), model()
        ctrl = T.train_arm(net, *args(d), **T.TRAIN_KW)
        zero = T.train_arm(net, *args(d), extra_prefixes=d["ext"], extra_y=d["ey"], extra_indices=d["eidx"],
                           extra_weight=0.0, **T.TRAIN_KW)
        self.assertTrue(same_state(zero["state_dict"], ctrl["state_dict"]))
        self.assertTrue(np.array_equal(zero["losses"][:, [0, 3, 4, 5]], ctrl["losses"][:, [0, 3, 4, 5]]))
        self.assertTrue(np.isfinite(zero["losses"][:, 1]).all())
        self.assertTrue((zero["losses"][:, 2] == 0).all())

    def test_candidate_head_only_and_old_schedule_unchanged(self):
        d, net = synth(), model()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        ctrl = T.train_arm(net, *args(d), **T.TRAIN_KW)
        cand = T.train_arm(net, *args(d), extra_prefixes=d["ext"], extra_y=d["ey"], extra_indices=d["eidx"],
                           extra_weight=0.125, arm="cand", **T.TRAIN_KW)
        sd = cand["state_dict"]
        self.assertTrue(same_state(net.state_dict(), before))                      # input model untouched
        self.assertTrue(all(n.startswith("head.") for n in cand["trainable_names"]))
        self.assertTrue(all(torch.equal(sd[k], before[k]) for k in sd if not k.startswith("head.")))
        self.assertFalse(torch.equal(sd["head.weight"], before["head.weight"]))
        self.assertFalse(torch.equal(sd["head.weight"], ctrl["state_dict"]["head.weight"]))
        # old exposures identical and not replaced: same schedule, identical step-0 old CE/output
        self.assertTrue(np.array_equal(cand["base_indices"], d["bidx"]))
        self.assertTrue(np.array_equal(cand["context_indices"], d["cidx"]))
        self.assertTrue(np.array_equal(cand["extra_indices"], d["eidx"]))
        self.assertEqual(cand["losses"][0, 0], ctrl["losses"][0, 0])
        self.assertEqual(cand["losses"][0, 4], ctrl["losses"][0, 4])
        self.assertTrue(np.array_equal(cand["losses"][:, 2], cand["losses"][:, 1] * 0.125))
        np.testing.assert_allclose(cand["losses"][:, 5], cand["losses"][:, [0, 2, 3, 4]].sum(1), rtol=1e-12)

    def test_extra_requires_data(self):
        d, net = synth(steps=2), model()
        with self.assertRaises(ValueError):
            T.train_arm(net, *args(d), extra_weight=0.125)


class SplitTests(unittest.TestCase):
    def test_bucket_and_fold_deterministic(self):
        self.assertEqual(T.bucket_of("16/00_16.MP4"), "00")
        self.assertEqual(T.bucket_of("48\\00_48.MP4"), "00")
        with self.assertRaises(ValueError):
            T.bucket_of("16/0016.MP4")
        b1, f1, o1 = T.assign_folds(MEMBERS)
        b2, f2, o2 = T.assign_folds(list(reversed(MEMBERS)))
        self.assertEqual(dict(zip(b1, f1.tolist())), dict(zip(b2, f2.tolist())))
        want = sorted({m.split("/")[1].split("_")[0] for m in MEMBERS},
                      key=lambda p: hashlib.sha256(("wis-20260928-fold:" + p).encode()).hexdigest())
        self.assertEqual(o1, want)
        self.assertEqual({b: f for b, f in zip(b1, f1.tolist())}, {p: i % 3 for i, p in enumerate(want)})

    def _data(self):
        buckets, folds, _ = T.assign_folds(MEMBERS)
        ids = [m.replace("/", "_") for m in MEMBERS]
        y = np.array([0 if m.startswith("16/") else 1 for m in MEMBERS], dtype=np.int64)
        return ids, y, folds, buckets

    def test_no_leakage_and_schedule_uses_training_rows_only(self):
        ids, y, folds, buckets = self._data()
        info = T.check_split(ids, y, folds, buckets, {"source": ids, "input": ids})
        self.assertEqual(info["same_fold_duplicates"], {})
        for f in range(3):
            groups = T.fold_training_rows(ids, y, folds, f, [0, 1])
            for g in groups:
                self.assertEqual([ids[i] for i in g], sorted(ids[i] for i in g))
            rows = np.array([i for g in groups for i in g], dtype=np.int64)
            sched = T.cyclic_schedule([len(g) for g in groups], 660, 2)
            used = rows[sched]
            self.assertTrue((folds[used] != f).all())
            self.assertFalse(set(np.flatnonzero(folds == f)) & set(used.tolist()))
            self.assertEqual(set(used.tolist()), set(rows.tolist()))            # every training row cycled
            self.assertTrue((y[used[:, :2]] == 0).all() and (y[used[:, 2:]] == 1).all())
            self.assertFalse({buckets[i] for i in used.ravel()} & {buckets[i] for i in np.flatnonzero(folds == f)})

    def test_cyclic_rule(self):
        s = T.cyclic_schedule([5, 7], 9, 2)
        for t in range(9):
            self.assertEqual(s[t].tolist(), [(2 * t) % 5, (2 * t + 1) % 5, 5 + (2 * t) % 7, 5 + (2 * t + 1) % 7])
        with self.assertRaises(ValueError):
            T.cyclic_schedule([1, 4], 3, 2)

    def test_crossfold_duplicate_hash_rejected(self):
        ids, y, folds, buckets = self._data()
        i = 0
        j = next(k for k in range(len(ids)) if folds[k] != folds[i])
        src = list(ids)
        src[j] = src[i]
        with self.assertRaises(ValueError):
            T.check_split(ids, y, folds, buckets, {"source": src})
        k = next(k for k in range(1, len(ids)) if folds[k] == folds[i])
        src = list(ids)
        src[k] = src[i]
        self.assertIn("source", T.check_split(ids, y, folds, buckets, {"source": src})["same_fold_duplicates"])

    def test_too_few_training_rows_rejected(self):
        ids, y, folds, buckets = self._data()
        keep = [i for i in range(len(ids)) if y[i] == 0 or folds[i] == 0]
        with self.assertRaises(ValueError):
            T.check_split([ids[i] for i in keep], y[keep], folds[keep], [buckets[i] for i in keep], {})


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-evidence", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c)
                                for c in (LearnerTests, SplitTests)])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if a.write_evidence:
        now = datetime.datetime.now(datetime.timezone.utc)
        ok = res.wasSuccessful()
        rec = {"status": "PASSED" if ok else "FAILED",
               "created_utc": now.isoformat(),
               "tests_run": res.testsRun,
               "failures": [str(t) for t, _ in res.failures], "errors": [str(t) for t, _ in res.errors],
               "scripts": {n: _sha(HERE / n) for n in SCRIPTS},
               "prior_generated_path": str(T.PRIOR_GENERATED), "prior_generated_sha256": T.prior_sha256(),
               "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "scope": "synthetic data only; no real data, camera, network or training run"}
        name = "TEST_RECEIPT.json" if ok else "TEST_FAILED_" + now.strftime("%Y%m%dT%H%M%S%fZ") + ".json"
        with (HERE / name).open("x", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
            f.write("\n")
        print("wrote", HERE / name)
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
