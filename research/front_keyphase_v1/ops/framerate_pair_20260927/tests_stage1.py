"""Synthetic tests for framerate_train.py: tiny compatible model + random arrays only.

No dataset, checkpoint, camera or project loader is touched: an import guard denies camera and
project-data modules before framerate_train is imported. Runs as a script (or under pytest).

    py -3.13 ops/framerate_pair_20260927/tests_stage1.py                   # run only
    py -3.13 ops/framerate_pair_20260927/tests_stage1.py --write-evidence  # + STAGE1_SYNTHETIC_TESTS.json
"""
import copy
import hashlib
import importlib
import importlib.abc
import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DENIED = ("cv2", "mediapipe", "pyrealsense2", "wis_lab", "pilot", "tools_prep_union93_candidate",
          "tools_visibility_experiment", "local_io", "fr_prior_local_io")


class _DenyCameraAndData(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in DENIED:
            raise ImportError(f"denied in synthetic tests: {name}")
        return None


assert not [m for m in DENIED if m in sys.modules], "camera/data module loaded before the guard"
sys.meta_path.insert(0, _DenyCameraAndData())

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
import torch.nn.functional as Fnn                                    # noqa: E402
from torch import nn                                                 # noqa: E402

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import framerate_train as T                                          # noqa: E402

G = T.G
W, C, K, STEPS, HALF = 8, 6, 5, 5, 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(W, W, 3, padding=1)
        self.norm = nn.BatchNorm1d(W)
        self.drop = nn.Dropout(0.5)                                  # would break equalities if not eval
        with torch.no_grad():
            self.norm.running_mean.uniform_(-0.2, 0.2)
            self.norm.running_var.uniform_(0.5, 1.5)

    def forward(self, h, m):
        return (h + self.drop(torch.tanh(self.norm(self.conv(h * m)))) * m) * m


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(C, W, 1)
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.head = nn.Linear(W, K)


def setup(seed=0, n_base=12, n_ctx=6):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    clip = lambda: rng.standard_normal((int(rng.integers(3, 10)), W)).astype(np.float32)
    base, ctx = [clip() for _ in range(n_base)], [clip() for _ in range(n_ctx)]
    eligible = np.arange(n_base) % 3 != 0                           # rows 0,3,6,9 ineligible
    bidx = rng.integers(0, n_base, (STEPS, HALF)).astype(np.int64)
    bidx[0] = [0, 1, 2, 3]
    return {"model": Tiny().eval(), "base": base, "by": rng.integers(0, K, n_base).astype(np.int64),
            "ctx": ctx, "cy": rng.integers(0, K, n_ctx).astype(np.int64),
            "teacher": rng.standard_normal((n_base, K)), "bidx": bidx,
            "cidx": rng.integers(0, n_ctx, (STEPS, HALF)).astype(np.int64),
            "dec": [np.ascontiguousarray(base[i][::2] * 1.5) if eligible[i] else None for i in range(n_base)],
            "eligible": eligible, "coins": np.ones((STEPS, HALF), dtype=bool)}


def run(d, **over):
    a = {**d, **over}
    return T.run_framerate_pair(a["model"], a["base"], a["by"], a["ctx"], a["cy"], a["teacher"], a["bidx"],
                                a["cidx"], a["dec"], a["eligible"], a["coins"], learning_rate=0.05)


def same_state(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def step0_losses(d, row_mask):
    """CE and weighted output loss of step 0, by hand, from the untouched parent."""
    work = copy.deepcopy(d["model"]).eval()
    clips, labels, brow = T.select_batch(0, d["bidx"], d["cidx"], d["base"], d["ctx"], d["dec"], d["by"], d["cy"], row_mask)
    h, m = G._pad_batch(clips)
    with torch.no_grad():
        logits = work.head(G._masked_mean(work.blocks[-1](h, m), m))
    ce = Fnn.cross_entropy(logits, torch.from_numpy(labels))
    delta = logits[:HALF].double() - torch.from_numpy(d["teacher"][brow])   # ORIGINAL-row teacher
    delta = delta - delta.mean(dim=1, keepdim=True)
    return float(ce), float(0.5 * delta.pow(2).mean())


# --------------------------------------------------------------------------- #
def test_coins_predeclared_generator():
    c = T.make_coins()
    assert c.dtype == np.bool_ and c.shape == (660, 16)
    assert np.array_equal(c, np.random.default_rng(20260927).random((660, 16)) < 0.5)
    assert np.array_equal(c, T.make_coins()) and not np.array_equal(c, T.make_coins(20260928))
    for kw in ({"seed": 1.0}, {"seed": True}, {"shape": [660, 16]}, {"shape": (660, 0)}):
        try:
            T.make_coins(**kw)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"make_coins accepted {kw}")


def test_eligibility_and_batch_order():
    d = setup()
    mask = T.replacement_mask(d["bidx"], d["coins"], d["eligible"])
    assert np.array_equal(mask, d["eligible"][d["bidx"]])            # coins all 1 -> eligibility decides
    assert mask[0].tolist() == [False, True, True, False]            # ineligible rows 0 and 3 never replaced
    clips, labels, _ = T.select_batch(0, d["bidx"], d["cidx"], d["base"], d["ctx"], d["dec"], d["by"], d["cy"], mask[0])
    for k, i in enumerate(d["bidx"][0]):
        assert clips[k] is (d["dec"][i] if mask[0, k] else d["base"][i])
    for k, j in enumerate(d["cidx"][0]):
        assert clips[HALF + k] is d["ctx"][j]                        # context never replaced
    assert labels.tolist() == d["by"][d["bidx"][0]].tolist() + d["cy"][d["cidx"][0]].tolist()
    coins = np.zeros_like(d["coins"])
    coins[0, 1] = coins[0, 0] = True
    assert T.replacement_mask(d["bidx"], coins, d["eligible"]).sum() == 1   # (0,0) ineligible
    try:
        T.select_batch(0, d["bidx"], d["cidx"], d["base"], d["ctx"], d["dec"], d["by"], d["cy"],
                       np.array([True, False, False, False]))
    except ValueError:
        pass
    else:
        raise AssertionError("replacement of an ineligible row was not refused")


def test_control_equals_prior_run_pair_control():
    d = setup()
    ours = run(d)
    prior = G.run_pair(d["model"], d["base"], d["by"], d["ctx"], d["cy"], d["teacher"], d["bidx"], d["cidx"],
                       learning_rate=0.05)
    assert same_state(ours["control"]["state_dict"], prior["control"]["state_dict"])
    assert np.array_equal(ours["control"]["losses"], prior["control"]["losses"])
    assert ours["control"]["trainable_names"] == prior["control"]["trainable_names"]


def test_no_replacement_gives_identical_arms():
    d = setup()
    for over in ({"coins": np.zeros_like(d["coins"])},
                 {"eligible": np.zeros_like(d["eligible"]), "dec": [None] * len(d["base"])}):
        r = run(d, **over)
        assert r["candidate"]["replacements"] == 0 == r["control"]["replacements"]
        assert same_state(r["control"]["state_dict"], r["candidate"]["state_dict"])
        assert np.array_equal(r["control"]["losses"], r["candidate"]["losses"])


def test_candidate_replacement_fixed_teacher_and_loss():
    d = setup()
    r = run(d)
    mask = T.replacement_mask(d["bidx"], d["coins"], d["eligible"])
    assert np.array_equal(r["candidate"]["replaced"], mask) and r["candidate"]["replacements"] == int(mask.sum()) > 0
    assert not r["control"]["replaced"].any()
    assert not same_state(r["control"]["state_dict"], r["candidate"]["state_dict"])
    for arm, row_mask in (("control", np.zeros(HALF, dtype=bool)), ("candidate", mask[0])):
        ce, out = step0_losses(d, row_mask)
        l0 = r[arm]["losses"][0]
        assert abs(l0[0] - ce) < 1e-6 and abs(l0[2] - out) < 1e-9 and l0[1] == 0.0, (arm, l0, ce, out)
    assert r["control"]["losses"][0, 2] != r["candidate"]["losses"][0, 2]   # replaced draws are regularised


def test_head_only_parent_and_inputs_untouched():
    d = setup()
    before = {k: v.clone() for k, v in d["model"].state_dict().items()}
    snap = copy.deepcopy({k: d[k] for k in ("base", "ctx", "dec", "teacher", "bidx", "cidx", "coins", "eligible", "by", "cy")})
    r = run(d)
    head = [n for n, _ in d["model"].named_parameters() if n.startswith("head.")]
    for arm in ("control", "candidate"):
        a = r[arm]
        assert a["trainable_names"] == head and a["frozen_parameters_unchanged"] and a["buffers_unchanged"]
        changed = [k for k, v in a["state_dict"].items() if not torch.equal(v, before[k])]
        assert changed and all(k.startswith("head.") for k in changed), changed
        assert a["updates"] == STEPS
    assert same_state(d["model"].state_dict(), before) and not d["model"].training
    for k, v in snap.items():
        now = d[k]
        if isinstance(v, list):
            assert all((x is None and y is None) or np.array_equal(x, y) for x, y in zip(v, now)), k
        else:
            assert np.array_equal(v, now), k


def test_bad_inputs_refused():
    d = setup()

    def dec_with(i, v):
        x = list(d["dec"])
        x[i] = v
        return x

    cases = {
        "coins shape": {"coins": np.ones((STEPS, HALF + 1), dtype=bool)},
        "coins dtype": {"coins": np.ones((STEPS, HALF), dtype=np.int64)},
        "coins list": {"coins": np.ones((STEPS, HALF), dtype=bool).tolist()},
        "eligible length": {"eligible": d["eligible"][:-1]},
        "eligible dtype": {"eligible": d["eligible"].astype(np.int64)},
        "dec length": {"dec": d["dec"][:-1]},
        "dec missing for eligible": {"dec": dec_with(1, None)},
        "dec for ineligible": {"dec": dec_with(0, d["base"][0][::2].copy())},
        "dec width": {"dec": dec_with(1, np.zeros((3, W + 1), np.float32))},
        "dec float64": {"dec": dec_with(1, np.zeros((3, W)))},
        "dec empty": {"dec": dec_with(1, np.zeros((0, W), np.float32))},
        "dec nan": {"dec": dec_with(1, np.full((3, W), np.nan, np.float32))},
    }
    for name, over in cases.items():
        try:
            run(d, **over)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"accepted bad input: {name}")


def test_camera_and_data_import_denied():
    for name in ("cv2", "mediapipe", "pilot", "wis_lab"):
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        raise AssertionError(f"{name} import was not denied")
    assert not [m for m in DENIED if m in sys.modules]


# --------------------------------------------------------------------------- #
def main() -> int:
    torch.set_num_threads(2)
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
            print("PASS", name, flush=True)
        except Exception:
            failed.append(name)
            print("FAIL", name, "\n" + traceback.format_exc(), flush=True)
    print(json.dumps({"passed": len(passed), "failed": failed}), flush=True)
    if "--write-evidence" in sys.argv and not failed:
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        evidence = {"status": "PASS", "at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                    "tests": [n for n, _ in tests], "passed": passed, "failed": failed,
                    "framerate_train_sha256": sha(H / "framerate_train.py"), "tests_sha256": sha(Path(__file__)),
                    "prior_generated_v1_sha256": T.prior_module_sha256(),
                    "prior_generated_v1_path": str(T.PRIOR_GENERATED),
                    "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                    "synthetic_only": True, "dataset_or_checkpoint_reads": 0, "camera": 0}
        with (H / "STAGE1_SYNTHETIC_TESTS.json").open("x", encoding="utf-8", newline="\n") as fh:
            json.dump(evidence, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
