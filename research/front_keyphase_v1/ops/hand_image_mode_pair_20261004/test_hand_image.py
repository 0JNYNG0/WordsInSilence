"""Synthetic-only checks for hand_image_pair.py. No media, models, MediaPipe, torch or parent cadence import.

Run: python test_hand_image.py   (or pytest)
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_image_pair as H  # noqa: E402

LS, RS = slice(33, 54), slice(54, 75)
TMP = Path(tempfile.mkdtemp(prefix="hand_image_test_"))
HAND_TASK, POSE_TASK = TMP / "hand.task", TMP / "pose.task"
HAND_TASK.write_bytes(b"hand")
POSE_TASK.write_bytes(b"pose")
N = 4
TS = [0, 66_733, 133_466, 200_200]
FIDS = [0, 2, 4, 6]
FROZEN_BUDGETS = {"decoder_opened": 12, "decoded_frames": 1706, "selected_frames_per_arm": 428,
                  "landmarker_process_calls": 856, "pose_detect_for_video": 856, "hand_detect_for_video": 428,
                  "hand_detect_image": 428, "forward_calls": 8, "landmarker_constructed": 12,
                  "pose_tasks_constructed": 12, "hand_video_tasks_constructed": 12,
                  "hand_video_tasks_discarded_without_inference": 6, "hand_image_tasks_constructed": 6,
                  "independent_forward_calls": "separately authorized and reported", "training": 0, "camera": 0}


def hand_result(n):
    pts = [NS(x=0.1 * (k + 1), y=0.2 + 0.01 * j, z=0.0) for k in range(n) for j in range(21)]
    return NS(hand_landmarks=[pts[k * 21:(k + 1) * 21] for k in range(n)],
              hand_world_landmarks=[pts[k * 21:(k + 1) * 21] for k in range(n)],
              handedness=[[NS(category_name="Left" if k else "Right", score=0.9, index=k)] for k in range(n)])


HANDS = [hand_result(n) for n in (0, 2, 1, 2)]                 # frame 0: no detection
POSES = [NS(pose=np.full((33, 2), 10.0 + r, np.float32)) for r in range(N)]


class FakeTask:
    def __init__(self, mode, results, fail_close=False):
        self.mode, self.results, self.fail_close, self.calls, self.closed = mode, results, fail_close, [], 0

    def detect_for_video(self, image, ms):
        assert self.mode == "VIDEO"
        self.calls.append(("video", ms))
        return self.results[len(self.calls) - 1]

    def detect(self, image):                                   # IMAGE API takes no timestamp
        assert self.mode == "IMAGE"
        self.calls.append(("image",))
        return self.results[len(self.calls) - 1]

    def close(self):
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("close boom")


class FakeLM:
    """Same call shape as the reference Landmarker: process -> pose/hand detect_for_video -> self._assign_sides."""

    def __init__(self, fail_pose_close=False):
        self.hand_model_path, self.pose_model_path = str(HAND_TASK), str(POSE_TASK)
        self.hand, self.pose = FakeTask("VIDEO", HANDS), FakeTask("VIDEO", POSES, fail_pose_close)
        self.raw_hand, self.raw_pose = self.hand, self.pose
        self.handedness_comparisons = self.handedness_disagreements = self.unassigned_hands = 0
        self.returned, self.sides_made, self.sides_seen = [], [], []

    def extractor_id(self):
        return "base0"

    def close(self):
        for m in (self.hand, self.pose):
            try:
                m.close()
            except Exception:
                pass

    def process(self, rgb, timestamp_us, out, row):
        h, w = rgb.shape[:2]
        ms = int(timestamp_us // 1000)
        pose_res = self.pose.detect_for_video(rgb, ms)
        hand_res = self.hand.detect_for_video(rgb, ms)
        self.returned.append((pose_res, hand_res))
        out["xy_px"][row, :33] = pose_res.pose
        out["usable"][row, :33] = True
        if hand_res.hand_landmarks:
            sides = self._assign_sides(hand_res, out["xy_px"][row], out["usable"][row], w, h)
            self.sides_seen.append(sides)
            for i, (s, _sc) in sides.items():
                sl = LS if s == 0 else RS
                for j, lm in enumerate(hand_res.hand_landmarks[i]):
                    out["xy_px"][row, sl.start + j] = (lm.x * w, lm.y * h)
                    out["usable"][row, sl.start + j] = True

    def _assign_sides(self, hand_res, pose_xy, pose_usable, w, h):
        self.handedness_comparisons += 1
        sides = {i: (i % 2, 0.9) for i in range(len(hand_res.hand_landmarks))}
        self.sides_made.append(sides)
        return sides


class Opts:
    def __init__(self, base_options, running_mode, num_hands, min_hand_detection_confidence=0.5,
                 min_hand_presence_confidence=0.5, min_tracking_confidence=0.5):
        self.base_options, self.running_mode, self.num_hands = base_options, running_mode, num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence


def fake_mod(create):
    return NS(POSE_L_WRIST=15, POSE_R_WRIST=16, mp_python=NS(BaseOptions=lambda model_asset_path: NS(path=model_asset_path)),
              mp_vision=NS(RunningMode=NS(VIDEO="VIDEO", IMAGE="IMAGE"), HandLandmarkerOptions=Opts,
                           HandLandmarker=NS(create_from_options=create)))


def image_create(log):
    def create(opts):
        log.append(opts)
        return FakeTask("IMAGE", HANDS)
    return create


def drive(factory):
    """Minimal stand-in for extract_arm: fresh LM, sequential rows, try/finally close."""
    lm = factory()
    out = {"xy_px": np.full((N, 75, 2), np.nan, np.float32), "usable": np.zeros((N, 75), bool),
           "frame_id": np.array(FIDS, np.int64)}
    rgb = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)
    try:
        for row in range(N):
            lm.process(rgb, TS[row], out, row)
    finally:
        lm.close()
    return lm, out


def test_mode_dispatch_identity_lifecycle():
    sink, opts_log = [], []
    mod = fake_mod(image_create(opts_log))
    lb, _ = drive(H.make_factory(FakeLM, mod, "B_video", "VIDEO", sink))
    lc, _ = drive(H.make_factory(FakeLM, mod, "C_hand_image", "IMAGE", sink))
    ib, ic = sink
    assert lb.raw_hand.calls == [("video", t // 1000) for t in TS] and lb.raw_pose.calls == lb.raw_hand.calls
    image_task = ic.tasks[2].task
    assert image_task.calls == [("image",)] * N                  # no timestamp reaches IMAGE
    assert lc.raw_hand.calls == [] and lc.raw_hand.closed == 1      # unused VIDEO hand: created, 0 calls, closed
    assert lc.raw_pose.calls == [("video", t // 1000) for t in TS]
    assert len(opts_log) == 1 and opts_log[0].running_mode == "IMAGE" and opts_log[0].num_hands == 2
    for k in range(N):                                              # proxies return the task's own objects
        assert lc.returned[k][1] is HANDS[k] and lc.returned[k][0] is POSES[k]
    assert len(lc.sides_seen) == 3 and all(a is b for a, b in zip(lc.sides_seen, lc.sides_made))
    f0, f1 = ic.frames[0], ic.frames[1]
    assert f0["n_hands"] == 0 and f0["hand_raw"] == [] and f0["assignment"] is None
    assert f1["hand_api"] == "detect" and f1["hand"]["timestamp_ms_passed"] is None and f1["pose_ms"] == TS[1] // 1000
    assert [e["side"] for e in f1["assignment"]["index_to_side"]] == [0, 1] and len(f1["hand_raw"][0]["world_xyz"]) == 21
    assert ib.frames[1]["hand"]["timestamp_ms_passed"] == TS[1] // 1000
    assert [(f["row"], f["timestamp_us"], f["frame_id"]) for f in ic.frames] == list(zip(range(N), TS, FIDS))
    assert ib.construction["base_factory_ns"] >= 0 and ib.construction["image_create_ns"] is None
    assert ic.construction["base_factory_ns"] >= 0 and ic.construction["image_create_ns"] >= 0
    json.dumps(ic.frames, allow_nan=False)
    r = {"selected_obs": {"timestamp_us": np.array(TS, np.int64)}}
    assert H.gate_diag(ib, r, N) == [] and H.gate_diag(ic, r, N) == []
    assert ib.provenance["base_extractor_id"] == ic.provenance["base_extractor_id"]
    assert ib.provenance["experiment_extractor_id"] != ic.provenance["experiment_extractor_id"]
    assert ic.provenance["hand_options"]["num_hands"] == 2
    tc = H.task_counts(sink)
    assert (tc["pose_video_tasks"], tc["hand_video_tasks"], tc["hand_image_tasks"]) == (2, 2, 1)
    assert (tc["pose_video_calls"], tc["hand_video_calls"], tc["hand_image_calls"]) == (2 * N, N, N)
    assert tc["hand_video_closed_ok"] == 2 and tc["hand_image_closed_ok"] == 1


def test_diag_does_not_modify_observations():
    _, plain = drive(FakeLM)
    _, inst_b = drive(H.make_factory(FakeLM, fake_mod(image_create([])), "B_video", "VIDEO", []))
    _, inst_c = drive(H.make_factory(FakeLM, fake_mod(image_create([])), "C_hand_image", "IMAGE", []))
    for k in plain:
        assert H.bitexact(plain[k], inst_b[k]) and H.bitexact(plain[k], inst_c[k])


def test_c_setup_failure_and_close_errors():
    def boom(_opts):
        raise RuntimeError("no image task")
    sink = []
    try:
        H.make_factory(FakeLM, fake_mod(boom), "C_hand_image", "IMAGE", sink)()
        raise AssertionError("expected failure")
    except RuntimeError:
        pass
    ins = sink[0]
    assert [t.record()["task"] for t in ins.tasks] == ["pose_VIDEO", "hand_VIDEO"]
    assert all(t.close_rec["ok"] for t in ins.tasks) and "C setup" in ins.errors[0]

    sink = []
    drive(H.make_factory(lambda: FakeLM(fail_pose_close=True), fake_mod(image_create([])), "B_video", "VIDEO", sink))
    rec = sink[0].tasks[0].record()                                # Landmarker.close swallowed it; proxy kept it
    assert rec["close"] == {"attempted": True, "ok": False, "error": "RuntimeError: close boom"}
    assert any("not closed ok" in b for b in H.gate_diag(sink[0], {"selected_obs": {"timestamp_us": TS}}, N))
    budgets = dict.fromkeys(H.BUDGET_INTS, 0)
    budgets.update(pose_tasks_constructed=1, hand_video_tasks_constructed=1, pose_detect_for_video=N,
                   hand_detect_for_video=N)
    assert "pose_video_closed_ok 0 != 1" in H.check_budgets({}, sink, budgets)


OBS = ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score")


def arm():
    obs = {"xy_px": np.full((3, 75, 2), np.nan, np.float32), "timestamp_us": np.array([0, 2, 4], np.int64),
           "frame_id": np.array([0, 2, 4], np.int64), "observed": np.zeros((3, 75), bool),
           "usable": np.zeros((3, 75), bool), "conf": np.full((3, 75), np.nan, np.float32),
           "hand_score": np.full((3, 2), np.nan, np.float32)}
    return {"stored_obs": obs, "selected_obs": {k: v.copy() for k, v in obs.items()},
            "decoded_meta": {"frame_id": np.arange(5), "rgb_sha256": np.array(["a", "b", "c", "d", "e"])},
            "selected_meta": {"frame_id": np.array([0, 2, 4]), "timestamp_us": np.array([0, 2, 4])},
            "counts": {"decoded_frames": 5}, "time_base": "1/30000"}


def test_bitexact_nan_payload_and_gates():
    r = arm()
    npz = {f"obs_{k}": v.copy() for k, v in r["stored_obs"].items()}
    npz.update({f"decoded_{k}": v.copy() for k, v in r["decoded_meta"].items()}, selected_frame_id=np.array([0, 2, 4]))
    meta = {"counts": {"decoded_frames": 5}, "time_base": "1/30000"}
    assert H.gate_b_baseline(r, npz, meta, OBS) == []
    other_nan = npz["obs_conf"].copy()
    other_nan.view(np.uint32)[0, 0] = 0x7FC00001                  # still NaN, different payload
    assert np.isnan(other_nan[0, 0])
    assert H.gate_b_baseline(r, dict(npz, obs_conf=other_nan), meta, OBS) == ["conf", "selected:conf"]
    assert (H.gate_b_baseline(r, dict(npz, obs_frame_id=npz["obs_frame_id"].astype(np.int32)), meta, OBS)
            == ["frame_id", "selected:frame_id"])
    assert H.gate_b_baseline(r, npz, dict(meta, time_base="1/1000"), OBS) == ["meta:time_base"]

    rb, rc = arm(), arm()
    rc["selected_obs"]["xy_px"][1, 40] = (1.0, 2.0)                 # hand-only change is allowed
    assert H.gate_bc(rb, rc, OBS, ("frame_id", "timestamp_us")) == []
    rc["selected_obs"]["usable"][2, 15] = True                      # pose change is a technical failure
    rc["selected_meta"]["timestamp_us"] = np.array([0, 2, 5])
    assert H.gate_bc(rb, rc, OBS, ("frame_id", "timestamp_us")) == ["selected_meta:timestamp_us", "pose:usable"]


def test_budgets_and_decide():
    budgets = dict.fromkeys(H.BUDGET_INTS, 0)
    budgets["forward_calls"] = 8
    assert H.check_budgets({"tcn_forwards": 7, "model_loads": 1}, [], budgets) == ["forward_calls: 7 != 8"]
    assert H.check_budgets({"tcn_forwards": 8, "model_loads": 1, "downloads": 1}, [], budgets) == ["downloads != 0"]
    assert H.check_budgets({"tcn_forwards": 8}, [], budgets) == ["model_loads: 0 != 1"]
    assert H.check_budgets({}, [], budgets, final=False) == []      # extraction stage: forward/model not yet due
    assert H.decide([(True, True)] * 3 + [(False, True)])["code"] == "LIMITED_RECOVERY_FIXED_DEV_SAMPLE"
    assert H.decide([(True, True)] * 2 + [(True, False), (False, True)])["code"] == "MIXED_STOP"
    assert H.decide([(True, True)] * 2 + [(True, False), (False, False)])["code"] == "WORSENED_STOP"
    assert H.decide([(True, True)] * 3 + [(False, False)])["code"] == "NO_CHANGE_STOP"
    assert H.direction(False, True) == "C_recovers" and H.direction(True, False) == "C_regresses"


def frozen_run_instrs(per_clip=(72, 72, 71, 71, 71, 71)):
    """Synthetic task records shaped like 6 clips x (B, C) under the frozen budgets. No MediaPipe."""
    ok = {"attempted": True, "ok": True, "error": None}
    task = lambda kind, mode, calls: NS(kind=kind, mode=mode, calls=calls, close_rec=dict(ok))
    instrs = []
    for n in per_clip:
        instrs.append(NS(tag="B_video", frames=[None] * n, tasks=[task("pose", "VIDEO", n), task("hand", "VIDEO", n)]))
        instrs.append(NS(tag="C_hand_image", frames=[None] * n,
                         tasks=[task("pose", "VIDEO", n), task("hand", "VIDEO", 0), task("hand", "IMAGE", n)]))
    counters = {"decoder_opened": 12, "decoded_frames": 1706, "landmarker_process_calls": 856,
                "landmarker_constructed": 12, "landmarker_closed": 12, "model_loads": 1, "tcn_forwards": 8,
                "camera_calls": 0, "training_runs": 0, "downloads": 0}
    return counters, instrs


def test_frozen_budget_mapping():
    counters, instrs = frozen_run_instrs()
    assert H.check_budgets(counters, instrs, FROZEN_BUDGETS) == []
    act = H.actual_counts(counters, instrs)
    assert act["hand_video_tasks_discarded_without_inference"] == 6 and act["selected_frames_per_arm:C_hand_image"] == 428
    instrs[1].tasks[1].calls = 1                                     # a "discarded" VIDEO hand did inference
    bad = H.check_budgets(counters, instrs, FROZEN_BUDGETS)
    assert "hand_video_tasks_discarded_without_inference: 5 != 6" in bad and "hand_detect_for_video: 429 != 428" in bad
    counters, instrs = frozen_run_instrs()
    assert H.check_budgets(dict(counters, landmarker_closed=11), instrs, FROZEN_BUDGETS) == ["landmarker_closed: 11 != 12"]
    instrs[3].frames.pop()
    assert "selected_frames_per_arm:C_hand_image: 427 != 428" in H.check_budgets(counters, instrs, FROZEN_BUDGETS)


def test_receipt():
    root = Path(tempfile.mkdtemp(prefix="hand_image_receipt_"))
    good = {"status": "PASS", "runner_sha256": H.sha256_file(H.SRC), "test_sha256": H.sha256_file(H.TEST_SRC),
            "tests_run": 8}
    p = root / "receipt.json"
    p.write_text(json.dumps(good), "utf-8")
    assert H.check_receipt(p) == good
    for bad, text in ((dict(good, status="FAIL"), "status"), (dict(good, runner_sha256="0" * 64), "runner_sha256"),
                      (dict(good, test_sha256="0" * 64), "test_sha256"), (dict(good, tests_run="8"), "tests_run")):
        p.write_text(json.dumps(bad), "utf-8")
        try:
            H.check_receipt(p)
            raise AssertionError(f"expected receipt failure: {text}")
        except H.GateFail as exc:
            assert text in str(exc), exc
    try:
        H.check_receipt(None)
        raise AssertionError("expected missing receipt failure")
    except H.GateFail as exc:
        assert "--test-receipt" in str(exc)


def make_cfg(root: Path) -> dict:
    files = {n: root / n for n in ("cadence_pair.py", "old.json", "old_result.json", "runtime.json", "mech.json",
                                   "c1.mp4", "c1_b.npz", "c1_b.meta.json", "contract.json", "h.task", "p.task", "m.pt")}
    for n, p in files.items():
        p.write_bytes(n.encode())
    sha = {n: hashlib.sha256(n.encode()).hexdigest() for n in files}
    s = lambda n: str(files[n])
    return {"schema": H.SCHEMA, "unit_dir": str(root), "output_dir": str(root / "result"), "workspace_root": str(root),
            "front_root": str(root),
            "parent_cadence": {"module_path": s("cadence_pair.py"), "config_path": s("old.json"),
                               "result_path": s("old_result.json"), "runtime_path": s("runtime.json")},
            "mechanism_evidence": {"result": s("mech.json")},
            "clips": [{"clip_id": "c1", "path": s("c1.mp4"), "sha256": sha["c1.mp4"], "expected_frames_from_prior": 3,
                       "expected_selected_even_frames": 2, "score": True, "baseline_b_npz": s("c1_b.npz"),
                       "baseline_b_meta": s("c1_b.meta.json"), "baseline_b_npz_sha256": sha["c1_b.npz"],
                       "baseline_b_meta_sha256": sha["c1_b.meta.json"]}],
            "pins": {s(n): sha[n] for n in files},
            "pin_inventory": [{"path": s(n), "role": n, "bytes": len(n), "sha256": sha[n]} for n in files],
            "extraction": {"contract_path": s("contract.json"), "hand_model_path": s("h.task"),
                           "pose_model_path": s("p.task"),
                           "hand_running_mode_by_arm": {"B_video": "VIDEO", "C_hand_image": "IMAGE"}},
            "scoring": {"checkpoint_path": s("m.pt"), "checkpoint_sha256": sha["m.pt"]},
            "budgets": dict(FROZEN_BUDGETS), "isolation": {}, "baseline_parity": {}, "diagnostics": {}}


def expect_fail(cfg, text):
    try:
        H.preflight(cfg)
    except H.GateFail as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"expected GateFail containing {text!r}")


def test_preflight_hash_only():
    root = Path(tempfile.mkdtemp(prefix="hand_image_cfg_"))
    cfg = make_cfg(root)
    assert H.preflight(cfg)["status"] == "PREFLIGHT_OK_HASH_ONLY"
    assert "cadence_pair" not in sys.modules and "mediapipe" not in sys.modules
    expect_fail({k: v for k, v in cfg.items() if k != "diagnostics"}, "missing keys")
    for m in ({"B": "VIDEO", "C": "IMAGE"}, {"B_video": "VIDEO", "C_hand_image": "VIDEO"}):
        expect_fail(dict(cfg, extraction=dict(cfg["extraction"], hand_running_mode_by_arm=m)), "hand_running_mode_by_arm")
    expect_fail(dict(cfg, budgets={"tcn_forwards": 8}), "budgets")
    expect_fail(dict(cfg, budgets={k: v for k, v in FROZEN_BUDGETS.items() if k != "camera"}), "budgets")
    expect_fail(dict(cfg, budgets=dict(FROZEN_BUDGETS, training=True)), "budgets")
    expect_fail(dict(cfg, budgets=dict(FROZEN_BUDGETS, invented=1)), "unknown budgets keys")
    no_note = {k: v for k, v in FROZEN_BUDGETS.items() if k != "independent_forward_calls"}
    assert H.preflight(dict(cfg, budgets=no_note))["status"] == "PREFLIGHT_OK_HASH_ONLY"
    (root / "result").mkdir()
    expect_fail(cfg, "fresh only")
    (root / "result").rmdir()
    (root / "c1_b.npz").write_bytes(b"changed")
    expect_fail(cfg, "pin mismatch")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} synthetic tests passed")
