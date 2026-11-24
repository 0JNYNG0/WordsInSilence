"""Independent state-machine safety contracts, without camera/server/model training.

The injected shapes and synthetic coordinates isolate controller behavior. They
are not learner-accuracy evidence and never enter the live input path. Root runs
the separately required public learned-observer regression gate.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys

import numpy as np

V = Path(__file__).resolve().parent
U = V.parent
sys.path.insert(0, str(U))
sys.path.insert(0, str(U.parent / "basic_demo_feedback_20261115"))
sys.path.insert(0, str(U.parent / "word100_rtmw_shared_input_20261108"))


def shape(state):
    return dict(state=state, fingers=[1 if state == 0 else 0 if state == 1 else -1] * 4)


def shapes(left=0, right=0):
    return {"L": shape(left), "R": shape(right)}


def points(q=.2, y=None, left_y=None, right_y=None):
    """Body-relative fixture: 200 px shoulder width, stable left forearm."""
    p = np.tile(np.array([500., 300.], dtype=np.float32), (133, 1))
    p[5] = [400, 200]
    p[6] = [600, 200]
    p[7] = [600, 300]
    p[8] = [350, 300]
    p[9] = [400, 300]
    p[91:112] = p[9]
    p[112:133] = [600 - 200 * q, 300]
    p[10] = p[112]
    if y is not None:
        ly = y if left_y is None else left_y
        ry = y if right_y is None else right_y
        p[91:112] = [430, 260 + 200 * ly]
        p[112:133] = [570, 260 + 200 * ry]
        p[9], p[10] = p[91], p[112]
    return p


class HoldAction:
    def reset(self):
        self.queries = []

    def movement(self, start, end, axis, moving, anchor):
        self.queries.append((start, end, axis, moving, anchor))
        return dict(eligible=True, state=1, reason="independent_hold_fixture")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coach", type=Path, default=U / "coach.py")
    parser.add_argument("--output", type=Path, default=V / "INDEPENDENT_CHECKS.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous review results; use a fresh output filename.")
    before = hashlib.sha256(args.coach.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("independent_coach_target", args.coach)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Coach = module.Coach
    checks = []

    def run(name, fn):
        try:
            details = fn() or {}
            checks.append(dict(name=name, passed=True, **details))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, error=type(exc).__name__ + ": " + str(exc)))

    def new(stage=0, action=None):
        c = Coach(model=object(), action=action)
        if stage:
            c.stage_index = stage
            c.completed[0] = True
        c.test_ts = 1000000
        return c

    def feed(c, p=None, obs=None, dt=33333):
        c.test_ts += dt
        out = c.consume_observation(points() if p is None else p, c.test_ts, 1000, 800,
                                    shapes() if obs is None else obs)
        assert out["whole_word_correct"] is None
        assert all(x["state"] in {"ok", "wrong", "unknown", "pending"} for x in out.get("checks", []))
        json.dumps(out, allow_nan=False)
        return out

    def no_wrong(out):
        assert all(x.get("state") != "wrong" for x in out.get("checks", []) + out.get("highlights", [])), out

    def start_sweep():
        c = new()
        feed(c, points(.2)); feed(c, points(.4))
        assert c.start_q is not None and not c.completed[0]
        return c

    def repeat_one():
        c = new(1)
        feed(c, points(y=0), shapes(1, 1))
        feed(c, points(y=.08), shapes(1, 1))
        assert c.repeats == 1 and c.turn == "up" and not c.completed[1]
        return c

    def sweep_role(left, missing=False):
        c = new()
        for q in [.2, .4, .6, .9]:
            p = points(q)
            if missing:
                p[92:112] = np.nan
            out = feed(c, p, shapes(left, 0))
            assert not any(x.get("id") == "L_shape" and x["state"] in {"wrong", "unknown"}
                           for x in out.get("checks", []))
        assert c.completed == [True, False]
        return dict(completed=c.completed)

    run("sweep_support_shape_unknown_not_required", lambda: sweep_role(-1))
    run("sweep_support_closed_shape_not_required", lambda: sweep_role(1))
    run("sweep_support_finger_landmarks_not_required", lambda: sweep_role(-1, True))

    for joint, label in [(5, "shoulder"), (7, "support_elbow"), (9, "support_wrist")]:
        def required_anchor(joint=joint):
            c = new()
            for q in [.2, .4, .6, .9]:
                p = points(q); p[joint] = np.nan
                out = feed(c, p)
                no_wrong(out)
            assert not any(c.completed)
        run("sweep_missing_" + label + "_cannot_complete", required_anchor)

    def sweep_unknown():
        c = start_sweep(); old = (c.start_q, c.repeats, len(c.events))
        out = feed(c, points(.4), shapes(0, -1), dt=100000)
        no_wrong(out)
        assert (c.start_q, c.repeats, len(c.events)) == old
        for q in [.4, .6, .9]:
            out = feed(c, points(q))
        assert c.completed == [True, False]
    run("sweep_short_shape_uncertainty_preserves_observed_progress", sweep_unknown)

    def sweep_missing():
        c = start_sweep(); old = (c.start_q, c.repeats, len(c.events))
        p = points(.4); p[120] = np.nan
        out = feed(c, p, dt=100000); no_wrong(out)
        assert (c.start_q, c.repeats, len(c.events)) == old
        for q in [.4, .6, .9]:
            feed(c, points(q))
        assert c.completed == [True, False]
    run("sweep_short_missing_landmark_preserves_observed_progress", sweep_missing)

    def sweep_endpoint_skip():
        c = start_sweep()
        feed(c, points(.4), shapes(0, -1), dt=100000)
        for q in [.91, .91, .92, .93, .94]:
            feed(c, points(q))
        assert c.completed == [False, False] and not c.events
    run("sweep_hidden_endpoint_jump_and_hold_cannot_complete", sweep_endpoint_skip)

    def sweep_whole_missing():
        c = new()
        for q in [.2, .4, .6]:
            feed(c, points(q), shapes(0, -1))
        for q in [.9] * 8:
            feed(c, points(q))
        assert not any(c.completed)
    run("sweep_whole_motion_unobserved_cannot_complete", sweep_whole_missing)

    def sweep_long_gap():
        c = start_sweep()
        feed(c, points(.4), shapes(0, -1), dt=100000)
        feed(c, points(.4), shapes(0, -1), dt=300000)
        assert c.start_q is None
        for q in [.9, .91, .92]:
            feed(c, points(q))
        assert not any(c.completed)
    run("sweep_long_uncertainty_resets_attempt", sweep_long_gap)

    def sweep_wrong():
        c = start_sweep()
        out = feed(c, points(.4), shapes(0, 1))
        assert c.start_q is None and out["phase"] == "correction"
        assert any(x["state"] == "wrong" for x in out.get("checks", []))
        for q in [.9] * 4:
            feed(c, points(q))
        assert not any(c.completed)
    run("sweep_confirmed_wrong_shape_breaks_continuity", sweep_wrong)

    def sweep_reverse():
        c = new()
        for q in [.2, .5, .25]:
            out = feed(c, points(q))
        assert c.start_q is None and not any(c.completed) and out["phase"] == "correction"
    run("sweep_observed_reverse_still_requires_restart", sweep_reverse)

    def learned_sweep_veto():
        action = HoldAction(); c = new(action=action)
        for q in [.2, .4, .6, .9]:
            feed(c, points(q))
        assert action.queries and not any(c.completed)
    run("sweep_learned_direction_veto_retained", learned_sweep_veto)

    for hand in ["L", "R"]:
        def repeat_wrong(hand=hand):
            c = repeat_one()
            obs = shapes(1, 1); obs[hand] = shape(0)
            out = feed(c, points(y=.08), obs)
            assert c.repeats == 0 and c.high is None
            assert c.completed == [True, False] and out["phase"] == "correction"
        run("repeat_" + hand + "_known_open_resets_attempt", repeat_wrong)

        def repeat_gap(hand=hand):
            c = repeat_one(); old_events = len(c.events)
            obs = shapes(1, 1); obs[hand] = shape(-1)
            out = feed(c, points(y=.08), obs, dt=100000); no_wrong(out)
            assert c.repeats == 1 and len(c.events) == old_events and c.completed == [True, False]
            feed(c, points(y=.08), shapes(1, 1))
            assert c.repeats == 1 and len(c.events) == old_events
            for y in [.04, 0, .08]:
                feed(c, points(y=y), shapes(1, 1))
            assert c.completed == [True, True] and c.repeats == 2
        run("repeat_" + hand + "_short_uncertainty_preserves_count_and_visible_return", repeat_gap)

    def repeat_hidden_return():
        c = repeat_one(); old_events = len(c.events)
        feed(c, points(y=.08), shapes(-1, -1), dt=100000)
        feed(c, points(y=0), shapes(1, 1))
        assert c.repeats == 1 and c.turn == "up" and len(c.events) == old_events
        for y in [.04, .08, .08, .08]:
            feed(c, points(y=y), shapes(1, 1))
        assert c.repeats == 1 and not c.completed[1]
    run("repeat_hidden_upward_return_cannot_enable_second_downstroke", repeat_hidden_return)

    def repeat_hidden_down():
        c = new(1)
        feed(c, points(y=0), shapes(1, 1))
        feed(c, points(y=0), shapes(-1, -1), dt=100000)
        for y in [.08, .08, .08]:
            feed(c, points(y=y), shapes(1, 1))
        assert c.repeats == 0 and not c.events and not c.completed[1]
        for y in [.04, 0, .08]:
            feed(c, points(y=y), shapes(1, 1))
        assert c.repeats == 1 and not c.completed[1]
    run("repeat_hidden_downstroke_is_not_counted_on_recovery", repeat_hidden_down)

    def repeat_long():
        c = repeat_one()
        feed(c, points(y=.08), shapes(-1, -1), dt=100000)
        feed(c, points(y=.08), shapes(-1, -1), dt=300000)
        assert c.repeats == 0 and c.high is None and c.completed == [True, False]
    run("repeat_long_uncertainty_resets_unfinished_stage", repeat_long)

    for side in ["L", "R"]:
        def one_hand(side=side):
            c = new(1)
            for y in [0, .04, .08, .04, 0, .04, .08]:
                p = points(y=0, left_y=y if side == "L" else 0, right_y=y if side == "R" else 0)
                feed(c, p, shapes(1, 1))
            assert c.repeats == 0 and not c.completed[1]
        run("repeat_only_" + side + "_moving_cannot_count", one_hand)

    def static_repeat():
        c = new(1)
        for _ in range(10):
            feed(c, points(y=0), shapes(1, 1))
        assert c.repeats == 0 and not c.completed[1]
    run("repeat_static_fists_cannot_complete", static_repeat)

    def reset():
        c = repeat_one()
        feed(c, points(y=.08), shapes(-1, -1))
        out = c.reset()
        assert c.completed == [False, False] and c.repeats == 0 and not c.events and c.start_q is None
        assert out["whole_word_correct"] is None
    run("session_reset_clears_all_progress", reset)

    def degenerate_anchor():
        c = start_sweep()
        p = points(.4); p[7] = p[9]
        for _ in range(4):
            out = feed(c, p, dt=100000)
            no_wrong(out)
        assert c.start_q is None and not any(c.completed)
    run("long_degenerate_forearm_does_not_restart_pause_clock", degenerate_anchor)

    def no_jump_travel():
        c = start_sweep()
        feed(c, points(.4), shapes(0, -1), dt=100000)
        feed(c, points(.75))
        for q in [.8, .87, .9]:
            feed(c, points(q))
        assert not any(c.completed), "Only .35 q travel was actually observed"
    run("sweep_gap_displacement_is_not_added_to_observed_travel", no_jump_travel)

    def many_pause_updates():
        c = repeat_one(); events = copy.deepcopy(c.events)
        for _ in range(5):
            out = feed(c, points(y=.08), shapes(-1, -1), dt=30000)
            no_wrong(out)
            assert c.repeats == 1 and c.events == events and c.completed == [True, False]
    run("all_short_gap_updates_have_zero_completion_and_count_increase", many_pause_updates)

    def category(kind):
        c = start_sweep()
        p = points(.4); obs = shapes()
        if kind == "missing_keypoints":
            p[120] = np.nan
        elif kind == "shape_uncertain":
            obs["R"] = shape(-1)
        elif kind == "shape_mismatch":
            obs["R"] = shape(1)
        elif kind == "rule_unmet":
            p[112:133, 1] += 100; p[10] = p[112]
        elif kind == "motion_uncertain":
            c = new(action=HoldAction())
            feed(c, points(.2)); p = points(.9)
        out = feed(c, p, obs)
        assert out["diagnostic"]["category"] == kind, out.get("diagnostic")
        if kind in {"missing_keypoints", "shape_uncertain"}:
            no_wrong(out)
            assert out["diagnostic"]["held_progress"]
    for kind in ["missing_keypoints", "shape_uncertain", "shape_mismatch", "rule_unmet", "motion_uncertain"]:
        run("diagnostic_separates_" + kind, lambda kind=kind: category(kind))

    def delay():
        c = new()
        for i in range(7):
            out = c.update(points(.2), np.ones(133, dtype=np.float32), i * 150000, 1000, 800)
        assert out["diagnostic"]["category"] == "observation_delay"
        assert c.start_q is None and not any(c.completed)
        no_wrong(out)
    run("diagnostic_separates_observation_delay", delay)

    report = dict(
        scope=__doc__, passed=all(x["passed"] for x in checks),
        passes=sum(x["passed"] for x in checks), total=len(checks), checks=checks,
        target=str(args.coach), coach_sha256=before,
        code_unchanged_during_review=hashlib.sha256(args.coach.read_bytes()).hexdigest() == before,
        camera_opened=False, server_accessed=False, private_data=False, training=False,
        model_evaluated=False, independent_person_accuracy=None, whole_word_accuracy=None,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["passed", "passes", "total", "code_unchanged_during_review"]}))
    for row in checks:
        if not row["passed"]:
            print(json.dumps(row, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
