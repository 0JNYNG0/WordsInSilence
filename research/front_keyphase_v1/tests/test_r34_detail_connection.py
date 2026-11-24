"""observational_detail_v1: the detail blocks reach the ordinary web path, additively.

Public reference observations and synthetic edits of them only. No camera, session, server,
network model or participant file. The live path is app._finish's scoring order
(predict -> task_match -> _split_result); the replay path is _explicit_comparison ->
_split_result -> _replay_following, the same order the replay handler uses.

Run: py -3.13 tests/test_r34_detail_connection.py
"""
import copy
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core import features as F  # noqa: E402
from wis_lab.core import recognize as R  # noqa: E402
from wis_lab.core.observation import Observation  # noqa: E402
from wis_lab.core.trajectory import location_path  # noqa: E402
from wis_lab.service.app import _explicit_comparison, _replay_following, _split_result  # noqa: E402

CONTRACT_PATH = os.path.join(ROOT, "config", "contract.json")
CONTRACT = json.load(open(CONTRACT_PATH, encoding="utf-8"))
CANDIDATE = json.load(open(os.path.join(ROOT, "config", "observed_feedback_candidate_v1.json"),
                           encoding="utf-8"))
MANIFEST = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
REFS, _ = load_reference_set()
ST = SimpleNamespace(refs=REFS, contract=CONTRACT)
W, OTHER = REFS[0]["word_id"], REFS[1]["word_id"]
POLICY = "observational_detail_v1"
FAILS = []


def ok(cond, name):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAILS.append(name)


def dump(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)


def clip(word_id):
    return next(c for c in MANIFEST["clips"] if c["word_id"] == word_id)


def obs_for(word_id):
    return Observation.load(os.path.join(ROOT, clip(word_id)["observation_npz"]))


def binding(word_id, status="BOUND"):
    return {"status": status, "demonstration_clip_id": clip(word_id)["clip_id"],
            "demonstration_word_id": word_id}


def live(obs, requested):
    result = R.predict(obs, REFS, CONTRACT)
    result["task"] = R.task_match(result, requested, observation=obs,
                                  reference_set=REFS, frozen_config=CONTRACT)
    _split_result(result, {"reference_binding": binding(requested)})
    return result


def replay(obs, chosen):
    result = R.predict(obs, REFS, CONTRACT)
    erc = _explicit_comparison(obs, chosen, ST)
    result["explicit_reference_comparison"] = erc
    _split_result(result, {"reference_binding": binding(chosen)})
    _replay_following(result, binding(chosen), erc)
    return result


def without_detail(dev):
    return {k: v for k, v in dev.items() if k not in (
        "movement_observation", "movement_observation_role", "movement_observation_scope",
        "detail_policy")}


# ------------------------------------------------ 1. default path carries both detail blocks
for obs_word, chosen in ((W, W), (OTHER, W)):
    for name, res in (("live", live(obs_for(obs_word), chosen)),
                      ("replay", replay(obs_for(obs_word), chosen))):
        df = res["demonstration_following"]
        iv = (df.get("location_path") or {}).get("interval_review") or {}
        ok(df["status"] == "AVAILABLE" and iv.get("status") == "AVAILABLE"
           and iv.get("detail_policy") == POLICY,
           f"{name} {obs_word}->{chosen}: interval detail is available on the default contract")
        ok(iv.get("reference_word_id") == chosen
           and iv.get("reference_clip_id") == clip(chosen)["clip_id"],
           "  and it is about the bound demonstration, not the prediction")
        ok([(s["i"], s["j"]) for s in iv["steps"]]
           == [(s["i"], s["j"]) for s in df["location_path"]["steps"]],
           "  and it reads exactly the trajectory's own path")
        ok(iv.get("review_window") == {"status": "NOT_CONFIGURED"},
           "  and no preset review window is invented")
        dev = df["axis_deviation"]
        mo = dev.get("movement_observation") or {}
        ok(mo.get("status") == "AVAILABLE" and dev.get("movement_observation_role") == "SUPPLEMENTAL",
           "  full-rate movement observation is attached as SUPPLEMENTAL")
        ok("movement" in dev["axes"] and "movement" in dev["order"],
           "  and the movement aggregate distance stays beside it")

# ------------------------------------------------ 2. live and replay use the same policy
a = live(obs_for(W), W)["demonstration_following"]
b = replay(obs_for(W), W)["demonstration_following"]
ok(dump(a["location_path"]["interval_review"]) == dump(b["location_path"]["interval_review"]),
   "live and replay produce the identical interval detail for the same clip and reference")
ok(dump(a["axis_deviation"]) == dump(b["axis_deviation"]),
   "and the identical axis deviation including the movement observation")

# ------------------------------------------------ 3. recognition is blind to the request
r1, r2 = live(obs_for(OTHER), W), live(obs_for(OTHER), OTHER)
ok(dump(r1["ranking"]) == dump(r2["ranking"]) and r1["top1_word_id"] == r2["top1_word_id"],
   "the requested word does not change the ranking or its distances")

# ------------------------------------------------ 4. detail never leaks into scoring
seq = F.sequence_features(obs_for(W))
q = F.uniform_time_sample(seq, int(CONTRACT["sampling_A"]["budget"]))
bare = {k: v for k, v in q.items() if k not in ("movement_observation", "feedback_body_context")}
ref = next(r for r in REFS if r["word_id"] == W)
d_full = R.axis_deviation(q, W, REFS, CONTRACT)
d_bare = R.axis_deviation(bare, W, REFS, CONTRACT)
ok(dump(without_detail(d_full)) == dump(without_detail(d_bare)),
   "removing the detail sources changes no axis distance, deviation, order or support")
ok(d_bare["movement_observation"]["status"] == "UNAVAILABLE"
   and d_bare["movement_observation"]["query"]["status"] == "UNAVAILABLE",
   "a query without its full-rate source is UNAVAILABLE, not reconstructed from samples")
lp_full = location_path(q, ref, CONTRACT)
lp_bare = location_path(bare, ref, CONTRACT)
ok(dump({k: v for k, v in lp_full.items() if k != "interval_review"})
   == dump({k: v for k, v in lp_bare.items() if k != "interval_review"}),
   "and no trajectory step, point or segment")
qi = lp_bare["interval_review"]["inputs"]["query"]
ok(qi["body_points"] == {} and qi["normalization"] is None,
   "a query without body context draws no anchors and claims no normalization")
r_default = R.predict(obs_for(W), REFS, CONTRACT)
opt = copy.deepcopy(CONTRACT)
opt["feedback"] = {"selected_interval_review": True}
r_opt = R.predict(obs_for(W), REFS, opt)
ok(dump(r_default) == dump(r_opt),
   "the old opt-in flag no longer changes anything on the default path")

# ------------------------------------------------ 5. no binding, no display
res = R.predict(obs_for(W), REFS, CONTRACT)
res["task"] = R.task_match(res, W, observation=obs_for(W), reference_set=REFS,
                           frozen_config=CONTRACT)
_split_result(res, {"reference_binding": {"status": "UNDETERMINED_NO_DEMONSTRATION_SPECIFIED"}})
df = res["demonstration_following"]
ok(df["location_path"] is None and df["axis_deviation"] is None,
   "a missing demonstration binding still prevents any detail from being shown")

# ------------------------------------------------ 6. the candidate still REPLACES movement
cand = copy.deepcopy(CONTRACT)
cand["feedback"] = copy.deepcopy(CANDIDATE.get("feedback") or {})
ok(cand["feedback"].get("exclude_both_resting_hands") is True
   and cand["feedback"].get("observed_movement_structure") is True,
   "fixture: only the candidate's feedback block is borrowed, not its whole config")
dc = R.axis_deviation(q, W, REFS, cand)
ok("movement" not in dc["axes"] and dc["movement_observation"]["status"] == "AVAILABLE"
   and "movement_observation_role" not in dc,
   "the role-filtered candidate keeps replacing the movement aggregate, unchanged")
ok(dump(dc["movement_observation"]) == dump(d_full["movement_observation"]),
   "and its movement observation is the same compare_sources output as the default path's")

# ------------------------------------------------ 7. frozen contract untouched
same = open(CONTRACT_PATH, "rb").read() == open(os.path.join(
    ROOT, "ops", "web_detail_connection_20260926", "before", "config", "contract.json"),
    "rb").read()
ok(same and "feedback" not in CONTRACT, "config/contract.json is byte-identical, still no feedback block")

print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL'}")
sys.exit(1 if FAILS else 0)
