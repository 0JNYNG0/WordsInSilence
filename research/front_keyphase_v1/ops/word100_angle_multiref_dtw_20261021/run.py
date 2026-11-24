import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import sys, json, hashlib, shutil, traceback, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import numpy as np
from adapter import pair, aggregate

U = Path(__file__).resolve().parent
O = U.parent
B = O.parent
ROOT = B.parents[1]
A = O / "word100_angle_evidence_20261021"
M = O / "word100_multiref_observation_compare_20261019"
sys.dont_write_bytecode = True
sys.path.insert(0, str(B / "src"))
from wis_lab.core.observation import Observation
from wis_lab.core import features as F
from wis_lab.core.dtw import frame_cost_matrix, _optimal_path
sys.path.insert(0, str(A))
from candidate import angle_evidence

def read(p): return json.loads(Path(p).read_text("utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
def write(p, x):
    with Path(p).open("x", encoding="utf8") as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
def now(): return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def freeze():
    assert not (U / "FROZEN.json").exists()
    assert read(U / "CONTRACTS.json")["status"] == "PASS"
    prior, bank = read(A / "FROZEN.json"), read(M / "FROZEN.json")
    quality = {r["clip_id"]: r for r in prior["rows"]}
    index = {r["clip_id"]: r for r in read(A / "INDEX.json")}
    rows = []
    pins = {}
    for row in bank["inputs"]:
        if row["role"] != "train_candidate":
            continue
        cid = row["clip_id"]
        r = dict(clip_id=cid, signer_id=row["signer_id"], word_code=row["word_code"],
                 label=row["canonical_label"], intervals=row["intervals"], query=cid in quality)
        if cid in quality:
            r.update(observation=quality[cid]["obs"], evidence=index[cid]["path"], source="merged136")
            assert sha(r["evidence"]) == index[cid]["sha256"]
            pins[r["evidence"]] = index[cid]["sha256"]
        else:
            r.update(observation=row["inputs"]["masked"], evidence=None, source="masked763_unassessed")
            assert sha(r["observation"]) == row["input_hashes"]["masked"]
        pins[r["observation"]] = sha(r["observation"])
        side = Path(r["observation"]).with_suffix(".sidecar.json")
        if side.exists(): pins[str(side)] = sha(side)
        assert 1 <= r["signer_id"] <= 9
        rows.append(r)
    assert len(rows) == 899 and sum(r["query"] for r in rows) == 136
    for p, h in prior["protected"].items(): assert sha(p) == h, p
    for p in [A / "FROZEN.json", A / "VERIFY.json", A / "FINAL_AUDIT.json", A / "candidate.py",
              M / "FROZEN.json", U / "run.py", U / "adapter.py", U / "contracts.py", U / "CONTRACTS.json"]:
        pins[str(p)] = sha(p)
    for p in (B / "src/wis_lab/core").glob("*.py"): pins[str(p)] = sha(p)
    (U / "before").mkdir()
    for p in [ROOT / "WordsInSilence_PROJECT_CONTEXT.md", ROOT / "NEW_CHAT_HANDOFF.md", O / "STATE.json"]:
        shutil.copyfile(p, U / "before" / p.name)
    write(U / "FROZEN.json", dict(at=now(), rows=rows, pins=pins, protected=prior["protected"],
        roles=bank["roles"], implementer="Codex local", verifier="Codex separate formula, not independent reviewer",
        authorization="2026-10-21 user: angle evidence + public multireference DTW, no learning until review",
        conditions=dict(band=.2, path_support=.35, min_frames=8, min_support=.5, min_people=5,
            sampling="ALL saved observation frames, no new video extraction; full clip, trim=False",
            alignment="unchanged existing raw feature cost; evidence applied AFTER one shared path",
            reference_expected="fixed public annotation core, all 15 angles on required sides; never shrinks with missing query or reference",
            aggregation="equal median over distinct qualifying reference people; weighted within each reference by unique source frame cell duration",
            evidence="state4 both ends for supported-only difference; single-view stays unassessed",
            normal_safety="no coordinate removal, no identity reassignment, no wrong/correct sign verdict",
            limitations="136 merged with quality +763 masked fullframe-to-ROI without second-view quality; identical snapshot for raw/strict comparison, not model performance experiment. Public core annotation is offline-only. Existing repeat guard is not replaced."),
        forbidden=["user/mentor records", "DEV or heldout", "training", "classifier inference",
                   "detector", "camera", "download", "external transfer", "Git", "production change"]))
    print("FROZEN899 / queries136", flush=True)


def prepared(row):
    obs = Observation.load(row["observation"])
    seq = F.sequence_features(obs, trim=False)
    if row["evidence"]:
        e = load(row["evidence"])
        for k in ("xy_px", "observed", "usable", "conf", "timestamp_us", "frame_id"):
            assert np.array_equal(e[k], getattr(obs, k), equal_nan=True), (row["clip_id"], k)
        deg, state = e["angle_deg"], e["angle_state"]
    else:
        xy = np.stack((obs.xy_px[:, 33:54], obs.xy_px[:, 54:75]), axis=1)
        usable = np.stack((obs.usable[:, 33:54], obs.usable[:, 54:75]), axis=1)
        e = angle_evidence(xy, usable, np.zeros((obs.n_frames, 2), bool),
                           np.zeros((obs.n_frames, 2, 15), bool), np.zeros((obs.n_frames, 2, 15), bool))
        deg, state = e["angle_deg"], e["angle_state"]
    core = np.zeros(obs.n_frames, bool)
    for a, b in row["intervals"]: core |= (obs.timestamp_us >= a * 1e6) & (obs.timestamp_us <= b * 1e6)
    assert core.any(), row["clip_id"]
    out = dict(features=seq["features"], present=seq["present"], angle_deg=deg, angle_state=state,
               timestamp_us=obs.timestamp_us, frame_id=obs.frame_id, core=core,
               xy_px=obs.xy_px, usable=obs.usable)
    np.savez_compressed(U / "prepared" / (row["clip_id"] + ".npz"), **out)
    return out


def alignment(q, r, required):
    weights = None if len(required) == 2 else {
        name: weight if name.endswith("_" + required[0]) else 0. for name, (_, weight) in F.GROUPS.items()}
    cost, evidence = frame_cost_matrix(q, r, group_weights=weights,
                                      hand_participation=False, one_sided_mode="flat")
    path = np.asarray(_optimal_path(cost, .2), int).reshape(-1, 2)
    assert len(path), "No DTW path"
    support = float(evidence[tuple(path.T)].mean())
    return path, support


def execute():
    f = read(U / "FROZEN.json")
    for p, h in {**f["pins"], **f["protected"]}.items(): assert sha(p) == h, p
    assert not (U / "RUNNING.json").exists()
    write(U / "RUNNING.json", dict(pid=os.getpid(), at=now()))
    start = time.monotonic()
    try:
        for sub in ("prepared", "pairs", "queries"): (U / sub).mkdir()
        counts = Counter()
        index = []
        for word in sorted({r["word_code"] for r in f["rows"]}):
            rows = sorted([r for r in f["rows"] if r["word_code"] == word], key=lambda r: r["signer_id"])
            samples = {r["clip_id"]: prepared(r) for r in rows}
            required = f["roles"].get(word, {}).get("required_sides", ["L", "R"])
            for row in rows:
                if not row["query"]: continue
                cid = row["clip_id"]; q = samples[cid]; pairs = []; people = []; pair_ids = []
                for ref in rows:
                    if ref["signer_id"] == row["signer_id"]: continue
                    rid = ref["clip_id"]; r = samples[rid]
                    path, support = alignment(q, r, required)
                    p = pair(q, r, path, required, support >= .35)
                    pid = cid + "__" + rid
                    np.savez_compressed(U / "pairs" / (pid + ".npz"),
                        **{k: p[k] for k in ("expected", "raw", "strict", "difference_deg", "path")})
                    write(U / "pairs" / (pid + ".json"), dict(query=cid, reference=rid, required=required,
                          alignment_support=support, rows=p["rows"]))
                    pairs.append(p); people.append(ref["signer_id"]); pair_ids.append(pid)
                    counts["pairs"] += 1
                    counts["quality_both_pairs"] += int(ref["query"])
                ag = aggregate(q, pairs, required, people)
                np.savez_compressed(U / "queries" / (cid + ".npz"), votes=ag["votes"], expected_votes=ag["expected_votes"])
                record = dict(**row, required=required, role_reviewed=word in f["roles"],
                    rows=ag["rows"], pairs=pair_ids, accepted=None, sign_error=None, whole_hand_score=None)
                write(U / "queries" / (cid + ".json"), record)
                index.append(dict(clip_id=cid, label=row["label"], word_code=word, signer_id=row["signer_id"]))
                counts.update("angle_" + x["status"] for x in ag["rows"])
                counts["query_clips"] += 1
            print(word, dict(counts), flush=True)
        write(U / "INDEX.json", index)
        write(U / "RUN_RESULT.json", dict(status="COMPLETE_PENDING_VERIFY", counts=dict(counts), seconds=time.monotonic()-start, at=now()))
    except BaseException:
        p = U / "FIRST_ERROR.txt"
        if not p.exists(): p.write_text(traceback.format_exc(), encoding="utf8")
        raise
    finally:
        (U / "RUNNING.json").unlink()

if __name__ == "__main__":
    {"freeze": freeze, "run": execute}[sys.argv[1]]()
