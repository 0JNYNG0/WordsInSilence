"""Synthetic-only tests; never loads public observations or saved results."""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
import candidate as C


def sample(y, x=None):
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.full(n, .4) if x is None else np.asarray(x, dtype=float)
    values = np.zeros((n, 69), dtype=float)
    values[:, 28] = x
    values[:, 29] = y
    values[:, 56] = x
    values[:, 57] = y
    values[:, 58] = np.hypot(x, y)
    return dict(x=values, mask=np.ones_like(values, dtype=bool),
                frame_id=np.arange(n, dtype=np.int64),
                timestamp_us=np.arange(n, dtype=np.int64)*33333)


def types(e):
    return [x["type"] for x in e["events"]]


def all_paths(n, m):
    def walk(path):
        i, j = path[-1]
        if i == n-1 and j == m-1:
            yield path
            return
        for di, dj in [(1, 1), (1, 0), (0, 1)]:
            ni, nj = i+di, j+dj
            if ni < n and nj < m and abs(ni/max(n-1, 1)-nj/max(m-1, 1)) <= .25:
                yield from walk(path+[[ni, nj]])
    return list(walk([[0, 0]]))


def contains(path, gates):
    return all(any(a <= i <= b and c <= j <= d for i, j in path)
               for a, b, c, d in gates)


def run():
    checks = []
    def ck(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    def raises(name, f):
        try:
            f()
        except ValueError:
            ck(name, True)
        else:
            raise AssertionError(name)
    q = sample([-.5, -.5, -.5, .5, .5, .5, .5, .5])
    r = sample([-.5, -.5, -.5, -.5, -.5, .5, .5, .5])
    diagonal = [[i, i] for i in range(8)]
    snapshot = copy.deepcopy((q, r))
    result = C.align(q, r, diagonal)
    ck("single observed transition can constrain a different full path", result["alignment_changed"])
    ck("candidate visits rectangle", contains(result["path"], [[2, 3, 4, 5]]))
    ck("all query frames retained", set(i for i, j in result["path"]) == set(range(8)))
    ck("all reference frames retained", set(j for i, j in result["path"]) == set(range(8)))
    for a, b in zip((q, r), snapshot):
        for k in a:
            ck("input immutable "+k, np.array_equal(a[k], b[k], equal_nan=True))
    ck("no physical occurrence certainty", result["occurrence_unresolved"] and not result["physical_occurrence_verified"])
    ck("no educational verdict", result["whole_sign_verdict"] is None and result["correction_advice"] is None)
    same = C.align(q, q, diagonal)
    ck("identity baseline unchanged", same["path"] == diagonal and not same["alignment_changed"])
    ck("identity needs no new DP", "candidate_solver" not in same)

    e = C.events(sample([-1, 0, 0, 1]))
    ck("zero plateau has one bracket", len(e["events"]) == 1 and e["events"][0]["index_range"] == [0, 3])
    ck("zero plateau is not restored or called hold", e["events"][0]["zero_plateau_samples"] == 2 and not e["exact_zero_is_semantic_hold"])
    ck("constant zero has no crossing", not C.events(sample([0, 0, 0]))["events"])
    a = sample([-1, 0, 1]);a["mask"][1, 0] = False
    ck("missing wrist breaks event bracket", not C.events(a)["events"])
    a = sample([-1, -1, 1]);a["frame_id"][-1] = 3
    ck("missing native index breaks event bracket", not C.events(a)["events"])
    a = sample([-1, 1]);a["x"][1, 57] = np.nan
    ck("nonfinite relation is unknown", not C.events(a)["events"])
    a = sample([-1, 1], [-1, 1]);e = C.events(a)
    ck("two axes recorded without arbitrary order anchors", len(e["events"]) == 2 and len(e["excluded_overlap_event_ids"]) == 2)
    ck("overlapping axes fallback exact", C.align(a, a, [[0,0],[1,1]])["fallback"])

    # Repetition is retained as distinct event instances. A one-event target
    # may correspond to either same-kind occurrence: neither is forced.
    a, b = sample([-1, 1, -1, 1]), sample([-1, 1])
    m = C.match_events(C.events(a), C.events(b))
    ck("same-kind repeated events not merged", len(C.events(a)["events"]) == 3)
    ck("no nth-event forcing", not m["forced_pairs"] and not m["combinatorially_unique"])
    ck("extra events retained", m["unmatched_count_per_maximum_match"]["query"] == 2)
    a = sample([-1, 1, -1, 1]);result = C.align(a, a, [[i,i] for i in range(4)])
    ck("touching forced intervals do not fake two gates", result["status"] == "FALLBACK_OVERLAPPING_OR_CONFLICTING_ANCHOR_INTERVALS")
    # Unique among observed events cannot prove which cycle preceded a gap.
    a = sample([-1, -1, -1, 1, 1, 1])
    b = sample([-1, 1, -1, -1, 1, 1]);b["mask"][:2, :] = False
    result = C.align(a, b, [[i,i] for i in range(6)])
    ck("hidden earlier occurrence still unknown", result["event_matching"]["combinatorially_unique"] and result["unknown_intervals_can_hide_occurrences"] and result["occurrence_unresolved"])

    a = sample([-.4, -.4, .4, .4])
    b = sample([-.4, -.4, -.4, .4, .4, .4])
    ck("hold duplication preserves transition types", types(C.events(a)) == types(C.events(b)))
    a = sample([-1e-10, 1e-10, -1e-10, 1e-10])
    result = C.align(a, a, [[i,i] for i in range(4)])
    ck("jitter not hidden by invented tolerance", len(result["events"]["query"]["events"]) == 3)
    ck("jitter events are not semantic phases", result["occurrence_unresolved"] and result["whole_sign_verdict"] is None)
    a = sample([.4]*6);a["mask"][:, :] = False
    baseline = [[i,i] for i in range(6)];result = C.align(a, a, baseline)
    ck("no support exact baseline fallback", result["fallback"] and result["path"] == baseline)
    ck("no support full denominator", result["events"]["query"]["possible_relation_samples"] == 12 and result["events"]["query"]["observed_relation_samples"] == 0)
    a = sample([-.5, -.5, .5, .5]);a["mask"][2, 11] = False
    before = copy.deepcopy(a)
    C.align(a, a, [[i,i] for i in range(4)])
    ck("one missing finger does not delete wrist evidence", len(C.events(a)["events"]) == 1)
    ck("partial finger mask preserved", np.array_equal(a["mask"], before["mask"]))
    changed = copy.deepcopy(a);changed["x"][1:3, 39] += .1
    ck("short finger change does not alter transition labels", C.events(changed) == C.events(a))
    ck("short finger change not overwritten", np.array_equal(changed["x"][1:3,39], a["x"][1:3,39]+.1))
    a = sample([-.5, -.5, .5, .5], [-.3, -.3, .3, .3]);before=copy.deepcopy(a)
    C.align(a, a, [[i,i] for i in range(4)])
    ck("normal crossing preserves anatomical columns", np.array_equal(a["x"], before["x"]))

    for forbidden in ["word", "alias", "source_index", "phase", "reference_kind"]:
        a = sample([-.5, .5]);a[forbidden] = "forbidden"
        raises("reject truth/identity metadata "+forbidden, lambda: C.events(a))
    a = sample([-.5,.5]);a["mask"] = a["mask"].astype(int)
    raises("explicit boolean mask", lambda: C.events(a))
    a = sample([-.5,.5]);a["timestamp_us"][1] = 0
    raises("strict timestamps", lambda: C.events(a))
    raises("full scope baseline required", lambda: C.align(q, q, [[1,1],[2,2]]))

    # Independent exhaustive enumeration of all tiny monotone paths. This
    # oracle neither calls C13's DP nor reproduces the gate-state recurrence.
    for n in [3,4]:
        paths = all_paths(n,n)
        gate_sets = [[], [[1,1,1,1]], [[0,0,0,0],[n-1,n-1,n-1,n-1]], [[1,2,0,1]]]
        costs = [np.zeros((n,n)), np.eye(n), np.arange(n*n,dtype=float).reshape(n,n)%3]
        near = np.ones((n,n));near[0,0] = 0;near[-1,-1] = 0;near[1,1] = 1e-12
        costs.append(near)
        for ci,cost in enumerate(costs):
            for gi,gates in enumerate(gate_sets):
                feasible = [p for p in paths if contains(p,gates)]
                result = C.solve(cost,gates)
                ck(f"oracle feasibility {n}/{ci}/{gi}", (result["path"] is not None) == bool(feasible))
                if feasible:
                    values = [sum(cost[i,j] for i,j in p) for p in feasible]
                    best = min(values)
                    ck(f"oracle exact optimum {n}/{ci}/{gi}", result["objective_sum"] == best)
                    ck(f"oracle representative {n}/{ci}/{gi}", sum(cost[i,j] for i,j in result["path"]) == best and contains(result["path"],gates))
                    ck(f"oracle exact path ambiguity {n}/{ci}/{gi}", result["exact_path_count_capped"] == min(2,sum(v==best for v in values)))
    ck("outside band gate has no path", C.solve(np.zeros((5,5)), [[0,0,4,4]])["path"] is None)
    raises("conflicting gate input", lambda: C.solve(np.zeros((5,5)), [[0,2,0,2],[2,4,3,4]]))

    # Exhaustively enumerate event matchings independently for tiny token
    # sequences; common pairs must equal the intersection of all maxima.
    def event_fixture(tokens):
        return dict(events=[dict(event_id=i,type=["y",-1 if t==0 else 1,1 if t==0 else -1],anchor_eligible=True) for i,t in enumerate(tokens)])
    for n,m in [(2,2),(3,2),(3,3)]:
        for aa in itertools.product(range(2),repeat=n):
            for bb in itertools.product(range(2),repeat=m):
                matches=[]
                for length in range(min(n,m)+1):
                    for ia in itertools.combinations(range(n),length):
                        for ib in itertools.combinations(range(m),length):
                            if all(aa[i]==bb[j] for i,j in zip(ia,ib)):
                                matches.append(set(zip(ia,ib)))
                maximum=max(map(len,matches));best=[s for s in matches if len(s)==maximum]
                common=set.intersection(*best)
                result=C.match_events(event_fixture(aa),event_fixture(bb))
                got={(p['query_event_id'],p['reference_event_id']) for p in result['forced_pairs']}
                ck(f"LCS common intersection {aa}/{bb}", got==common and result['maximum_observed_event_matches']==maximum)
    return dict(status="PASS", checks=len(checks), check_names=checks,
                public_data_loaded=False, new_inference=0, new_public_alignment=0,
                known_limitations=["Exact-sign jitter can create extra observed events.",
                                  "Unique observed event matching cannot identify hidden occurrences across missing evidence.",
                                  "Rectangle visits and monotonic paths do not verify physical action phases."])


if __name__ == "__main__":
    output = run()
    output["code_sha256"] = hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()
    output["selftest_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target = Path(__file__).with_name("SELFTEST.json")
    if target.exists():
        raise FileExistsError("Preserve an earlier selftest; use a new output explicitly")
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=output["status"],checks=output["checks"],path=str(target),code_sha256=output["code_sha256"])))
