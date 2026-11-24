"""Independent saved-artifact arithmetic. No model imports or model execution."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).resolve().parents[1]
OLD = UNIT.parent / "word100_semantic_episode_pilot_20261031"
KNOWN = {"rest", "preparation", "core_candidate", "return"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(segments, frame):
    """An overlap is ambiguous even when overlapping segments share a role."""
    hits = [s for s in segments if s["core"][0] <= frame <= s["core"][1]]
    if len(hits) != 1 or hits[0]["role"] not in KNOWN:
        return None
    return hits[0]["segment_id"], hits[0]["role"]


def transfer(segments, reference_frames):
    if not reference_frames:
        return None
    memberships = [locate(segments, f) for f in reference_frames]
    if any(m is None for m in memberships):
        return None
    if len(set(memberships)) != 1:
        return None
    return memberships[0][1]


def consensus(reference_votes, expected_references):
    identities = [x[0] for x in reference_votes]
    if len(identities) != 3 or len(set(identities)) != 3:
        return None
    if set(identities) != set(expected_references):
        return None
    votes = [x[1] for x in reference_votes]
    return votes[0] if votes[0] in KNOWN and len(set(votes)) == 1 else None


def role_frames(segments, count):
    return [None if (x := locate(segments, f)) is None else x[1] for f in range(count)]


def independent_projection(phase_clips, auth_rows, empirical_development):
    auth = {r["alias"]: r for r in auth_rows}
    result = {}
    for development in empirical_development:
        alias = development["alias"]
        query = auth[alias]
        count = query["frames"]
        refs = [r["alias"] for r in auth_rows if r["role"] == "fit" and r["word_id"] == query["word_id"]]
        assert len(refs) == 3 and len(set(refs)) == 3
        pairs = development["pairs"]
        assert sorted(p["reference"] for p in pairs) == sorted(refs)
        by_reference = {}
        mapping = {}
        for pair in pairs:
            ref = pair["reference"]
            groups = collections.defaultdict(list)
            path = pair["path_original_frames"]
            assert path[0] == [0, 0]
            assert path[-1] == [count - 1, auth[ref]["frames"] - 1]
            for k, (qf, rf) in enumerate(path):
                assert isinstance(qf, int) and isinstance(rf, int)
                assert 0 <= qf < count and 0 <= rf < auth[ref]["frames"]
                if k:
                    dq = qf - path[k-1][0]
                    dr = rf - path[k-1][1]
                    assert (dq, dr) in {(0, 1), (1, 0), (1, 1)}
                groups[qf].append(rf)
            assert set(groups) == set(range(count))
            assert {rf for _, rf in path} == set(range(auth[ref]["frames"]))
            assert pair["reference_frame_ids"] == list(range(auth[ref]["frames"]))
            by_reference[ref] = [transfer(phase_clips[ref]["segments"], groups[f]) for f in range(count)]
            mapping[ref] = [groups[f] for f in range(count)]
        predicted = [consensus([(r, by_reference[r][f]) for r in refs], refs) for f in range(count)]
        draft = role_frames(phase_clips[alias]["segments"], count)
        result[alias] = {
            "references": refs,
            "per_reference": by_reference,
            "mapped_reference_frames": mapping,
            "consensus": predicted,
            "dev_draft_roles": draft,
            "all_native_frames": count,
            "known_projection": sum(v is not None for v in predicted),
            "unknown_projection": sum(v is None for v in predicted),
            "known_dev_draft": sum(v is not None for v in draft),
            "unknown_dev_draft": sum(v is None for v in draft),
            "known_draft_match": sum(t is not None and t == p for t, p in zip(draft, predicted)),
            "known_draft_predicted_unknown": sum(t is not None and p is None for t, p in zip(draft, predicted)),
            "known_draft_wrong_role": sum(t is not None and p is not None and t != p for t, p in zip(draft, predicted)),
            "per_draft_role": {
                role: {
                    "draft_frames": sum(t == role for t in draft),
                    "match": sum(t == role and p == role for t, p in zip(draft, predicted)),
                    "projected_unknown": sum(t == role and p is None for t, p in zip(draft, predicted)),
                    "wrong_known_role": sum(t == role and p is not None and p != role for t, p in zip(draft, predicted)),
                } for role in sorted(KNOWN)
            },
        }
    return result


def synthetic_controls():
    a = {"segment_id": "a", "role": "core_candidate", "core": [2, 5], "start_bracket": [0, 2], "end_bracket": [5, 7]}
    b = {"segment_id": "b", "role": "core_candidate", "core": [7, 9]}
    c = {"segment_id": "c", "role": "rest", "core": [11, 13]}
    u = {"segment_id": "u", "role": "uncertain", "core": [15, 17]}
    segments = [a, b, c, u]
    cases = [
        ("empty_mapping", transfer(segments, []), None),
        ("inside_one_core", transfer(segments, [3, 4]), "core_candidate"),
        ("inclusive_core_endpoints", transfer(segments, [2, 5]), "core_candidate"),
        ("bracket_outside_core", transfer(segments, [1, 2]), None),
        ("gap_link_must_not_disappear", transfer(segments, [5, 6, 7]), None),
        ("same_role_different_core", transfer(segments, [5, 7]), None),
        ("different_roles", transfer(segments, [9, 11]), None),
        ("uncertain_not_known", transfer(segments, [15]), None),
        ("known_plus_uncertain", transfer(segments, [13, 15]), None),
        ("overlap_ambiguous", transfer([a, dict(b, core=[4, 9])], [4]), None),
        ("out_of_scope", transfer(segments, [99]), None),
        ("three_unique_agree", consensus([("x", "rest"), ("y", "rest"), ("z", "rest")], ["x", "y", "z"]), "rest"),
        ("two_known_one_unknown", consensus([("x", "rest"), ("y", "rest"), ("z", None)], ["x", "y", "z"]), None),
        ("three_unknown", consensus([("x", None), ("y", None), ("z", None)], ["x", "y", "z"]), None),
        ("two_of_three_majority_not_enough", consensus([("x", "rest"), ("y", "rest"), ("z", "return")], ["x", "y", "z"]), None),
        ("duplicate_reference", consensus([("x", "rest"), ("x", "rest"), ("z", "rest")], ["x", "y", "z"]), None),
        ("missing_reference", consensus([("x", "rest"), ("y", "rest")], ["x", "y", "z"]), None),
        ("wrong_reference_identity", consensus([("x", "rest"), ("y", "rest"), ("w", "rest")], ["x", "y", "z"]), None),
    ]
    return [{"name": name, "actual": actual, "expected": expected, "pass": actual == expected} for name, actual, expected in cases]


if __name__ == "__main__":
    # A separate command supplies comparison against the implementation's final
    # saved schema; this preliminary oracle never rewrites labels or results.
    tests = synthetic_controls()
    assert all(t["pass"] for t in tests)
    fit = read(UNIT / "PHASE_FIT.json")
    dev = read(UNIT / "PHASE_DEV.json")
    clips = {c["alias"]: c for c in fit["clips"] + dev["clips"]}
    oracle = independent_projection(clips, read(UNIT / "AUTHORIZATION.json")["rows"], read(OLD / "EMPIRICAL_RANGE.json")["development"])
    output = {
        "status": "INDEPENDENT_ORACLE_READY_NOT_YET_COMPARED_WITH_ROOT_OUTPUT",
        "synthetic_controls": tests,
        "clips": oracle,
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in [UNIT / "PHASE_FIT.json", UNIT / "PHASE_DEV.json", UNIT / "AUTHORIZATION.json", OLD / "EMPIRICAL_RANGE.json"]],
        "new_model_inference": False,
        "new_training": False,
        "old_results_modified": False,
    }
    path = UNIT / "review" / "PHASE_INDEPENDENT_ORACLE.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(path), "controls": len(tests), "clips": len(oracle)}, ensure_ascii=False))
