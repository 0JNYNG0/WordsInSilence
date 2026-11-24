"""Read-only compatibility check for grouping saved query evidence by seed.

Reads only the pinned existing 108 JSON gzip readouts. Does not import model,
alignment, readout, or scoring code, and does not aggregate any residuals.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN = HERE / "EVALUATION_PLAN.json"
plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
pins = {p: h for p, h in plan["pins"].items()
        if Path(p).parent.name == "readout_replay" and p.endswith(".json.gz")}
assert len(pins) == 108
groups = defaultdict(list)
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


for name, digest in pins.items():
    path = Path(name)
    raw = path.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == digest, f"source SHA: {path.name}")
    row = json.loads(gzip.decompress(raw))
    check(row["query"] != row["reference"], f"self reference: {path.name}")
    groups[row["query"], row["seed"]].append((path, row))

check(len(groups) == 48, "exact 48 query/seed groups")
inventory = []
axis_identity_fields = ["frame_id", "timestamp_us", "dimension_labels",
                        "possible_shape", "own_observed"]
for (query, seed), records in sorted(groups.items()):
    records.sort(key=lambda pair: pair[1]["reference"])
    base = records[0][1]
    base_groups = base["evidence"]["groups"]
    refs = [row["reference"] for _, row in records]
    check(len(refs) == len(set(refs)), f"unique references: {query}/{seed}")
    check(len(refs) in (2, 3), f"complete reference count: {query}/{seed}")
    n_frames = len(base["evidence"]["query_frame_ids"])
    dimensions = 0
    cells = 0
    for name, group in base_groups.items():
        axis = group["query"]
        n, d = axis["possible_shape"]
        dimensions += d
        cells += n * d
        check(n == n_frames, f"query frame count: {query}/{seed}/{name}")
        check(axis["frame_id"] == base["evidence"]["query_frame_ids"],
              f"top-level frame IDs: {query}/{seed}/{name}")
        check(len(axis["dimension_labels"]) == d, f"dimension count: {query}/{seed}/{name}")
        check(len(axis["own_observed"]) == n and all(len(x) == d for x in axis["own_observed"]),
              f"own-mask shape: {query}/{seed}/{name}")
    check(dimensions == 70 and len(base_groups) == 16,
          f"complete feature inventory: {query}/{seed}")
    for path, other in records[1:]:
        og = other["evidence"]["groups"]
        check(set(base_groups) == set(og), f"feature groups: {path.name}")
        check(other["evidence"]["query_frame_ids"] == base["evidence"]["query_frame_ids"],
              f"top-level query axis: {path.name}")
        for name, group in base_groups.items():
            for field in axis_identity_fields:
                check(group["query"][field] == og[name]["query"][field],
                      f"query identity {path.name}/{name}/{field}")
    inventory.append({"query": query, "seed": seed, "references": refs,
                      "frames": n_frames, "dimensions": dimensions,
                      "unique_query_cells": cells,
                      "sources": [str(path) for path, _ in records]})

distribution = Counter(len(x["references"]) for x in inventory)
check(dict(distribution) == {2: 36, 3: 12}, "36 two-reference and12 three-reference groups")
check(sum(x["frames"] for x in inventory) == 6252, "native query frames times3seeds")
check(sum(x["unique_query_cells"] for x in inventory) == 437640, "unique query opportunities")
out = {
    "schema": "wis_saved_readout_query_group_compatibility_v1",
    "status": "PASS",
    "at_utc": datetime.now(timezone.utc).isoformat(),
    "reviewer": "Codex semantic_data_audit",
    "checks": checks,
    "source_records": len(pins),
    "query_seed_groups": len(groups),
    "two_reference_groups": distribution[2],
    "three_reference_groups": distribution[3],
    "unique_query_native_frames_times_seeds": 6252,
    "unique_query_opportunities": 437640,
    "pairwise_both_axis_opportunities_preserved_separately": 1926120,
    "identical_across_references": axis_identity_fields,
    "claim": "Existing query axes and own masks are compatible for listing reference-specific evidence; no pooling or new observation is implied.",
    "limits": "Uses saved readout masks already checked against pinned inputs by READOUT_ONLY_VERIFICATION. Does not newly inspect raw input masks or validate linguistic correctness. No residuals or coordinates averaged, no reference selected.",
    "inventory": inventory,
    "pins": pins,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "actions": {"new_alignment": 0, "model_inference": 0, "training": 0,
                "scoring": 0, "raw_or_private_media_access": 0, "old_artifact_changes": 0},
}
destination = HERE / "SOURCE_GROUP_COMPATIBILITY.json"
destination.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"status": out["status"], "checks": checks,
                  "query_seed_groups": len(groups), "unique_query_opportunities": 437640,
                  "output": str(destination),
                  "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))
