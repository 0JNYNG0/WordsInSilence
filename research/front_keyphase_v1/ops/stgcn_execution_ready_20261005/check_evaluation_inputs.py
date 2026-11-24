"""Independent post-evaluation input audit; never loads or scores a model.

Requires completed evaluation artifacts. Audits all row metadata and directly
reconstructs every public condition, every masked condition, and one otherwise
uncovered group representative. No production input/normalization helpers.
"""
from pathlib import Path
from collections import Counter
import hashlib
import io
import json

import numpy as np
import torch

U = Path(__file__).resolve().parent
RUN = U.parent.parent / "runs/stgcn_full_path_v1"
GROUPS = {"annotation_dev": 699, "whole_dev": 74, "public": 32, "mp": 14, "ksl36": 36}
HAND = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
LAYOUTS = {
    "native67": ("xy_native", "coordinate_present", "timestamp_nominal_us", [0, 16, 15, 5, 2, 6, 3] + [25 + j for j in HAND] + [46 + j for j in HAND], {"left": (25, 46), "right": (46, 67)}),
    "mp75": ("xy_px", "observed", "timestamp_us", [0, 2, 5, 11, 12, 13, 14] + [33 + j for j in HAND] + [54 + j for j in HAND], {"left": (33, 54), "right": (54, 75)}),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_sha(tensor):
    array = np.ascontiguousarray(tensor.numpy())
    digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def reconstruct(row, arrays):
    xy_key, mask_key, _, joint_indices, hands = LAYOUTS[row["layout"]]
    start, end = row["start"], row["end_exclusive"]
    xy = arrays[xy_key][start:end].astype(np.float32, copy=True)
    mask = arrays[mask_key][start:end].astype(bool, copy=True)
    if row["mask_hand"] is not None:
        lo, hi = hands[row["mask_hand"]]
        mask[:, lo:hi] = False
    assert np.isfinite(xy[mask]).all()
    xy[~mask] = 0
    xy = xy[:, joint_indices]
    # Preserve CTV memory order and the official arithmetic order so the
    # independently computed result can be checked against the recorded SHA.
    ctv = torch.from_numpy(np.ascontiguousarray(xy.transpose(2, 0, 1)))
    tvc = ctv.permute(1, 2, 0)
    vtc = tvc.permute(1, 0, 2)
    left, right = vtc[3].reshape(-1, 2), vtc[4].reshape(-1, 2)
    center = torch.mean((left + right) / 2, dim=0)
    distance = torch.mean(torch.sqrt(((left - right) ** 2).sum(-1)))
    assert bool(torch.isfinite(distance)) and float(distance) > 0
    normalized = ((tvc - center) * (1.0 / distance)).permute(2, 0, 1)
    frames = len(xy)
    if frames <= 60:
        packed = torch.cat((normalized.new_zeros((2, 60 - frames, 27)), normalized), dim=1)
    else:
        indices = [j * (frames - 1) // 59 for j in range(60)]
        packed = normalized[:, indices]
    sample = packed.reshape(2, 6, 10, 27).permute(1, 0, 2, 3).contiguous()
    distance64 = float(np.linalg.norm(xy[:, 3].astype(np.float64) - xy[:, 4].astype(np.float64), axis=1).mean())
    return sample, distance64, int((~mask[:, joint_indices]).sum())


def main():
    # Fail before reading any final source material if evaluation is incomplete.
    required = [RUN / name for name in ("EVALUATION_ROWS.json", "EVALUATION_INPUT_DIAGNOSTICS.json", "EVALUATION_EXECUTION.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Evaluation outputs are not complete; no source data read: " + ", ".join(missing))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = read_json(U / "FROZEN_EXPERIMENT.json")
    context = read_json(U / "EXECUTION_CONTEXT.json")
    manifest_path = U.parent / "stgcn_transfer_compare_20261005/INPUT_MANIFEST.json"
    assert sha(manifest_path.read_bytes()) == plan["pins"]["input_manifest"]
    manifest = read_json(manifest_path)
    evaluation = read_json(required[0])
    diagnostics = read_json(required[1])
    execution = read_json(required[2])
    rows = evaluation["rows"]
    expected = [r for r in manifest["rows"] if r["group"] in GROUPS]
    assert rows == expected and len(rows) == len(diagnostics) == 855
    assert evaluation["classes"] == manifest["classes"] == context["classes"] and len(evaluation["classes"]) == 93
    assert dict(Counter(r["group"] for r in rows)) == GROUPS
    assert execution["rows"] == 855 and execution["classes"] == 93 and execution["model_state_unchanged"]
    assert execution["no_training_or_selection"] and not execution["user_data_used"]
    assert execution["frozen_model_sha256"] == sha((RUN / "FROZEN_MODEL.json").read_bytes())
    assert execution["final_model_sha256"] == read_json(RUN / "FROZEN_MODEL.json")["final_model_sha256"]
    for row, diag in zip(rows, diagnostics):
        assert row["role"] != "train" and "/runs/ses_" not in row["path"].replace("\\", "/")
        assert evaluation["classes"][row["y"]] == row["label"]
        assert (diag["id"], diag["group"], diag["condition"], diag["layout"]) == (row["id"], row["group"], row["condition"], row["layout"])
        assert diag["source_sha256"] == row["sha256"] and diag["slice"] == [row["start"], row["end_exclusive"]]
        assert diag["mask_hand"] == row["mask_hand"]
        expected_hand = list(LAYOUTS[row["layout"]][4][row["mask_hand"]]) if row["mask_hand"] is not None else None
        assert diag["hand_slice"] == expected_hand
        frames = row["end_exclusive"] - row["start"]
        indices = [-1] * (60 - frames) + list(range(frames)) if frames <= 60 else [j * (frames - 1) // 59 for j in range(60)]
        assert diag["prepare_diag"]["T"] == frames and diag["prepare_diag"]["indices"] == indices
        assert diag["prepare_diag"]["shape"] == [1, 6, 2, 10, 27]
        assert len(diag["input_tensor_sha256"]) == 64

    selected = {i for i, r in enumerate(rows) if r["group"] == "public" or r["mask_hand"] is not None}
    for group in GROUPS:
        selected.add(next(i for i, r in enumerate(rows) if r["group"] == group))
    sources, checks = {}, []
    for i in sorted(selected):
        row, diag = rows[i], diagnostics[i]
        path = Path(row["path"])
        if str(path) not in sources:
            before = path.stat()
            raw = path.read_bytes()
            assert sha(raw) == row["sha256"]
            xy_key, mask_key, time_key, _, _ = LAYOUTS[row["layout"]]
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key]) for key in (xy_key, mask_key, time_key)}
            times = arrays[time_key].astype(np.float64)
            assert np.isfinite(times).all() and (np.diff(times) > 0).all()
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            sources[str(path)] = (sha(raw), arrays)
        source_sha, arrays = sources[str(path)]
        assert source_sha == row["sha256"]
        sample, distance, missing_points = reconstruct(row, arrays)
        actual_sha = tensor_sha(sample)
        assert actual_sha == diag["input_tensor_sha256"], f"row {i}: independently reconstructed tensor SHA differs"
        assert distance == diag["prepare_diag"]["shoulder_distance"]
        checks.append({"row": i, "group": row["group"], "id": row["id"], "condition": row["condition"], "mask_hand": row["mask_hand"], "slice": diag["slice"], "source_sha256": source_sha, "tensor_sha256": actual_sha, "selected_missing_points": missing_points})

    report = {"status": "PASS", "mode": "post_evaluation_input_audit_no_model_or_scores", "all_metadata_rows": 855, "public_conditions_reconstructed": sum(r["group"] == "public" for r in checks), "all_masked_conditions_reconstructed": sum(r["mask_hand"] is not None for r in checks), "representative_conditions": len(checks), "unique_source_npz_read": len(sources), "manual_tensor_sha_exact": True, "checks": checks, "artifact_sha256": {p.name: sha(p.read_bytes()) for p in required}, "verifier_sha256": sha(Path(__file__).read_bytes()), "limitation": "Raw source content independently reconstructed only for selected conditions; other rows have manifest and diagnostic checks. No EVALUATION_SCORES, model state arrays, training or inference were read or run."}
    (U / "EVALUATION_INPUT_INDEPENDENT_CHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "artifact_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
