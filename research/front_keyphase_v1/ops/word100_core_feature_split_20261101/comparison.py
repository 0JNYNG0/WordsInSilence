"""Research-only joint versus protected-shape/separate-motion adapter.

No training CLI, data extraction, downloading, or previous-run inference. The
only CLI action is a synthetic interface/freeze self-test. Existing protected
predictions are read verbatim; new-clip frozen inference requires an explicit
call after the caller's readiness gate. This is a system comparison, not a
capacity- and pretraining-matched causal test of architectural separation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import torch
from torch import nn

U = Path(__file__).resolve().parent
OPS = U.parent
OLD = OPS / "word100_stage_feedback_cycle_20261031"
VISIBLE = OPS / "word100_visible_temporal_targets_20261101"
SEEDS = (42, 43, 44)
SHAPE_TARGETS = ("R_index_only", "R_open_four", "L_open_four")
DEFAULT_TARGETS = SHAPE_TARGETS + (
    "R_higher", "L_higher", "R_motion_down", "R_motion_still",
    "R_motion_mixed", "L_motion_down", "L_motion_still",
)
SUPPORTED_NEW_TARGETS = {"R_higher", "L_higher"} | {
    f"{hand}_motion_{kind}" for hand in "RL"
    for kind in ("up", "down", "horizontal", "still", "mixed")
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_seed(seed):
    if seed not in SEEDS:
        raise ValueError(f"Only prespecified seeds {SEEDS} are supported")


def target_mapping(targets):
    names = tuple(targets)
    if len(set(names)) != len(names) or not set(SHAPE_TARGETS).issubset(names):
        raise ValueError("Unique targets including all three protected shapes required")
    if set(names) - set(SHAPE_TARGETS) - SUPPORTED_NEW_TARGETS:
        raise ValueError("Unsupported shape/relation target; do not silently remap")
    shape = tuple(names.index(k) for k in SHAPE_TARGETS)
    motion = tuple(i for i, k in enumerate(names) if k not in SHAPE_TARGETS)
    if not motion:
        raise ValueError("A separate motion/relation target is required")
    return names, shape, motion


class OriginalFrameHead(nn.Module):
    """Exact stage_feedback_cycle Head forward and state_dict interface."""
    def __init__(self, targets=10):
        super().__init__()
        self.pose = nn.Linear(138, 32)
        self.aux = nn.Linear(768, 32)
        self.c1 = nn.Conv1d(64, 64, 3, padding=1)
        self.c2 = nn.Conv1d(64, 32, 3, padding=1)
        self.out = nn.Linear(32, targets)

    def forward(self, x, z, valid):
        h = torch.cat([nn.functional.gelu(self.pose(x)),
                       nn.functional.gelu(self.aux(z))], -1) * valid[:, :, None]
        h = nn.functional.gelu(self.c1(h.transpose(1, 2))) * valid[:, None, :]
        h = nn.functional.gelu(self.c2(h)) * valid[:, None, :]
        return self.out(h.transpose(1, 2))


class FrozenOriginalHead(OriginalFrameHead):
    def __init__(self, state):
        super().__init__(10)
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Parent split.train() must not put the protected branch in train mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, x, z, valid):
        return super().forward(x, z, valid)


def _contract_pin(path):
    """Once the contract exists, reject a changed original artifact."""
    contract_path = U / "review/MODEL_CONTRACT.json"
    if contract_path.exists():
        pins = read(contract_path)["source_pins"]
        expected = pins.get(str(Path(path)))
        if expected is None or sha(path) != expected:
            raise ValueError(f"Unpinned or changed protected artifact: {path}")


def load_frozen(seed):
    check_seed(seed)
    directory = OLD / "runs" / f"pose_plus_signrep_{seed}"
    manifest_path, checkpoint = directory / "RUN.json", directory / "head.pt"
    for p in (manifest_path, checkpoint, OLD / "NORMALIZATION.json",
              OLD / "LEARNING_INPUTS.json", OLD / "learning.py"):
        _contract_pin(p)
    manifest = read(manifest_path)
    if (manifest["seed"] != seed or manifest["condition"] != "pose_plus_signrep"
            or sha(checkpoint) != manifest["checkpoint_sha256"]):
        raise ValueError("Original checkpoint identity mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return FrozenOriginalHead(state)


class WindowHead(nn.Module):
    """Same hidden architecture as the completed 13-frame study, new weights."""
    def __init__(self, targets):
        super().__init__()
        self.pose = nn.Linear(138, 32)
        self.aux = nn.Linear(768, 32)
        self.c1 = nn.Conv1d(64, 48, 3, padding=1)
        self.c2 = nn.Conv1d(48, 32, 3, padding=2, dilation=2)
        self.out = nn.Linear(128, targets)

    def forward(self, x, z):
        if x.shape[1:] != (13, 138) or z.shape != (len(x), 13, 768):
            raise ValueError("Expected aligned [N,13,138] and [N,13,768]")
        h = torch.cat([nn.functional.gelu(self.pose(x)),
                       nn.functional.gelu(self.aux(z))], -1)
        h = nn.functional.gelu(self.c1(h.transpose(1, 2)))
        h = nn.functional.gelu(self.c2(h)).transpose(1, 2)
        summary = torch.cat([h[:, :4].mean(1), h[:, 4:9].mean(1),
                             h[:, 9:].mean(1), h[:, 6]], -1)
        return self.out(summary)


@dataclass
class PreparedBatch:
    x: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    old_x: torch.Tensor
    old_z: torch.Tensor
    old_valid: torch.Tensor
    clip_index: torch.Tensor
    center_index: torch.Tensor
    targets: tuple
    rows: tuple
    clips: tuple

    def subset(self, indices):
        ix = torch.as_tensor(indices, dtype=torch.long)
        return replace(self, x=self.x[ix], z=self.z[ix], y=self.y[ix],
                       mask=self.mask[ix], clip_index=self.clip_index[ix],
                       center_index=self.center_index[ix],
                       rows=tuple(self.rows[int(i)] for i in ix))


def _validate_raw(a):
    n = len(a["frame_id"])
    if (a["x"].shape != (n, 69) or a["mask"].shape != (n, 69)
            or a["rgb"].shape != (n, 768) or a["timestamp_us"].shape != (n,)
            or a["mask"].dtype != np.dtype(bool) or n < 13):
        raise ValueError("Raw feature schema mismatch")
    if (not np.all(np.diff(a["frame_id"]) > 0)
            or not np.all(np.diff(a["timestamp_us"]) > 0)
            or not np.isfinite(a["x"][a["mask"]]).all()
            or not np.isfinite(a["rgb"]).all()):
        raise ValueError("Invalid time order or observed feature; do not repair silently")


def _indices(row, a):
    lookup = {int(f): i for i, f in enumerate(a["frame_id"])}
    frame_ids = row["frame_ids"]
    if len(frame_ids) != 13 or any(b - a != 1 for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError("Exactly 13 consecutive original frame IDs required")
    try:
        ix = [lookup[int(f)] for f in frame_ids]
    except KeyError as exc:
        raise ValueError("Requested original frame is absent; nearest-frame fallback forbidden") from exc
    if (ix != list(range(ix[0], ix[0] + 13))
            or row["center_frame"] != frame_ids[6]
            or a["timestamp_us"][ix].tolist() != row["timestamps_us"]
            or int(a["timestamp_us"][ix[6]]) != row["center_timestamp_us"]):
        raise ValueError("Frame/time identity mismatch")
    return ix


def _normalize(a, stats):
    pm, ps = np.asarray(stats["pose_mean"]), np.asarray(stats["pose_std"])
    zm, zs = np.asarray(stats["rgb_mean"]), np.asarray(stats["rgb_std"])
    if (pm.shape != (69,) or ps.shape != (69,) or zm.shape != (768,)
            or zs.shape != (768,) or not all(np.isfinite(v).all() for v in (pm, ps, zm, zs))
            or not (ps > 0).all() or not (zs > 0).all()):
        raise ValueError("Normalization schema mismatch")
    pose = np.where(a["mask"], (a["x"] - pm) / ps, 0)
    return (np.concatenate([pose, a["mask"]], -1).astype(np.float32),
            ((a["rgb"] - zm) / zs).astype(np.float32))


def assemble(info, stats=None):
    """Prepare common new windows and exact old full clips, without inference.

    New-branch normalization is fit-only. The old branch always uses its original
    frozen full-clip fit normalization. Their raw features/time keys are shared;
    normalized tensors are deliberately not claimed identical across branches.
    """
    names, _, _ = target_mapping(info["selected_targets"])
    ids = [info["targets"].index(k) for k in names]
    if "selected_indices" in info and info["selected_indices"] != ids:
        raise ValueError("Selected target name/index mismatch")
    rows = tuple(info["rows"])
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    raw, clips, by_clip, window_raw, clip_ix, centers = [], [], {}, [], [], []
    old_info = read(OLD / "LEARNING_INPUTS.json")
    feature_names = old_info["rows"][0]["feature_names"]
    known_input_hashes = {r["input_sha256"] for r in old_info["rows"]}
    for row in rows:
        declared_features = row.get("feature_names", info.get("feature_names"))
        if declared_features is None and row["input_sha256"] not in known_input_hashes:
            raise ValueError("New input must declare the 69 feature_names in its row or dataset")
        if declared_features is not None and declared_features != feature_names:
            raise ValueError("Feature order differs from original 69-feature interface")
        clip_id = row["clip_id"]
        if clip_id not in by_clip:
            p = Path(row["input"])
            if sha(p) != row["input_sha256"]:
                raise ValueError(f"Input SHA mismatch: {clip_id}")
            with np.load(p, allow_pickle=False) as archive:
                a = {k: archive[k] for k in ("x", "mask", "rgb", "frame_id", "timestamp_us")}
            _validate_raw(a)
            by_clip[clip_id] = len(raw)
            raw.append(a)
            clips.append({k: row[k] for k in ("clip_id", "input", "input_sha256")})
        c = by_clip[clip_id]
        if row["input_sha256"] != clips[c]["input_sha256"]:
            raise ValueError("One clip identity has multiple input arrays")
        a = raw[c]
        ix = _indices(row, a)
        window_raw.append({k: a[k][ix] for k in ("x", "mask", "rgb")})
        clip_ix.append(c)
        centers.append(ix[6])
    if stats is None:
        fit = [a for a, r in zip(window_raw, rows) if r["role"] == "fit"]
        if not fit:
            raise ValueError("Fit windows required to create normalization")
        xf = np.concatenate([a["x"] for a in fit])
        mf = np.concatenate([a["mask"] for a in fit])
        zf = np.concatenate([a["rgb"] for a in fit])
        pm = np.array([xf[mf[:, j], j].mean() if mf[:, j].any() else 0 for j in range(69)])
        ps = np.array([max(float(xf[mf[:, j], j].std()), .001) if mf[:, j].any() else 1 for j in range(69)])
        stats = dict(pose_mean=pm.tolist(), pose_std=ps.tolist(),
                     rgb_mean=zf.mean(0).tolist(), rgb_std=np.maximum(zf.std(0), .001).tolist(),
                     scope="new fit13-frame windows only; overlapping frame occurrences retained")
    old_stats_path = OLD / "NORMALIZATION.json"
    _contract_pin(old_stats_path)
    old_stats = read(old_stats_path)
    windows = [_normalize(a, stats) for a in window_raw]
    full = [_normalize(a, old_stats) for a in raw]
    max_t = max(len(a["x"]) for a in raw)
    ox = np.zeros((len(raw), max_t, 138), np.float32)
    oz = np.zeros((len(raw), max_t, 768), np.float32)
    valid = np.zeros((len(raw), max_t), np.float32)
    for i, (x, z) in enumerate(full):
        ox[i, :len(x)], oz[i, :len(x)], valid[i, :len(x)] = x, z, 1
    y = np.asarray([r["y"] for r in rows], np.float32)[:, ids]
    mask = np.asarray([r["target_mask"] for r in rows], bool)[:, ids]
    if not np.isin(y[mask], [0, 1]).all():
        raise ValueError("Known labels must be binary")
    tensor = torch.from_numpy
    batch = PreparedBatch(tensor(np.stack([a[0] for a in windows])),
                          tensor(np.stack([a[1] for a in windows])), tensor(y), tensor(mask),
                          tensor(ox), tensor(oz), tensor(valid),
                          torch.tensor(clip_ix), torch.tensor(centers), names, rows, tuple(clips))
    return batch, stats


class JointModel(nn.Module):
    def __init__(self, targets):
        super().__init__()
        self.targets, _, _ = target_mapping(targets)
        self.trainable_target_indices = tuple(range(len(self.targets)))
        self.head = WindowHead(len(self.targets))

    def forward(self, batch):
        return self.head(batch.x, batch.z)

    def optimizer_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


class SplitModel(nn.Module):
    def __init__(self, targets, frozen):
        super().__init__()
        self.targets, self.shape_indices, self.trainable_target_indices = target_mapping(targets)
        self.frozen = frozen
        self.head = WindowHead(len(self.trainable_target_indices))

    def forward(self, batch):
        # Deliberately never runs the old branch, including during training.
        return self.head(batch.x, batch.z)

    def optimizer_parameters(self):
        return (p for p in self.head.parameters() if p.requires_grad)

    def combine_probabilities(self, motion_probabilities, protected_probabilities):
        if (motion_probabilities.shape[-1] != len(self.trainable_target_indices)
                or protected_probabilities.shape != (len(motion_probabilities), 3)):
            raise ValueError("Prediction columns do not match the frozen target mapping")
        if not torch.isfinite(protected_probabilities).all():
            raise ValueError("Protected predictions incomplete; do not fill unknowns")
        out = motion_probabilities.new_empty((len(motion_probabilities), len(self.targets)))
        out[:, self.trainable_target_indices] = motion_probabilities
        out[:, self.shape_indices] = protected_probabilities
        return out


def build_pair(seed, targets=DEFAULT_TARGETS):
    """Identical shared-trunk and motion-output initialization for each pair."""
    check_seed(seed)
    torch.manual_seed(seed)
    joint = JointModel(targets)
    split = SplitModel(targets, load_frozen(seed))
    copied = joint.head.state_dict()
    for key in ("out.weight", "out.bias"):
        copied[key] = copied[key][list(split.trainable_target_indices)].clone()
    split.head.load_state_dict(copied, strict=True)
    return joint, split


def stored_old_shape_probabilities(info, seed):
    """Read original values verbatim; missing new clips stay NaN, never inferred."""
    check_seed(seed)
    old_info_path = OLD / "LEARNING_INPUTS.json"
    _contract_pin(old_info_path)
    prior = read(old_info_path)
    old_rows = {r["clip_id"]: (i, r) for i, r in enumerate(prior["rows"])}
    cols = [prior["targets"].index(k) for k in SHAPE_TARGETS]
    directory = OLD / "runs" / f"pose_plus_signrep_{seed}"
    predictions, manifest = directory / "probabilities.npy", directory / "RUN.json"
    for p in (predictions, manifest):
        _contract_pin(p)
    if sha(predictions) != read(manifest)["prediction_sha256"]:
        raise ValueError("Original stored predictions changed")
    p = np.load(predictions, allow_pickle=False)
    if p.ndim != 3 or p.shape[0] != len(prior["rows"]) or p.shape[-1] != len(prior["targets"]):
        raise ValueError("Original prediction schema mismatch")
    output = np.full((len(info["rows"]), 3), np.nan, dtype=p.dtype)
    missing, cache = [], {}
    for i, row in enumerate(info["rows"]):
        if row["clip_id"] not in old_rows:
            missing.append(i)
            continue
        c, previous = old_rows[row["clip_id"]]
        if row["input_sha256"] != previous["input_sha256"]:
            raise ValueError("Known clip has changed input; protected inference rerun forbidden")
        if c not in cache:
            if sha(previous["input"]) != previous["input_sha256"]:
                raise ValueError("Original input file changed")
            with np.load(previous["input"], allow_pickle=False) as archive:
                cache[c] = {k: archive[k] for k in ("frame_id", "timestamp_us")}
        center = _indices(row, cache[c])[6]
        output[i] = p[c, center, cols]
    return output, missing


@torch.no_grad()
def complete_protected_probabilities(batch, seed, *, allow_new_inference=False, frozen=None):
    """Caller may cache this once per seed, after readiness. Old clips NEVER infer.

    This function neither writes a cache nor authorizes training. The caller must
    save its returned values, source SHA, and audit once, then pass the cached
    values to combine_probabilities. No per-epoch invocation is needed or wanted.
    """
    probabilities, missing = stored_old_shape_probabilities({"rows": batch.rows}, seed)
    if missing and not allow_new_inference:
        raise RuntimeError("New clip inference requires the caller's readiness gate")
    new_clips = sorted({int(batch.clip_index[i]) for i in missing})
    if new_clips:
        h = frozen if frozen is not None else load_frozen(seed)
        if h.training or any(p.requires_grad for p in h.parameters()):
            raise ValueError("Protected head is not frozen")
        prior = read(OLD / "LEARNING_INPUTS.json")
        cols = [prior["targets"].index(k) for k in SHAPE_TARGETS]
        # Only new clips are evaluated; no old full-clip batch slips through.
        logits = h(batch.old_x[new_clips], batch.old_z[new_clips], batch.old_valid[new_clips])
        p = logits.sigmoid().cpu().numpy()
        local = {c: i for i, c in enumerate(new_clips)}
        for i in missing:
            probabilities[i] = p[local[int(batch.clip_index[i])], int(batch.center_index[i]), cols]
    if not np.isfinite(probabilities).all():
        raise ValueError("Protected probability cache incomplete")
    return torch.from_numpy(probabilities), dict(
        seed=seed, old_rows_reused=len(batch.rows) - len(missing),
        new_rows_inferred=len(missing), new_clips_inferred=len(new_clips),
        previous_clip_inference_count=0,
        original_probability_sha256=sha(OLD / "runs" / f"pose_plus_signrep_{seed}" / "probabilities.npy"),
        checkpoint_sha256=sha(OLD / "runs" / f"pose_plus_signrep_{seed}" / "head.pt"),
    )


def masked_loss(logits, y, mask, class_weights, event_weights):
    safe = torch.where(mask, y, torch.zeros_like(y))
    raw = nn.functional.binary_cross_entropy_with_logits(logits, safe, reduction="none")
    w = torch.where(safe > .5, class_weights[1], class_weights[0]) * event_weights
    return torch.where(mask, raw * w, torch.zeros_like(raw)).sum() / event_weights[mask].sum().clamp_min(1)


def training_weights(batch, fit_indices, target_indices):
    """Same per-clip/target exposure rule, restricted to an arm's learned targets."""
    ids, fit = list(target_indices), list(fit_indices)
    if not fit or any(batch.rows[i]["role"] != "fit" for i in fit):
        raise ValueError("Training weights may use fit rows only")
    y, mask = batch.y[:, ids], batch.mask[:, ids]
    counts = torch.stack([((y[fit] == c) & mask[fit]).sum(0) for c in (0, 1)]).float()
    if not (counts > 0).all():
        raise ValueError("Both classes required; support thresholds belong to the readiness gate")
    weights = counts.sum(0)[None] / (2 * counts)
    event = torch.zeros_like(y)
    for clip in {batch.rows[i]["clip_id"] for i in fit}:
        ix = [i for i in fit if batch.rows[i]["clip_id"] == clip]
        event[ix] = 1 / mask[ix].sum(0).clamp_min(1)
    return weights, event


def _load_module(path, name):
    # Independent source comparison without creating files in frozen directories.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def selftest():
    """Tiny random tensors only. No real sample prediction, score, or timing run."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    canonical = _load_module(OLD / "learning.py", "original_stage_learning_contract")
    canonical_window = _load_module(VISIBLE / "head.py", "original_visible_head_contract")
    torch.manual_seed(20261101)
    n, t = 8, 19
    x, z = torch.randn(n, 13, 138), torch.randn(n, 13, 768)
    old_x, old_z = torch.randn(n, t, 138), torch.randn(n, t, 768)
    valid = torch.ones(n, t)
    valid[0, -4:] = 0
    old_x[0, -4:], old_z[0, -4:] = 0, 0
    y = torch.tensor([[float((i + j) % 2) for j in range(10)] for i in range(n)])
    mask = torch.ones(n, 10, dtype=torch.bool)
    mask[0, 0] = False
    rows = tuple(dict(sample_id=f"synthetic_{i}", clip_id=f"synthetic_{i // 2}", role="fit") for i in range(n))
    batch = PreparedBatch(x, z, y, mask, old_x, old_z, valid, torch.arange(n),
                          torch.full((n,), 6), DEFAULT_TARGETS, rows, tuple())
    checks, counts = [], {}
    for seed in SEEDS:
        joint, split = build_pair(seed)
        old_reference = canonical.Head(10)
        old_reference.load_state_dict(split.frozen.state_dict())
        new_reference = canonical_window.Head(10)
        new_reference.load_state_dict(joint.head.state_dict())
        with torch.no_grad():
            original_delta = (split.frozen(old_x, old_z, valid) - old_reference(old_x, old_z, valid)).abs().max().item()
            assert torch.allclose(split.frozen(old_x, old_z, valid), old_reference(old_x, old_z, valid), atol=2e-6, rtol=2e-5)
            assert torch.allclose(joint(batch), new_reference(x, z), atol=2e-6, rtol=2e-5)
            assert torch.allclose(split(batch), joint(batch)[:, list(split.trainable_target_indices)], atol=2e-6, rtol=2e-5)
        protected_before = {k: v.clone() for k, v in split.frozen.state_dict().items()}
        split.train()
        assert split.training and not split.frozen.training
        ids = list(split.trainable_target_indices)
        weights, event = training_weights(batch, range(n), ids)
        logits = split(batch)
        a = masked_loss(logits, y[:, ids], mask[:, ids], weights, event)
        changed = y[:, ids].clone()
        m = mask[:, ids].clone()
        m[0, 0] = False
        changed[~m] = float("nan")
        first = masked_loss(logits, y[:, ids], m, weights, event)
        second = masked_loss(logits, changed, m, weights, event)
        ga = torch.autograd.grad(first, tuple(split.head.parameters()), retain_graph=True)
        gb = torch.autograd.grad(second, tuple(split.head.parameters()), retain_graph=True)
        assert torch.equal(first, second) and all(torch.equal(s, q) for s, q in zip(ga, gb))
        opt = torch.optim.AdamW(split.optimizer_parameters(), lr=.001, weight_decay=.01)
        opt.zero_grad()
        a.backward()
        assert all(p.grad is None for p in split.frozen.parameters())
        opt.step()
        assert all(torch.equal(v, split.frozen.state_dict()[k]) for k, v in protected_before.items())
        protected = torch.rand(n, 3)
        combined = split.combine_probabilities(split(batch).sigmoid(), protected)
        assert torch.equal(combined[:, list(split.shape_indices)], protected)
        assert torch.allclose(joint(batch.subset([2, 0])), joint(batch)[[2, 0]], atol=2e-6, rtol=2e-5)
        counts = dict(joint_trainable=sum(p.numel() for p in joint.parameters() if p.requires_grad),
                      split_trainable=sum(p.numel() for p in split.parameters() if p.requires_grad),
                      split_frozen=sum(p.numel() for p in split.parameters() if not p.requires_grad),
                      split_total=sum(p.numel() for p in split.parameters()))
        checks.append(dict(seed=seed, original_synthetic_forward_allclose=True,
                           original_synthetic_max_abs_difference=original_delta,
                           new_synthetic_forward_allclose=True, initial_motion_output_allclose=True,
                           frozen_unchanged_after_synthetic_step=True,
                           protected_probability_routing_exact=True, unknown_gradient_invariant=True))
    a = dict(frame_id=np.arange(20, 39), timestamp_us=np.arange(19) * 33333)
    row = dict(frame_ids=list(range(23, 36)), center_frame=29,
               timestamps_us=a["timestamp_us"][3:16].tolist(), center_timestamp_us=int(a["timestamp_us"][9]))
    assert _indices(row, a) == list(range(3, 16))
    bad = dict(row, center_timestamp_us=row["center_timestamp_us"] + 1)
    try:
        _indices(bad, a)
    except ValueError:
        pass
    else:
        raise AssertionError("Timestamp mismatch accepted")
    return dict(status="PASS_SYNTHETIC_INTERFACE_AND_FREEZE_ONLY", seeds=list(SEEDS),
                checks=checks, parameters=counts, exact_frame_time_rejection=True,
                actual_clip_inference_count=0, actual_training_runs=0, previous_score_reruns=0,
                note="Three synthetic optimizer steps are interface tests, not research learning or evidence of improvement")


if __name__ == "__main__":
    if sys.argv[1:] != ["selftest"]:
        raise SystemExit("Usage: comparison.py selftest (no training CLI)")
    print(json.dumps(selftest(), ensure_ascii=False, indent=2))
