"""Head-only fine-tuning of an existing frozen CPU temporal model (research only).

This module performs a fixed-schedule, classifier-head-only optimisation pass on a
model that is otherwise completely frozen: every non-head parameter AND every
non-head buffer (batch-normalisation running statistics, tracking counters, and
anything else registered as a buffer) must be bit-identical before and after
training.  The encoder is executed in evaluation mode for the whole run, so no
normalisation statistic can drift and no dropout mask is applied.

Nothing here registers, promotes, exports or otherwise publishes a model.  All
file I/O, metadata preflight, checkpointing, evaluation and data loading remain
the responsibility of the unchanged host modules; this module only supplies the
training callable and two thin, self-restoring delegation wrappers.

Design notes:
  * `run_head_only_training` validates everything it can *before* touching the
    model, so a misconfigured run (in particular a nonzero encoder learning
    rate) can never leave a partially mutated module behind.
  * The freeze is *proved*, not asserted by construction: snapshots of the
    non-head state dict are compared tensor-by-tensor (dtype, shape, values)
    after the final optimiser step.  Encoder values are never restored, so a
    passing check means the encoder genuinely never moved.
  * The legacy augmentation / `nikl` branch is rejected outright.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

import tools_native_data_pair as base
import tools_prep_union93_candidate as u

# Captured at import time so that `main`'s temporary install of this module's
# `run` into `base.run` can never cause `run` to re-enter itself.
_ORIGINAL_BASE_RUN = base.run

TRAINING_SCOPE = "head_only_frozen_encoder_and_buffers"
HEAD_PREFIX = "head."
FEATURE_DIM = 280
LOG_EVERY = 60
TAIL_WINDOW = 20

REQUIRED_HEAD_ONLY_PLAN_KEYS = frozenset(
    {"encoder_lr", "freeze_encoder_buffers", "training_scope"}
)


# --------------------------------------------------------------------------- #
# small validation helpers
# --------------------------------------------------------------------------- #
def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value):
    return _is_number(value) and math.isfinite(float(value))


def _is_head_key(name):
    return name.startswith(HEAD_PREFIX)


def _tensors_identical(a, b):
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.dtype == b.dtype
        and tuple(a.shape) == tuple(b.shape)
        and bool(torch.equal(a, b))
    )


# --------------------------------------------------------------------------- #
# validation of the caller-supplied training inputs
# --------------------------------------------------------------------------- #
def _validate_classes(classes):
    _require(isinstance(classes, (list, tuple)) and len(classes) > 0,
             "classes must be a nonempty list")
    classes = list(classes)
    _require(len(set(classes)) == len(classes), "classes must be unique")
    return classes


def _validate_source_train(source_train, classes):
    _require(isinstance(source_train, (list, tuple)) and len(source_train) > 0,
             "source_train must be a nonempty list of sample dicts")
    seen_ids = set()
    for position, item in enumerate(source_train):
        where = "source_train[%d]" % position
        _require(isinstance(item, dict), "%s must be a dict" % where)

        sample_id = item.get("sample_id")
        _require(isinstance(sample_id, str) and sample_id != "",
                 "%s.sample_id must be a nonempty string" % where)
        _require(sample_id not in seen_ids,
                 "%s.sample_id duplicated: %r" % (where, sample_id))
        seen_ids.add(sample_id)

        label = item.get("label")
        _require(isinstance(label, str), "%s.label must be a string" % where)

        class_index = item.get("class_index")
        _require(isinstance(class_index, int) and not isinstance(class_index, bool),
                 "%s.class_index must be an int" % where)
        _require(0 <= class_index < len(classes),
                 "%s.class_index %d out of range for %d classes"
                 % (where, class_index, len(classes)))
        _require(classes[class_index] == label,
                 "%s.class_index %d maps to %r but label is %r"
                 % (where, class_index, classes[class_index], label))

        x = item.get("x")
        _require(isinstance(x, np.ndarray), "%s.x must be a NumPy array" % where)
        _require(x.dtype == np.float32, "%s.x must be float32, got %s" % (where, x.dtype))
        _require(x.ndim == 2, "%s.x must be 2-D [T,%d]" % (where, FEATURE_DIM))
        _require(x.shape[0] > 0, "%s.x must have T > 0" % where)
        _require(x.shape[1] == FEATURE_DIM,
                 "%s.x must have %d features, got %d" % (where, FEATURE_DIM, x.shape[1]))
        _require(bool(np.isfinite(x).all()), "%s.x contains non-finite values" % where)

    return list(source_train)


def _validate_nikl(nikl):
    if nikl is None:
        return
    _require(hasattr(nikl, "__len__"), "nikl must be an empty sized container or None")
    _require(len(nikl) == 0,
             "nikl must be empty: the legacy augmentation branch is disabled "
             "(%d items supplied)" % len(nikl))


def _validate_setting(setting, steps_override):
    """Validate the schedule.  Rejects a nonzero encoder lr before any mutation."""
    _require(isinstance(setting, dict), "setting must be a dict")

    lr = setting.get("lr")
    _require(isinstance(lr, dict), "setting['lr'] must be a dict")
    encoder_lr = lr.get("encoder")
    _require(_finite_number(encoder_lr), "setting['lr']['encoder'] must be a finite number")
    _require(float(encoder_lr) == 0.0,
             "setting['lr']['encoder'] must be exactly 0.0 for head-only training, got %r"
             % (encoder_lr,))
    head_lr = lr.get("head")
    _require(_finite_number(head_lr) and float(head_lr) > 0.0,
             "setting['lr']['head'] must be a positive finite number")

    _require(setting.get("augmentation") is False,
             "setting['augmentation'] must be exactly False")

    optimizer_steps = setting.get("optimizer_steps")
    _require(_pos_int(optimizer_steps),
             "setting['optimizer_steps'] must be a positive int (bool rejected)")
    batch_size = setting.get("batch_size")
    _require(_pos_int(batch_size),
             "setting['batch_size'] must be a positive int (bool rejected)")

    batch = setting.get("batch")
    _require(isinstance(batch, dict), "setting['batch'] must be a dict")
    for key in ("total", "source", "nikl"):
        _require(key in batch, "setting['batch'] missing %r" % key)
    _require(batch["nikl"] == 0 and not isinstance(batch["nikl"], bool),
             "setting['batch']['nikl'] must be 0")
    _require(batch["total"] == batch_size,
             "setting['batch']['total'] (%r) must equal batch_size (%r)"
             % (batch["total"], batch_size))
    _require(batch["source"] == batch_size,
             "setting['batch']['source'] (%r) must equal batch_size (%r)"
             % (batch["source"], batch_size))

    weight_decay = setting.get("weight_decay")
    _require(_finite_number(weight_decay) and float(weight_decay) >= 0.0,
             "setting['weight_decay'] must be a nonnegative finite number")

    seed = setting.get("seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool),
             "setting['seed'] must be an int")

    cpu_threads = setting.get("cpu_threads")
    _require(_pos_int(cpu_threads), "setting['cpu_threads'] must be a positive int")

    if steps_override is None:
        n_steps = int(optimizer_steps)
    else:
        _require(_pos_int(steps_override),
                 "steps override must be a positive int (bool rejected)")
        n_steps = int(steps_override)

    return {
        "steps": n_steps,
        "batch_size": int(batch_size),
        "head_lr": float(head_lr),
        "weight_decay": float(weight_decay),
        "cpu_threads": int(cpu_threads),
    }


def _validate_model(model, classes):
    _require(isinstance(model, torch.nn.Module), "model must be a torch.nn.Module")
    head = getattr(model, "head", None)
    _require(isinstance(head, torch.nn.Linear),
             "model.head must be a torch.nn.Linear, got %r" % (type(head),))

    named_params = dict(model.named_parameters())
    for key in ("head.weight", "head.bias"):
        _require(key in named_params, "model is missing named parameter %r" % key)
    _require(head.out_features == len(classes),
             "model.head.out_features (%d) must equal len(classes) (%d)"
             % (head.out_features, len(classes)))

    head_buffers = [name for name, _ in head.named_buffers()]
    _require(not head_buffers,
             "model.head must own no buffers (a Linear head is expected), found %r"
             % (head_buffers,))

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        _require(tensor.device.type == "cpu",
                 "tensor %r must live on CPU, found device %r" % (name, tensor.device))

    return head


# --------------------------------------------------------------------------- #
# class-balanced sampling (exactly two RNG draws per sampled example)
# --------------------------------------------------------------------------- #
def _group_by_class(source_train):
    by_class = {}
    for item in source_train:
        by_class.setdefault(int(item["class_index"]), []).append(item)
    class_keys = sorted(by_class)
    return by_class, class_keys


def _sample_batch(by_class, class_keys, batch_size, rng):
    n_classes = len(class_keys)
    batch = []
    for _ in range(batch_size):
        key = class_keys[int(rng.integers(n_classes))]
        items = by_class[key]
        batch.append(items[int(rng.integers(len(items)))])
    return batch


# --------------------------------------------------------------------------- #
# the training entry point handed to the host as `u.run_training`
# --------------------------------------------------------------------------- #
def run_head_only_training(model, source_train, nikl, classes, setting, rng,
                           log=print, steps=None):
    """Train only `model.head`; every other parameter and buffer stays frozen.

    Returns a proof dictionary describing the schedule, the observed losses and
    the verified freeze of the non-head state.
    """
    # ---- validation only: nothing below this block mutates the model -------
    classes = _validate_classes(classes)
    source_train = _validate_source_train(source_train, classes)
    _validate_nikl(nikl)
    schedule = _validate_setting(setting, steps)
    _require(hasattr(rng, "integers"), "rng must expose .integers (numpy Generator API)")
    _require(callable(log), "log must be callable")
    head = _validate_model(model, classes)

    from pilot.data import collate  # lazy: keep module import stdlib/torch only

    n_steps = schedule["steps"]
    batch_size = schedule["batch_size"]
    head_lr = schedule["head_lr"]
    weight_decay = schedule["weight_decay"]
    torch.set_num_threads(schedule["cpu_threads"])

    # ---- declare the scope on the mutable setting dict the host persists ---
    setting["training_scope"] = TRAINING_SCOPE
    setting["encoder_buffers_frozen"] = True

    # ---- freeze everything, then unfreeze the head ------------------------
    model.eval()          # encoder stays in eval mode for the entire run
    head.train()          # a Linear head has no mode-dependent behaviour
    for name, param in model.named_parameters():
        param.requires_grad_(_is_head_key(name))
    for param in model.parameters():
        param.grad = None  # clear every stale gradient, head and non-head alike

    head_params = [param for name, param in model.named_parameters() if _is_head_key(name)]
    _require(len(head_params) > 0, "no head parameters found to optimise")
    optimizer = torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)

    # ---- snapshots used later to prove the freeze -------------------------
    non_head_snapshot = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not _is_head_key(name)
    }
    head_snapshot = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if _is_head_key(name)
    }
    non_head_parameter_count = sum(
        1 for name, _ in model.named_parameters() if not _is_head_key(name)
    )
    non_head_buffer_count = sum(
        1 for name, _ in model.named_buffers() if not _is_head_key(name)
    )

    by_class, class_keys = _group_by_class(source_train)
    _require(len(class_keys) > 0, "source_train yielded no classes")

    log("head-only training: %d steps x batch %d over %d clips / %d present classes "
        "(encoder_lr=0.0, head_lr=%g, weight_decay=%g, scope=%s)"
        % (n_steps, batch_size, len(source_train), len(class_keys),
           head_lr, weight_decay, TRAINING_SCOPE))

    losses = []
    seen_sample_ids = set()

    for step in range(1, n_steps + 1):
        batch_items = _sample_batch(by_class, class_keys, batch_size, rng)
        for item in batch_items:
            seen_sample_ids.add(item["sample_id"])

        batch_x, batch_pad, batch_y = collate([
            {
                "x": item["x"],
                "frames": int(item["x"].shape[0]),
                "label": int(item["class_index"]),
            }
            for item in batch_items
        ])

        x_t = torch.from_numpy(batch_x)
        pad_t = torch.from_numpy(batch_pad)
        y_t = torch.from_numpy(batch_y)
        if y_t.dtype != torch.int64:
            y_t = y_t.long()

        logits = model(x_t, pad_t)
        loss = F.cross_entropy(logits, y_t)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite loss %r at step %d" % (loss.item(), step))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in model.named_parameters():
            if not _is_head_key(name):
                continue
            if param.grad is None:
                raise RuntimeError("missing gradient for head parameter %r at step %d"
                                   % (name, step))
            if not bool(torch.isfinite(param.grad).all()):
                raise RuntimeError("non-finite gradient for head parameter %r at step %d"
                                   % (name, step))
        optimizer.step()

        for name, param in model.named_parameters():
            if _is_head_key(name) and not bool(torch.isfinite(param).all()):
                raise RuntimeError("non-finite head parameter %r after step %d"
                                   % (name, step))

        loss_value = float(loss.item())
        losses.append(loss_value)
        if step % LOG_EVERY == 0 or step == n_steps:
            tail = losses[-TAIL_WINDOW:]
            log("  step %d/%d loss=%.6f mean_last_%d=%.6f"
                % (step, n_steps, loss_value, len(tail), sum(tail) / len(tail)))

    # ---- prove the freeze; never restore encoder values to fake it --------
    final_state = {
        name: tensor for name, tensor in model.state_dict().items()
        if not _is_head_key(name)
    }
    if set(final_state) != set(non_head_snapshot):
        raise RuntimeError("non-head state_dict key set changed during training")
    drifted = [
        name for name, tensor in final_state.items()
        if not _tensors_identical(tensor, non_head_snapshot[name])
    ]
    if drifted:
        raise RuntimeError("non-head state changed during training: %r" % (drifted[:8],))
    all_non_head_state_identical = True

    leaked = [
        name for name, param in model.named_parameters()
        if not _is_head_key(name) and param.grad is not None
    ]
    if leaked:
        raise RuntimeError("non-head gradients present after training: %r" % (leaked[:8],))
    all_non_head_gradients_none = True

    changed_head_keys = sorted(
        name for name, param in model.named_parameters()
        if _is_head_key(name) and not _tensors_identical(param.detach(), head_snapshot[name])
    )
    if not changed_head_keys:
        raise RuntimeError("head weight/bias did not change: training had no effect")
    for name, param in model.named_parameters():
        if _is_head_key(name) and not bool(torch.isfinite(param).all()):
            raise RuntimeError("non-finite head parameter %r at end of training" % name)

    model.eval()

    tail = losses[-TAIL_WINDOW:]
    effective_head_lr = float(optimizer.param_groups[0]["lr"])
    log("head-only training complete: %d steps, loss %.6f -> %.6f, "
        "%d distinct clips, changed head keys %r, encoder frozen (%d tensors)"
        % (n_steps, losses[0], losses[-1], len(seen_sample_ids),
           changed_head_keys, len(non_head_snapshot)))

    return {
        "steps": n_steps,
        "loss_first": round(losses[0], 6),
        "loss_last": round(losses[-1], 6),
        "loss_mean_last_20": round(sum(tail) / len(tail), 6),
        "losses": [round(value, 6) for value in losses],
        "distinct_source_clips_seen": len(seen_sample_ids),
        "distinct_nikl_clips_seen": 0,
        "augmentation_examples": [],
        "intermediate_checkpoints_kept": 0,
        "learning_rates": {"encoder": 0.0, "head": effective_head_lr},
        "training_scope": TRAINING_SCOPE,
        "freeze_proof": {
            "non_head_state_tensor_count": len(non_head_snapshot),
            "non_head_parameter_count": non_head_parameter_count,
            "non_head_buffer_count": non_head_buffer_count,
            "all_non_head_state_identical": all_non_head_state_identical,
            "all_non_head_gradients_none": all_non_head_gradients_none,
            "changed_head_keys": changed_head_keys,
        },
    }


# --------------------------------------------------------------------------- #
# host delegation wrappers
# --------------------------------------------------------------------------- #
def _require_head_only_plan(plan):
    _require(isinstance(plan, dict), "plan must be a dict")
    head_only = plan.get("head_only")
    _require(isinstance(head_only, dict),
             "plan['head_only'] is required for head-only training")
    _require(set(head_only) == set(REQUIRED_HEAD_ONLY_PLAN_KEYS),
             "plan['head_only'] must have exactly the keys %r, got %r"
             % (sorted(REQUIRED_HEAD_ONLY_PLAN_KEYS), sorted(head_only)))

    encoder_lr = head_only["encoder_lr"]
    _require(_finite_number(encoder_lr) and float(encoder_lr) == 0.0,
             "plan['head_only']['encoder_lr'] must be 0.0, got %r" % (encoder_lr,))
    _require(head_only["freeze_encoder_buffers"] is True,
             "plan['head_only']['freeze_encoder_buffers'] must be True")
    _require(head_only["training_scope"] == TRAINING_SCOPE,
             "plan['head_only']['training_scope'] must be %r, got %r"
             % (TRAINING_SCOPE, head_only["training_scope"]))


def run(plan, prepared, out):
    """Head-only variant of `base.run`.

    The inherited base plan carries a legacy positive `settings.encoder_lr`
    purely so the unchanged base metadata preflight keeps accepting the run; it
    must never become the effective rate.  A deep copy of `prepared` (metadata
    only) is overridden to 0.0 before being handed to the original `base.run`,
    so the effective settings that base persists record a zero encoder lr, and
    `run_head_only_training` additionally tags the mutable setting dict with the
    training scope and buffer-freeze flag.  `plan`, `prepared` and the frozen
    host modules are left untouched.
    """
    _require_head_only_plan(plan)

    prepared_head_only = copy.deepcopy(prepared)
    _require(isinstance(prepared_head_only, dict), "prepared must be a dict")
    settings = prepared_head_only.get("settings")
    _require(isinstance(settings, dict), "prepared['settings'] must be a dict")
    settings["encoder_lr"] = 0.0

    original_run_training = u.run_training
    u.run_training = run_head_only_training
    try:
        # Delegate to the base.run captured at import time: `main` temporarily
        # points base.run at this wrapper, so calling base.run here would recurse.
        return _ORIGINAL_BASE_RUN(plan, prepared_head_only, out)
    finally:
        u.run_training = original_run_training


def main(argv=None):
    """Run the unchanged host CLI (preflight, provenance, output refusal) with
    this module's head-only `run` temporarily installed as base's module-global
    `run`.  The original is always restored."""
    previous_run = base.run
    base.run = run
    try:
        return base.main(argv)
    finally:
        base.run = previous_run


if __name__ == "__main__":
    raise SystemExit(main())
