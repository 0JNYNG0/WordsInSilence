"""tools_parent_anchor_pair.py

One fixed, uniform, parent-centred quadratic anchor on the classifier head of
the verified frozen-encoder head-only trainer:

    loss = mean_cross_entropy
         + 0.5 * 1.0 * sum_over({head.weight, head.bias}) (p - p0)^2

``p0`` are detached clones of the head parameters captured before that arm
trains, so both arms are anchored to the shared parent they start from.  There
is no division by parameter or class count, no anchor to zero, no teacher and
no class-specific term.  Data, split, RNG, sampler, steps, augmentation,
architecture, encoder freeze, optimiser settings and every host provenance
check are reused unchanged; training itself is delegated to the existing
verified trainer.  No new optimiser and no copied trainer code.

Public surface: ``make_trainer_wrapper``, ``run``, ``main``.
"""

from __future__ import annotations

import copy
import math

import torch

import tools_native_data_pair as base
import tools_head_only_native_pair as h

SCHEMA = "uniform_classifier_parent_anchor_v1"
PARAMETERS = ("head.weight", "head.bias")
COEFFICIENT = 1.0
HALF_FACTOR = 0.5
REDUCTION = "sum"

# The host trainer stores its losses rounded to 6 decimals.
_LOSS_ABS_TOL = 5e-6
_LOSS_REL_TOL = 1e-6

# Keys whose values must survive delegation exactly as passed by the host.
_WATCHED_SETTING_KEYS = (
    "batch_size", "batch", "steps", "seed", "lr", "learning_rate",
    "head_lr", "encoder_lr", "weight_decay",
)

# torch.nn.functional.cross_entropy positional layout.
_POS_SIZE_AVERAGE, _POS_REDUCE, _POS_REDUCTION = 3, 5, 6

_RUN_ACTIVE = False
_MAIN_ACTIVE = False
_ARM_ACTIVE = False


def _number(value, name, expected):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("plan['parent_anchor'][%r] must be a non-boolean number" % name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("plan['parent_anchor'][%r] must be finite" % name)
    if value != expected:
        raise ValueError("plan['parent_anchor'][%r] must be %r" % (name, expected))
    return value


def _text(block, name):
    value = block.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("plan['parent_anchor'][%r] must be a non-empty string" % name)
    return value


def validate_plan(plan):
    """Validate the anchor semantics; return the descriptive strings."""
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dict")
    if plan.get("augmentation") is True:
        raise ValueError("plan['augmentation'] must stay false for this scope")
    head_only = plan.get("head_only")
    if not isinstance(head_only, dict):
        raise ValueError("plan['head_only'] block is required")
    if head_only.get("encoder_lr") != 0.0 or isinstance(head_only.get("encoder_lr"), bool):
        raise ValueError("plan['head_only']['encoder_lr'] must be 0.0")
    if head_only.get("freeze_encoder_buffers") is not True:
        raise ValueError("plan['head_only']['freeze_encoder_buffers'] must be True")
    if head_only.get("training_scope") != "head_only_frozen_encoder_and_buffers":
        raise ValueError("plan['head_only']['training_scope'] is not the head-only scope")

    block = plan.get("parent_anchor")
    if not isinstance(block, dict):
        raise ValueError("plan['parent_anchor'] block is required")
    if block.get("schema") != SCHEMA:
        raise ValueError("plan['parent_anchor']['schema'] must be %r" % SCHEMA)
    _number(block.get("coefficient"), "coefficient", COEFFICIENT)
    _number(block.get("half_factor"), "half_factor", HALF_FACTOR)
    if block.get("reduction") != REDUCTION:
        raise ValueError("plan['parent_anchor']['reduction'] must be %r" % REDUCTION)
    names = block.get("parameters")
    if not isinstance(names, (list, tuple)) or tuple(names) != PARAMETERS:
        raise ValueError("plan['parent_anchor']['parameters'] must be %s" % list(PARAMETERS))
    return {key: _text(block, key) for key in ("anchor", "loss", "selection")}


def _head_parameters(model):
    """Return [(name, Parameter)] for exactly head.weight and head.bias."""
    head = getattr(model, "head", None)
    if head is None:
        raise AttributeError("model.head is required for this scope")
    found = sorted(n for n, _ in model.named_parameters() if n.split(".")[0] == "head")
    if found != sorted(PARAMETERS):
        raise ValueError("head parameter names must be exactly %s, got %s" % (list(PARAMETERS), found))
    lookup = dict(model.named_parameters())
    items = [(name, lookup[name]) for name in PARAMETERS]
    if items[0][1] is not head.weight or items[1][1] is not head.bias:
        raise ValueError("named_parameters head entries do not alias model.head parameters")
    for name, param in items:
        if not isinstance(param, torch.nn.Parameter):
            raise TypeError("%s must be a Parameter" % name)
    return items


def _check_reduction(args, kwargs):
    if len(args) > _POS_SIZE_AVERAGE and args[_POS_SIZE_AVERAGE] is not None:
        raise ValueError("legacy size_average is not supported by the anchored loss")
    if len(args) > _POS_REDUCE and args[_POS_REDUCE] is not None:
        raise ValueError("legacy reduce is not supported by the anchored loss")
    if kwargs.get("size_average") is not None or kwargs.get("reduce") is not None:
        raise ValueError("legacy size_average/reduce are not supported by the anchored loss")
    reduction = kwargs.get("reduction", "mean")
    if len(args) > _POS_REDUCTION:
        reduction = args[_POS_REDUCTION]
    if reduction != "mean":
        raise ValueError("anchored loss requires mean cross entropy, got reduction=%r" % (reduction,))


class _AnchoredFunctional:
    """Minimal proxy over the trainer's functional namespace.

    Every attribute is forwarded unchanged except ``cross_entropy``, which adds
    the differentiable parent anchor to the host's own mean cross entropy.
    """

    def __init__(self, original, live, references, records):
        self._original = original
        self._live = live
        self._references = references
        self._records = records

    def __getattr__(self, name):
        return getattr(self._original, name)

    def cross_entropy(self, *args, **kwargs):
        _check_reduction(args, kwargs)
        ce = self._original.cross_entropy(*args, **kwargs)
        if not isinstance(ce, torch.Tensor) or ce.dim() != 0:
            raise ValueError("anchored loss requires a scalar cross entropy")
        accumulator = None
        for name, param, expected in self._live:
            current = getattr(param[0], param[1])
            if current is not expected:
                raise RuntimeError("%s was rebound during training" % name)
            diff = current - self._references[name]
            term = (diff * diff).sum()
            accumulator = term if accumulator is None else accumulator + term
        penalty = accumulator * (HALF_FACTOR * COEFFICIENT)
        total = ce + penalty
        for key, value in (("cross_entropy", ce), ("penalty", penalty),
                           ("total", total), ("squared_distance", accumulator)):
            number = float(value.detach())
            if not math.isfinite(number):
                raise ValueError("non-finite %s at step %d" % (key, len(self._records["total"]) + 1))
            self._records[key].append(number)
        return total


def _close(observed, recorded):
    return abs(observed - recorded) <= _LOSS_ABS_TOL + _LOSS_REL_TOL * abs(observed)


def make_trainer_wrapper(original_trainer, plan):
    """Wrap the verified trainer so each arm trains against its own parent."""
    described = validate_plan(plan)
    if not callable(original_trainer):
        raise TypeError("original_trainer must be callable")

    def wrapper(model, source_train, nikl, classes, setting, rng, log=print, steps=None):
        global _ARM_ACTIVE
        if _ARM_ACTIVE:
            raise RuntimeError("parent anchor is sequential only; recursive use rejected")
        if not isinstance(setting, dict):
            raise TypeError("setting must be a dict")
        owned = [plan] + list(plan.values()) + list(plan["head_only"].values())
        if any(setting is candidate for candidate in owned):
            raise ValueError("setting aliases a plan-owned object; refusing to mutate the plan")
        if "parent_anchor" in setting:
            raise ValueError("setting already carries a 'parent_anchor' block")

        items = _head_parameters(model)
        head = model.head
        live = [(PARAMETERS[0], (head, "weight"), items[0][1]),
                (PARAMETERS[1], (head, "bias"), items[1][1])]
        references = {name: param.detach().clone() for name, param in items}
        records = {"cross_entropy": [], "penalty": [], "total": [], "squared_distance": []}

        sampler = h._sample_batch
        watched = {k: setting[k] for k in _WATCHED_SETTING_KEYS if k in setting}
        original_functional = h.F
        if not callable(getattr(original_functional, "cross_entropy", None)):
            raise TypeError("h.F does not expose cross_entropy")
        torch_functional = torch.nn.functional.cross_entropy

        _ARM_ACTIVE = True
        h.F = _AnchoredFunctional(original_functional, live, references, records)
        try:
            stats = original_trainer(model, source_train, nikl, classes, setting, rng,
                                     log=log, steps=steps)
        finally:
            h.F = original_functional
            _ARM_ACTIVE = False

        if h.F is not original_functional:
            raise RuntimeError("cross entropy namespace was not restored")
        if torch.nn.functional.cross_entropy is not torch_functional:
            raise RuntimeError("torch.nn.functional was modified")
        if h._sample_batch is not sampler:
            raise RuntimeError("h._sample_batch identity changed")
        if {k: setting[k] for k in _WATCHED_SETTING_KEYS if k in setting} != watched:
            raise RuntimeError("optimizer settings were rewritten during training")
        if not isinstance(stats, dict) or "parent_anchor" in stats:
            raise RuntimeError("unexpected trainer result")

        observed = records["total"]
        count = len(observed)
        if any(len(values) != count for values in records.values()):
            raise RuntimeError("anchor component lists have different lengths")
        if stats.get("steps") != count:
            raise RuntimeError("cross entropy calls %d != reported steps %r" % (count, stats.get("steps")))
        legacy = stats.get("losses")
        if not isinstance(legacy, (list, tuple)) or len(legacy) != count:
            raise RuntimeError("trainer losses do not match the observed step count")
        for index, recorded in enumerate(legacy):
            if not _close(observed[index], float(recorded)):
                raise RuntimeError("trainer loss %d disagrees with the anchored total" % index)
        for key, value in (("loss_first", observed[0] if count else None),
                           ("loss_last", observed[-1] if count else None)):
            if key in stats and value is not None and not _close(value, float(stats[key])):
                raise RuntimeError("%s disagrees with the anchored total" % key)

        final_distance = records["squared_distance"][-1] if count else 0.0
        setting["parent_anchor"] = {
            "schema": SCHEMA, "coefficient": COEFFICIENT, "half_factor": HALF_FACTOR,
            "reduction": REDUCTION, "parameters": list(PARAMETERS),
            "origin": "arm initial values", "applied": True, "steps": count,
            "final_squared_distance": final_distance,
            "recorded_losses_are_total_including_anchor": True,
        }
        result = dict(stats)
        result["parent_anchor"] = {
            "schema": SCHEMA, "coefficient": COEFFICIENT, "half_factor": HALF_FACTOR,
            "reduction": REDUCTION, "parameters": list(PARAMETERS),
            "origin": "arm initial values",
            "anchor": described["anchor"], "loss": described["loss"],
            "selection": described["selection"],
            "steps": count,
            "cross_entropy": list(records["cross_entropy"]),
            "penalty": list(records["penalty"]),
            "total_loss": list(observed),
            "squared_distance": list(records["squared_distance"]),
            "final_squared_distance": final_distance,
            "legacy_losses_are_total_including_anchor": True,
            "legacy_losses_are_pure_cross_entropy": False,
            "normalisation": "none: no division by parameter or class count",
            "sampler_unchanged": True,
            "cross_entropy_namespace_restored": True,
            "encoder_changes": "none",
            "class_specific_changes": "none",
            "quality_claim": "none: the anchor limits drift from the parent and is not evidence of better accuracy",
        }
        return result

    return wrapper


def run(plan, prepared, out):
    """Install the anchored trainer for one sequential pair run."""
    global _RUN_ACTIVE
    if _RUN_ACTIVE:
        raise RuntimeError("parent anchor is sequential only; recursive run rejected")
    validate_plan(plan)
    snapshot = copy.deepcopy(plan)
    original_trainer = h.run_head_only_training
    wrapper = make_trainer_wrapper(original_trainer, plan)
    _RUN_ACTIVE = True
    h.run_head_only_training = wrapper
    try:
        result = h.run(plan, prepared, out)
    finally:
        h.run_head_only_training = original_trainer
        _RUN_ACTIVE = False
    try:
        changed = plan != snapshot
    except Exception:
        changed = False
    if changed:
        raise RuntimeError("plan was mutated during the run")
    return result


def main(argv=None):
    """Bind this run into the host CLI and delegate; restore in finally."""
    global _MAIN_ACTIVE
    if _MAIN_ACTIVE:
        raise RuntimeError("parent anchor is sequential only; recursive main rejected")
    original_base_run = base.run
    _MAIN_ACTIVE = True
    base.run = run
    try:
        return base.main(argv)
    finally:
        base.run = original_base_run
        _MAIN_ACTIVE = False


if __name__ == "__main__":
    raise SystemExit(main())
