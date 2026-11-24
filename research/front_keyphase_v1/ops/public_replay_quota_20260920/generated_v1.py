"""tools_public_replay_quota.py

Narrow sampling wrapper for one preregistered comparison: class-balanced
training with a *fixed half-batch quota* of original public-domain examples.

What this module does
---------------------
It changes **where half of each training batch comes from** and nothing else:

    per step (batch_size = 32):
      * samples  1..16 : drawn by the original ``h._sample_batch`` from the
                         unchanged merged pool / class keys (2 rng draws each)
      * samples 17..32 : drawn uniformly over the 18 ordered public-original
                         ids from ``plan['public_replay']['sample_ids']``;
                         2 rng draws each -- ``rng.integers(n_merged_classes)``
                         is drawn and *ignored*, ``rng.integers(18)`` selects
                         the replay item.

    => 32 samples and 64 rng draws per step; for a 660-step arm that is
       21,120 samples and 42,240 draws.

What this module does *not* do
------------------------------
* No new training algorithm: ``h.run_head_only_training`` (frozen encoder,
  head-only AdamW+CE, encoder-invariance proof) is *reused*, never copied.
* No transformations, no threshold changes, no augmentation, no re-seeding or
  replacement of the caller rng, no hyperparameter search, no model promotion.
* No file, model, array, network or user-data access of its own; arrays
  (``item['x']``) are never read, copied or mutated -- only item *references*
  are passed through.
* No source file is patched on disk; ``h._sample_batch``,
  ``h.run_head_only_training`` and ``base.run`` are temporarily rebound and
  always restored in ``finally``, including on error.
* Provenance / split preflight, exclusive output creation and persistence stay
  in the host (``base.main`` / ``base.run``); they are not reimplemented.

Sampler stats schema (``make_sampler`` second return value) -- see
``SAMPLER_STATS_SCHEMA`` for the authoritative per-key documentation.  All
values are JSON-safe, so the dict can be embedded in persisted metadata.

Honesty notes baked into the metadata
-------------------------------------
* ``setting['batch']`` (legacy ``source=32 / nikl=0``) is an *input-argument
  accounting field*.  It is preserved untouched and is explicitly labelled as
  NOT evidence that zero public originals were trained on.
* ``merged`` rng draws are *contract-derived* (the original sampler is
  documented to perform exactly 2 draws per example); the explicit replay draws
  are *observed* by this wrapper.  The two are labelled separately.
* ``distinct_nikl_clips_seen`` is corrected to the number of unique public ids
  actually observed, and labelled as coming from the merged input pool plus the
  explicit replay half.
* Origin is read only from ``item['origin']`` / plan ids -- never inferred from
  word labels or from input paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import operator
from collections.abc import Mapping

import tools_native_data_pair as base
import tools_head_only_native_pair as h

# --------------------------------------------------------------------------
# Fixed, preregistered constants.  Intentionally not configurable.
# --------------------------------------------------------------------------

PLAN_SCHEMA = "fixed_public_original_quota_v1"
SAMPLER_STATS_SCHEMA_ID = "public_replay_sampler_stats_v1"
PROOF_SCHEMA_ID = "public_replay_quota_proof_v1"

BATCH_SIZE = 32
QUOTA = 16                      # explicit public-original examples per batch
MERGED_PER_BATCH = BATCH_SIZE - QUOTA
DRAWS_PER_SAMPLE = 2
DRAWS_PER_BATCH = BATCH_SIZE * DRAWS_PER_SAMPLE
PUBLIC_COUNT = 18
EXPECTED_ORIGIN = "nikl_original"
TRANSFORMS = "none"

DRAW_RULE = (
    "per step: 16 merged-pool examples via the original h._sample_batch on the "
    "unchanged by_class/class_keys (2 rng.integers draws each), then 16 explicit "
    "public-original replay examples (draw 1 = rng.integers(n_merged_classes), "
    "drawn and ignored; draw 2 = rng.integers(18) indexes the ordered plan "
    "public_replay.sample_ids); 32 samples and 64 rng draws per step; "
    "transforms=none; item references reused, arrays untouched"
)

MERGED_DRAWS_SOURCE = (
    "contract-derived: the original h._sample_batch performs exactly 2 "
    "rng.integers draws per example (class, then within-class item)"
)

LEGACY_BATCH_NOTE = (
    "setting['batch'] (e.g. source=32 / nikl=0) is an input-argument accounting "
    "field describing how examples were handed to the trainer; it is NOT evidence "
    "that zero public-original examples were trained on. See public_replay.sampling "
    "for the examples actually drawn."
)

SAMPLER_STATS_SCHEMA = {
    "schema": "constant stats-schema id",
    "quota": "explicit public replay examples per batch (16)",
    "batch_size": "required batch size handed to the sampler (32)",
    "merged_per_batch": "examples taken from the merged pool per batch (16)",
    "public_ids": "ordered plan public-original sample ids (18)",
    "calls": "number of sampler invocations (must equal trainer stats['steps'])",
    "total_samples": "items returned overall (32 * calls)",
    "merged_samples": "items returned by the original sampler (16 * calls)",
    "explicit_replay_samples": "items injected by the replay half (16 * calls)",
    "merged_rng_draws_expected": "contract-derived merged draws (32 * calls)",
    "explicit_replay_rng_draws_observed": "draws this wrapper made (32 * calls)",
    "ignored_class_draws": "rng.integers(n_classes) draws made and discarded",
    "total_rng_draws": "merged_rng_draws_expected + explicit_replay_rng_draws_observed",
    "merged_rng_draws_source": "how merged draws were accounted for",
    "public_originals_observed": "items with origin=='nikl_original' in either half",
    "public_originals_observed_merged_half": "same, merged half only",
    "public_originals_observed_replay_half": "same, replay half only",
    "public_replay_counts": "per-public-id explicit replay count (all 18 keys, zeros kept)",
    "public_ids_observed": "sorted public ids seen in either half",
    "distinct_public_ids_observed": "len(public_ids_observed)",
    "drawn_origin_counts": "histogram of item['origin'] over every returned item",
    "drawn_non_public_samples": "returned items whose origin != 'nikl_original'",
    "pool_size": "len(source_train) for this arm",
    "pool_origin_counts": "histogram of item['origin'] over source_train",
    "pool_public_count": "source_train items with origin=='nikl_original' (must be 18)",
    "pool_non_public_count": "pool_size - pool_public_count (extra candidates, by origin/id only)",
    "grouped_sample_count": "distinct ids present in by_class",
    "grouped_class_count": "number of class keys",
    "grouped_equals_pool": "whether by_class covers exactly source_train's ids",
    "class_group_signature_sha256": "digest of the first-call ordered (key, ids) grouping",
    "ordered_public_replay_id_sha256": "incremental digest of ordered replay picks",
    "ordered_sample_digest_sha256": "incremental digest of ordered (mode, id, class) per item",
    "transforms": "'none'",
    "draw_rule": "human-readable description of the draw rule",
}

_RUN_ACTIVE = False
_MAIN_ACTIVE = False
_TRAINER_ACTIVE = False


# --------------------------------------------------------------------------
# small, strict helpers
# --------------------------------------------------------------------------

def _field(item, name):
    """Read ``name`` from a mapping-like or attribute-like item."""
    if isinstance(item, Mapping):
        if name not in item:
            raise KeyError("item is missing required field %r" % (name,))
        return item[name]
    try:
        return getattr(item, name)
    except AttributeError:
        raise KeyError("item is missing required field %r" % (name,)) from None


def _plan_int(value, what):
    """Strict int: rejects bool-as-int and any non-int type."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be a non-bool int, got %r" % (what, type(value).__name__))
    return value


def _index(value, what):
    """Integer-like coercion (numpy ints allowed), bool rejected."""
    if isinstance(value, bool):
        raise TypeError("%s must not be a bool" % (what,))
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError("%s must be integer-like, got %r" % (what, type(value).__name__)) from None


def _text(value, what):
    if not isinstance(value, str) or not value:
        raise TypeError("%s must be a non-empty str" % (what,))
    return value


def _true(value, what):
    if value is not True:
        raise ValueError("%s must be exactly True" % (what,))
    return True


def _false(value, what):
    if value is not False:
        raise ValueError("%s must be exactly False" % (what,))
    return False


def _mapping(value, what):
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % (what,))
    return value


def _seq(value, what):
    if not isinstance(value, (list, tuple)):
        raise TypeError("%s must be a list or tuple" % (what,))
    return value


def _log(log, message):
    if callable(log):
        log(message)


def _hash_line(hasher, payload):
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    hasher.update((line + "\n").encode("utf-8"))


def _check_rng(rng):
    """Accept any caller rng exposing ``.integers``; never reseed or replace it."""
    if rng is None or not callable(getattr(rng, "integers", None)):
        raise TypeError("rng must expose a callable .integers(n)")
    return rng


def snapshot_stats(stats):
    """JSON-safe deep copy of a sampler stats dict (documented schema above)."""
    return copy.deepcopy(dict(stats))


# --------------------------------------------------------------------------
# validation: plan / prepared, then per-arm source_train
# --------------------------------------------------------------------------

def validate_plan(plan, prepared):
    """Validate the replay plan against ``prepared``; return the 18 ordered ids.

    Runs before any training or delegation.  Neither object is modified.
    """
    _mapping(plan, "plan")
    _mapping(prepared, "prepared")

    block = _mapping(_field(plan, "public_replay"), "plan['public_replay']")
    if _text(_field(block, "schema"), "public_replay.schema") != PLAN_SCHEMA:
        raise ValueError("public_replay.schema must be %r" % (PLAN_SCHEMA,))

    raw_ids = _seq(_field(block, "sample_ids"), "public_replay.sample_ids")
    ids = tuple(_text(sid, "public_replay.sample_ids[%d]" % i) for i, sid in enumerate(raw_ids))
    if len(ids) != PUBLIC_COUNT:
        raise ValueError("public_replay.sample_ids must hold %d ids, got %d"
                         % (PUBLIC_COUNT, len(ids)))
    if len(set(ids)) != PUBLIC_COUNT:
        raise ValueError("public_replay.sample_ids contains duplicates")

    if _plan_int(_field(block, "quota"), "public_replay.quota") != QUOTA:
        raise ValueError("public_replay.quota must be %d (fixed, preregistered)" % (QUOTA,))
    if _plan_int(_field(block, "batch_size"), "public_replay.batch_size") != BATCH_SIZE:
        raise ValueError("public_replay.batch_size must be %d" % (BATCH_SIZE,))
    if _text(_field(block, "expected_origin"), "public_replay.expected_origin") != EXPECTED_ORIGIN:
        raise ValueError("public_replay.expected_origin must be %r" % (EXPECTED_ORIGIN,))
    if _text(_field(block, "transforms"), "public_replay.transforms") != TRANSFORMS:
        raise ValueError("public_replay.transforms must be %r" % (TRANSFORMS,))
    _text(_field(block, "draw_rule"), "public_replay.draw_rule")

    settings = _mapping(_field(plan, "settings"), "plan['settings']")
    if _plan_int(_field(settings, "batch_size"), "settings.batch_size") != BATCH_SIZE:
        raise ValueError("plan['settings']['batch_size'] must be %d" % (BATCH_SIZE,))

    seen_aug = False
    for owner, holder in (("plan", plan), ("plan['settings']", settings)):
        if "augmentation" in holder:
            _false(holder["augmentation"], "%s['augmentation']" % (owner,))
            seen_aug = True
    if not seen_aug:
        raise ValueError("augmentation must be present and False in plan or plan['settings']")

    head_only = _mapping(_field(plan, "head_only"), "plan['head_only']")
    _true(_field(head_only, "encoder_lr0"), "head_only.encoder_lr0")
    _true(_field(head_only, "freeze_encoder_buffers"), "head_only.freeze_encoder_buffers")

    nikl_items = _seq(_field(prepared, "nikl_items"), "prepared['nikl_items']")
    if len(nikl_items) != PUBLIC_COUNT:
        raise ValueError("prepared['nikl_items'] must hold %d entries, got %d"
                         % (PUBLIC_COUNT, len(nikl_items)))
    seen_ids, seen_classes = set(), set()
    for pos, meta in enumerate(nikl_items):
        where = "prepared['nikl_items'][%d]" % (pos,)
        sid = _text(_field(meta, "sample_id"), where + ".sample_id")
        _text(_field(meta, "label"), where + ".label")
        cls = _index(_field(meta, "class_index"), where + ".class_index")
        if sid in seen_ids:
            raise ValueError("duplicate sample_id %r in prepared['nikl_items']" % (sid,))
        if cls in seen_classes:
            raise ValueError("duplicate class_index %d in prepared['nikl_items']" % (cls,))
        seen_ids.add(sid)
        seen_classes.add(cls)
    if seen_ids != set(ids):
        raise ValueError("public_replay.sample_ids do not match prepared['nikl_items'] ids")
    if len(seen_classes) != PUBLIC_COUNT:
        raise ValueError("prepared['nikl_items'] must cover %d distinct classes" % (PUBLIC_COUNT,))

    return ids


def validate_source_train(source_train, public_ids):
    """Validate one arm's merged pool; return ``(id_to_item, ordered_replay_items)``.

    Requires: unique ids; the 18 replay ids present exactly once each; the items
    carrying ``origin == 'nikl_original'`` are *all and only* those 18 ids; and
    each of them sits on a distinct class.  Order, arrays and metadata of
    ``source_train`` are left untouched (``x`` is never read).
    """
    public_ids = tuple(public_ids)
    if len(public_ids) != PUBLIC_COUNT or len(set(public_ids)) != PUBLIC_COUNT:
        raise ValueError("public_ids must hold %d distinct ids" % (PUBLIC_COUNT,))

    items = _seq(source_train, "source_train")
    if not items:
        raise ValueError("source_train is empty")

    id_to_item = {}
    origin_counts = {}
    public_classes = {}
    for pos, item in enumerate(items):
        where = "source_train[%d]" % (pos,)
        sid = _text(_field(item, "sample_id"), where + ".sample_id")
        if sid in id_to_item:
            raise ValueError("duplicate sample_id %r in source_train" % (sid,))
        origin = _text(_field(item, "origin"), where + ".origin")
        cls = _index(_field(item, "class_index"), where + ".class_index")
        id_to_item[sid] = item
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        if origin == EXPECTED_ORIGIN:
            public_classes[sid] = cls

    missing = [sid for sid in public_ids if sid not in id_to_item]
    if missing:
        raise ValueError("replay ids missing from source_train: %r" % (missing,))
    if set(public_classes) != set(public_ids):
        raise ValueError(
            "origin=%r items in source_train (%d) do not match the plan replay ids (%d)"
            % (EXPECTED_ORIGIN, len(public_classes), PUBLIC_COUNT))
    if len(set(public_classes.values())) != PUBLIC_COUNT:
        raise ValueError("the %d public originals must occupy %d distinct classes"
                         % (PUBLIC_COUNT, PUBLIC_COUNT))

    replay_items = [id_to_item[sid] for sid in public_ids]
    return id_to_item, replay_items, origin_counts


# --------------------------------------------------------------------------
# sampler factory (public, independently testable on synthetic item dicts)
# --------------------------------------------------------------------------

def make_sampler(original_sampler, source_train, public_ids):
    """Build the half-batch replay sampler.

    Returns ``(sample_callable, mutable_stats)``.  ``sample_callable`` has the
    ``h._sample_batch(by_class, class_keys, batch_size, rng)`` signature and
    returns 32 *original item references* (16 merged + 16 explicit replay).
    ``mutable_stats`` follows ``SAMPLER_STATS_SCHEMA`` and is updated in place.
    """
    if not callable(original_sampler):
        raise TypeError("original_sampler must be callable")
    public_ids = tuple(public_ids)
    id_to_item, replay_items, pool_origin_counts = validate_source_train(source_train, public_ids)

    pool_size = len(source_train)
    pool_public = pool_origin_counts.get(EXPECTED_ORIGIN, 0)

    stats = {
        "schema": SAMPLER_STATS_SCHEMA_ID,
        "quota": QUOTA,
        "batch_size": BATCH_SIZE,
        "merged_per_batch": MERGED_PER_BATCH,
        "public_ids": list(public_ids),
        "calls": 0,
        "total_samples": 0,
        "merged_samples": 0,
        "explicit_replay_samples": 0,
        "merged_rng_draws_expected": 0,
        "explicit_replay_rng_draws_observed": 0,
        "ignored_class_draws": 0,
        "total_rng_draws": 0,
        "merged_rng_draws_source": MERGED_DRAWS_SOURCE,
        "public_originals_observed": 0,
        "public_originals_observed_merged_half": 0,
        "public_originals_observed_replay_half": 0,
        # every replay id is initialised, zero counts included
        "public_replay_counts": {sid: 0 for sid in public_ids},
        "public_ids_observed": [],
        "distinct_public_ids_observed": 0,
        "drawn_origin_counts": {},
        "drawn_non_public_samples": 0,
        "pool_size": pool_size,
        "pool_origin_counts": dict(pool_origin_counts),
        "pool_public_count": pool_public,
        "pool_non_public_count": pool_size - pool_public,
        "grouped_sample_count": 0,
        "grouped_class_count": 0,
        "grouped_equals_pool": None,
        "class_group_signature_sha256": None,
        "ordered_public_replay_id_sha256": hashlib.sha256().hexdigest(),
        "ordered_sample_digest_sha256": hashlib.sha256().hexdigest(),
        "transforms": TRANSFORMS,
        "draw_rule": DRAW_RULE,
    }

    replay_hasher = hashlib.sha256()
    all_hasher = hashlib.sha256()
    observed_public = set()
    state = {"signature": None, "active": False}

    def _validate_groups(by_class, class_keys):
        keys = _seq(class_keys, "class_keys")
        if not keys:
            raise ValueError("class_keys is empty")
        _mapping(by_class, "by_class")
        signature = []
        seen = set()
        for key in keys:
            int_key = _index(key, "class key")
            if key not in by_class:
                raise KeyError("class key %r missing from by_class" % (key,))
            group = _seq(by_class[key], "by_class[%r]" % (key,))
            if not group:
                raise ValueError("class group %r is empty" % (key,))
            group_ids = []
            for item in group:
                sid = _text(_field(item, "sample_id"), "grouped item sample_id")
                if sid not in id_to_item:
                    raise ValueError("by_class item %r is not part of source_train" % (sid,))
                if sid in seen:
                    raise ValueError("sample_id %r appears in more than one class group" % (sid,))
                seen.add(sid)
                group_ids.append(sid)
            signature.append((int_key, tuple(group_ids)))
        signature = tuple(signature)

        for sid in public_ids:
            if sid not in seen:
                raise ValueError("public replay id %r is absent from by_class" % (sid,))
        replay_keys = {key for key, gids in signature for gid in gids if gid in set(public_ids)
                       for _ in (0,)} if False else set()
        for key, gids in signature:
            for gid in gids:
                if gid in stats["public_replay_counts"]:
                    replay_keys.add(key)
        if len(replay_keys) != PUBLIC_COUNT:
            raise ValueError("the %d public originals must sit on %d distinct class keys, saw %d"
                             % (PUBLIC_COUNT, PUBLIC_COUNT, len(replay_keys)))

        if state["signature"] is None:
            state["signature"] = signature
            digest = hashlib.sha256()
            _hash_line(digest, [[key, list(gids)] for key, gids in signature])
            stats["class_group_signature_sha256"] = digest.hexdigest()
            stats["grouped_sample_count"] = len(seen)
            stats["grouped_class_count"] = len(signature)
            stats["grouped_equals_pool"] = (seen == set(id_to_item))
        elif signature != state["signature"]:
            raise ValueError("by_class id/class grouping changed between sampler calls")
        return keys

    def _observe(mode, item, position, call_index):
        sid = _text(_field(item, "sample_id"), "drawn item sample_id")
        origin = _text(_field(item, "origin"), "drawn item origin")
        cls = _index(_field(item, "class_index"), "drawn item class_index")
        stats["drawn_origin_counts"][origin] = stats["drawn_origin_counts"].get(origin, 0) + 1
        if origin == EXPECTED_ORIGIN:
            stats["public_originals_observed"] += 1
            stats["public_originals_observed_%s_half" % ("replay" if mode == "replay"
                                                         else "merged")] += 1
            observed_public.add(sid)
        else:
            stats["drawn_non_public_samples"] += 1
        _hash_line(all_hasher, {"call": call_index, "pos": position,
                                "mode": mode, "id": sid, "class": cls})
        return sid

    def sample(by_class, class_keys, batch_size, rng):
        if state["active"]:
            raise RuntimeError("recursive use of the public-replay sampler is not supported")
        state["active"] = True
        try:
            if _index(batch_size, "batch_size") != BATCH_SIZE:
                raise ValueError("public replay quota %d is preregistered for batch_size %d, "
                                 "refusing batch_size %r" % (QUOTA, BATCH_SIZE, batch_size))
            _check_rng(rng)
            keys = _validate_groups(by_class, class_keys)
            call_index = stats["calls"] + 1

            merged = _seq(original_sampler(by_class, class_keys, MERGED_PER_BATCH, rng),
                          "original sampler result")
            if len(merged) != MERGED_PER_BATCH:
                raise ValueError("original sampler returned %d items, expected %d"
                                 % (len(merged), MERGED_PER_BATCH))

            batch = list(merged)
            for position, item in enumerate(batch):
                _observe("merged", item, position, call_index)

            n_classes = len(keys)
            for k in range(QUOTA):
                _index(rng.integers(n_classes), "ignored class draw")   # drawn, discarded
                pick = _index(rng.integers(PUBLIC_COUNT), "replay index draw")
                if not 0 <= pick < PUBLIC_COUNT:
                    raise ValueError("replay index draw %d out of range" % (pick,))
                item = replay_items[pick]          # original reference, no copy
                position = MERGED_PER_BATCH + k
                sid = _observe("replay", item, position, call_index)
                if sid != public_ids[pick]:
                    raise RuntimeError("replay index/id mismatch for %r" % (sid,))
                stats["public_replay_counts"][sid] += 1
                _hash_line(replay_hasher, {"call": call_index, "pos": position, "id": sid})
                batch.append(item)

            if len(batch) != BATCH_SIZE:
                raise RuntimeError("assembled batch of %d, expected %d" % (len(batch), BATCH_SIZE))

            stats["calls"] = call_index
            stats["merged_samples"] += MERGED_PER_BATCH
            stats["explicit_replay_samples"] += QUOTA
            stats["total_samples"] += BATCH_SIZE
            stats["merged_rng_draws_expected"] += MERGED_PER_BATCH * DRAWS_PER_SAMPLE
            stats["explicit_replay_rng_draws_observed"] += QUOTA * DRAWS_PER_SAMPLE
            stats["ignored_class_draws"] += QUOTA
            stats["total_rng_draws"] = (stats["merged_rng_draws_expected"]
                                        + stats["explicit_replay_rng_draws_observed"])
            stats["public_ids_observed"] = sorted(observed_public)
            stats["distinct_public_ids_observed"] = len(observed_public)
            stats["ordered_public_replay_id_sha256"] = replay_hasher.hexdigest()
            stats["ordered_sample_digest_sha256"] = all_hasher.hexdigest()
            return batch
        finally:
            state["active"] = False

    sample._public_replay_sampler = True
    return sample, stats


# --------------------------------------------------------------------------
# per-arm trainer wrapper around the existing verified trainer
# --------------------------------------------------------------------------

_PRESERVED_SETTING_KEYS = ("batch", "batch_size", "seed", "optimizer_steps", "augmentation")


def _preserved_snapshot(setting):
    snap = {}
    for key in setting.keys():
        name = str(key)
        if name in _PRESERVED_SETTING_KEYS or "lr" in name or "learning_rate" in name:
            snap[name] = copy.deepcopy(setting[key])
    return snap


def make_trainer_wrapper(original_trainer, plan, public_ids):
    """Wrap the *captured* original ``h.run_head_only_training`` for one arm."""
    if not callable(original_trainer):
        raise TypeError("original_trainer must be callable")
    if getattr(original_trainer, "_public_replay_wrapper", False):
        raise RuntimeError("original_trainer is already a public-replay wrapper (recursive use)")
    public_ids = tuple(public_ids)

    plan_owned = [plan]
    for key in ("settings", "head_only", "public_replay"):
        if isinstance(plan, Mapping) and key in plan:
            plan_owned.append(plan[key])

    setting_block = {
        "schema": PLAN_SCHEMA,
        "effective_quota_per_batch": QUOTA,
        "batch_size": BATCH_SIZE,
        "merged_per_batch": MERGED_PER_BATCH,
        "public_ids": list(public_ids),
        "expected_origin": EXPECTED_ORIGIN,
        "transforms": TRANSFORMS,
        "draw_rule": DRAW_RULE,
    }

    def wrapper(model, source_train, nikl, classes, setting, rng, log=print, steps=None):
        global _TRAINER_ACTIVE
        if _TRAINER_ACTIVE:
            raise RuntimeError("public-replay trainer wrapper is sequential-only; "
                               "recursive/concurrent use refused")
        _check_rng(rng)
        _mapping(setting, "setting")
        if "augmentation" in setting:
            _false(setting["augmentation"], "setting['augmentation']")
        if "public_replay" in setting and setting["public_replay"] != setting_block:
            raise ValueError("setting already carries a conflicting 'public_replay' block")

        # per-arm provenance: exact 18 replay ids, all-and-only nikl_original, distinct classes
        validate_source_train(source_train, public_ids)

        if isinstance(nikl, (list, tuple)) and nikl:
            try:
                nikl_ids = {_text(_field(m, "sample_id"), "nikl sample_id") for m in nikl}
            except KeyError:
                nikl_ids = None
            if nikl_ids is not None and nikl_ids != set(public_ids):
                raise ValueError("nikl argument ids do not match the plan replay ids")

        annotate_in_place = not any(setting is owned for owned in plan_owned)
        before = _preserved_snapshot(setting)

        original_sampler = h._sample_batch
        if getattr(original_sampler, "_public_replay_sampler", False):
            raise RuntimeError("h._sample_batch is already patched by this wrapper")
        sampler, sampling = make_sampler(original_sampler, source_train, public_ids)

        _TRAINER_ACTIVE = True
        try:
            h._sample_batch = sampler
            try:
                stats = original_trainer(model, source_train, nikl, classes,
                                         setting, rng, log=log, steps=steps)
            finally:
                h._sample_batch = original_sampler
        finally:
            _TRAINER_ACTIVE = False

        stats = _mapping(stats, "trainer stats")
        steps_done = _index(_field(stats, "steps"), "stats['steps']")
        if steps_done <= 0:
            raise ValueError("trainer reported %d steps" % (steps_done,))

        checks = (
            ("sampler calls", sampling["calls"], steps_done),
            ("total samples", sampling["total_samples"], BATCH_SIZE * steps_done),
            ("merged samples", sampling["merged_samples"], MERGED_PER_BATCH * steps_done),
            ("explicit replay samples", sampling["explicit_replay_samples"], QUOTA * steps_done),
            ("rng draws", sampling["total_rng_draws"], DRAWS_PER_BATCH * steps_done),
        )
        for name, got, want in checks:
            if got != want:
                raise RuntimeError("public replay accounting mismatch: %s = %d, expected %d "
                                   "(steps=%d)" % (name, got, want, steps_done))

        after = _preserved_snapshot(setting)
        if after != before:
            changed = sorted(k for k in set(before) | set(after)
                             if before.get(k, object()) != after.get(k, object()))
            raise RuntimeError("protected setting keys changed during training: %r" % (changed,))

        proof = {
            "schema": PROOF_SCHEMA_ID,
            "plan_schema": PLAN_SCHEMA,
            "quota_per_batch": QUOTA,
            "merged_per_batch": MERGED_PER_BATCH,
            "batch_size": BATCH_SIZE,
            "expected_origin": EXPECTED_ORIGIN,
            "transforms": TRANSFORMS,
            "draw_rule": DRAW_RULE,
            "public_replay_ids": list(public_ids),
            "steps": steps_done,
            "sampling": snapshot_stats(sampling),
            "legacy_batch_setting": copy.deepcopy(setting.get("batch")),
            "legacy_batch_note": LEGACY_BATCH_NOTE,
            "distinct_nikl_clips_seen_reported_by_trainer":
                copy.deepcopy(stats.get("distinct_nikl_clips_seen")),
            "distinct_nikl_clips_seen_source":
                "unique origin='nikl_original' ids observed in the merged input half "
                "plus the explicit public replay half",
            "setting_annotated_in_place": bool(annotate_in_place),
            "scope_note": "sampling wrapper only; loss, freeze and encoder-invariance proofs "
                          "are produced by h.run_head_only_training and are neither modified "
                          "nor re-validated here",
        }

        if annotate_in_place:
            setting["public_replay"] = copy.deepcopy(setting_block)
        else:
            _log(log, "[public_replay] setting object is plan-owned; leaving it unmodified to "
                      "preserve the original plan -- the replay block is recorded in "
                      "stats['public_replay'] instead.")

        stats["public_replay"] = proof
        stats["distinct_nikl_clips_seen"] = sampling["distinct_public_ids_observed"]
        stats["distinct_nikl_clips_seen_source"] = proof["distinct_nikl_clips_seen_source"]

        _log(log, "[public_replay] steps=%d samples=%d (merged=%d, explicit replay=%d) "
                  "rng_draws=%d distinct_public_ids=%d"
             % (steps_done, sampling["total_samples"], sampling["merged_samples"],
                sampling["explicit_replay_samples"], sampling["total_rng_draws"],
                sampling["distinct_public_ids_observed"]))
        return stats

    wrapper._public_replay_wrapper = True
    return wrapper


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def run(plan, prepared, out):
    """Validate, install the trainer wrapper, and delegate to ``h.run``."""
    global _RUN_ACTIVE
    if _RUN_ACTIVE:
        raise RuntimeError("tools_public_replay_quota.run is sequential-only; "
                           "recursive use refused")

    public_ids = validate_plan(plan, prepared)          # before any delegation

    original_trainer = h.run_head_only_training
    original_sampler = h._sample_batch
    if getattr(original_trainer, "_public_replay_wrapper", False):
        raise RuntimeError("h.run_head_only_training is already patched (recursive use)")
    if getattr(original_sampler, "_public_replay_sampler", False):
        raise RuntimeError("h._sample_batch is already patched (recursive use)")

    wrapper = make_trainer_wrapper(original_trainer, plan, public_ids)

    _RUN_ACTIVE = True
    try:
        h.run_head_only_training = wrapper
        return h.run(plan, prepared, out)
    finally:
        h.run_head_only_training = original_trainer
        h._sample_batch = original_sampler
        _RUN_ACTIVE = False


run._public_replay_run = True


def main(argv=None):
    """Delegate to the unchanged host ``base.main`` with ``base.run`` rebound.

    Preflight-only invocations never reach ``run`` (the host calls ``base.run``
    only for ``--execute``), and this module opens no model or data arrays.
    """
    global _MAIN_ACTIVE
    if _MAIN_ACTIVE:
        raise RuntimeError("tools_public_replay_quota.main is sequential-only; "
                           "recursive use refused")
    original_base_run = base.run
    if getattr(original_base_run, "_public_replay_run", False):
        raise RuntimeError("base.run is already patched by this module (recursive use)")

    _MAIN_ACTIVE = True
    try:
        base.run = run
        return base.main(argv)
    finally:
        base.run = original_base_run
        _MAIN_ACTIVE = False


if __name__ == "__main__":
    raise SystemExit(main())
