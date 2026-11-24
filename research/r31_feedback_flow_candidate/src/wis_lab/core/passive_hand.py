"""Prepare a paired intact/masked condition for a hand a reviewer marked as non-signing.

The one way in is prepare_from_public_clip(annotation): it resolves the clip through the public
manifest AND the frozen training list, requires them to agree on path and hash, computes the cache
file's ACTUAL sha256, and loads the arrays itself. A caller cannot hand it arrays. That is not
fussiness - an earlier version accepted one clip's annotation with another clip's arrays and
returned the pair stamped with the first clip's provenance.

PREPARATION ONLY. Nothing here trains, infers, loads a model, opens a camera, or touches the web
recogniser or the functional_17 training specification. The default call writes no file, loads no
model and uses no randomness.

What this is for: a root diagnostic held timestamps, body and the opposite hand fixed and removed
one detailed hand from six public NIKL training references. Two of the six changed label. That is
a sensitivity observation on training references - not accuracy, and not an account of any webcam
error. To study it properly the two conditions have to be built the same way, over the same span,
from the same bytes. This module builds that pair and refuses to build it from anything weaker.

What grants eligibility is a REVIEW, never a measurement. A static supporting hand carries
meaning, and so does a static handshape, so low motion is not evidence of a passive hand. Neither
is a hand_participation number, a model's output, a word's name, or a user simply not moving. One
anatomical hand must carry an explicit reviewed role of `non_sign_hand` covering the whole sign
interval, and an annotation is review evidence - not a certification that the review is correct.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from .observation import (HAND_SLICES, LEFT_HAND_SLICE, Observation, RIGHT_HAND_SLICE,
                          sha256_file)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MANIFEST_REL = os.path.join("config", "source_manifest.json")
TRAINING_INPUTS_REL = os.path.join("runs", "union93_candidate_v1", "inputs.json")
CACHE_REL = "cache"
PUBLIC_SOURCE_KINDS = ("nikl_interpreter_word_clip",)

ANNOTATION_SCHEMA = "wis_passive_hand_annotation_v1"
PREPARED_SCHEMA = "wis_passive_hand_pair_v1"
REQUIRED_ROLE = "non_sign_hand"
SIDES = ("left", "right")
SIDE_INDEX = {"left": 0, "right": 1}
SIDE_SLICE = {"left": LEFT_HAND_SLICE, "right": RIGHT_HAND_SLICE}
# the split role the frozen training inputs use for a NIKL reference
TRAINING_ROLES = ("training_original_and_every_augmentation",)
# a basis that is a measurement, a guess or an output - never a review
REJECTED_BASES = ("low_motion", "hand_participation", "model_prediction", "prediction",
                  "word_name", "user_did_not_move", "unknown", "auto", "heuristic")
REVIEW_BASIS = "reviewed_public_footage"
NOT_CERTIFICATION = ("review evidence about one public recording. It is NOT a certification that "
                     "the review is correct, not an expert verification, and not a verdict about "
                     "any learner's execution.")


# --------------------------------------------------------------------------- public sources
def load_sources(manifest_path: str, training_inputs_path: str) -> dict:
    """The two existing public lists, joined by clip id. Nothing else is a source of truth.

    `manifest_path` is config/source_manifest.json; `training_inputs_path` is the frozen
    runs/.../inputs.json whose `nikl` list says which clip held which split role.
    """
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(training_inputs_path, encoding="utf-8") as fh:
        inputs = json.load(fh)
    clips = {c["clip_id"]: c for c in manifest.get("clips", [])}
    out = {}
    for row in inputs.get("nikl", []):
        clip = clips.get(row["id"])
        if clip is None:
            continue                        # in the training list but not in the public manifest
        out[row["id"]] = {"clip_id": row["id"], "word_id": row["word_id"],
                          "split_role": row.get("role"), "npz": row.get("npz"),
                          "npz_sha256": row.get("npz_sha256"),
                          "manifest_word_id": clip.get("word_id"),
                          "manifest_npz": clip.get("observation_npz"),
                          "manifest_observation_sha256": clip.get("observation_sha256"),
                          "source_kind": clip.get("source_kind"),
                          "source_video_sha256": clip.get("sha256"),
                          "source_group": row["id"]}
    return out


def _safe_relative(path: str) -> bool:
    """A cache path from the frozen list, not somewhere a caller chose."""
    if not path or os.path.isabs(path) or path.startswith("\\\\"):
        return False
    parts = path.replace("\\", "/").split("/")
    return ".." not in parts and parts[0] == CACHE_REL


def _inside_cache(path: str, root: str) -> bool:
    """Canonical containment: after resolving links and '..', the file is under <root>/cache."""
    cache = os.path.realpath(os.path.join(root, CACHE_REL))
    real = os.path.realpath(os.path.join(root, path))
    return os.path.commonpath([real, cache]) == cache and real != cache


def resolve_public_source(clip_id: str, root: str = ROOT) -> dict:
    """-> {ok, reasons, source}. The one place a clip id becomes bytes on disk.

    Every step is a refusal point: the two public lists must both know the clip, agree on its
    word, its cache path and its cache hash; the split role must be a training role; the source
    kind must be a public one; the path must resolve INSIDE the repository cache; the file must
    exist; and its ACTUAL sha256 must equal what both lists declare. A declared hash that nobody
    ever computed is not provenance.
    """
    reasons = []
    sources = load_sources(os.path.join(root, MANIFEST_REL),
                           os.path.join(root, TRAINING_INPUTS_REL))
    src = sources.get(clip_id)
    if src is None:
        return {"ok": False, "reasons": ["UNKNOWN_CLIP_ID"], "source": None}

    if src["split_role"] not in TRAINING_ROLES:
        reasons.append("NOT_A_TRAINING_SPLIT_CLIP")
    if src["word_id"] != src["manifest_word_id"]:
        reasons.append("WORD_ID_DISAGREES_BETWEEN_LISTS")
    if src["source_kind"] not in PUBLIC_SOURCE_KINDS:
        reasons.append("NOT_A_PUBLIC_SOURCE_KIND")
    if src["manifest_npz"] and src["manifest_npz"] != src["npz"]:
        reasons.append("CACHE_PATH_DISAGREES_BETWEEN_LISTS")
    if src["manifest_observation_sha256"] != src["npz_sha256"]:
        # one list saying a different hash than the other is not a detail to prefer one of
        reasons.append("MANIFEST_AND_TRAINING_CACHE_HASH_DISAGREE")
    if not _safe_relative(src["npz"] or ""):
        reasons.append("CACHE_PATH_NOT_IN_PUBLIC_CACHE")
    elif not _inside_cache(src["npz"], root):
        reasons.append("CACHE_PATH_ESCAPES_THE_PUBLIC_CACHE")

    abs_npz = os.path.join(root, src["npz"] or "")
    actual = None
    if not reasons:
        if not os.path.isfile(abs_npz):
            reasons.append("CACHE_FILE_MISSING")
        else:
            actual = sha256_file(abs_npz)
            if actual != src["npz_sha256"]:
                reasons.append("CACHE_FILE_SHA256_MISMATCH")
    if reasons:
        return {"ok": False, "reasons": reasons, "source": None}
    return {"ok": True, "reasons": [], "source": dict(src, abs_npz=abs_npz,
                                                      actual_cache_sha256=actual)}



# --------------------------------------------------------------------------- the annotation
def _evidence_reasons(ann: dict, src: dict) -> list:
    """The review itself must be present, tied to the public recording, and say what was seen."""
    ev = ann.get("evidence")
    if not isinstance(ev, dict) or not ev:
        return ["REVIEW_EVIDENCE_MISSING"]
    out = []
    seen = ev.get("visual_observation")
    if not (isinstance(seen, str) and seen.strip()):
        out.append("REVIEW_EVIDENCE_HAS_NO_VISUAL_OBSERVATION")
    vid = ev.get("source_video_sha256")
    if not vid:
        out.append("REVIEW_EVIDENCE_HAS_NO_SOURCE_VIDEO")
    elif src.get("source_video_sha256") and vid != src["source_video_sha256"]:
        # a review of some other footage cannot license removing a hand from this recording
        out.append("REVIEW_EVIDENCE_VIDEO_IS_NOT_THIS_CLIP")
    if ev.get("every_decoded_frame_reviewed") is not True:
        out.append("REVIEW_EVIDENCE_DOES_NOT_COVER_EVERY_FRAME")
    if ann.get("is_expert_verification") is True:
        out.append("ANNOTATION_MAY_NOT_CLAIM_EXPERT_VERIFICATION")
    return out


def check_annotation(ann: dict, sources: dict) -> dict:
    """-> {eligible, reasons, record}. Every refusal is named; none is silent.

    This function never looks at the clip's content. Eligibility is a property of the review, and
    letting the data argue for its own removal is exactly what must not happen here.
    """
    reasons = []

    def need(cond, why):
        if not cond:
            reasons.append(why)
        return cond

    need(isinstance(ann, dict), "ANNOTATION_NOT_AN_OBJECT")
    if reasons:
        return {"eligible": False, "reasons": reasons, "record": None}
    need(ann.get("schema") == ANNOTATION_SCHEMA, "WRONG_ANNOTATION_SCHEMA")

    clip_id = ann.get("clip_id")
    src = sources.get(clip_id)
    if not need(src is not None, "UNKNOWN_CLIP_ID"):
        return {"eligible": False, "reasons": reasons, "record": None}

    need(src["split_role"] in TRAINING_ROLES, "NOT_A_TRAINING_SPLIT_CLIP")
    need(ann.get("word_id") == src["word_id"] == src["manifest_word_id"], "WORD_ID_MISMATCH")
    need(ann.get("cache_sha256") == src["npz_sha256"], "CACHE_SHA256_MISMATCH")
    need(src["manifest_observation_sha256"] == src["npz_sha256"],
         "MANIFEST_AND_TRAINING_CACHE_HASH_DISAGREE")
    need(_safe_relative(src["npz"] or ""), "CACHE_PATH_NOT_IN_PUBLIC_CACHE")
    need("npz" not in ann and "path" not in ann and "absolute_path" not in ann,
         "ANNOTATION_MAY_NOT_CHOOSE_A_PATH")

    # exactly one anatomical hand, reviewed as non-signing. The other side may be described, but
    # whatever it says, it is never removed.
    roles = ann.get("hand_roles")
    if need(isinstance(roles, dict) and set(roles) <= set(SIDES) and roles, "MALFORMED_HAND_ROLES"):
        marked = [s for s in SIDES if roles.get(s) == REQUIRED_ROLE]
        need(len(marked) == 1, "BOTH_HANDS_MARKED_NON_SIGNING" if len(marked) > 1
             else "NO_HAND_REVIEWED_AS_NON_SIGNING")
        for s in SIDES:
            if s in roles:
                need(isinstance(roles[s], str) and roles[s] != "unknown",
                     "HAND_ROLE_UNKNOWN_IS_NOT_ELIGIBLE")
    else:
        marked = []

    basis = ann.get("basis")
    need(basis == REVIEW_BASIS, "BASIS_IS_NOT_A_REVIEW")
    need(basis not in REJECTED_BASES, "BASIS_IS_A_MEASUREMENT_OR_OUTPUT_NOT_A_REVIEW")
    need(bool(ann.get("reviewed_by")), "NO_REVIEWER_RECORDED")
    reasons.extend(_evidence_reasons(ann, src))

    iv = ann.get("reviewed_interval")
    if need(isinstance(iv, dict), "MALFORMED_REVIEWED_INTERVAL"):
        a, b = iv.get("start_frame"), iv.get("end_frame")
        need(isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool)
             and not isinstance(b, bool), "MALFORMED_REVIEWED_INTERVAL")
        need(isinstance(a, int) and isinstance(b, int) and a >= 0 and b >= a,
             "MALFORMED_REVIEWED_INTERVAL")
    need(ann.get("covers_whole_sign_interval") is True, "REVIEW_DOES_NOT_COVER_THE_WHOLE_SIGN")

    if reasons:
        return {"eligible": False, "reasons": sorted(set(reasons)), "record": None}
    side = marked[0]
    ev = ann["evidence"]
    return {"eligible": True, "reasons": [], "record": {
        "clip_id": clip_id, "word_id": src["word_id"], "side": side,
        "side_index": SIDE_INDEX[side], "npz": src["npz"], "cache_sha256": src["npz_sha256"],
        "split_role": src["split_role"], "source_group": src["source_group"],
        "reviewed_by": ann.get("reviewed_by"), "basis": ann.get("basis"),
        "reviewed_interval": dict(ann["reviewed_interval"]),
        "covers_whole_sign_interval": True,
        "hand_roles": dict(ann["hand_roles"]),
        # the reason the removal was allowed travels WITH the pair. Dropping it left a prepared
        # artifact that no longer said why one of its hands was gone.
        "review_evidence": {
            "source_video_sha256": ev.get("source_video_sha256"),
            "every_decoded_frame_reviewed": ev.get("every_decoded_frame_reviewed"),
            "frame_count": ev.get("frame_count"),
            "visual_observation": ev.get("visual_observation"),
            "contact_sheet": os.path.basename(ev["contact_sheet"]) if ev.get("contact_sheet")
            else None,
            "contact_sheet_sha256": ev.get("contact_sheet_sha256"),
            "reviewed_interval_description": (
                f"frames {ann['reviewed_interval']['start_frame']}-"
                f"{ann['reviewed_interval']['end_frame']}, reviewed as covering the whole sign"),
        },
        "limitation": ann.get("limitation"),
        "is_expert_verification": False,
        "annotation_is": NOT_CERTIFICATION,
    }}



# --------------------------------------------------------------------------- the masked copy
def mask_hand(obs: Observation, side: str) -> Observation:
    """A COPY with one detailed hand absent. The original object is never modified.

    Absent means what the schema already means by absent: NaN coordinates, observed false, usable
    false, a NaN hand score. It is never a zero, because a zero is a position and would read as a
    hand resting at the origin.
    """
    if side not in SIDE_SLICE:
        raise ValueError(f"side must be left or right, got {side!r}")
    out = Observation(
        xy_px=np.array(obs.xy_px, dtype=np.float32, copy=True),
        timestamp_us=np.array(obs.timestamp_us, dtype=np.int64, copy=True),
        frame_id=np.array(obs.frame_id, dtype=np.int64, copy=True),
        observed=np.array(obs.observed, dtype=bool, copy=True),
        usable=np.array(obs.usable, dtype=bool, copy=True),
        conf=np.array(obs.conf, dtype=np.float32, copy=True),
        hand_score=np.array(obs.hand_score, dtype=np.float32, copy=True),
        sidecar=copy.deepcopy(obs.sidecar),
    )
    sl = SIDE_SLICE[side]
    out.xy_px[:, sl, :] = np.nan
    out.observed[:, sl] = False
    out.usable[:, sl] = False
    out.conf[:, sl] = np.nan
    out.hand_score[:, SIDE_INDEX[side]] = np.nan
    out.sidecar["passive_hand_masked_side"] = side
    out.sidecar["passive_hand_note"] = ("one detailed hand was removed for a paired condition. "
                                        "Body pose including the pose wrist, the opposite hand, "
                                        "the timestamps and every other value are unchanged.")
    out.validate()
    return out


# --------------------------------------------------------------------------- the paired build
def _pilot():
    """The existing 280-channel builder and the existing span rule. No new dependency."""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    src = os.path.normpath(os.path.join(root, "..", "sign_model_pilot_v1", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)
    from pilot.data import build_channels
    from pilot.web_input import active_span, to_native_like
    return build_channels, active_span, to_native_like


def _prepare_pair_from_arrays(ann: dict, obs: Observation, sources: dict,
                              verified: dict | None = None) -> dict:
    """INTERNAL. Build the pair from arrays a caller already holds.

    This is the array-level helper, kept for synthetic tests. It is deliberately NOT the public
    entry point: a caller can hand it any arrays at all, so what it produces carries
    provenance.verified = False. Passing one clip's annotation with another clip's arrays used to
    be possible here and came back stamped with the first clip's cache hash - which is exactly the
    kind of claim this boundary must not be able to make. Verified provenance is only ever added
    by prepare_from_public_clip(), which loads the bytes itself.
    """
    checked = check_annotation(ann, sources)
    if not checked["eligible"]:
        return {"prepared": False, "reasons": checked["reasons"], "pair": None}
    rec = checked["record"]
    build_channels, active_span, to_native_like = _pilot()

    nat, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                  np.asarray(obs.usable, dtype=bool))
    span = active_span(present)
    if span is None:
        return {"prepared": False, "reasons": ["NO_USABLE_HAND_FRAME"], "pair": None}
    lo, hi = span

    iv = rec["reviewed_interval"]
    if not (iv["start_frame"] <= lo and iv["end_frame"] >= hi):
        # a review of part of the sign cannot license removing the hand for all of it
        return {"prepared": False, "pair": None,
                "reasons": ["REVIEWED_INTERVAL_DOES_NOT_COVER_THE_SIGN"],
                "detail": {"active_span": [lo, hi], "reviewed_interval": iv}}

    masked = mask_hand(obs, rec["side"])
    other = RIGHT_HAND_SLICE if rec["side"] == "left" else LEFT_HAND_SLICE
    opposite_frames = int(masked.usable[lo:hi + 1, other].any(axis=1).sum())
    if opposite_frames < 1:
        # with both hands gone there is nothing left to learn the sign from
        return {"prepared": False, "pair": None, "reasons": ["NO_OBSERVED_OPPOSITE_HAND_FRAME"],
                "detail": {"active_span": [lo, hi]}}

    ts = np.asarray(obs.timestamp_us, dtype=np.int64)
    nat_m, present_m = to_native_like(np.asarray(masked.xy_px, dtype=np.float32),
                                      np.asarray(masked.usable, dtype=bool))
    intact = build_channels(nat[lo:hi + 1], present[lo:hi + 1], ts[lo:hi + 1])
    variant = build_channels(nat_m[lo:hi + 1], present_m[lo:hi + 1], ts[lo:hi + 1])

    provenance = {
        "verified": False,
        "meaning": "these arrays were supplied by the caller. Nothing here checked that they are "
                   "the clip the annotation names, so this is NOT public provenance.",
        "cache_npz_declared_by_list": rec["npz"],
        "cache_sha256_declared_by_list": rec["cache_sha256"],
        "split_role": rec["split_role"], "source_group": rec["source_group"],
    }
    if verified:
        provenance = dict(verified, split_role=rec["split_role"],
                          source_group=rec["source_group"])

    return {"prepared": True, "reasons": [], "pair": {
        "schema": PREPARED_SCHEMA,
        "clip_id": rec["clip_id"], "word_id": rec["word_id"],
        "masked_side": rec["side"],
        "annotation": rec,
        "review_evidence": rec["review_evidence"],
        "annotation_is": NOT_CERTIFICATION,
        "provenance": provenance,
        "span": {"first_frame": lo, "last_frame": hi, "frames": int(hi - lo + 1),
                 "taken_from": "the ORIGINAL clip, and held fixed for both variants"},
        "opposite_hand_frames_in_span": opposite_frames,
        "variants": {"original": intact, "masked": variant},
        "grouping": {
            "source_group": rec["source_group"],
            "role": rec["split_role"],
            "independent_samples_added": 0,
            "meaning": "the masked variant is a CONDITION of the same source clip, not another "
                       "person and not another sample. It keeps the source group and the training "
                       "role of its original, and never enters validation.",
        },
        "is_not": "a training run, a remedy, a model setting or a claim about any word. It is one "
                  "prepared pair, waiting for a review that has not happened yet.",
    }}


def prepare_from_public_clip(ann: dict, root: str = ROOT) -> dict:
    """THE entry point. Resolve the clip from the public lists, verify the bytes, then load them.

    The caller supplies an annotation and nothing else. The arrays come from the file the two
    public lists agree on, after its actual sha256 has been computed and matched, so an annotation
    can never be paired with another clip's arrays and still be called verified.
    """
    if not isinstance(ann, dict) or not ann.get("clip_id"):
        return {"prepared": False, "reasons": ["ANNOTATION_HAS_NO_CLIP_ID"], "pair": None}
    resolved = resolve_public_source(ann["clip_id"], root=root)
    if not resolved["ok"]:
        return {"prepared": False, "reasons": resolved["reasons"], "pair": None}
    src = resolved["source"]

    sources = load_sources(os.path.join(root, MANIFEST_REL),
                           os.path.join(root, TRAINING_INPUTS_REL))
    checked = check_annotation(ann, sources)
    if not checked["eligible"]:
        return {"prepared": False, "reasons": checked["reasons"], "pair": None}

    try:
        obs = Observation.load(src["abs_npz"])
    except (OSError, ValueError, AssertionError) as exc:
        return {"prepared": False, "pair": None,
                "reasons": [f"CACHE_UNREADABLE:{type(exc).__name__}"]}

    verified = {
        "verified": True,
        "meaning": "the arrays were loaded from the file both public lists name, after its actual "
                   "sha256 was computed and matched. The caller supplied no arrays.",
        "cache_npz": src["npz"], "cache_sha256_declared": src["npz_sha256"],
        "cache_sha256_actual": src["actual_cache_sha256"],
        "manifest_observation_sha256": src["manifest_observation_sha256"],
        "source_kind": src["source_kind"],
        "source_video_sha256": src["source_video_sha256"],
        "resolved_from": ["config/source_manifest.json",
                          "runs/union93_candidate_v1/inputs.json"],
    }
    out = _prepare_pair_from_arrays(ann, obs, sources, verified=verified)
    if out["prepared"]:
        out["pair"]["provenance"]["observation_frames"] = obs.n_frames
    return out



# --------------------------------------------------------------------------- optional artifact
def write_prepared(pair: dict, out_dir: str) -> dict:
    """Write the pair only when asked, only into a NEW directory, and never over a file.

    prepare_pair() writes nothing. This is separate so that preparing cannot quietly leave state.
    """
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} exists; an output directory must be new")
    os.makedirs(out_dir)
    written = {}
    for name in ("original", "masked"):
        path = os.path.join(out_dir, f"{pair['clip_id']}.{name}.npz")
        if os.path.exists(path):
            raise FileExistsError(path)
        ch = pair["variants"][name]
        np.savez_compressed(path, **{k: v for k, v in ch.items()
                                     if isinstance(v, np.ndarray)})
        written[name] = {"path": path, "sha256": sha256_file(path)}
    meta = os.path.join(out_dir, f"{pair['clip_id']}.pair.json")
    doc = {k: v for k, v in pair.items() if k != "variants"}
    doc["written"] = written
    with open(meta, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
    written["meta"] = {"path": meta, "sha256": sha256_file(meta)}
    return written


# prepare_from_public_clip is the boundary. The array-level helper is private on purpose.
__all__ = ["ANNOTATION_SCHEMA", "PREPARED_SCHEMA", "REQUIRED_ROLE", "HAND_SLICES",
           "NOT_CERTIFICATION", "load_sources", "resolve_public_source", "check_annotation",
           "mask_hand", "prepare_from_public_clip", "write_prepared"]
