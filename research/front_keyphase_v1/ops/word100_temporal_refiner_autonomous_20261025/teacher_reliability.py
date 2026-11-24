"""Loss-only reliability for saved B0 pseudo-targets; never repairs coordinates.

No class, target word, user recording, RGB prediction or held-out label enters
these rules. Weights are uncertainty discounts, not correctness probabilities.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

UNIT = Path(__file__).resolve().parent
BUNDLE = UNIT.parent / "word100_temporal_public_bundle_20261025"
TRAIN_IDS = [1, 2, 3, 4, 5, 6]
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
         (15,16),(0,17),(17,18),(18,19),(19,20)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def load_adapter():
    spec = importlib.util.spec_from_file_location("teacher_safe_adapter", BUNDLE / "load_inputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _basis_scale(base, basis=None):
    if basis is not None:
        scale = float(basis[1])
    else:
        xy = np.asarray(base["xy_px"])
        ok = np.asarray(base["usable"], bool)[:,11] & np.asarray(base["usable"], bool)[:,12]
        widths = np.linalg.norm(xy[:,11] - xy[:,12], axis=1)
        good = ok & np.isfinite(widths) & (widths > 1)
        if not good.any():
            raise ValueError("No independent body shoulder scale; hand scale cannot substitute")
        scale = float(np.median(widths[good]))
    if not np.isfinite(scale) or scale <= 1:
        raise ValueError("Invalid fixed body basis")
    return scale


def _metrics(base, basis=None):
    xy = np.asarray(base["xy_px"], dtype=np.float64)
    usable = np.asarray(base["usable"], bool)
    if xy.ndim != 3 or xy.shape[1:] != (75,2) or usable.shape != xy.shape[:2]:
        raise ValueError("Expected safe B0 payload [T,75,2] and [T,75]")
    valid = usable & np.isfinite(xy).all(axis=-1)
    scale = _basis_scale(base, basis)
    confidence = np.asarray(base["conf"])
    confidence_available = np.asarray(base["conf_available"], bool)
    ts = np.asarray(base["timestamp_us"], dtype=np.float64)
    if ts.shape != (len(xy),) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError("Nonmonotonic source timestamps")
    wrist_distance = np.full((len(xy),2), np.nan)
    for side, body_wrist in enumerate((15,16)):
        hand_wrist = 33 + 21*side
        # Body confidence is an additional observation, not anatomical truth.
        ok = valid[:,body_wrist] & valid[:,hand_wrist] & confidence_available[:,body_wrist]
        ok &= confidence[:,body_wrist] >= 0.5
        wrist_distance[ok,side] = np.linalg.norm(xy[ok,body_wrist]-xy[ok,hand_wrist],axis=1)/scale
    excursion = np.full(valid.shape, np.nan)
    if len(xy) >= 3:
        ok = valid[:-2] & valid[1:-1] & valid[2:]
        ratio = ((ts[1:-1]-ts[:-2])/(ts[2:]-ts[:-2]))[:,None,None]
        expected = xy[:-2] + ratio*(xy[2:]-xy[:-2])
        values = np.linalg.norm(xy[1:-1]-expected,axis=-1)/scale
        excursion[1:-1] = np.where(ok, values, np.nan)
    bone_length = np.full((len(xy),2,len(EDGES)), np.nan)
    for side in (0,1):
        offset = 33+21*side
        for i,(a,b) in enumerate(EDGES):
            ok = valid[:,offset+a] & valid[:,offset+b]
            bone_length[ok,side,i] = np.linalg.norm(xy[ok,offset+a]-xy[ok,offset+b],axis=1)/scale
    return valid, wrist_distance, excursion, bone_length, scale


def weights_and_evidence(base, policy=None, basis=None):
    """Return loss-only weights and diagnostics; no arrays in base are mutated."""
    if policy is None:
        policy = read(UNIT / "TEACHER_POLICY.json")
    if policy.get("schema") != "b0_loss_reliability_v1":
        raise ValueError("Unsupported teacher policy")
    valid, wrist_dist, excursion, bones, scale = _metrics(base, basis)
    thresholds = policy["thresholds"]
    weights = valid.astype(np.float32)
    wrist_uncertain = np.isfinite(wrist_dist) & (wrist_dist > thresholds["wrist_disagreement_shoulder_units"])
    geometric_uncertain = np.zeros(valid.shape, bool)
    wrist_point_uncertain = np.zeros(valid.shape, bool)
    for side in (0,1):
        offset=33+21*side
        wrist_point_uncertain[:,offset:offset+21] = wrist_uncertain[:,side,None]
        for i,(a,b) in enumerate(EDGES):
            # A projected/foreshortened bone may be valid: discount, never delete.
            suspect = np.isfinite(bones[:,side,i]) & ((bones[:,side,i] < thresholds["projected_bone_short_shoulder_units"]) |
                                                     (bones[:,side,i] > thresholds["projected_bone_long_shoulder_units"]))
            geometric_uncertain[:,offset+a] |= suspect
            geometric_uncertain[:,offset+b] |= suspect
    temporal_large = np.isfinite(excursion) & (excursion > thresholds["isolated_excursion_shoulder_units"])
    # Speed/reversal alone is not unreliable. Another concern must corroborate it.
    temporal_corroborated = temporal_large & (wrist_point_uncertain | geometric_uncertain)
    weights[valid & (wrist_point_uncertain | geometric_uncertain)] = 0.75
    weights[valid & temporal_corroborated] = 0.5
    return weights, dict(valid=valid, wrist_uncertain=wrist_uncertain,
        geometric_uncertain=geometric_uncertain, temporal_large=temporal_large,
        temporal_corroborated=temporal_corroborated,
        wrist_disagreement_shoulder_units=wrist_dist, scale=scale)


def weights_for(base, policy=None, basis=None):
    """[T,75] float32 loss weights. Never include these in model inputs."""
    return weights_and_evidence(base,policy,basis)[0]


target_weights = weights_for


def fit_policy(rows, adapter):
    # Only global distributions from TRAIN01--06. No labels or reviewed boxes.
    wrist_values, excursion_values, pins = [], [], {}
    train_clips=[]
    for index,row in enumerate(rows):
        if int(row["signer_id"]) not in TRAIN_IDS:
            continue
        assert sha(row["arrays"]) == row["arrays_sha256"]
        payload=adapter.load_observations(index,"B0")
        _,w,e,_,_= _metrics(payload)
        wrist_values.extend(w[np.isfinite(w)].tolist())
        hand_e=e[:,33:]; excursion_values.extend(hand_e[np.isfinite(hand_e)].tolist())
        pins[row["arrays"]]=row["arrays_sha256"]
        train_clips.append(dict(clip_id=row["clip_id"],signer_id=int(row["signer_id"])))
    w99=float(np.quantile(wrist_values,0.99)); e99=float(np.quantile(excursion_values,0.99))
    return dict(schema="b0_loss_reliability_v1", status="FROZEN_BEFORE_HELDOUT_AUDIT",
        purpose="Robust pseudo-target learning; not anatomical truth, not correctness probability",
        provenance="Saved B0 observations only; teacher weights are loss-only and absent from model inputs",
        train_signer_ids=TRAIN_IDS, threshold_fit_clips=train_clips, source_pins=pins,
        fitting=dict(quantile=0.99,wrist_samples=len(wrist_values),excursion_samples=len(excursion_values),
                     wrist_train_q99=w99,excursion_train_q99=e99),
        thresholds=dict(wrist_disagreement_shoulder_units=max(0.15,w99),
            isolated_excursion_shoulder_units=max(0.10,e99),
            projected_bone_short_shoulder_units=0.005,projected_bone_long_shoulder_units=0.25,
            body_wrist_min_provided_confidence=0.5),
        loss_weights=dict(valid_default=1.0,independent_concern=0.75,corroborated_temporal_concern=0.5,
                          unavailable_or_nonfinite=0.0),
        safeguards=["All original coordinates, masks and timestamps remain unchanged",
            "No x-order, crossing, word-name or class-specific rule",
            "Unknown per-point confidence is not a negative finding",
            "Large speed/acceleration or short movement alone causes no discount",
            "Valid finite targets retain weight >=0.5; no sample exclusion",
            "Held-out IDs07--09 and already-seen24 image annotations never fit thresholds"],
        limitations=["Correlated detector errors may survive these rules, including a wrong wrist target",
            "Projected bone collapse may be legitimate; its 0.75 discount is caution, not error detection",
            "Body wrist and hand wrist are both predictions, not independent ground truth",
            "Proxy-error improvement cannot establish true21-joint or hidden-hand reconstruction accuracy",
            "This does not claim to correct the previously diagnosed V05 wrist mismatch"],
        implementation_sha256=sha(__file__))


def _swap_sides(base):
    out={k:np.array(v,copy=True) for k,v in base.items()}
    permutation=np.arange(75)
    permutation[33:54]=np.arange(54,75);permutation[54:75]=np.arange(33,54)
    for a,b in ((11,12),(13,14),(15,16)):
        permutation[a],permutation[b]=b,a
    for key in ("xy_px","usable","observed","conf","conf_available","point_origin"):
        out[key]=out[key][:,permutation].copy()
    for key in ("hand_score","hand_score_available"):
        out[key]=out[key][:,::-1].copy()
    return out,permutation


def audit(rows,adapter,policy):
    counts=Counter(); by_split={k:Counter() for k in ("train","dev","test")}; details=[]; checks=Counter()
    for index,row in enumerate(rows):
        assert sha(row["arrays"]) == row["arrays_sha256"]
        base=adapter.load_observations(index,"B0")
        before={k:np.array(v,copy=True) for k,v in base.items()}
        w,e=weights_and_evidence(base,policy)
        assert w.shape==base["usable"].shape and w.dtype==np.float32 and np.isfinite(w).all()
        assert np.all(w[~e["valid"]]==0) and np.all(w[e["valid"]]>=.5)
        assert np.all(w<=1)
        for k in before:
            assert np.array_equal(before[k],base[k],equal_nan=True),k
        checks["source_arrays_unchanged"]+=1
        slow={k:np.array(v,copy=True) for k,v in base.items()}
        slow["timestamp_us"]=slow["timestamp_us"]*2
        assert np.array_equal(w,weights_for(slow,policy));checks["uniform_speed_scaling_invariant"]+=1
        swapped,perm=_swap_sides(base)
        assert np.array_equal(w[:,perm],weights_for(swapped,policy));checks["anatomical_side_swap_equivariant"]+=1
        unknown={k:np.array(v,copy=True) for k,v in base.items()}
        unknown["conf"][:,33:]=0;unknown["conf_available"][:,33:]=False
        assert np.array_equal(w,weights_for(unknown,policy));checks["unknown_hand_confidence_preserved"]+=1
        c=dict(valid_hand_targets=int(e["valid"][:,33:].sum()),
            weight_1=int((w[:,33:]==1).sum()),weight_075=int((w[:,33:]==.75).sum()),
            weight_050=int((w[:,33:]==.5).sum()),wrist_concern_hand_frames=int(e["wrist_uncertain"].sum()),
            geometry_concern_points=int(e["geometric_uncertain"][:,33:].sum()),
            large_temporal_only_points=int((e["temporal_large"][:,33:] & ~e["temporal_corroborated"][:,33:]).sum()),
            valid_removed=0)
        split="train" if int(row["signer_id"])<=6 else "dev" if int(row["signer_id"])==7 else "test"
        counts.update(c);by_split[split].update(c)
        details.append(dict(clip_id=row["clip_id"],signer_id=int(row["signer_id"]),split=split,**c))
    assert set(x["signer_id"] for x in policy["threshold_fit_clips"])<=set(TRAIN_IDS)
    return dict(status="PASS_LOSS_ONLY_PROXY_RELIABILITY", clips=len(rows),checks=dict(checks),totals=dict(counts),
        by_split={k:dict(v) for k,v in by_split.items()},clips_detail=details,
        policy_sha256=sha(UNIT/"TEACHER_POLICY.json"),implementation_sha256=sha(__file__),
        source_coordinate_change=0,valid_target_deleted=0,test_threshold_fitting=0,user_recording_access=0,
        interpretation="Weighting limits influence of uncertain pseudo-targets; it does not validate or repair anatomical coordinates",
        readiness="Suitable for bounded robust proxy learning with independent image diagnostics and unweighted results also reported")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");args=parser.parse_args()
    if not args.prepare:
        parser.error("Use --prepare to freeze train-only policy and audit public inputs")
    rows=read(BUNDLE/"MANIFEST.json");assert len(rows)==61
    adapter=load_adapter()
    policy_path=UNIT/"TEACHER_POLICY.json"
    if policy_path.exists():
        policy=read(policy_path)
        assert policy["implementation_sha256"]==sha(__file__),"Policy implementation changed; do not silently refit"
    else:
        policy=fit_policy(rows,adapter);write_new(policy_path,policy)
    result=audit(rows,adapter,policy);write_new(UNIT/"TEACHER_AUDIT.json",result)
    print(json.dumps({"status":result["status"],"clips":result["clips"],"totals":result["totals"],"thresholds":policy["thresholds"]},ensure_ascii=False))


if __name__=="__main__":
    main()
