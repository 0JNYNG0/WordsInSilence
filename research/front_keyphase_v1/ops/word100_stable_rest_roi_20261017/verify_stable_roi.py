"""Independent stable preparation/causal wrist guard audit. No detector/model run.

The expected preparation and guard state machines below are implemented directly
from SCOPE, without using the producer's decision helpers. The prior independent
reviewer's 280-channel arithmetic is reused, never the producer feature builder.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
U = Path(__file__).resolve().parent
LAB = U.parents[1]
PRIOR = U.with_name("word100_wrist_roi_pair_20261017")
ROI = U.with_name("word100_fullframe_roi_pair_20261016")
sys.path[:0] = [str(LAB / "src"), str(ROI), str(U)]
from wis_lab.core.observation import Observation, empty_frames
from roi_adapter import apply_roi, build_roi_channels
import stable_rest as producer

spec = importlib.util.spec_from_file_location("prior_independent_roi_verifier", PRIOR / "verify_wrist_pair.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def eq(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    assert np.array_equal(a, b, equal_nan=True), "array mismatch"


def down(s, e, w, scale):
    return bool(e[1] - s[1] >= .2 * scale and w[1] - e[1] >= .2 * scale
                and abs(w[0] - e[0]) <= .5 * (w[1] - e[1]))


def calibration_reference(ob, width, height):
    times = ob.timestamp_us
    if not len(times):
        return {"status": "NOT_CALIBRATED", "selected_frame_indices": []}
    last = [i for i, t in enumerate(times) if int(t - times[0]) <= 1000000]
    for end in last:
        for start in range(end + 1):
            ids = list(range(start, end + 1))
            if len(ids) < 8 or int(times[end] - times[start]) < 300000:
                continue
            if np.any(np.diff(times[ids]) > 100000):
                continue
            points = ob.xy_px[ids].astype(np.float64)
            required = [11, 12, 13, 14, 33, 54]
            if not (ob.observed[np.ix_(ids, required)] & ob.usable[np.ix_(ids, required)]).all():
                continue
            if not np.isfinite(points[:, required]).all():
                continue
            sw = float(np.median([np.linalg.norm(p[11] - p[12]) for p in points]))
            if not np.isfinite(sw) or sw <= 0:
                continue
            if not all(down(p[s], p[e], p[w], sw) for p in points
                       for s, e, w in ((11, 13, 33), (12, 14, 54))):
                continue
            stable_ids = [11, 12, 33, 54]
            med = np.median(points[:, stable_ids], axis=0)
            if any(np.linalg.norm(p - med[j]) > .05 * sw
                   for row in points[:, stable_ids] for j, p in enumerate(row)):
                continue
            wrists = {name: np.median(points[:, w], axis=0) for name, w in (("left", 33), ("right", 54))}
            center = np.median(np.stack([(p[11] + p[12]) / 2 for p in points]), axis=0)
            bottom = math.floor(min(w[1] for w in wrists.values()) - .02 * sw)
            if not 0 < bottom <= height:
                continue
            return {"status": "CALIBRATED", "selected_frame_indices": ids,
                    "selected_frame_ids": ob.frame_id[ids].tolist(), "selected_timestamp_us": times[ids].tolist(),
                    "shoulder_width": sw, "median_shoulder_center": center.tolist(),
                    "wrist_anchors_normalized": {k: ((v-center)/sw).tolist() for k, v in wrists.items()},
                    "bottom": bottom, "box_source_pixels": [0, 0, width, bottom]}
    return {"status": "NOT_CALIBRATED", "selected_frame_indices": []}


def check_calibration(actual, expected):
    for k, v in expected.items():
        assert actual[k] == v, ("calibration", k, actual.get(k), v)


def guard_reference(ob, cal):
    out = copy.deepcopy(ob)
    suppress = np.zeros((ob.n_frames, 2), bool)
    dwell = np.zeros((ob.n_frames, 2), np.int64)
    state = [None, None]
    if cal["status"] != "CALIBRATED":
        return out, suppress, dwell
    bottom = cal["bottom"]
    fixed = cal["shoulder_width"]
    for f, now in enumerate(ob.timestamp_us):
        for side, (name, s, e, w) in enumerate((("left", 11, 13, 33), ("right", 12, 14, 54))):
            p = ob.xy_px[f].astype(np.float64)
            required = [11, 12, e, w]
            ok = (ob.observed[f, required] & ob.usable[f, required]).all() and np.isfinite(p[required]).all()
            ok = ok and not ob.observed[f, w+1:w+21].any()
            if not ok:
                state[side] = None
                continue
            sw = float(np.linalg.norm(p[11]-p[12]))
            if not np.isfinite(sw) or sw <= 0:
                state[side] = None
                continue
            center = (p[11]+p[12])/2
            wr, el = (p[w]-center)/sw, (p[e]-center)/sw
            ok = (0 < bottom-p[w, 1] <= .03*fixed
                  and np.linalg.norm(wr-np.array(cal["wrist_anchors_normalized"][name])) <= .08
                  and down(p[s], p[e], p[w], sw))
            if not ok:
                state[side] = None
                continue
            previous = state[side]
            if (previous is None or int(now-previous[1]) > 100000
                    or np.linalg.norm(wr-previous[2]) > .02 or np.linalg.norm(el-previous[3]) > .02):
                previous = [int(now), int(now), wr, el]
                state[side] = previous
            previous[1] = int(now)
            dwell[f, side] = int(now)-previous[0]
            if dwell[f, side] >= 300000:
                suppress[f, side] = True
                out.observed[f, w] = out.usable[f, w] = False
                out.xy_px[f, w] = np.nan
                out.conf[f, w] = np.nan
                out.hand_score[f, side] = np.nan
    return out, suppress, dwell


def equal_observations(a, b):
    for k in ("xy_px", "observed", "usable", "conf", "hand_score", "frame_id", "timestamp_us"):
        eq(getattr(a, k), getattr(b, k))


def verify_guard(ob, cal, actual=None, trace=None):
    expected, suppress, dwell = guard_reference(ob, cal)
    if actual is None:
        actual, trace = producer.suppress_rest_fragments(ob, cal)
    equal_observations(actual, expected)
    assert len(trace["frames"]) == ob.n_frames
    for f, row in enumerate(trace["frames"]):
        assert row["frame_id"] == int(ob.frame_id[f]) and row["timestamp_us"] == int(ob.timestamp_us[f])
        for side, label in enumerate(("left", "right")):
            assert row[label]["suppressed"] == bool(suppress[f, side]), (f, label, "suppression")
            assert row[label]["dwell_us"] == int(dwell[f, side]), (f, label, "dwell")
    assert trace["suppressed_frame_counts"] == dict(left=int(suppress[:, 0].sum()), right=int(suppress[:, 1].sum()))
    assert not (actual.observed & ~ob.observed).any()
    assert not (actual.usable & ~ob.usable).any()
    lost = ob.observed & ~actual.observed
    assert not lost[:, [i for i in range(75) if i not in (33, 54)]].any()
    for f, side in zip(*np.where(suppress)):
        w = 33+21*side
        assert not ob.observed[f, w+1:w+21].any()
        assert 0 < cal["bottom"]-ob.xy_px[f, w, 1] <= .03*cal["shoulder_width"]
    return actual, suppress, dwell


def fixture(frames=30):
    a = empty_frames(frames)
    a["frame_id"][:] = np.arange(frames)*2+7
    a["timestamp_us"][:] = 17000+np.arange(frames)*40000
    positions = {0:(400,100), 11:(300,200), 12:(500,200), 13:(300,350), 14:(500,350),
                 15:(300,580), 16:(500,580), 23:(350,500), 24:(450,500), 33:(300,600), 54:(500,600)}
    for w in (33, 54):
        for j in range(1,21):
            positions[w+j] = (300 if w == 33 else 500, 610+j)
    for p, xy in positions.items():
        a["xy_px"][:,p] = xy
        a["observed"][:,p] = a["usable"][:,p] = True
        a["conf"][:,p] = .9
    a["hand_score"][:] = .95
    return Observation(**a, sidecar={"clip_id":"independent_synthetic", "fps":25.0})


def missing(ob, frames, points):
    for f in np.atleast_1d(frames):
        for p in np.atleast_1d(points):
            ob.xy_px[f,p] = np.nan
            ob.observed[f,p] = ob.usable[f,p] = False
            ob.conf[f,p] = np.nan


def prefix(ob, stop):
    ob = copy.deepcopy(ob)
    for key in ("xy_px", "observed", "usable", "conf", "hand_score", "frame_id", "timestamp_us"):
        setattr(ob,key,getattr(ob,key)[:stop].copy())
    return ob


def synthetic_checks():
    checks = []
    def calibrated_case(name, ob, expected_start=None, fail=False):
        before = copy.deepcopy(ob)
        c = producer.calibrate_rest(ob,800,800)
        check_calibration(c,calibration_reference(ob,800,800))
        equal_observations(ob,before)
        assert (c["status"] == "NOT_CALIBRATED") == fail, name
        if expected_start is not None:
            assert c["selected_frame_indices"][0] == expected_start, (name,c)
        checks.append(name)
        return c

    base = fixture()
    cal = calibrated_case("stable_window_earliest_confirmation_and_start",base,0)
    assert cal["selected_frame_indices"] == list(range(9)) and cal["bottom"] == 596
    a=fixture(); a.xy_px[:5,[33,54],1] = np.array([500,520,540,560,580])[:,None]
    calibrated_case("transient_initial_lowering_not_stable",a,5)
    a=fixture(); missing(a,[4],[33]); calibrated_case("missing_wrist_breaks_initial_window",a,5)
    a=fixture(); a.timestamp_us[4:] += 100001; calibrated_case("initial_time_gap_not_accumulated",a,4)
    a=fixture(); a.xy_px[:,[33,54],1] += np.arange(a.n_frames)[:,None]*4
    calibrated_case("continually_unstable_initial_wrists_fail_without_fallback",a,fail=True)
    a=fixture(); a.xy_px[:,[13,14],1]=590
    calibrated_case("stationary_low_hand_not_arms_down",a,fail=True)
    a=fixture(); missing(a,range(a.n_frames),[33,54]); calibrated_case("pose_wrists_cannot_replace_hand_wrists",a,fail=True)
    a=fixture(60); a.xy_px[:26,[33,54],1]=300
    calibrated_case("stable_pose_after_first_second_not_used",a,fail=True)
    a=fixture(); a.usable[:,13]=False; calibrated_case("unusable_elbow_rejects_calibration",a,fail=True)
    a=fixture(); a.xy_px[:,12]=a.xy_px[:,11]; calibrated_case("zero_shoulder_scale_rejected",a,fail=True)
    calibrated_case("empty_observation_has_no_fallback",fixture(0),fail=True)
    a=fixture(); a.xy_px[:,[11,12],0]+=np.arange(a.n_frames)[:,None]*4
    calibrated_case("moving_shoulders_reject_initial_window",a,fail=True)
    a=fixture(); a.xy_px[:,33,0]=500; calibrated_case("nonvertical_forearm_rejects_initial_window",a,fail=True)

    def view(n=20):
        v=fixture(n); v.xy_px[:,[33,54],1]=595
        return apply_roi(v,cal["box_source_pixels"])
    def guarded_case(name, ob, zero=False):
        before=copy.deepcopy(ob)
        out, suppressed, dwell=verify_guard(ob,cal)
        equal_observations(ob,before)
        assert not any(np.shares_memory(getattr(ob,k),getattr(out,k)) for k in ("xy_px","observed","usable","frame_id","timestamp_us"))
        if zero: assert not suppressed.any(),name
        checks.append(name)
        return out,suppressed,dwell
    out,sp,dw=guarded_case("stable_wrist_fragment_suppressed_only_after_causal_dwell",view())
    assert not sp[:8].any() and sp[8:].all()
    guarded_case("brief_lowered_fragment_preserved",view(7),True)
    a=view(); a.xy_px[4:,33,0]+=5
    _,s,d=guarded_case("wrist_motion_resets_duration",a); assert d[4,0]==0 and not s[4:12,0].any()
    a=view(); a.xy_px[4:,13,0]+=5
    _,s,d=guarded_case("elbow_motion_resets_duration",a); assert d[4,0]==0 and not s[4:12,0].any()
    a=view(); a.timestamp_us[5:]+=100001
    _,s,d=guarded_case("guard_large_time_gap_resets_duration",a); assert d[5,0]==0 and not s[5:13].any()
    a=view(); missing(a,[4],[33])
    _,s,d=guarded_case("missing_wrist_then_reentry_requires_new_dwell",a); assert d[5,0]==0 and not s[5:13,0].any()
    a=view(); missing(a,[4],[11])
    _,s,d=guarded_case("missing_shoulder_resets_both_sides",a); assert (d[5]==0).all() and not s[5:13].any()
    a=view(); a.xy_px[:,34]=(300,500); a.observed[:,34]=True; a.usable[:,34]=False
    _,s,_=guarded_case("observed_unusable_finger_protects_hand",a); assert not s[:,0].any() and s[8:,1].all()
    a=view(); a.xy_px[:,34]=(300,500); a.observed[:,34]=a.usable[:,34]=True
    _,s,_=guarded_case("interior_finger_is_never_deleted",a); assert not s[:,0].any()
    a=view(); a.xy_px[:,[33,54],1]=585
    guarded_case("stationary_interior_wrist_outside_band_preserved",a,True)
    a=view(); a.xy_px[:,[33,54],0]+=20
    guarded_case("stationary_low_nonanchor_hand_preserved",a,True)
    a=view(); a.xy_px[:,[13,14],1]=592
    guarded_case("nonneutral_low_stationary_geometry_preserved",a,True)
    a=view(); a.usable[:,[33,54]]=False
    guarded_case("unusable_wrists_never_promoted_or_suppressed",a,True)
    a=view(); missing(a,[10],[33]); _,s,d=guarded_case("absence_after_suppression_resets_without_hallucination",a)
    assert not s[10:19,0].any() and d[11,0]==0
    a=view(); a.xy_px[4,34]=(300,500); a.observed[4,34]=True
    _,s,d=guarded_case("one_finger_reentry_resets_wrist_history",a); assert d[5,0]==0 and not s[5:13,0].any()
    a=view(); failed=producer.calibrate_rest(fixture(0),800,800)
    r,_,_=verify_guard(a,failed); equal_observations(a,r); checks.append("not_calibrated_guard_is_explicit_identity")
    a=view(); allout,_,_=verify_guard(a,cal)
    for stop in (1,7,9,13,20):
        part,_,_=verify_guard(prefix(a,stop),cal)
        equal_observations(part,prefix(allout,stop))
    checks.append("prefix_causality_no_lookahead_or_backdating")
    actual=build_roi_channels(allout)
    expected,_,_=prior.independent_channels(allout)
    eq(actual["x"],expected); checks.append("synthetic_280channels_recomputed_after_guard")
    for bad,label in ((base,"unmasked_guard_input_rejected"),):
        try: producer.suppress_rest_fragments(bad,cal)
        except ValueError: checks.append(label)
        else: raise AssertionError(label)
    a=view(); a.xy_px[0,33,1]=cal["bottom"]
    try: producer.suppress_rest_fragments(a,cal)
    except ValueError: checks.append("tampered_roi_boundary_rejected")
    else: raise AssertionError("tampered_roi_boundary_rejected")
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--synthetic-only",action="store_true")
    args=parser.parse_args()
    start=time.perf_counter()
    checks=synthetic_checks()
    if args.synthetic_only:
        print(json.dumps({"status":"PASS","synthetic_checks":len(checks),"checks":checks},ensure_ascii=False,indent=2))
        return
    plan=read(U/"PLAN.json")
    comparison=read(U/"COMPARISON.json")
    capture=read(U/"CAPTURE_RESULT.json")
    captures={r["clip_id"]:r for r in capture["rows"]}
    pins=plan.get("pins",{})
    for path,digest in pins.items(): assert sha(path)==digest,path
    oldplan=read(PRIOR/"PLAN.json")
    assert [i["clip_id"] for i in plan["items"]]==[i["clip_id"] for i in oldplan["items"]]
    assignment_spec=importlib.util.spec_from_file_location("independent_stable_joint_v3",U.with_name("word100_hand_assignment_joint_20261016")/"assignment.py")
    assigner=importlib.util.module_from_spec(assignment_spec)
    sys.modules[assignment_spec.name]=assigner
    assignment_spec.loader.exec_module(assigner)
    contract=read(LAB/"config/contract.json")
    savedrows={r["clip_id"]:r for r in comparison["rows"]}
    pairs={"full_boundary":("wrist_full","stable_full_raw"),"sensor_boundary":("wrist_sensor","stable_sensor_raw"),
           "full_guard":("stable_full_raw","stable_full_guarded"),"sensor_guard":("stable_sensor_raw","stable_sensor_guarded"),
           "full_combined":("wrist_full","stable_full_guarded"),"sensor_combined":("wrist_sensor","stable_sensor_guarded")}
    audit=[]; values=frames=removed=interior_removed=hidden_points=raw_hands=0
    recomputed=[]
    for item in plan["items"]:
        cid=item["clip_id"]
        assert item["source_role"]=="train_candidate" and item["signer_id"] in (1,2,3)
        native=Observation.load(item["cached"]["native"]["observation"])
        cal=read(U/"outputs"/cid/"calibration.json")
        expected=calibration_reference(native,*item["original_size"])
        check_calibration(cal,expected)
        if cal["status"]!="CALIBRATED":
            assert cid not in captures and cid not in savedrows
            audit.append({"clip_id":cid,"calibration_status":cal["status"],"not_evaluated_without_fallback":True})
            continue
        rec=captures[cid]
        assert rec["sensor_box"]==cal["box_source_pixels"]==item["stable_box"]
        assert rec["plan_sha256"]==sha(U/"PLAN.json")
        assert rec["reused"]==(item["stable_box"]==item["wrist_box"])
        for path,digest in rec["outputs"].items(): assert sha(path)==digest,path
        sensor_path=Path(rec["observation"])
        sensor=Observation.load(str(sensor_path))
        assert sensor.sidecar["source_bbox_px"]==item["stable_box"]
        nf,nr=prior.replay_check(sensor_path.parent,sensor_path,sensor_path.parent/"assignment.npz",assigner.HandAssigner,contract,item["stable_box"][2:])
        assert nf==native.n_frames
        raw_hands+=nr
        saved=savedrows[cid]
        allobs={name:Observation.load(path) for name,path in item["old_models"].items()}
        checked_metrics={}
        record={"clip_id":cid,"label":item["label"],"calibration_status":cal["status"],"selected_frames":cal["selected_frame_indices"],"bottom":cal["bottom"],"conditions":{}}
        for mode in ("full","sensor"):
            rawdir=U/"outputs"/cid/f"stable_{mode}_raw"
            guarddir=U/"outputs"/cid/f"stable_{mode}_guarded"
            raw=Observation.load(str(rawdir/"model_input.npz"))
            guarded=Observation.load(str(guarddir/"model_input.npz"))
            tr=read(guarddir/"guardtrace.json")
            source=native if mode=="full" else sensor
            mask=prior.mask_reference(source,cal["box_source_pixels"],(0,0))
            for key,expected_array in zip(("xy_px","observed","usable","conf","hand_score"),mask[:5]):
                eq(getattr(raw,key),expected_array)
            eq(raw.timestamp_us,native.timestamp_us); eq(raw.frame_id,native.frame_id)
            _,sp,dw=verify_guard(raw,cal,guarded,tr)
            removed+=int(sp.sum())
            interior_removed+=int(((raw.observed & ~guarded.observed)&(raw.xy_px[...,1]<cal["bottom"]-.03*cal["shoulder_width"])).sum())
            row={"removed_wrist_points":int(sp.sum()),"suppressed_counts":tr["suppressed_frame_counts"],"raw":prior.extra_metrics_reference(raw),"guarded":prior.extra_metrics_reference(guarded)}
            allobs[f"stable_{mode}_raw"]=raw
            allobs[f"stable_{mode}_guarded"]=guarded
            changed=copy.deepcopy(source)
            hidden=~mask[5]&source.observed
            changed.xy_px[hidden]=(-100000,-200000)
            changed.conf[hidden]=.123
            changed_view=apply_roi(changed,cal["box_source_pixels"])
            equal_observations(changed_view,raw)
            changed_guard,_=producer.suppress_rest_fragments(changed_view,cal)
            equal_observations(changed_guard,guarded)
            eq(build_roi_channels(changed_guard)["x"],prior.independent_channels(guarded)[0])
            hidden_points+=int(hidden.sum())
            record["conditions"][mode]=row
        for name,ob in allobs.items():
            directory=Path(item["old_models"][name]).parent if name in item["old_models"] else U/"outputs"/cid/name
            with np.load(directory/"channels.npz",allow_pickle=False) as z:
                channels={k:z[k] for k in z.files}
            x,_,_=prior.independent_channels(ob)
            eq(x,channels["x"]); eq(channels["timestamp_us"],ob.timestamp_us); eq(channels["frame_id"],ob.frame_id)
            values+=x.size; frames+=ob.n_frames
            checked_metrics[name]=prior.extra_metrics_reference(ob)
            assert checked_metrics[name]==saved["metrics"][name],(cid,name,"metrics")
        checked_pairs={name:prior.diff_reference(allobs[a],allobs[b]) for name,(a,b) in pairs.items()}
        assert checked_pairs==saved["pairs"],(cid,"paired metrics")
        recomputed.append({"clip_id":cid,"label":item["label"],"metrics":checked_metrics,"pairs":checked_pairs})
        audit.append(record)
        print(json.dumps({"verified_clip":cid,"bottom":cal["bottom"]},ensure_ascii=False),flush=True)
    assert interior_removed==0
    aggregate={pair:{metric:{field:sum(r["pairs"][pair][side][metric][field] for r in recomputed for side in ("left","right"))
                            for field in ("old","new","gain","loss")} for metric in ("any","palm5","all21","points")} for pair in pairs}
    assert aggregate==comparison["aggregate"]
    ends={}
    for mode in plan["scope"]["conditions"]:
        ends[mode]={}
        for win in ("first5","last5","first_half_second","last_half_second"):
            rr=[dict(clip_id=r["clip_id"],label=r["label"],side=side,**r["metrics"][mode]["end_windows"][win]["hands"][side])
                for r in recomputed for side in ("left","right") if r["metrics"][mode]["end_windows"][win]["hands"][side]["points"]>0]
            ends[mode][win]=dict(total_hand_point_frames=sum(q["points"] for q in rr),nonempty_clip_sides=len(rr),rows=rr)
    assert ends==comparison["end_window_audit"]
    assert len(recomputed)==comparison["calibrated_clips"]==capture["conditions"]
    assert 6*len(recomputed)==comparison["conditions"]
    assert comparison["unique_video_frames"]*6==frames
    assert comparison["training"]==comparison["classifier_inference"]==comparison["production_changes"]==capture["training"]==0
    fresh=[r for r in capture["rows"] if not r["reused"]]
    assert len(set(r["detector_instance"] for r in fresh))==len(fresh)==capture["new_capture"]
    for row in comparison.get("rows",[]):
        for path,digest in row.get("outputs",{}).items(): assert sha(path)==digest,path
    for path,digest in pins.items(): assert sha(path)==digest,path
    result={"status":"PASS","reviewer":"stable_roi_verify","producer":"stable_roi_impl/root",
            "independent_of":"producer calibration and guard decisions; prior independent 280-channel formulas reused",
            "synthetic_checks":len(checks),"checks":checks,"clips":len(audit),"condition_frames":frames,
            "conditions":comparison["conditions"],"unique_video_frames":comparison["unique_video_frames"],
            "independently_recomputed_280channel_values":values,"removed_hand_wrist_point_frames":removed,
            "removed_points_interior_to_bottom_band":interior_removed,"protected_pins_checked":len(pins),
            "sensor_detector_conditions_replayed":len(captures),"sensor_raw_hands_replayed":raw_hands,
            "hidden_source_points_mutated":hidden_points,"independently_recomputed_aggregate":aggregate,
            "independently_recomputed_end_window_audit":ends,
            "code_sha256":{"producer":sha(U/"stable_rest.py"),"verifier":sha(Path(__file__))},
            "limitations":["Engineering validation only; no hand ground truth or classifier accuracy claim",
                           "Full-source initial calibration is offline preparation, not live cropped-camera calibration",
                           "Only isolated hand wrist is eligible; pose wrist channels are intentionally preserved",
                           "A stationary interior sign resembling the calibrated pose cannot be semantically identified by geometry alone; only narrow wrist fragments may be removed",
                           "No training, detector execution, camera access, production change or user data in this verifier"],
            "elapsed_seconds":time.perf_counter()-start,"rows":audit}
    destination=U/"INDEPENDENT_VERIFICATION.json"
    assert not destination.exists(),"completed independent report is immutable"
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","checks","limitations")},ensure_ascii=False))


if __name__=="__main__":
    main()
