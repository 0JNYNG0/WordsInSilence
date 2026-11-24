"""Independently verify saved image-recovery geometry; no detector/producer import."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import math
import traceback
import numpy as np

UNIT = Path(__file__).resolve().parents[1]
PALM = (0,5,9,13,17)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k].copy() for k in archive.files}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks=0
        self.failures=[]
        self.counts=Counter()

    def check(self,ok,label,detail=None):
        self.checks+=1
        if not ok:self.failures.append({"check":label,"detail":detail})

    def eq(self,actual,expected,label):
        self.check(actual==expected,label,None if actual==expected else {"actual":actual,"expected":expected})

    def array(self,actual,expected,label):
        self.check(np.array_equal(actual,expected,equal_nan=True),label)


def finite_inside(p,box):
    return all(math.isfinite(float(v)) for v in p) and box[0]<=p[0]<box[2] and box[1]<=p[1]<box[3]


def distance(a,b):
    return math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))


def geometry_context(pose,usable,w,h):
    good=[bool(usable[i]) and finite_inside(p,[0,0,w,h]) for i,p in enumerate(pose)]
    scale=distance(pose[11],pose[12]) if good[11] and good[12] else 0.
    return good,scale


def identify_arm(hand,pose,good,scale,p):
    if scale<=1 or not np.isfinite(hand[0]).all():return -1
    ds=[distance(hand[0],pose[wrist])/scale if good[elbow] and good[wrist] else math.inf
        for elbow,wrist in ((13,15),(14,16))]
    side=0 if ds[0]<=ds[1] else 1
    return side if ds[side]<=p["arm_gate"] and ds[1-side]-ds[side]>=p["arm_margin"] else -1


def box_overlap(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():return 0.
    amin,amax=np.min(a,axis=0),np.max(a,axis=0)
    bmin,bmax=np.min(b,axis=0),np.max(b,axis=0)
    area=min(float(np.prod(amax-amin)),float(np.prod(bmax-bmin)))
    extent=np.maximum(0,np.minimum(amax,bmax)-np.maximum(amin,bmin))
    return float(np.prod(extent))/area if area>0 else 0.


def classify_relation(a,b,sa,sb,scale,p):
    if scale<=1:return "unknown"
    if sa>=0 and sb>=0 and sa!=sb:return "different_current_arms"
    if not np.isfinite(a[list(PALM)]).all() or not np.isfinite(b[list(PALM)]).all():return "unknown"
    wrist=distance(a[0],b[0])/scale
    med=sorted(distance(a[j],b[j]) for j in PALM)[2]/scale
    if wrist<=p["duplicate_wrist"] and med<=p["duplicate_median"]:return "same_geometry"
    overlap=box_overlap(a,b)
    if sa>=0 and sa==sb and overlap>.25:return "same_current_arm_overlap"
    if overlap>=p["ambiguous_overlap"] and wrist<=p["ambiguous_wrist"]:return "ambiguous_overlap"
    return "separate"


def expected_decisions(full,crops,raw_hands,base_sides,pose,usable,w,h,params):
    good,scale=geometry_context(pose,usable,w,h)
    old_arms=[identify_arm(hand,pose,good,scale,params) for hand in raw_hands]
    decisions=[]
    for index,det in enumerate(full):
        hand=np.asarray(det["xy"])
        side=identify_arm(hand,pose,good,scale,params) if hand.shape==(21,2) else -1
        row={"index":index,"side":side,"accepted":False,"reason":"current_arm_uncertain",
             "matched_old":[],"relations":[],"joint_supported":[False]*21,"crop_index":None}
        if side<0:
            decisions.append(row);continue
        if not all(finite_inside(hand[j],[0,0,w,h]) for j in PALM):
            row["reason"]="palm_outside_source";decisions.append(row);continue
        relations=[classify_relation(hand,np.asarray(old),side,sa,scale,params) for old,sa in zip(raw_hands,old_arms)]
        row["relations"]=relations
        matches=[j for j,r in enumerate(relations) if r.startswith("same_")]
        row["matched_old"]=matches
        if matches:
            row["reason"]="existing_identity_conflict" if any(base_sides[j]!=side for j in matches) else "confirm_existing_hand"
        elif "ambiguous_overlap" in relations:
            row["reason"]="unresolved_existing_overlap"
        elif side in base_sides:
            row["reason"]="existing_side_preserved"
        else:
            agreements=[]
            for j,crop in enumerate(crops):
                other=np.asarray(crop["xy"]);box=crop["box"]
                if crop["requested_side"]!=side or identify_arm(other,pose,good,scale,params)!=side:continue
                valid=[finite_inside(hand[k],[0,0,w,h]) and finite_inside(other[k],[0,0,w,h])
                       and finite_inside(other[k],box) and distance(hand[k],other[k])<=params["point_agreement"]*scale for k in range(21)]
                if all(valid[k] for k in PALM):agreements.append((j,valid))
            if len(agreements)==1:
                row.update(accepted=True,reason="new_hand_current_arm_and_two_image_scales",
                           crop_index=agreements[0][0],joint_supported=agreements[0][1])
            else:row["reason"]="image_scales_disagree_or_ambiguous"
        decisions.append(row)
    for side in (0,1):
        group=[r for r in decisions if r["accepted"] and r["side"]==side]
        if len(group)>1:
            for row in group:row.update(accepted=False,reason="multiple_new_same_side")
    return decisions


def eligibility(native_observed,pose,usable,box,w,h,base_sides=()):
    good,scale=geometry_context(pose,usable,w,h)
    result=[]
    for side,(elbow,wrist) in enumerate(((13,15),(14,16))):
        count=int(np.count_nonzero(native_observed[33+21*side:54+21*side]))
        reason=("EXISTING_NATIVE_HAND_PRESERVED" if count else
                "EXISTING_RAW_ASSIGNED_SIDE_PRESERVED" if side in base_sides else
                "CURRENT_POSE_ARM_UNSUPPORTED" if scale<=1 or not(good[elbow] and good[wrist]) else
                "POSE_WRIST_OUTSIDE_FIXED_ROI" if not finite_inside(pose[wrist],box) else
                "ELIGIBLE_NATIVE_SIDE_UNOBSERVED_CURRENT_ARM_IN_ROI")
        result.append({"side":side,"eligible":reason.startswith("ELIGIBLE_"),"reason":reason,"native_observed_points":count})
    return result


def verify():
    a=Audit()
    prep=read(UNIT/"RECOVERY_PREPARED.json")
    frozen=read(UNIT/"RECOVERY_FROZEN.json")
    result=read(UNIT/"RECOVERY_RESULT_MANIFEST.json")
    params=prep["method"]["parameters"]
    by_alias={r["alias"]:r for r in result["records"]}
    point_threshold=read(UNIT.parent.parent/"config/contract.json")["usable_rule"]["hand_score_threshold"]
    a.eq(len(by_alias),8,"eight fixed public clips")
    seen=[]
    for clip in prep["clips"]:
        alias=clip["alias"];files=clip["files"];saved=by_alias[alias]
        a.eq(sha(saved["output"]),saved["output_sha256"],alias+":output hash")
        a.eq(sha(saved["ledger"]),saved["ledger_sha256"],alias+":ledger hash")
        native=arrays(files["native"]);raw=arrays(files["raw"]);sides=arrays(files["assignment"])["sides"]
        old_input=arrays(files["input"])
        output=arrays(saved["output"]);ledger=read(saved["ledger"])
        a.eq(output["source_index"].tolist(),clip["indices"],alias+":fixed source indices")
        a.eq(len(ledger["frames"]),len(clip["indices"]),alias+":every fixed frame ledger")
        for out_index,row in enumerate(ledger["frames"]):
            i=row["source_index"];fid=row["frame_id"];prefix=f"{alias}:f{fid}"
            seen.append((alias,fid))
            a.eq(fid,int(native["frame_id"][i]),prefix+":native identity")
            a.eq(row["timestamp_us"],int(native["timestamp_us"][i]),prefix+":native PTS")
            pose=raw["pose_xy_px"][i];usable=raw["pose_usable"][i]
            w,h=int(raw["width"][i]),int(raw["height"][i])
            n=int(raw["raw_count"][i]);base_sides=sides[i,:n].tolist()
            eligible=eligibility(native["observed"][i],pose,usable,clip["box"],w,h,base_sides)
            a.eq(row["eligibility"],eligible,prefix+":eligibility independent")
            allowed={r["side"] for r in eligible if r["eligible"]}
            for side in allowed:a.check(side not in base_sides,prefix+":raw assigned side absent")
            expected_xy=native["xy_px"][i].copy();expected_observed=native["observed"][i].copy()
            expected_usable=native["usable"][i].copy();expected_score=native["hand_score"][i].copy()
            if not allowed:
                a.eq(row["status"],"SKIPPED_BY_FIXED_ELIGIBILITY",prefix+":skip preserved")
                for k in ("full","crop","decisions","additions"):a.eq(row[k],[],prefix+":no skip inference "+k)
                a.eq(row["full_calls"]+row["crop_calls"],0,prefix+":zero skipped calls")
                a.counts["skipped_frames"]+=1
            else:
                a.eq(row["status"],"NEW_IMAGE_CANDIDATE_EXECUTED",prefix+":executed once")
                a.eq(row["full_calls"],1,prefix+":one full IMAGE call")
                good,scale=geometry_context(pose,usable,w,h)
                boxes={}
                for side in allowed:
                    if side in base_sides:continue
                    x,y=pose[15 if side==0 else 16];d=scale*params["crop_half_shoulder"]
                    box=[max(0,math.floor(x-d)),max(0,math.floor(y-d)),min(w,math.ceil(x+d)),min(h,math.ceil(y+d))]
                    if min(box[2]-box[0],box[3]-box[1])>=32:boxes[str(side)]=box
                a.eq(row["boxes"],boxes,prefix+":fixed arm crop geometry")
                a.eq(row["crop_calls"],len(boxes),prefix+":one crop per eligible side")
                for crop in row["crop"]:a.eq(crop["box"],boxes[str(crop["requested_side"])],prefix+":crop coordinate origin")
                exp=expected_decisions(row["full"],row["crop"],raw["hand_xy_px"][i,:n],base_sides,pose,usable,w,h,params)
                a.eq(len(row["decisions"]),len(exp),prefix+":all full proposals retained")
                additions=[]
                for actual,e in zip(row["decisions"],exp):
                    for k,v in e.items():a.eq(actual[k],v,prefix+":independent merge "+k)
                    a.eq(actual["old_merge_accepted"],e["accepted"],prefix+":old acceptance retained")
                    apply=e["accepted"] and e["side"] in allowed
                    a.eq(actual["applied"],apply,prefix+":eligibility application")
                    if not apply:continue
                    side=e["side"];sl=slice(33+21*side,54+21*side)
                    full=row["full"][e["index"]];crop=row["crop"][e["crop_index"]]
                    score=min(full["score"],crop["score"])
                    points=np.asarray(full["xy"],dtype=np.float32)
                    support=np.asarray(e["joint_supported"],bool)&np.isfinite(points).all(axis=1)
                    us=support&(score>=point_threshold)
                    expected_xy[sl]=np.where(support[:,None],points,np.nan)
                    expected_observed[sl]=support;expected_usable[sl]=us;expected_score[side]=score
                    roi=sum(bool(us[j]) and finite_inside(points[j],clip["box"]) for j in range(21))
                    for k,v in {"actual_full_handedness_score":full["score"],"actual_crop_handedness_score":crop["score"],"stored_hand_score":score,"supported_points":int(support.sum()),"final_roi_usable_points":roi}.items():a.eq(actual[k],v,prefix+":actual measurement "+k)
                    additions.append({"side":side,"full_index":e["index"],"crop_index":e["crop_index"],"observed_points":int(support.sum()),"usable_points":int(us.sum()),"roi_usable_points":roi})
                    a.counts["added_native_hand_frames"]+=1
                    a.counts["added_roi_usable_joint_frames"]+=roi
                a.eq(row["additions"],additions,prefix+":all addition records")
                a.counts["new_full_image_calls"]+=row["full_calls"]
                a.counts["new_crop_image_calls"]+=row["crop_calls"]
                a.counts["executed_frames"]+=1
            expected_usable &= np.asarray([finite_inside(p,clip["box"]) for p in expected_xy])
            for k,expected in (("xy_px",expected_xy),("observed",expected_observed),("usable",expected_usable),("hand_score",expected_score)):
                a.array(output[k][out_index],expected,prefix+":exact isolated observation "+k)
        for k in ("frame_id","timestamp_us"):a.array(output[k],native[k][clip["indices"]],alias+":original "+k)
        a.array(output["baseline_x"],old_input["x"][clip["indices"]],alias+":baseline values exact source")
        a.array(output["baseline_mask"],old_input["mask"][clip["indices"]],alias+":baseline support exact source")
        a.array(output["newly_supported"],output["mask"]&~output["baseline_mask"],alias+":new support ledger")
        a.check(output["mask"][output["baseline_mask"]].all(),alias+":original valid support retained")
        a.array(output["x"][output["baseline_mask"]],output["baseline_x"][output["baseline_mask"]],alias+":original valid features unchanged")
        a.counts["new_supported_selected_feature_cells"]+=int(output["newly_supported"].sum())
        a.counts["preserved_original_selected_feature_cells"]+=int(output["baseline_mask"].sum())
        # Its new69 dependency derivation is independently verified by the data auditor;
        # this geometry verifier checks ledger-to-manifest addition and selected scope.
        a.counts["out_of_scope_derived_support_discarded"]+=ledger["outside_scope_derived_support_discarded"]
    a.eq(len(seen),303,"all303 source frame opportunities")
    a.eq(len(set(seen)),303,"unique303 source frames")
    for k,v in result["counts"].items():a.eq(a.counts[k],v,"manifest count "+k)
    for k in ("skipped_frames","executed_frames"):a.eq(a.counts[k],result[k],"manifest "+k)
    a.eq(a.counts["executed_frames"],prep["selection"]["eligible_frames"],"prepared eligible frame scope")
    a.eq(result["frozen_sha256"],sha(UNIT/"RECOVERY_FROZEN.json"),"recovery freeze pin")
    for p,h in frozen["pins"].items():a.eq(sha(p),h,"unchanged frozen pin "+Path(p).name)
    return a


def selftest():
    a=Audit()
    p={"arm_gate":.25,"arm_margin":.10,"point_agreement":.08,"duplicate_median":.10,
       "duplicate_wrist":.20,"ambiguous_overlap":.50,"ambiguous_wrist":.30}
    pose=np.zeros((33,2));pose[11]=[100,100];pose[12]=[200,100]
    pose[13]=[100,150];pose[15]=[100,200];pose[14]=[200,150];pose[16]=[200,200]
    usable=np.ones(33,bool);good,scale=geometry_context(pose,usable,400,400)
    hand=np.array([[100+j%5,200-j//5] for j in range(21)],float)
    other=hand+[100,0];full=[{"xy":hand.tolist(),"score":.8}];crop=[{"xy":hand.tolist(),"score":.9,"box":[40,140,160,260],"requested_side":0}]
    a.eq(identify_arm(hand,pose,good,scale,p),0,"clear current left arm")
    a.eq(classify_relation(hand,hand,0,0,scale,p),"same_geometry","duplicate geometry")
    a.eq(classify_relation(hand,hand,0,1,scale,p),"different_current_arms","crossing different current arms not box merged")
    base=expected_decisions(full,crop,[other],[1],pose,usable,400,400,p)
    a.eq(base[0]["accepted"],True,"full/crop new separate side")
    a.eq(base[0]["joint_supported"],[True]*21,"all same-frame corroborated joints")
    conflict=expected_decisions(full,crop,[hand],[-1],pose,usable,400,400,p)
    a.eq(conflict[0]["reason"],"existing_identity_conflict","unassigned raw remains duplicate blocker")
    a.eq(conflict[0]["accepted"],False,"no unassigned duplicate promotion")
    a.eq(expected_decisions(full,[],[],[],pose,usable,400,400,p)[0]["accepted"],False,"no crop corroboration rejected")
    tied=expected_decisions(full+full,crop,[],[],pose,usable,400,400,p)
    a.check(not any(r["accepted"] for r in tied),"multiple proposed same side ambiguous")
    native=np.zeros(75,bool);native[33]=True
    e=eligibility(native,pose,usable,[0,0,400,190],400,400)
    a.eq(e[0]["reason"],"EXISTING_NATIVE_HAND_PRESERVED","partial native preserved")
    a.eq(e[1]["reason"],"POSE_WRIST_OUTSIDE_FIXED_ROI","resting outside ROI not restored")
    a.eq(eligibility(np.zeros(75,bool),pose,usable,[0,0,400,400],400,400,[0])[0]["reason"],
         "EXISTING_RAW_ASSIGNED_SIDE_PRESERVED","raw assignment guard independent of native mask")
    return a


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--selftest",action="store_true");parser.add_argument("--output",type=Path);args=parser.parse_args()
    out=args.output or UNIT/"review"/("RECOVERY_MATH_SELFTEST.json" if args.selftest else "RECOVERY_MATH_VERIFICATION.json")
    if out.exists():raise FileExistsError(out)
    try:a=selftest() if args.selftest else verify()
    except Exception:
        a=Audit();a.check(False,"verifier exception",traceback.format_exc())
    result={"status":"PASS" if not a.failures else "FAIL","checks":a.checks,"failures":a.failures,"counts":dict(a.counts),
            "verifier_sha256":sha(__file__),"method":"Independent saved-coordinate arithmetic and original source invariants; no candidate/merge/detector imported",
            "division_of_review":"New69 feature dependency derivation and discarded outside-scope derivative counts are independently checked by semantic_data_audit; this verifier audits merge mathematics, native additions and aggregate counts.",
            "new_model_inference":0,"producer_reexecution":0,"new_DTW":0,"educational_accuracy":None,"independent_human_truth":False}
    with out.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"status":result["status"],"checks":a.checks,"failures":len(a.failures),"output":str(out)}))
    raise SystemExit(0 if not a.failures else 1)
