"""Research-only fixed-model/DTW measurement on the public 61-clip bundle.

No camera, user history, training, runtime replacement, or synthetic z. Estimates
remain explicit estimates. A separate experimental 280-channel view can test
their effect on frozen classifiers without changing any observation file.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

UNIT=Path(__file__).resolve().parent
LAB=UNIT.parents[1]
REPO=LAB.parents[1]
BUNDLE=UNIT.parent/"word100_temporal_public_bundle_20261025"
sys.dont_write_bytecode=True
for p in (LAB/"src",REPO/"research/sign_model_pilot_v1/src"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from pilot import data as D
from pilot.web_input import to_native_like,active_span
from wis_lab.core.observation import Observation
from wis_lab.core import features as F
from wis_lab.core.dtw import frame_cost_matrix,_optimal_path,MIN_PATH_EVIDENCE_FRACTION

SEEDS=(17,29,43)
CHAINS=((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20))
TRIPLES=np.array([(c[i-1],c[i],c[i+1]) for c in CHAINS for i in (1,2,3)])

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def new_json(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m
def split_for(s):return "train" if int(s)<=6 else "dev" if int(s)==7 else "test"

def check_plan():
    plan=read(UNIT/"DOWNSTREAM_PLAN.json")
    for p,h in plan["source_pins"].items():assert sha(p)==h,(p,"source changed")
    assert plan["production_change"] is False and plan["user_data_access"] is False
    return plan

def load_sources():
    adapter=module(BUNDLE/"load_inputs.py","downstream_safe_adapter")
    clips={}
    for row in read(BUNDLE/"MANIFEST.json"):
        assert sha(row["arrays"])==row["arrays_sha256"]
        with np.load(row["arrays"],allow_pickle=False) as z:
            base=adapter.from_arrays(z,"B0");image=adapter.from_arrays(z,"B1")
        assert np.array_equal(base["xy_px"][base["usable"]],image["xy_px"][base["usable"]])
        assert np.all(image["usable"][base["usable"]])
        native,present=to_native_like(base["xy_px"],base["usable"])
        span=active_span(present)
        assert span is not None,(row["clip_id"],"no baseline span")
        lo,hi=span;legacy=D.build_channels(native[lo:hi+1],present[lo:hi+1],base["timestamp_us"][lo:hi+1])
        clips[row["clip_id"]]=dict(row=row,B0=base,B1=image,span=(lo,hi),
            scales=dict(body=legacy["body_scale"],hands=legacy["hand_scales"]),legacy=legacy)
    assert len(clips)==61
    return clips

def fixed_channels(view,clip):
    """Exact frozen 280 layout, but B0 scales/span fixed across all conditions."""
    lo,hi=clip["span"];slc=slice(lo,hi+1)
    native,present=to_native_like(view["xy_px"],view["available"])
    xy,pr=D.to_common(native[slc],present[slc]);n=len(xy)
    shoulders=pr[:,D.L_SHOULDER]&pr[:,D.R_SHOULDER]
    origin=(xy[:,D.L_SHOULDER]+xy[:,D.R_SHOULDER])/2
    bv=pr&shoulders[:,None]
    br=np.zeros((n,51,2),np.float32)
    br[bv]=(xy[bv]-np.repeat(origin[:,None,:],51,axis=1)[bv])/clip["scales"]["body"]
    hr=np.zeros((n,42,2),np.float32);hv=np.zeros((n,42),np.float32)
    for side,hand in enumerate(D.HAND_SLICES):
        scale=clip["scales"]["hands"]["left" if side==0 else "right"]
        if not np.isfinite(scale):continue
        for j in range(21):
            good=pr[:,hand.start]&pr[:,hand.start+j]
            hr[good,side*21+j]=(xy[good,hand.start+j]-xy[good,hand.start])/scale
            hv[good,side*21+j]=1
    dt=np.zeros((n,1),np.float32)
    dt[1:,0]=np.diff(view["timestamp_us"][slc]).astype(np.float32)/1e6
    x=np.concatenate([br.reshape(n,102),hr.reshape(n,84),bv.astype(np.float32),hv,dt],axis=1).astype(np.float32)
    assert x.shape==(n,280) and np.isfinite(x).all()
    return x

def observed_view(payload):
    return dict(xy_px=payload["xy_px"],available=payload["usable"],
                observed=payload["usable"]&payload["observed"],estimated=np.zeros_like(payload["usable"]),
                timestamp_us=payload["timestamp_us"],frame_id=payload["frame_id"])

def candidate_view(clip,path):
    with np.load(path,allow_pickle=False) as z:
        v={k:z[k].copy() for k in ("xy_px","estimated","observed","usable_observed","timestamp_us","frame_id")}
    base=clip["B1"]
    for k in ("timestamp_us","frame_id"):assert np.array_equal(v[k],base[k]),(str(path),k)
    assert np.array_equal(v["observed"],base["observed"])
    assert np.array_equal(v["usable_observed"],base["usable"])
    assert not np.any(v["estimated"]&base["usable"])
    assert not v["estimated"][:,:33].any(),"Refiner must not modify body"
    assert np.array_equal(v["xy_px"][base["usable"]],base["xy_px"][base["usable"]])
    x0,y0,x1,y1=np.asarray(base["roi_box"]).reshape(-1)[:4]
    pts=v["xy_px"][v["estimated"]]
    assert np.isfinite(pts).all() and np.all((pts[:,0]>=x0)&(pts[:,0]<x1)&(pts[:,1]>=y0)&(pts[:,1]<y1))
    return dict(xy_px=v["xy_px"],available=base["usable"]|v["estimated"],
                observed=base["usable"]&base["observed"],estimated=v["estimated"],
                timestamp_us=v["timestamp_us"],frame_id=v["frame_id"])

def condition_views(clips,include_candidates):
    output={"B0":{k:observed_view(c["B0"]) for k,c in clips.items()},
            "B1":{k:observed_view(c["B1"]) for k,c in clips.items()}}
    if include_candidates:
        for condition in ("B2","B3"):
            for seed in SEEDS:
                name=f"{condition}_seed{seed}"
                folder=UNIT/"runs"/name/"natural_estimates"
                assert {p.stem for p in folder.glob("*.npz")}==set(clips),(name,"missing/extra clips")
                output[name]={cid:candidate_view(c,folder/(cid+".npz")) for cid,c in clips.items()}
    return output

def evidence(view,clip):
    xy=view["xy_px"];available=view["available"];observed=view["observed"];estimated=view["estimated"]
    n=len(xy);position=np.full((n,2,2),np.nan,np.float32)
    pv=np.zeros((n,2),bool);po=pv.copy();pe=pv.copy()
    angle=np.full((n,2,15),np.nan,np.float32);av=np.zeros((n,2,15),bool);ao=av.copy();ae=av.copy()
    b=clip["B0"];bodyok=b["usable"][:,11]&b["usable"][:,12]
    origin=(b["xy_px"][:,11]+b["xy_px"][:,12])/2;scale=clip["scales"]["body"]
    arm=np.full((n,2,2,2),np.nan,np.float32);arm_ok=np.zeros((n,2,2),bool)
    for side in (0,1):
        off=33+21*side;sl=slice(off,off+21)
        pv[:,side]=available[:,off]&bodyok
        po[:,side]=observed[:,off]&bodyok;pe[:,side]=estimated[:,off]&bodyok
        position[pv[:,side],side]=(xy[pv[:,side],off]-origin[pv[:,side]])/scale
        hand=xy[:,sl]
        u=hand[:,TRIPLES[:,0]]-hand[:,TRIPLES[:,1]];v=hand[:,TRIPLES[:,2]]-hand[:,TRIPLES[:,1]]
        nu=np.linalg.norm(u,axis=-1);nv=np.linalg.norm(v,axis=-1)
        valid=available[:,sl][:,TRIPLES].all(axis=-1)&(nu>1e-6)&(nv>1e-6)
        cosine=np.clip((u*v).sum(-1)/np.where(valid,nu*nv,1),-1,1)
        angle[:,side]=np.where(valid,np.degrees(np.arccos(cosine)),np.nan)
        av[:,side]=valid
        ao[:,side]=valid&observed[:,sl][:,TRIPLES].all(axis=-1)
        ae[:,side]=valid&estimated[:,sl][:,TRIPLES].any(axis=-1)
        for j,point in enumerate((13+side,15+side)):
            ok=b["usable"][:,point]&bodyok;arm_ok[:,side,j]=ok
            arm[ok,side,j]=(b["xy_px"][ok,point]-origin[ok])/scale
    return dict(position=position,position_valid=pv,position_observed=po,position_estimated=pe,
                angle=angle,angle_valid=av,angle_observed=ao,angle_estimated=ae,
                arm=arm,arm_valid=arm_ok)

def interval_list(flags,times):
    edges=np.diff(np.r_[False,flags,False].astype(np.int8))
    return [dict(first_index=int(a),last_index=int(b-1),start_us=int(times[a]),end_us=int(times[b-1]))
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]

def coverage(view,e):
    n=len(view["xy_px"])
    return dict(frames=n,potential_joint_slots=n*42,
        observed_joint_frames=int(view["observed"][:,33:].sum()),estimated_joint_frames=int(view["estimated"][:,33:].sum()),
        potential_angle_slots=n*30,observed_angle_slots=int(e["angle_observed"].sum()),
        estimated_angle_slots=int(e["angle_estimated"].sum()),
        per_hand=[dict(side=s,observed_wrist_frames=int(e["position_observed"][:,s].sum()),
                       estimated_wrist_frames=int(e["position_estimated"][:,s].sum()),
                       no_observed_wrist_intervals=interval_list(~e["position_observed"][:,s],view["timestamp_us"])) for s in (0,1)],
        interpretation="Potential slots are fixed accounting denominators, not required actions or errors; normal rest can be unobserved.")

def to_observation(payload):
    observed=payload["observed"]&payload["usable"]
    xy=payload["xy_px"].copy();xy[~observed]=np.nan
    conf=payload["conf"].copy();conf[~observed]=np.nan
    obs=Observation(xy_px=xy.astype(np.float32),timestamp_us=payload["timestamp_us"].copy(),
        frame_id=payload["frame_id"].copy(),observed=observed.copy(),usable=observed.copy(),
        conf=conf,hand_score=payload["hand_score"].copy())
    obs.validate();return obs

def build_paths(clips):
    # Alignment is built ONCE from B1 observed coordinates, never estimates.
    sampled={cid:F.uniform_time_sample(F.sequence_features(to_observation(c["B1"]),trim=False),96)
             for cid,c in clips.items()}
    pairs=[];counts=Counter()
    for cid,clip in clips.items():
        refs=[rid for rid,r in clips.items() if r["row"]["word_code"]==clip["row"]["word_code"]
              and int(r["row"]["signer_id"])<=6 and r["row"]["signer_id"]!=clip["row"]["signer_id"] and rid!=cid]
        counts[cid]=len(refs)
        for rid in refs:
            q,r=sampled[cid],sampled[rid]
            cost,support=frame_cost_matrix(q,r,hand_participation=False,one_sided_mode="flat")
            path=_optimal_path(cost,.2)
            row=dict(query_clip=cid,reference_clip=rid,word_code=clip["row"]["word_code"],
                     query_signer=clip["row"]["signer_id"],reference_signer=clips[rid]["row"]["signer_id"],
                     split=split_for(clip["row"]["signer_id"]))
            if not path:
                row.update(status="NO_PATH",query_indices=[],reference_indices=[])
            else:
                qi,ri=np.asarray(path).T;observed_support=float(support[qi,ri].mean())
                row.update(status="SUPPORTED" if observed_support>=MIN_PATH_EVIDENCE_FRACTION else "INSUFFICIENT_ALIGNMENT_EVIDENCE",
                    observed_alignment_support=observed_support,
                    query_indices=q["picked_rows"][qi].tolist(),reference_indices=r["picked_rows"][ri].tolist(),
                    path_pairs=len(path),baseline_alignment_cost=float(cost[qi,ri].mean()))
            pairs.append(row)
    return pairs,dict(counts)

def measured(values,mask):
    selected=values[mask]
    return dict(compared_slots=int(mask.sum()),mean=float(selected.mean()) if len(selected) else None,
                median=float(np.median(selected)) if len(selected) else None,
                p95=float(np.quantile(selected,.95)) if len(selected) else None)

def dtw_residual(pair,query,reference):
    identity={k:pair[k] for k in ("query_clip","reference_clip","word_code","query_signer","reference_signer","split","status")}
    if pair["status"]!="SUPPORTED":return identity|dict(measurements=None,reason="Observed alignment evidence insufficient; no low-distance success claim")
    qi=np.asarray(pair["query_indices"]);ri=np.asarray(pair["reference_indices"]);n=len(qi)
    position=np.linalg.norm(query["position"][qi]-reference["position"][ri],axis=-1)*100
    angle=np.abs(query["angle"][qi]-reference["angle"][ri])
    arm=np.linalg.norm(query["arm"][qi]-reference["arm"][ri],axis=-1)*100
    result=dict(path_pairs=n,position_potential_slots=n*2,angle_potential_slots=n*30,
        position_observed=measured(position,query["position_observed"][qi]&reference["position_observed"][ri]),
        position_estimated=measured(position,query["position_estimated"][qi]&reference["position_observed"][ri]),
        angle_observed=measured(angle,query["angle_observed"][qi]&reference["angle_observed"][ri]),
        angle_estimated=measured(angle,query["angle_estimated"][qi]&reference["angle_observed"][ri]),
        arm_observed=measured(arm,query["arm_valid"][qi]&reference["arm_valid"][ri]),
        units=dict(position="fixed shoulder width percent",angle="projected2D degrees",arm="fixed shoulder width percent"),
        verdict=None,interpretation="Reference difference only; estimated residuals are not observed accuracy. Same fixed B1 path across conditions.")
    return identity|dict(measurements=result)

def classify(models,x,word_code,codes):
    import torch
    tensor=torch.from_numpy(np.ascontiguousarray(x[None],dtype=np.float32))
    pad=torch.ones(tensor.shape[:2],dtype=torch.float32)
    target=codes.index(word_code);result={}
    with torch.inference_mode():
        for name,model in models.items():
            logits=model(tensor,pad)[0].cpu().numpy()
            assert logits.shape==(100,) and np.isfinite(logits).all()
            order=np.argsort(-logits,kind="stable");rank=int(np.flatnonzero(order==target)[0])+1
            result[name]=dict(target_rank=rank,top1=rank==1,top5=rank<=5,
                top5_word_codes=[codes[i] for i in order[:5]],logits=logits.tolist(),
                input_sha256=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest())
    return result

def classification_summary(rows):
    summaries={}
    for condition in sorted({r["condition"] for r in rows}):
        summaries[condition]={}
        for model in ("C","T3-F"):
            summaries[condition][model]={}
            for split in ("all","train","dev","test"):
                selected=[r for r in rows if r["condition"]==condition and (split=="all" or r["split"]==split)]
                n=len(selected);a=sum(r["classification"][model]["top1"] for r in selected)
                b=sum(r["classification"][model]["top5"] for r in selected)
                summaries[condition][model][split]=dict(clips=n,top1_correct=a,top5_correct=b,
                    top1_percent=100*a/n if n else None,top5_percent=100*b/n if n else None)
    all_changes={}
    for baseline in ("B0","B1"):
        base={r["clip_id"]:r for r in rows if r["condition"]==baseline};changes={}
        for condition in summaries:
            changes[condition]={};selected=[r for r in rows if r["condition"]==condition]
            for model in ("C","T3-F"):
                gains=[r["clip_id"] for r in selected if r["classification"][model]["top1"] and not base[r["clip_id"]]["classification"][model]["top1"]]
                losses=[r["clip_id"] for r in selected if not r["classification"][model]["top1"] and base[r["clip_id"]]["classification"][model]["top1"]]
                changes[condition][model]=dict(gains=gains,regressions=losses,delta_percentage_points=100*(len(gains)-len(losses))/len(selected))
        all_changes[baseline]=changes
    seed_aggregate={}
    for family in ("B2","B3"):
        seed_aggregate[family]={}
        for model in ("C","T3-F"):
            seed_aggregate[family][model]={}
            for split in ("all","train","dev","test"):
                vals=[summaries[f"{family}_seed{s}"][model][split]["top1_percent"] for s in SEEDS]
                seed_aggregate[family][model][split]=dict(per_seed=vals,mean=float(np.mean(vals)),min=min(vals),max=max(vals))
    return dict(per_condition=summaries,changes_from_B0=all_changes["B0"],changes_from_B1=all_changes["B1"],three_seed_summary=seed_aggregate,
                limitation="Previously exposed public people; not new-user or unseen-classifier generalization")

def candidate_source_pins(clips):
    pins={}
    for family in ("B2","B3"):
        for seed in SEEDS:
            run=UNIT/"runs"/f"{family}_seed{seed}"
            result=read(run/"RESULT.json")
            assert result["condition"]==family and result["seed"]==seed
            assert result["checkpoint_sha256"]==sha(run/"best.pt")
            for path in (run/"RESULT.json",run/"best.pt"):
                pins[str(path)]=sha(path)
            for cid in clips:
                path=run/"natural_estimates"/(cid+".npz");pins[str(path)]=sha(path)
    for name in ("FROZEN_PROTOCOL.json","COMPARISON.json"):
        path=UNIT/name
        if path.exists():pins[str(path)]=sha(path)
    return pins

def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("check","run"))
    ap.add_argument("--approved-plan-sha256");args=ap.parse_args()
    plan=check_plan();clips=load_sources()
    run_pins=candidate_source_pins(clips) if args.command=="run" else {}
    views=condition_views(clips,args.command=="run")
    passed=0
    for cid,clip in clips.items():
        recreated=fixed_channels(views["B0"][cid],clip)
        assert np.array_equal(recreated,clip["legacy"]["x"]),(cid,"B0 frozen channel reproduction differs")
        passed+=1
    if args.command=="check":
        result=dict(status="PASS",baseline_exact_channel_matches=passed,model_inference=0,
                    shape_280=True,uses_z=False,plan_sha256=sha(UNIT/"DOWNSTREAM_PLAN.json"))
        new_json(UNIT/"DOWNSTREAM_PREFLIGHT.json",result);print(json.dumps(result));return
    assert args.approved_plan_sha256==sha(UNIT/"DOWNSTREAM_PLAN.json"),"Root must explicitly approve this exact plan before inference"
    preflight=read(UNIT/"DOWNSTREAM_PREFLIGHT.json")
    assert preflight["status"]=="PASS" and preflight["plan_sha256"]==args.approved_plan_sha256
    out=UNIT/"downstream";assert not out.exists(),"No repeated scoring or overwrite";out.mkdir()
    new_json(out/"INPUT_PINS.json",run_pins)
    # Importing these modules alone never enumerates recordings; only _load is used.
    import torch
    from wis_lab.service import word100,word100_candidate as C,word100_t3f as T
    torch.set_num_threads(4)
    models={"C":C._load(),"T3-F":T._load()};codes=word100.config()["word_codes"]
    paths,reference_counts=build_paths(clips);new_json(out/"FIXED_PATHS.json",paths)
    reference_evidence={cid:evidence(views["B1"][cid],clip) for cid,clip in clips.items()}
    rows=[];residuals=[];started=time.perf_counter()
    for condition,by_clip in views.items():
        evidence_by_clip={cid:evidence(view,clips[cid]) for cid,view in by_clip.items()}
        for cid,view in by_clip.items():
            clip=clips[cid];x=fixed_channels(view,clip)
            rows.append(dict(condition=condition,clip_id=cid,word_code=clip["row"]["word_code"],
                signer=clip["row"]["signer_id"],split=split_for(clip["row"]["signer_id"]),
                classifier_span=list(clip["span"]),classification=classify(models,x,clip["row"]["word_code"],codes),
                evidence=coverage(view,evidence_by_clip[cid])))
        for pair in paths:
            residuals.append(dict(condition=condition,**dtw_residual(pair,evidence_by_clip[pair["query_clip"]],reference_evidence[pair["reference_clip"]])))
        print(json.dumps(dict(condition_complete=condition,clips=len(by_clip))),flush=True)
    # Existing B1 observed difference measurements must remain EXACT for B2/B3.
    baseline={(r["query_clip"],r["reference_clip"]):r for r in residuals if r["condition"]=="B1"}
    observed_checks=0
    for row in residuals:
        if not row["condition"].startswith(("B2_","B3_")) or row["measurements"] is None:continue
        base=baseline[(row["query_clip"],row["reference_clip"])]["measurements"]
        for key in ("position_observed","angle_observed","arm_observed"):
            assert row["measurements"][key]==base[key],(row["condition"],key,"observed residual altered")
        observed_checks+=1
    new_json(out/"ROWS.json",rows);new_json(out/"DTW_RESIDUALS.json",residuals)
    summary=classification_summary(rows)
    summary.update(status="COMPLETE_RESEARCH_DIAGNOSTIC",plan_sha256=args.approved_plan_sha256,
        reference_counts=reference_counts,clips_without_other_person_train_reference=[cid for cid,n in reference_counts.items() if n==0],
        fixed_reference_pairs=len(paths),supported_reference_pairs=sum(r["status"]=="SUPPORTED" for r in paths),
        observed_residual_preservation_checks=observed_checks,elapsed_seconds=time.perf_counter()-started,
        checkpoint_pins=dict(C=C.PIN,T3F=T.PIN),production_changed=False,user_recording_access=0,
        DTW_interpretation="Observed differences remain distinct from estimates. Low estimated-reference differences do not prove correct recovery.")
    new_json(out/"RESULT.json",summary)
    for path,pin in run_pins.items():assert sha(path)==pin,(path,"candidate input changed during scoring")
    for path,pin in plan["source_pins"].items():assert sha(path)==pin,(path,"protected source changed")
    print(json.dumps(dict(status=summary["status"],rows=len(rows),dtw_rows=len(residuals))),flush=True)

if __name__=="__main__":main()
