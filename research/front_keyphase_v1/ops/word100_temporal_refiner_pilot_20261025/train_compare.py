"""Gated public pilot; no C/T3-F/web edits. Targets are saved MP proxies.

Synthetic holes are applied AFTER B1 integration. B0/B1 stay missing there;
cached original B1 coordinates never become recovery predictions. Natural
estimates are separately stored and never promoted to observations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from refiner_model import POSE_DIM, TemporalRefiner, parameter_count

UNIT = Path(__file__).resolve().parent
SPLITS = {"train": [1,2,3,4,5,6], "dev": [7], "test": [8,9]}
BUDGET = {"seeds": [17,29,43], "epochs": 80, "batch_size": 16, "lr": .001, "weight_decay": .0001}
MODEL = {"window": 11, "hidden": 48, "pose_projection": 64, "rgb_projection": 32, "rgb_dim": 576}

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def validate_protocol(path, readiness=False):
    p = read(path)
    assert p["status"] == "FROZEN" and p["training_authorized"] is True
    assert p["splits"] == SPLITS
    for key, value in BUDGET.items():
        assert p["training"][key] == value, (key, "budget changed")
    for key, value in MODEL.items():
        assert p["model"][key] == value, (key, "architecture changed")
    pins = {str(Path(k).resolve()): v for k,v in p["source_pins"].items()}
    required = [Path(__file__), UNIT/"refiner_model.py", Path(p["features_manifest"])]
    required += [Path(p["bundle_dir"])/name for name in ("MANIFEST.json", "SCENARIOS.json", "load_inputs.py")]
    for f in required:
        assert str(f.resolve()) in pins, (str(f), "not pinned")
    for f,h in pins.items():
        assert sha(f) == h, (f, "source changed after freeze")
    if readiness:
        r = read(UNIT/"VISIBLE_READINESS.json")
        assert r["status"] == "PASS" and r["training_allowed"] is True
        assert r["protocol_sha256"] == sha(path)
    return p

def model_for(p, condition):
    return TemporalRefiner(**{k:v for k,v in p["model"].items() if k != "window"}, use_rgb=condition=="B3")

def split_for(signer):
    return next(k for k,v in SPLITS.items() if int(signer) in v)

def body_basis(base):
    valid = base["usable"][:,11] & base["usable"][:,12]
    left, right = base["xy_px"][valid,11], base["xy_px"][valid,12]
    width = np.linalg.norm(left-right, axis=1)
    good = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1) & (width>1)
    assert good.any(), "No reliable shoulder basis; hidden hand cannot substitute"
    return np.median((left[good]+right[good])/2, axis=0).astype(np.float32), float(np.median(width[good]))

def load_public(p):
    bundle = Path(p["bundle_dir"])
    adapter = load_module(bundle/"load_inputs.py", "pilot_safe_adapter")
    rows = read(bundle/"MANIFEST.json")
    assert len(rows) == 61
    feature_doc = read(p["features_manifest"])
    frows = feature_doc if isinstance(feature_doc, list) else feature_doc["rows"]
    feature_index = {r["clip_id"]:r for r in frows}
    assert {r["clip_id"] for r in rows} == set(feature_index)
    clips = {}
    for row in rows:
        cid = row["clip_id"]
        assert sha(row["arrays"]) == row["arrays_sha256"]
        with np.load(row["arrays"], allow_pickle=False) as z:
            raw = {k:z[k].copy() for k in z.files}  # controller only, never model input
        frow = feature_index[cid]
        assert sha(frow["path"]) == frow["sha256"]
        with np.load(frow["path"], allow_pickle=False) as z:
            rgb = z["features"].copy()
            for key in ("frame_id", "timestamp_us"):
                assert np.array_equal(z[key], raw[key]), (cid,key)
        assert rgb.shape == (row["frames"],2,576) and np.isfinite(rgb).all()
        base, merged = adapter.from_arrays(raw,"B0"), adapter.from_arrays(raw,"B1")
        assert np.all(merged["usable"][base["usable"]])
        assert np.array_equal(base["xy_px"][base["usable"]], merged["xy_px"][base["usable"]])
        clips[cid] = dict(row=row, raw=raw, rgb=rgb, base=base, merged=merged,
                          basis=body_basis(base), split=split_for(row["signer_id"]))
    return adapter, clips, read(bundle/"SCENARIOS.json")

def interpolate(payload, side, indices, basis):
    center, scale = basis
    sl = slice(33+21*side,54+21*side)
    xy = (payload["xy_px"][indices,sl]-center)/scale
    valid = payload["usable"][indices,sl]
    times = payload["timestamp_us"][indices].astype(np.float64)
    out = np.empty_like(xy)
    for joint in range(21):
        known = np.flatnonzero(valid[:,joint])
        assert len(known)>=2, "No two temporal anchors"
        for axis in range(2):
            out[:,joint,axis] = np.interp(times,times[known],xy[known,joint,axis])
    return out.reshape(11,42).astype(np.float32)

def encode(payload, rgb, side, indices, basis):
    center, scale = basis
    valid = payload["usable"][indices]
    xy = (payload["xy_px"][indices]-center)/scale
    xy[~valid] = 0
    interpolation = interpolate(payload,side,indices,basis)
    times = payload["timestamp_us"][indices].astype(np.float64)/1e6
    relative = (times-times[5]).astype(np.float32)[:,None]
    delta = np.r_[0.,np.diff(times)].astype(np.float32)[:,None]
    side_flag = np.zeros((11,2),np.float32); side_flag[:,side] = 1
    # Crop flags are audit only, absent from BOTH conditions.
    pose = np.concatenate([xy.reshape(11,150), valid, payload["observed"][indices],
        payload["conf"][indices], payload["conf_available"][indices],
        payload["hand_score"][indices], payload["hand_score_available"][indices],
        payload["point_origin"][indices], relative,delta,side_flag,interpolation],axis=1).astype(np.float32)
    feature = rgb[indices,side].astype(np.float32)
    assert pose.shape==(11,POSE_DIM) and feature.shape==(11,576)
    assert np.isfinite(pose).all() and np.isfinite(feature).all()
    return pose,feature,interpolation

def synthetic_sample(adapter,clip,case,raw_override=None):
    t = case["model_input"]["coordinate_transform"]
    side = t["side_index"]; start,stop=t["frame_range_half_open"]
    indices = np.arange(t["shared_center_index"]-5,t["shared_center_index"]+6)
    assert indices[0]>=0 and indices[-1]<clip["row"]["frames"]
    raw = clip["raw"] if raw_override is None else raw_override
    payload = adapter.from_arrays(raw,"B1",dict(side=side,start_index=start,end_index_exclusive=stop))
    pose,rgb,interpolation = encode(payload,clip["rgb"],side,indices,clip["basis"])
    sl = slice(33+side*21,54+side*21)
    target = ((clip["base"]["xy_px"][indices,sl]-clip["basis"][0])/clip["basis"][1]).reshape(11,42).astype(np.float32)
    mask = np.zeros((11,42),bool); mask[(indices>=start)&(indices<stop)] = True
    assert clip["base"]["usable"][start:stop,sl].all()
    assert not payload["usable"][start:stop,sl].any()
    return dict(pose=pose,rgb=rgb,interpolation=interpolation,target=target,loss_mask=mask,
                side=side,indices=indices,start=start,stop=stop,case_id=case["case_id"],
                clip_id=clip["row"]["clip_id"],split=clip["split"],hole_frames=stop-start)

def build_samples(adapter,clips,scenarios):
    samples = [synthetic_sample(adapter,clips[c["evaluator_only"]["clip_id"]],c)
               for c in scenarios["cases"] if c["model_input"]["coordinate_transform"]["kind"]!="none"]
    assert len(samples)==333
    return samples

def tensors(samples):
    return tuple(torch.from_numpy(np.stack([s[k] for s in samples]))
                 for k in ("pose","rgb","interpolation","target","loss_mask"))

def check_inputs(adapter,clips,scenarios,samples,p):
    checks = Counter()
    for case in scenarios["cases"]:
        t = case["model_input"]["coordinate_transform"]
        if t["kind"]=="none":
            continue
        clip=clips[case["evaluator_only"]["clip_id"]]
        a=synthetic_sample(adapter,clip,case)
        raw={k:v.copy() for k,v in clip["raw"].items()}
        side=t["side_index"]; start,stop=t["frame_range_half_open"]
        sl=slice(33+side*21,54+side*21)
        for prefix in ("baseline_roi","redetected_roi"):
            raw[prefix+"_xy_px"][start:stop,sl]=np.nan
            raw[prefix+"_conf"][start:stop,sl]=1e30
            raw[prefix+"_observed"][start:stop,sl]=False
            raw[prefix+"_hand_score"][start:stop,side]=1e30
        b=synthetic_sample(adapter,clip,case,raw_override=raw)
        for key in ("pose","rgb","interpolation"):
            assert np.array_equal(a[key],b[key]),(case["case_id"],key,"target leak")
        checks["hidden_target_input_invariance"]+=1
    torch.manual_seed(17); b2=model_for(p,"B2")
    torch.manual_seed(17); b3=model_for(p,"B3")
    assert parameter_count(b2)==parameter_count(b3)
    with torch.no_grad():
        pose,rgb,interpolation,_,_=tensors(samples[:2])
        assert torch.equal(b2(pose,rgb,interpolation),interpolation)
        assert torch.equal(b3(pose,rgb,interpolation),interpolation)
    checks.update(zero_residual_smoke=2,parameter_count_each=parameter_count(b2),pose_dimension=POSE_DIM)
    return dict(status="PASS",checks=dict(checks),
                scenario_counts=dict(Counter(s["split"] for s in samples)),
                clip_counts=dict(Counter(c["split"] for c in clips.values())),
                targets="Saved B0 MP proxy; no manual truth or user data training",
                synthetic_B0_B1="Both remain missing at hidden targets; no cached copy-back",
                normalization="Fixed per-clip shoulder center/width, independent of hand targets")

def metrics(predictions,samples,breakdown=True):
    assert np.isfinite(predictions).all(), "Nonfinite predicted coordinates"
    target=np.stack([s["target"] for s in samples])
    coord_mask=np.stack([s["loss_mask"] for s in samples])
    pred_joints=predictions.reshape(-1,11,21,2)
    target_joints=target.reshape(-1,11,21,2)
    error=np.linalg.norm(pred_joints-target_joints,axis=-1)
    mask=coord_mask.reshape(-1,11,21,2).all(axis=-1)
    values=error[mask]
    wrist_error=error[:,:,0]
    relative_error=np.linalg.norm((pred_joints[:,:,1:]-pred_joints[:,:,:1])-
                                  (target_joints[:,:,1:]-target_joints[:,:,:1]),axis=-1)
    wrist_mask=mask[:,:,0]
    relative_mask=mask[:,:,1:]&wrist_mask[:,:,None]
    rows=[]
    for i,s in enumerate(samples):
        v=error[i][mask[i]]
        w=wrist_error[i][wrist_mask[i]]; shape=relative_error[i][relative_mask[i]]
        rows.append({k:s[k] for k in ("case_id","clip_id","split","hole_frames","side")} |
                    dict(joint_frames=len(v),mean_euclidean=float(v.mean()),
                         median_euclidean=float(np.median(v)),p95_euclidean=float(np.quantile(v,.95)),
                         wrist_mean_euclidean=float(w.mean()),
                         wrist_relative_finger_mean_euclidean=float(shape.mean())))
    result=dict(cases=len(samples),clips=len({s["clip_id"] for s in samples}),
                joint_frames=int(mask.sum()),coordinate_values=int(coord_mask.sum()),
                mean_l1=float(np.abs(predictions-target)[coord_mask].mean()),
                mean_euclidean=float(values.mean()),median_euclidean=float(np.median(values)),
                p95_euclidean=float(np.quantile(values,.95)),per_case=rows)
    for prefix,values in (("wrist",wrist_error[wrist_mask]),
                          ("wrist_relative_finger",relative_error[relative_mask])):
        result[prefix+"_point_frames"]=len(values)
        result[prefix+"_mean_euclidean"]=float(values.mean())
        result[prefix+"_median_euclidean"]=float(np.median(values))
        result[prefix+"_p95_euclidean"]=float(np.quantile(values,.95))
    if breakdown:
        result["by_hole_frames"]={}
        for hole in (1,3,7):
            indices=[i for i,s in enumerate(samples) if s["hole_frames"]==hole]
            if indices:
                result["by_hole_frames"][str(hole)]=metrics(predictions[indices],[samples[i] for i in indices],False)
    return result

def infer(model,samples,batch_size):
    inputs=tensors(samples); predictions=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(samples),batch_size):
            predictions.append(model(*(x[start:start+batch_size] for x in inputs[:3])).cpu().numpy())
    return np.concatenate(predictions)

def ranges(flags):
    edges=np.diff(np.r_[False,flags,False].astype(np.int8))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))

def natural_predictions(model,clips,folder):
    """Bounded, unconfirmed estimates only; B1 observed values stay exact."""
    totals=Counter(); records=[]; model.eval()
    out=folder/"natural_estimates"; out.mkdir()
    for cid,clip in clips.items():
        base=clip["merged"]; xy=base["xy_px"].copy()
        estimated=np.zeros(base["usable"].shape,bool); n=len(xy)
        center,scale=clip["basis"]
        for side in (0,1):
            sl=slice(33+side*21,54+side*21)
            complete=(base["usable"][:,sl]&base["observed"][:,sl]).all(axis=1)
            for start,stop in ranges(~complete):
                if stop-start>7 or start==0 or stop==n:
                    totals["unbounded_or_long_intervals_not_estimated"]+=1
                    continue
                missing=~base["usable"][start:stop,sl]
                if not missing.any():
                    continue
                middle=(start+stop-1)//2; lo=max(0,min(middle-5,n-11))
                if n<11 or not(lo<=start-1 and stop<lo+11):
                    totals["insufficient_window_not_estimated"]+=1
                    continue
                indices=np.arange(lo,lo+11)
                pose,rgb,interpolation=encode(base,clip["rgb"],side,indices,clip["basis"])
                with torch.no_grad():
                    pred=model(torch.from_numpy(pose[None]),torch.from_numpy(rgb[None]),
                               torch.from_numpy(interpolation[None]))[0].numpy().reshape(11,21,2)
                local=(pred*scale+center)[start-lo:stop-lo]
                x0,y0,x1,y1=np.asarray(base["roi_box"]).reshape(-1)[:4]
                inside=np.isfinite(local).all(axis=-1)&(local[...,0]>=x0)&(local[...,0]<x1)&(local[...,1]>=y0)&(local[...,1]<y1)
                added=missing&inside
                section=xy[start:stop,sl]; section[added]=local[added]
                estimated[start:stop,sl]|=added
                totals["eligible_intervals"]+=1
                totals["estimated_joint_frames"]+=int(added.sum())
                totals["out_of_roi_estimates_withheld"]+=int((missing&~inside).sum())
                records.append(dict(clip_id=cid,side=side,range=[int(start),int(stop)],
                                    split=clip["split"],added_joint_frames=int(added.sum()),
                                    truth="unknown; temporal estimate, not observed"))
        assert not np.any(estimated&base["usable"])
        assert np.array_equal(xy[base["usable"]],base["xy_px"][base["usable"]])
        np.savez_compressed(out/(cid+".npz"),xy_px=xy,estimated=estimated,observed=base["observed"],
                            usable_observed=base["usable"],frame_id=base["frame_id"],timestamp_us=base["timestamp_us"])
        totals["clips_original_points_preserved"]+=1
    return dict(counts=dict(totals),intervals=records,accuracy=None,
                reason="No independent natural-gap truth; estimates remain unobserved",production_promoted=False)

def run_one(condition,seed,p,clips,samples,output_root):
    folder=output_root/f"{condition}_seed{seed}"; folder.mkdir()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=model_for(p,condition)
    train=[s for s in samples if s["split"]=="train"]
    dev=[s for s in samples if s["split"]=="dev"]
    assert train and dev and any(s["split"]=="test" for s in samples)
    b=p["training"]; generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(*tensors(train)),batch_size=b["batch_size"],shuffle=True,
                      drop_last=False,num_workers=0,generator=generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=b["lr"],weight_decay=b["weight_decay"])
    history=[]; best=float("inf"); best_epoch=None; best_state=None; started=time.perf_counter()
    for epoch in range(1,b["epochs"]+1):
        model.train(); loss_sum=0.; coordinates=0
        for pose,rgb,interpolation,target,mask in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.abs(model(pose,rgb,interpolation)-target)[mask].mean()
            assert torch.isfinite(loss),"Nonfinite loss; preserve failure, do not skip"
            loss.backward(); optimizer.step()
            loss_sum+=float(loss.detach())*int(mask.sum()); coordinates+=int(mask.sum())
        dev_score=metrics(infer(model,dev,b["batch_size"]),dev)["mean_l1"]
        history.append(dict(epoch=epoch,train_l1=loss_sum/coordinates,dev_l1=dev_score,
                            train_examples=len(train),optimizer_steps=len(loader)))
        if dev_score<best:
            best=dev_score; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    checkpoint=folder/"best.pt"
    torch.save(dict(condition=condition,seed=seed,selected_epoch=best_epoch,state_dict=best_state,
                    model_config=p["model"],dev_l1=best,class_independent=True),checkpoint)
    scores={}; saved={}
    for split in SPLITS:
        selected=[s for s in samples if s["split"]==split]
        pred=infer(model,selected,b["batch_size"])
        scores[split]=metrics(pred,selected)
        saved[split+"_prediction_normalized"]=pred
        saved[split+"_case_id"]=np.array([s["case_id"] for s in selected])
    np.savez_compressed(folder/"synthetic_predictions.npz",**saved)
    natural=natural_predictions(model,clips,folder)
    result=dict(condition=condition,seed=seed,selected_epoch=best_epoch,scores=scores,
                selection="DEV07 masked L1; earliest minimum; no TEST selection",
                parameter_count=parameter_count(model),epochs=b["epochs"],
                optimizer_steps=len(loader)*b["epochs"],training_examples_per_epoch=len(train),
                elapsed_seconds=time.perf_counter()-started,checkpoint_sha256=sha(checkpoint),natural=natural,
                limitations=["Proxy reconstruction, not physical-occlusion correctness",
                             "People08/09 held out only from refiner; C/T3-F already saw them",
                             "Natural estimates remain unobserved and are not runtime inputs"])
    write_new(folder/"history.json",history); write_new(folder/"RESULT.json",result)
    print(json.dumps(dict(completed=folder.name,selected_epoch=best_epoch,test_proxy_l1=scores["test"]["mean_l1"])),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("preflight","run"))
    parser.add_argument("--protocol",default=str(UNIT/"FROZEN_PROTOCOL.json"))
    args=parser.parse_args()
    p=validate_protocol(args.protocol,readiness=args.command=="run")
    torch.set_num_threads(int(p.get("cpu_threads",4)))
    torch.use_deterministic_algorithms(True)
    adapter,clips,scenarios=load_public(p); samples=build_samples(adapter,clips,scenarios)
    if args.command=="preflight":
        result=check_inputs(adapter,clips,scenarios,samples,p)
        result["protocol_sha256"]=sha(args.protocol)
        write_new(UNIT/"PREFLIGHT.json",result); print(json.dumps(result),flush=True)
        return
    pf=read(UNIT/"PREFLIGHT.json")
    assert pf["status"]=="PASS" and pf["protocol_sha256"]==sha(args.protocol)
    output_root=UNIT/"runs"
    assert not output_root.exists(),"Inspect existing/partial runs; no silent overwrite"
    lock=UNIT/"RUNNING.json"
    write_new(lock,dict(pid=os.getpid(),protocol_sha256=sha(args.protocol),started=time.time()))
    try:
        output_root.mkdir(); controls={}
        for split in SPLITS:
            selected=[s for s in samples if s["split"]==split]
            controls[split]=dict(B0=dict(observed_target_coverage=0.,error=None),
                                 B1=dict(observed_target_coverage=0.,error=None),
                                 linear_interpolation=metrics(np.stack([s["interpolation"] for s in selected]),selected))
        write_new(UNIT/"CONTROL_RESULTS.json",controls)
        results=[run_one(condition,seed,p,clips,samples,output_root)
                 for seed in p["training"]["seeds"] for condition in ("B2","B3")]
        aggregate={}
        for condition in ("B2","B3"):
            selected=[r for r in results if r["condition"]==condition]; aggregate[condition]={}
            for split in SPLITS:
                aggregate[condition][split]={}
                metric_names=("mean_l1","mean_euclidean","median_euclidean","p95_euclidean",
                              "wrist_mean_euclidean","wrist_median_euclidean","wrist_p95_euclidean",
                              "wrist_relative_finger_mean_euclidean","wrist_relative_finger_median_euclidean",
                              "wrist_relative_finger_p95_euclidean")
                for metric in metric_names:
                    values=[r["scores"][split][metric] for r in selected]
                    aggregate[condition][split][metric]=dict(per_seed=values,mean=float(np.mean(values)),min=min(values),max=max(values))
                aggregate[condition][split]["by_hole_frames"]={}
                for hole in ("1","3","7"):
                    aggregate[condition][split]["by_hole_frames"][hole]={}
                    for metric in metric_names:
                        values=[r["scores"][split]["by_hole_frames"][hole][metric] for r in selected]
                        aggregate[condition][split]["by_hole_frames"][hole][metric]=dict(per_seed=values,mean=float(np.mean(values)),min=min(values),max=max(values))
        write_new(UNIT/"COMPARISON.json",dict(status="COMPLETE_RESEARCH_ONLY",aggregate=aggregate,
            control_path=str(UNIT/"CONTROL_RESULTS.json"),protocol_sha256=sha(args.protocol),
            all_three_seeds_reported=True,production_changed=False,independent_true_coordinate_accuracy="not evaluated by proxy lane"))
    except Exception as exc:
        failure=UNIT/"FIRST_TRAINING_ERROR.json"
        if not failure.exists():
            write_new(failure,dict(type=type(exc).__name__,message=str(exc),partial_results_preserved=True))
        raise
    finally:
        if lock.exists() and read(lock).get("pid")==os.getpid():
            lock.unlink()

if __name__=="__main__":
    main()
