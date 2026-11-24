"""Independent saved-input reconstruction. Does not import candidate/production feature code."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
MAP = [0,11,12,13,14,15,16,23,24] + list(range(33,75))
PALM = [0,5,9,13,17]
TRIPLES = [(a,b,c) for chain in ([0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],
                               [0,13,14,15,16],[0,17,18,19,20])
           for a,b,c in zip(chain,chain[1:],chain[2:])]


def read(p):
    return json.loads(Path(p).read_text("utf-8-sig"))


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same(a,b):
    return np.array_equal(a,b,equal_nan=True)


def reconstruct(obs, override=None):
    xy=obs["xy_px"][:,MAP].copy()
    present=obs["usable"][:,MAP].copy()
    xy[~present]=0
    t=len(xy)
    shoulders=present[:,1]&present[:,2]
    width=np.median(np.linalg.norm(xy[shoulders,1]-xy[shoulders,2],axis=1))
    origin=(xy[:,1]+xy[:,2])/2
    out=np.zeros((t,280),np.float32)
    good=present&shoulders[:,None]
    coords=(xy-origin[:,None,:])/width
    coords[~good]=0
    out[:,:102]=coords.reshape(t,102)
    out[:,186:237]=good
    scales=[]
    for s,offset in enumerate((9,30)):
        lens=[]
        for frame in range(t):
            if not present[frame,offset]:
                continue
            for m in (5,9,13,17):
                if present[frame,offset+m]:
                    lens.append(float(np.linalg.norm(xy[frame,offset+m]-xy[frame,offset])))
        hs=float(np.median(lens)) if lens else np.nan
        if hs<=1e-6:hs=np.nan
        scales.append(hs)
        value=hs if override is None else override[s]
        if not np.isfinite(value):
            continue
        hand_ok=present[:,offset:offset+21]&present[:,offset,None]
        hand=(xy[:,offset:offset+21]-xy[:,offset,None,:])/value
        hand[~hand_ok]=0
        out[:,102+42*s:144+42*s]=hand.reshape(t,42)
        out[:,237+21*s:258+21*s]=hand_ok
    out[1:,279]=np.diff(obs["timestamp_us"]).astype(np.float32)/1e6
    return out,scales


def direct_angles(obs):
    xy=obs["xy_px"][:,33:].reshape(-1,2,21,2).astype(np.float64)
    valid=obs["usable"][:,33:].reshape(-1,2,21)
    a=np.full((len(xy),2,15),np.nan)
    mask=np.zeros(a.shape,bool)
    for k,(i,j,l) in enumerate(TRIPLES):
        u=xy[...,i,:]-xy[...,j,:];v=xy[...,l,:]-xy[...,j,:]
        good=valid[...,[i,j,l]].all(-1)&(np.linalg.norm(u,axis=-1)>1e-6)&(np.linalg.norm(v,axis=-1)>1e-6)
        # Independent stable atan2 form, unlike candidate acos.
        deg=np.degrees(np.arctan2(np.abs(u[...,0]*v[...,1]-u[...,1]*v[...,0]),(u*v).sum(-1)))
        a[...,k]=np.where(good,deg,np.nan);mask[...,k]=good
    return a,mask


def main():
    f=read(U/"FROZEN.json")
    assert not (U/"RUNNING.json").exists()
    idx={r["clip_id"]:r for r in read(U/"INDEX.json")}
    for p,h in {**f["pins"],**f["protected"]}.items():
        assert sha(p)==h,p
    cross=read(O/"word100_image_arm_connectivity_20261018/CROSSING.json")
    ann=read(O/"word100_image_arm_connectivity_20261018/ANNOTATED.json")
    counts=Counter();max_error=0.;angle_math_error=0.;norm_changed=0;checks=[]
    for row in f["rows"]:
        cid=row["clip_id"]
        for p,h in idx[cid]["outputs"].items():
            assert sha(p)==h,p
        old=load(row["old_masked"]);new=load(row["merged"])
        native=load(row["merged_native"]);obs=old["observed"];usable=old["usable"]
        for k in ("xy_px","observed","usable","conf"):
            assert same(old[k][obs],new[k][obs]),(cid,k)
        for k in ("frame_id","timestamp_us"):
            assert same(old[k],new[k])
        counts["retained_existing_joint_frames"]+=int(obs.sum())
        a,hs=reconstruct(old)
        b,_=reconstruct(new,hs)
        c,_=reconstruct(new)
        allarr={}
        for name,expected in zip(f["conditions"],(a,b,c)):
            z=load(U/"inputs"/cid/(name+".npz"))
            assert z["x"].shape==(len(a),280) and z["x"].dtype==np.float32
            assert np.isfinite(z["x"]).all()
            error=float(np.max(abs(expected-z["x"])))
            max_error=max(max_error,error)
            assert np.allclose(expected,z["x"],rtol=0,atol=1e-6),(cid,name,error)
            assert same(z["frame_id"],old["frame_id"]) and same(z["timestamp_us"],old["timestamp_us"])
            assert same(z["x"][:,186:237],z["body_valid"])
            assert same(z["x"][:,237:279],z["hand_valid"])
            assert not z["x"][:,:102][~z["body_valid"].repeat(2,axis=1).astype(bool)].any()
            assert not z["x"][:,102:186][~z["hand_valid"].repeat(2,axis=1).astype(bool)].any()
            allarr[name]=z["x"];counts["arrays"]+=1;counts["reconstructed_values"]+=expected.size
        a,b,c=[allarr[n] for n in f["conditions"]]
        body_good=a[:,186:237].repeat(2,axis=1).astype(bool)
        local_good=a[:,237:279].repeat(2,axis=1).astype(bool)
        assert same(a[:,:102][body_good],b[:,:102][body_good])
        assert same(a[:,102:186][local_good],b[:,102:186][local_good])
        assert same(a[:,186:279][a[:,186:279].astype(bool)],b[:,186:279][a[:,186:279].astype(bool)])
        assert same(b[:,:102],c[:,:102]) and same(b[:,186:],c[:,186:])
        counts["A_to_B_existing_value_changes"]+=int((a[:,102:186][local_good]!=b[:,102:186][local_good]).sum())
        counts["B_to_C_existing_local_changes"]+=int((b[:,102:186][local_good]!=c[:,102:186][local_good]).sum())
        norm_changed+=bool((b[:,102:186]!=c[:,102:186]).any())
        # Outside-ROI evidence is excluded before normalization, for all3 conditions.
        x=native["xy_px"];box=row["box"]
        inside=np.isfinite(x).all(-1)&(x[...,0]>=box[0])&(x[...,0]<box[2])&(x[...,1]>=box[1])&(x[...,1]<box[3])
        assert same(new["observed"],native["observed"]&inside)
        assert same(new["usable"],native["usable"]&inside)
        mutated={k:v.copy() for k,v in native.items()}
        outside=native["observed"]&~inside
        mutated["xy_px"][outside]=[1e9,1e9]
        mutated["observed"]&=inside;mutated["usable"]&=inside
        mutated["xy_px"][~mutated["observed"]]=np.nan
        assert same(reconstruct(mutated,hs)[0],b)
        assert same(reconstruct(mutated)[0],c)
        counts["outside_points_perturbed"]+=int(outside.sum())
        e=load(U/"evidence"/(cid+".npz"))
        expected_origin=np.where(usable[:,33:],1,np.where(new["usable"][:,33:],2,0)).reshape(-1,2,21)
        assert same(e["point_origin"],expected_origin)
        v=new["usable"][:,33:].reshape(-1,2,21)
        assert same(e["wrist_available"],v[...,0])
        assert same(e["palm_available"],v[...,PALM].all(-1))
        na,nm=direct_angles(new);oa,om=direct_angles(old)
        assert same(e["bend_available"],nm)
        if nm.any():
            err=float(np.max(abs(na[nm]-e["bend_degrees"][nm])))
            angle_math_error=max(angle_math_error,err);assert err<.1
        assert same(oa[om],na[om])
        counts["retained_existing_angles"]+=int(om.sum())
        for item in cross:
            if item["clip_id"]==cid and item["view"]=="masked":
                i=item["index"]
                assert same(old["xy_px"][i][obs[i]],new["xy_px"][i][obs[i]])
                assert same(a[i,102:186][local_good[i]],b[i,102:186][local_good[i]])
                counts["crossing_controls"]+=1
        for item in ann:
            if item["clip_id"]==cid and item["view"]=="masked":
                assert item["verdict"]=="normal_identity"
                i=item["index"]
                assert same(old["xy_px"][i][obs[i]],new["xy_px"][i][obs[i]])
                counts["normal_controls"]+=1
        counts["frames"]+=len(a)
        checks.append(dict(clip_id=cid,original_position_preserved=True,original_angles_preserved=True,
                           fixed_scale_original_features_preserved=True,three_arrays_reconstructed=True))
    assert counts["crossing_controls"]==53 and counts["normal_controls"]==3
    assert counts["A_to_B_existing_value_changes"]==0
    assert counts["B_to_C_existing_local_changes"]==153442
    result=dict(status="INPUT_ARITHMETIC_AND_PRESERVATION_PASS",counts=dict(counts),
                recomputed_scale_changed_clips=norm_changed, maximum_feature_abs_error=max_error,
                angle_atan2_vs_acos_max_deg=angle_math_error, checks=checks,
                implementation="Codex",verification="Separate direct equations by same Codex, no independent reviewer",
                limits="Computational input verification only. Existing3normal/53crossing saved controls retained, not new identity accuracy measurement. RGB precision unresolved.",
                training_runs=0,classifier_evaluations=0,user_data_reads=0,production_changes=0)
    with (U/"VERIFY.json").open("x",encoding="utf8") as fp:
        json.dump(result,fp,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},ensure_ascii=False))


if __name__=="__main__":
    main()
