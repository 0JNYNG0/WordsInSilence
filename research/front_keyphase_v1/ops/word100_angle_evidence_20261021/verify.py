"""Separate saved-result verification. Does not import candidate or run."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
same=lambda a,b:np.array_equal(a,b,equal_nan=True)
f=read(U/"FROZEN.json");index=read(U/"INDEX.json")
assert len(index)==len(f["rows"])==136 and not (U/"RUNNING.json").exists()
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
counts=Counter();maxerr=0.;ctrl=[]
cross=read(f["controls"]["crossing"]);normal=read(f["controls"]["normal"])
systematic=read(f["controls"]["systematic20"])["normal_controls"]
for r,item in zip(f["rows"],index):
    assert item["clip_id"]==r["clip_id"] and sha(item["path"])==item["sha256"]
    y=load(item["path"]);obs=load(r["obs"]);e=load(r["quality"]);x=load(r["inputs"])[r["input_key"]]
    for key in ["xy_px","observed","usable","conf","frame_id","timestamp_us"]:
        assert same(y[key],obs[key]),(r["clip_id"],key)
    assert same(y["x"],x)
    points=obs["xy_px"][:,33:].reshape(-1,2,21,2).astype(np.float64)
    vv=x[:,237:279].reshape(-1,2,21).astype(bool)
    out=np.full(y["angle_deg"].shape,np.nan);states=np.zeros(out.shape,np.uint8)
    chain_starts=[1,5,9,13,17]
    for digit,start in enumerate(chain_starts):
        chain=[0]+list(range(start,start+4))
        for local in range(3):
            k=digit*3+local;i,j,l=chain[local:local+3]
            z=points[...,0]+1j*points[...,1]
            present=vv[...,i]&vv[...,j]&vv[...,l]&np.isfinite(z[...,i])&np.isfinite(z[...,j])&np.isfinite(z[...,l])
            b=z[...,i]-z[...,j];d=z[...,l]-z[...,j]
            valid=present&(abs(b)>1e-6)&(abs(d)>1e-6)
            angle=np.abs(np.angle(b*np.conj(d)))*180/np.pi
            out[...,k]=np.where(valid,angle,np.nan)
            states[...,k][present]=1
            states[...,k][valid]=2
            check=valid&e["pair_available"]&e["angle_available"][...,k]
            states[...,k][check& e["angle_unresolved"][...,k]]=3
            states[...,k][check&~e["angle_unresolved"][...,k]]=4
    assert np.allclose(out,y["angle_deg"],rtol=0,atol=1e-9,equal_nan=True)
    assert same(states,y["angle_state"])
    assert same(np.isfinite(out),y["computable"]) and same(states==4,y["compare_eligible"])
    finite=np.isfinite(out)
    if finite.any():maxerr=max(maxerr,float(abs(out[finite]-y["angle_deg"][finite]).max()))
    assert not np.any(y["restored_eligible"]&e["angle_unresolved"])
    assert same(y["restored_eligible"],(states==4)&~y["old_point_filter_computable"])
    counts["clips"]+=1;counts["frames"]+=len(x);counts["coordinate_values_preserved"]+=obs["xy_px"].size
    counts["model_values_preserved"]+=x.size;counts["angle_states_checked"]+=states.size
    for label,items in [("prior_crossing",cross),("prior_normal",normal),("systematic20",systematic)]:
        for q in items:
            if q["clip_id"]!=r["clip_id"] or q.get("view","masked")!="masked":continue
            i=q["index"]
            ctrl.append(dict(group=label,clip_id=r["clip_id"],label=r["label"],index=i,
                computable=int(finite[i].sum()),eligible=int((states[i]==4).sum()),
                unassessed=int((states[i]==2).sum()),unresolved=int((states[i]==3).sum()),
                coordinate_changes=0,model_input_changes=0,
                shape_evidence_states=states[i].tolist()))
assert Counter(q["group"] for q in ctrl)==dict(prior_crossing=53,prior_normal=3,systematic20=20)
for p,h in f["protected"].items():assert sha(p)==h
result=dict(status="COORDINATES_AND_LOCAL_ANGLE_EVIDENCE_VERIFIED",counts=dict(counts),
    max_angle_difference_deg=maxerr,controls=ctrl,
    control_totals={g:dict(frames=sum(q["group"]==g for q in ctrl),
        computable=sum(q["computable"] for q in ctrl if q["group"]==g),
        eligible=sum(q["eligible"] for q in ctrl if q["group"]==g),
        unassessed=sum(q["unassessed"] for q in ctrl if q["group"]==g),
        unresolved=sum(q["unresolved"] for q in ctrl if q["group"]==g)) for g in ["prior_crossing","prior_normal","systematic20"]},
    protected_files=32,verification="Same Codex, independent complex-angle equations and saved-array checks; not an independent reviewer.",
    limits="No physical finger ground truth; normal identity/crossing is not proof each finger is visible or precisely measurable.")
with (U/"VERIFY.json").open("x",encoding="utf8") as fp:json.dump(result,fp,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!="controls"},ensure_ascii=False))
