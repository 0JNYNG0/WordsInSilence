"""Separate preservation and geometry arithmetic; no detector or classifier."""
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent
O=U.parent
def read(p): return json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open("x",encoding="utf8") as f: json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    f=read(U/"FROZEN.json"); rows=read(U/"PREPARED_INDEX.json")
    assert len(rows)==27 and not (U/"RUNNING.json").exists()
    for p,h in {**f["pins"],**f["protected"]}.items(): assert sha(p)==h,p
    counts=Counter(); details=[]; new_added=[]; normal=[]; max_angle=0.
    triples=[]
    for chain in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)):
        triples.extend([chain[i:i+3] for i in range(3)])
    for row in rows:
        cid=row["clip_id"]; old=load(row["old_masked"]); z=load(row["prepared"]); obs=load(row["observation"])
        qual=load(row["quality"])
        for p,h in row["output_hashes"].items(): assert sha(p)==h,p
        for k in ("xy_px","observed","usable","conf","frame_id","timestamp_us"):
            assert np.array_equal(z[k],obs[k],equal_nan=True),(cid,k)
        keep=old["usable"]
        assert np.array_equal(z["xy_px"][keep],old["xy_px"][keep],equal_nan=True)
        assert (z["usable"][keep]).all() and (z["observed"][old["observed"]]).all()
        # Both body and existing hands keep exact coordinates and side slots.
        assert np.array_equal(z["xy_px"][:,:33],old["xy_px"][:,:33],equal_nan=True)
        counts["existing_usable_points_preserved"]+=int(keep.sum())
        counts["coordinate_values_unmodified_by_quality"]+=z["xy_px"].size
        xy=z["xy_px"][:,33:].reshape(-1,2,21,2).astype(float)
        valid=z["usable"][:,33:].reshape(-1,2,21)
        new=z["usable"]&~old["usable"]
        inside=np.isfinite(z["xy_px"]).all(-1)
        x0,y0,x1,y1=row["box"]
        inside &= ((z["xy_px"][...,0]>=x0)&(z["xy_px"][...,0]<x1)&
                   (z["xy_px"][...,1]>=y0)&(z["xy_px"][...,1]<y1))
        assert not (new&~inside).any()
        for h in range(2):
            sl=slice(33+21*h,54+21*h)
            added=new[:,sl].any(-1)
            assert not (added&old["observed"][:,sl].any(-1)).any()
            for i in np.flatnonzero(added):
                new_added.append(dict(clip_id=cid,index=int(i),side=h,frame_id=int(z["frame_id"][i]),
                                      fresh=not row["reuse"],core=bool(z["core"][i])))
        max_this=0.
        for a,(i,j,k) in enumerate(triples):
            x=xy[:,:,i]-xy[:,:,j];y=xy[:,:,k]-xy[:,:,j]
            nx=np.linalg.norm(x,axis=-1);ny=np.linalg.norm(y,axis=-1)
            present=valid[:,:,i]&valid[:,:,j]&valid[:,:,k]&np.isfinite(x).all(-1)&np.isfinite(y).all(-1)
            ok=present&(nx>1e-6)&(ny>1e-6)
            value=np.degrees(np.arccos(np.clip((x*y).sum(-1)/np.where(nx*ny>0,nx*ny,1),-1,1)))
            if ok.any():
                err=float(np.max(abs(value[ok]-z["angle_deg"][:,:,a][ok])))
                max_this=max(max_this,err);assert err<1e-4,(cid,a,err)
            state=np.where(present,1,0).astype(np.uint8)
            state[ok]=2
            check=qual["pair_available"]&qual["angle_available"][:,:,a]&ok
            state[check&qual["angle_unresolved"][:,:,a]]=3
            state[check&~qual["angle_unresolved"][:,:,a]]=4
            assert np.array_equal(state,z["angle_state"][:,:,a]),(cid,a)
        max_angle=max(max_angle,max_this)
        required=f["roles"].get(row["word_code"],{}).get("required_sides",["L","R"])
        hs=[("L","R").index(s) for s in required]
        active=z["angle_state"][z["core"]][:,hs].ravel()
        c=Counter(map(int,active)); counts.update({"core_state_"+str(k):v for k,v in c.items()})
        coreids=np.flatnonzero(z["core"])
        for fraction in (.25,.5,.75):
            i=int(coreids[min(len(coreids)-1,int((len(coreids)-1)*fraction))])
            states=z["angle_state"][i,hs]
            normal.append(dict(clip_id=cid,index=i,frame_id=int(z["frame_id"][i]),fraction=fraction,
                               required=required,eligible=int((states==4).sum()),computable=int((states>=2).sum()),
                               total=states.size,state_counts=dict(Counter(map(int,states.ravel())))))
        details.append(dict(clip_id=cid,label=row["label"],word_code=row["word_code"],reuse=row["reuse"],
                            required_core_angles=int(active.size),core_states=dict(c),
                            new_points=int(new.sum()),frames=len(z["frame_id"])))
        counts["clips"]+=1;counts["frames"]+=len(z["frame_id"])
    prior_controls=read(O/"word100_angle_evidence_20261021/VERIFY.json")["controls"]
    controls=[]
    byid={r["clip_id"]:r for r in rows}
    for c in prior_controls:
        if c["clip_id"] not in byid: continue
        r=byid[c["clip_id"]]; z=load(r["prepared"]); prior=load(r["angle_saved"]); i=c["index"]
        for k in ("xy_px","usable","angle_deg","angle_state"):
            assert np.array_equal(z[k][i],prior[k][i],equal_nan=True)
        controls.append(c)
    write(U/"INPUT_VERIFY.json",dict(status="PASS",counts=dict(counts),max_degree_error=max_angle,
        details=details,normal_samples=normal,new_added_hand_frames=new_added,
        previous_controls=controls,protected=len(f["protected"]),
        statement="Same Codex separate arccos and preservation checks, not independent human/linguistic ground truth. Existing mistakes not corrected."))
    print(dict(counts),"added",len(new_added),"controls",len(controls))
if __name__=="__main__":main()
