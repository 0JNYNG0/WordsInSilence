"""Independent coverage audit, without importing the descriptive analyzer."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
TRIPLES=[(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),
         (0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),
         (14,15,16),(0,17,18),(17,18,19),(18,19,20)]
f=read(U/"FROZEN.json");report=read(U/"ANALYSIS.json")
reported={r["clip_id"]:r for r in report["rows"]};counts=Counter();motion=0.;removed=0.
for r in f["rows"]:
    cid=r["clip_id"];p=U/"outputs"/cid
    z=load(p/"inputs.npz");obs=load(p/"merged.npz");loss=load(p/"loss_audit.npz")
    e=load(p/"evidence.npz")
    xy=obs["xy_px"][:,33:].reshape(-1,2,21,2).astype(float)
    times=z["timestamp_us"]/1e6
    core=np.array([any(a<=t<=b for a,b in r["intervals"]) for t in times])
    assert np.array_equal(core,loss["core"])
    masks={}
    angles=np.full((len(xy),2,15),np.nan)
    for name in ("before","merged","safe"):
        valid=z[name][:,237:279].reshape(-1,2,21).astype(bool)
        out=np.zeros((len(xy),2,15),bool)
        for k,(a,b,c) in enumerate(TRIPLES):
            u=xy[...,a,:]-xy[...,b,:];v=xy[...,c,:]-xy[...,b,:]
            good=valid[...,[a,b,c]].all(-1)&(np.linalg.norm(u,axis=-1)>1e-6)&(np.linalg.norm(v,axis=-1)>1e-6)
            out[...,k]=good
            if name=="merged":
                # angle of complex quotient, independently from analyzer's dot/cross.
                uc=u[...,0]+1j*u[...,1];vc=v[...,0]+1j*v[...,1]
                product=uc*np.conj(vc)
                angles[...,k]=np.where(good,abs(np.angle(product,deg=True)),np.nan)
        masks[name]=out
        assert np.array_equal(out,loss["angle_valid_"+name]),(cid,name)
    lost=masks["merged"]&~masks["safe"];core3=core[:,None,None]
    allgone=masks["merged"].any(-1)&~masks["safe"].any(-1)
    c=dict(before_angles=int(masks["before"].sum()),merged_angles=int(masks["merged"].sum()),
           safe_angles=int(masks["safe"].sum()),lost_angles=int(lost.sum()),
           lost_core_angles=int((lost&core3).sum()),
           core_merged_angles=int((masks["merged"]&core3).sum()),
           core_safe_angles=int((masks["safe"]&core3).sum()),
           core_hand_frames_with_shape_loss=int((lost.any(-1)&core[:,None]).sum()),
           core_hand_frames_losing_all_computable_angles=int((allgone&core[:,None]).sum()),
           core_frames_losing_all_hands_angles=int((core&masks["merged"].any((1,2))&~masks["safe"].any((1,2))).sum()))
    for k,value in c.items():assert reported[cid][k]==value,(cid,k)
    assert np.array_equal(loss["lost_angles"],lost)
    transition=masks["merged"][1:]&masks["merged"][:-1]&core[1:,None,None]&core[:-1,None,None]
    kept=masks["safe"][1:]&masks["safe"][:-1]
    delta=np.where(transition,abs(np.nan_to_num(angles[1:]-angles[:-1])),0.)
    gone=np.where(~kept,delta,0.)
    # Analyzer uses source float32; direct equations here use float64.
    assert np.allclose(gone,loss["removed_angle_variation"],atol=1e-3,rtol=1e-6)
    motion+=float(delta.sum());removed+=float(gone.sum())
    counts.update(c)
for k,value in counts.items():assert report["counts"][k]==value,k
assert np.isclose(motion,report["core_angle_variation_deg"],rtol=1e-6,atol=1e-3)
assert np.isclose(removed,report["removed_core_angle_variation_deg"],rtol=1e-6,atol=1e-3)
result=dict(status="DESCRIPTIVE_LOSS_COUNTS_INDEPENDENTLY_RECONSTRUCTED",
    counts=dict(counts),core_angle_variation_deg=motion,removed_core_angle_variation_deg=removed,
    interpretation="No motion-semantics or joint ground truth. Counting reduced evidence only.",
    verification="Same Codex, separate complex-angle and mask reconstruction.")
with (U/"LOSS_VERIFY.json").open("x",encoding="utf8") as fp:json.dump(result,fp,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
