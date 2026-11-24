"""Independent complex-number geometry and saved-channel verification.

Does not import candidate.py or run.py. Same implementer; separate equations,
not an independent human/agent review.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

U=Path(__file__).resolve().parent
O=U.parent
Q=O/"word100_merge_scale_input_20261020"
PALM=[0,5,9,13,17]
CHAINS=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],
        [0,13,14,15,16],[0,17,18,19,20]]
EPS=1e-8
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):return np.array_equal(a,b,equal_nan=True)
def inside(x,b):
    return np.isfinite(x).all(-1)&(x[...,0]>=b[0])&(x[...,0]<b[2])&(x[...,1]>=b[1])&(x[...,1]<b[3])


def reference(a,va,b,vb):
    both=va&vb&np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
    if not both[PALM].all():return None
    ac=a[:,0]+1j*a[:,1];bc=b[:,0]+1j*b[:,1]
    am=ac[PALM].mean();bm=bc[PALM].mean()
    aa=ac[PALM]-am;bb=bc[PALM]-bm
    den=np.vdot(bb,bb).real
    if den<=EPS or np.vdot(aa,aa).real<=EPS:return None
    # Proper 2D similarity: complex multiplication cannot reflect a hand.
    z=np.vdot(bb,aa)/den
    if abs(z)<=EPS:return None
    aligned=(bc-bm)*z+am
    aligned=np.stack((aligned.real,aligned.imag),-1)
    radius=np.linalg.norm(a-aligned,axis=1)/2
    center=(a+aligned)/2
    ba=np.zeros(20,bool);bu=ba.copy();br=np.full(20,np.nan)
    avail=np.zeros(15,bool);unresolved=avail.copy()
    lo=np.full(15,np.nan);hi=lo.copy()
    flags=np.zeros(21,bool);fu=np.zeros(5,bool);fm=fu.copy();fc=np.zeros(5,np.uint8)
    widths=np.full(20,np.nan)
    for f,ch in enumerate(CHAINS):
        for j in range(4):
            k=f*4+j;i,l=ch[j:j+2]
            if not (both[i] and both[l]):continue
            ba[k]=True
            length=abs(complex(*(center[l]-center[i])))
            r=radius[i]+radius[l]
            shortest=min(np.linalg.norm(a[l]-a[i]),np.linalg.norm(aligned[l]-aligned[i]),length)
            bu[k]=shortest<=EPS or r>=length
            br[k]=r/max(length,EPS)
            widths[k]=180 if bu[k] else np.degrees(np.arcsin(min(1.,r/length)))
            if bu[k]:flags[[i,l]]=True
        for j in range(3):
            k=f*3+j;b1=f*4+j;b2=b1+1
            if not (ba[b1] and ba[b2]):continue
            avail[k]=True;i,m,l=ch[j:j+3]
            u=center[i]-center[m];v=center[l]-center[m]
            theta=np.degrees(np.arctan2(abs(np.linalg.det(np.stack((u,v)))),np.dot(u,v)))
            w=widths[b1]+widths[b2]
            lo[k]=max(0.,theta-w);hi[k]=min(180.,theta+w)
            unresolved[k]=bu[b1] or bu[b2] or (theta-w<=0 and theta+w>=180)
            if unresolved[k]:flags[[i,m,l]]=True
        fu[f]=bu[f*4:f*4+4].any() or unresolved[f*3:f*3+3].any()
        fm[f]=not both[ch].all()
        fc[f]=avail[f*3:f*3+3].sum()
    return dict(aligned_b=aligned,alignment_scale=abs(z),comparable=both,
        residual_px=np.where(both,2*radius,np.nan),point_uncertain=flags,
        bone_available=ba,bone_unresolved=bu,bone_discrepancy_ratio=br,
        angle_available=avail,angle_unresolved=unresolved,
        angle_envelope_low_deg=lo,angle_envelope_high_deg=hi,
        finger_uncertain=fu,finger_missing=fm,finger_available_count=fc)


def main():
    f=read(U/"FROZEN.json")
    assert not (U/"RUNNING.json").exists()
    for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
    index={r["clip_id"]:r for r in read(U/"INDEX.json")}
    cross=read(O/"word100_image_arm_connectivity_20261018/CROSSING.json")
    normals=read(O/"word100_image_arm_connectivity_20261018/ANNOTATED.json")
    counts=Counter();maxerr=0.;checks=[]
    for row in f["rows"]:
        cid=row["clip_id"]
        for p,h in index[cid]["outputs"].items():assert sha(p)==h,p
        e=load(U/"evidence"/(cid+".npz"));obs=load(row["merged"])
        old=load(row["old_masked"]);records=read(row["record"])["frames"]
        prior=load(Q/"evidence"/(cid+".npz"))
        fixed=load(Q/"inputs"/cid/"merge_fixed_scale.npz")["x"]
        baseline=load(Q/"inputs"/cid/"full_before.npz")["x"]
        gated=load(U/"inputs"/(cid+".npz"))
        for field in ("xy_px","observed","usable"):
            assert same(e["position_"+field],obs[field]),(cid,field)
        for field in ("frame_id","timestamp_us"):
            assert same(e[field],obs[field]) and same(gated[field],obs[field])
        for field in ("wrist_available","palm_available","palm_xy_px"):
            assert same(e[field],prior[field]),(cid,field)
        assert same(e["source_added"],prior["point_origin"]==2)
        for i in range(len(records)):
            fr=records[i]
            for s in range(2):
                start=33+21*s
                a=obs["xy_px"][i,start:start+21].astype(float)
                va=obs["usable"][i,start:start+21]
                old_present=old["usable"][i,start:start+21].any()
                b=np.full((21,2),np.nan);vb=np.zeros(21,bool);kind=0
                if va.any():
                    if not old_present:
                        selected=[d for d in fr["decisions"] if d["accepted"] and d["side"]==s]
                        assert len(selected)<=1
                        if selected:
                            b=np.array(fr["crop"][selected[0]["crop_index"]]["xy"])
                            vb=va&inside(b,row["box"]);kind=2
                    else:
                        selected=[d for d in fr["decisions"] if d["side"]==s and d["reason"]=="confirm_existing_hand"]
                        if len(selected)==1:
                            d=selected[0]
                            assert d["matched_old"] and all(fr["baseline_sides"][j]==s for j in d["matched_old"])
                            b=np.array(fr["full"][d["index"]]["xy"])
                            vb=inside(b,row["box"]);kind=1
                assert e["pair_kind"][i,s]==kind
                assert same(e["pair_b"][i,s],b) and same(e["pair_valid"][i,s],vb)
                expected=reference(a,va,b,vb)
                assert e["pair_available"][i,s]==(expected is not None)
                if expected is None:
                    for field in ("comparable","point_uncertain","bone_available","bone_unresolved",
                                  "angle_available","angle_unresolved","finger_uncertain","finger_available_count"):
                        assert not e[field][i,s].any()
                    assert e["finger_missing"][i,s].all()
                    assert np.isnan(e["aligned_b"][i,s]).all()
                else:
                    for key,value in expected.items():
                        actual=e[key][i,s]
                        if np.asarray(value).dtype.kind in "biu":
                            assert same(actual,value),(cid,i,s,key)
                        else:
                            assert np.allclose(actual,value,atol=1e-6,rtol=1e-8,equal_nan=True),(cid,i,s,key)
                            finite=np.isfinite(value)&np.isfinite(actual)
                            if np.any(finite):maxerr=max(maxerr,float(np.max(np.abs(np.asarray(value)[finite]-np.asarray(actual)[finite]))))
                    counts["independently_reconstructed_hand_comparisons"]+=1
        expected_hold=(e["point_uncertain"]|~e["comparable"])&e["source_added"]
        assert same(expected_hold,e["shape_hold_new"])
        y=fixed.copy()
        for i,s,j in np.argwhere(expected_hold):
            if j==0:continue
            n=21*s+j
            y[i,102+2*n:104+2*n]=0;y[i,237+n]=0
            if j not in PALM:
                n+=9
                y[i,2*n:2*n+2]=0;y[i,186+n]=0
        assert same(y,gated["x"]) and np.isfinite(y).all()
        assert same(y[:,186:237],gated["body_valid"])
        assert same(y[:,237:279],gated["hand_valid"])
        for left,right,mask in [(0,102,baseline[:,186:237].repeat(2,1).astype(bool)),
                                 (102,186,baseline[:,237:279].repeat(2,1).astype(bool)),
                                 (186,279,baseline[:,186:279].astype(bool))]:
            assert same(y[:,left:right][mask],baseline[:,left:right][mask]),(cid,left)
        assert same(y[:,:18],fixed[:,:18]) and same(y[:,279],fixed[:,279])
        for s in range(2):
            for j in PALM:
                n=9+s*21+j
                assert same(y[:,2*n:2*n+2],fixed[:,2*n:2*n+2])
                assert same(y[:,186+n],fixed[:,186+n])
        counts["body_finger_points_held"]+=int(((fixed[:,186:237]>0)&(y[:,186:237]==0)).sum())
        counts["local_shape_points_held"]+=int(((fixed[:,237:279]>0)&(y[:,237:279]==0)).sum())
        counts["reconstructed_input_values"]+=y.size
        counts["retained_raw_joint_frames"]+=int(obs["observed"].sum())
        for items,name in ((cross,"crossing_controls"),(normals,"normal_controls")):
            for item in items:
                if item["clip_id"]==cid and item["view"]=="masked":
                    i=item["index"]
                    if name=="normal_controls":assert item["verdict"]=="normal_identity"
                    mask=old["observed"][i]
                    assert same(e["position_xy_px"][i][mask],old["xy_px"][i][mask])
                    counts[name]+=1
        counts["clips"]+=1;counts["frames"]+=len(y)
        checks.append(dict(clip_id=cid,raw_position_preserved=True,
                           baseline_model_values_preserved=True,palm_body_values_preserved=True,
                           new_shape_mask_reconstructed=True))
    assert counts["clips"]==36 and counts["frames"]==4641
    assert counts["normal_controls"]==3 and counts["crossing_controls"]==53
    for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
    result=dict(status="GEOMETRY_AND_INPUT_PRESERVATION_PASS",counts=dict(counts),
        maximum_geometry_equation_error=maxerr,checks=checks,
        protected_files=len(f["protected"]),implementation="Codex",
        verification="Same Codex, separate complex-number equations and input reconstruction; no independent reviewer.",
        caveat="These are retained controls and prediction disagreement, not new ground-truth finger accuracy.",
        training_runs=0,classifier_evaluations=0,user_data_reads=0,production_changes=0)
    with (U/"VERIFY.json").open("x",encoding="utf8") as fp:json.dump(result,fp,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},ensure_ascii=False))

if __name__=="__main__":main()
