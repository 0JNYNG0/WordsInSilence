"""Independent saved geometry/input checks using already separate equations."""
import importlib.util
import json
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
geom=module("wider_independent_geometry",O/"word100_finger_evidence_20261020/verify.py")
feat=module("wider_independent_features",O/"word100_merge_scale_input_20261020/verify.py")
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
load=geom.load;same=geom.same
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    f=read(U/"FROZEN.json");idx=read(U/"INDEX.json")
    assert len(idx)==len(f["rows"])==100 and not (U/"RUNNING.json").exists()
    assert len({r["word_code"] for r in f["rows"]})==100
    assert all(r["role"]=="train_candidate" and r["signer_id"] in range(1,10) for r in f["rows"])
    for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
    counts=Counter();maxerr=0
    for r,item in zip(f["rows"],idx):
        cid=r["clip_id"];assert item["clip_id"]==cid
        for p,h in item["outputs"].items():assert sha(p)==h,p
        out=U/"outputs"/cid;old=load(r["old_masked"]);new=load(out/"merged.npz")
        native_old=load(r["native"]);native=load(out/"merged_native.npz")
        e=load(out/"evidence.npz");x=load(out/"inputs.npz")
        frames=read(out/"detections.json")["frames"]
        for field in ("xy_px","observed","usable","conf"):
            assert same(new[field][old["observed"]],old[field][old["observed"]]),(cid,field)
            assert same(native[field][native_old["observed"]],native_old[field][native_old["observed"]]),(cid,field)
        for field in ("frame_id","timestamp_us"):
            assert same(new[field],old[field]) and same(x[field],old[field]) and same(e[field],old[field])
        inside=geom.inside(native["xy_px"],r["box"])
        assert same(new["observed"],native["observed"]&inside)
        assert same(new["usable"],native["usable"]&inside)
        a,hs=feat.reconstruct(old);b,_=feat.reconstruct(new,hs)
        assert np.allclose(a,x["before"],rtol=0,atol=1e-6)
        assert np.allclose(b,x["merged"],rtol=0,atol=1e-6)
        assert np.allclose(hs,x["old_scales"],equal_nan=True)
        added=(~old["usable"]&new["usable"])[:,33:].reshape(-1,2,21)
        assert same(added,e["source_added"])
        for i,fr in enumerate(frames):
            for side in range(2):
                sl=slice(33+21*side,54+21*side)
                p=new["xy_px"][i,sl].astype(float);v=new["usable"][i,sl]
                pair=np.full((21,2),np.nan);valid=np.zeros(21,bool);kind=0
                if v.any():
                    if not old["usable"][i,sl].any():
                        ds=[d for d in fr["decisions"] if d["accepted"] and d["side"]==side];assert len(ds)<=1
                        if ds:
                            pair=np.array(fr["crop"][ds[0]["crop_index"]]["xy"])
                            valid=v&geom.inside(pair,r["box"]);kind=2
                    else:
                        ds=[d for d in fr["decisions"] if d["reason"]=="confirm_existing_hand" and d["side"]==side]
                        if len(ds)==1:
                            d=ds[0];assert d["matched_old"] and all(fr["baseline_sides"][j]==side for j in d["matched_old"])
                            pair=np.array(fr["full"][d["index"]]["xy"]);valid=geom.inside(pair,r["box"]);kind=1
                assert e["pair_kind"][i,side]==kind and same(e["pair_b"][i,side],pair) and same(e["pair_valid"][i,side],valid)
                g=geom.reference(p,v,pair,valid)
                assert e["pair_available"][i,side]==(g is not None)
                if g is None:
                    assert not e["comparable"][i,side].any() and not e["point_uncertain"][i,side].any()
                else:
                    for key,value in g.items():
                        actual=e[key][i,side]
                        if np.asarray(value).dtype.kind in "biu":assert same(actual,value),(cid,i,side,key)
                        else:
                            assert np.allclose(actual,value,rtol=1e-8,atol=1e-6,equal_nan=True),(cid,i,side,key)
                            good=np.isfinite(value)&np.isfinite(actual)
                            if good.any():maxerr=max(maxerr,float(np.max(abs(np.asarray(value)[good]-np.asarray(actual)[good]))))
                    counts["independent_geometry_pairs"]+=1
        hold=(e["point_uncertain"]|~e["comparable"])&added
        assert same(hold,e["shape_hold_new"])
        y=x["merged"].copy()
        for i,side,j in np.argwhere(hold):
            if j==0:continue
            n=21*side+j;y[i,102+2*n:104+2*n]=0;y[i,237+n]=0
            if j not in geom.PALM:
                n+=9;y[i,2*n:2*n+2]=0;y[i,186+n]=0
        assert same(y,x["safe"]) and np.isfinite(y).all()
        for start,stop,valid in [(0,102,x["before"][:,186:237].repeat(2,1).astype(bool)),
                                 (102,186,x["before"][:,237:279].repeat(2,1).astype(bool)),
                                 (186,279,x["before"][:,186:279].astype(bool))]:
            assert same(y[:,start:stop][valid],x["before"][:,start:stop][valid])
        for side in range(2):
            for j in geom.PALM:
                k=9+21*side+j
                assert same(y[:,2*k:2*k+2],x["merged"][:,2*k:2*k+2])
                assert same(y[:,186+k],x["merged"][:,186+k])
        assert same(y[:,279],x["merged"][:,279])
        counts["frames"]+=len(y);counts["input_values"]+=y.size*3
        counts["old_observed_points_preserved"]+=int(old["observed"].sum())
        counts["clips"]+=1
    old_verify=read(O/"word100_finger_evidence_20261020/VERIFY.json")
    assert old_verify["counts"]["crossing_controls"]==53 and old_verify["counts"]["normal_controls"]==3
    result=dict(status="PUBLIC100_INPUT_AND_GEOMETRY_VERIFIED",counts=dict(counts),
        prior_control_reuse=dict(normal=3,crossing=53,source="Previously verified hashes, no rerun"),
        maximum_geometry_equation_error=maxerr,protected_files=len(f["protected"]),
        implementation="Codex",verification="Same Codex, separate complex geometry and direct feature equations; no other reviewer",
        note="No new manual identity labels; preserved coordinates are not exact finger truth.",
        training_runs=0,classifier_runs=0,user_data_reads=0)
    with (U/"VERIFY.json").open("x",encoding="utf8") as fp:json.dump(result,fp,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":main()
