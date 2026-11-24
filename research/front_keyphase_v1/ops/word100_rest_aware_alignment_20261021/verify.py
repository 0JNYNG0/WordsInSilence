"""Separate arithmetic/path checks and missing-input counterfactuals."""
import sys,traceback
from collections import Counter
import numpy as np
import study as d
U,R=d.U,d.R

def independent(q,r,required,rest):
    n,m=len(q["features"]),len(r["features"])
    numerator=np.zeros((n,m));costden=np.zeros((n,m));evidence=np.zeros((n,m));expected=np.zeros(m)
    offset=0
    for g,(name,(width,w0)) in enumerate(d.F.GROUPS.items()):
        weight=np.full(m,float(w0) if len(required)==2 or name.endswith("_"+required[0]) else 0.)
        if name=="bimanual":
            use=np.zeros(m,bool)
            for h,side in enumerate(("L","R")):
                if side in required:use|=rest[:,h]
            weight[use]=0.
        for h,side in enumerate(("L","R")):
            if side in required and name.endswith("_"+side) and not name.startswith("arm_"):
                weight[rest[:,h]]=0.
        qa=q["features"][:,offset:offset+width];rb=r["features"][:,offset:offset+width];offset+=width
        pa=q["present"][:,g];pb=r["present"][:,g]
        both=pa[:,None]&pb[None,:];one=pa[:,None]^pb[None,:]
        valid=both&np.isfinite(qa).all(1)[:,None]&np.isfinite(rb).all(1)[None,:]
        delta=qa[:,None]-rb[None,:]
        # Features are float32. Runtime sums squares at that precision, then
        # divides by integer counts as float64. Keep the same arithmetic type
        # without reusing runtime's reducer; mean(float32) rounded too early.
        distances=np.sqrt(np.sum(delta**2,axis=2).astype(np.float64)/width)
        numerator+=(np.where(valid,distances,0)+1.5*one)*weight[None,:]
        costden+=(valid|one)*weight[None,:]
        evidence+=valid*weight[None,:]
        expected+=weight
    cost=np.divide(numerator,costden,out=np.zeros_like(numerator),where=costden>0)
    ev=evidence/np.maximum(expected[None,:],1e-9)
    return cost,ev,expected

def path_minimum(cost):
    n,m=cost.shape;band=max(1,round(.2*max(n,m)))
    a=np.full((n+1,m+1),np.inf);a[0,0]=0.
    for i in range(n):
        centre=i*(m-1)/(n-1)
        for j in range(max(0,int(np.floor(centre-band))),min(m,int(np.ceil(centre+band))+1)):
            a[i+1,j+1]=cost[i,j]+min(a[i,j],a[i,j+1],a[i+1,j])
    return float(a[-1,-1])

def verify():
    f=d.read(U/"FROZEN.json");d.prev.pipe.check(f["pins"]);d.prev.pipe.check(f["protected"])
    counts=Counter();not_changed=0
    for qr in f["rows"]:
        cid=qr["clip_id"];q=d.sample(qr);required=f["roles"].get(qr["word_code"],{}).get("required_sides",["L","R"])
        refs=[r for r in f["rows"] if r["word_code"]==qr["word_code"] and r["signer_id"]!=qr["signer_id"]]
        sumrows={c:np.zeros(30,int) for c in d.CONDITIONS}
        for rr in refs:
            rid=rr["clip_id"];r=d.sample(rr);pid=cid+"__"+rid;sch=d.load(R/"schedules"/(pid+".npz"))
            independent_cost,ev,expected=independent(q,r,required,sch["rest"])
            mat=d.C.matrices(q,r,required,sch["rest"],d.F.GROUPS,d.legacy)
            assert np.allclose(mat["cost"],independent_cost,atol=1e-12,rtol=1e-12)
            assert np.array_equal(mat["evidence"],ev)
            assert np.array_equal(mat["expected_group_weight"],expected)
            counts["cost_cells"]+=ev.size
            for condition in d.CONDITIONS:
                p=d.load(U/"pairs"/(condition+"__"+pid+".npz"));j=d.read(U/"pairs"/(condition+"__"+pid+".json"))
                base=d.load(R/"pairs"/("combined__"+pid+".npz"))
                assert np.array_equal(p["expected"],base["expected"])
                qi,ri=p["path"].T
                assert (qi[0],ri[0])==(0,0) and (qi[-1],ri[-1])==(len(q["features"])-1,len(r["features"])-1)
                steps=np.diff(p["path"],axis=0)
                assert ((steps>=0)&(steps<=1)).all() and (steps.sum(1)>=1).all()
                if condition=="baseline":
                    for k in ("raw","strict","path","expected","difference_deg"):
                        assert np.array_equal(p[k],base[k],equal_nan=True)
                else:
                    assert abs(float(ev[qi,ri].mean())-j["alignment_support"])<1e-12
                if condition=="rest_alignment":
                    assert abs(float(independent_cost[qi,ri].sum())-path_minimum(independent_cost))<1e-8
                    if not mat["rest_code"].any():
                        assert np.array_equal(p["path"],base["path"]);not_changed+=1
                raw=p["expected"][ri]&np.isfinite(q["angle_deg"][qi])&np.isfinite(r["angle_deg"][ri])
                strict=raw&(q["angle_state"][qi]==4)&(r["angle_state"][ri]==4)
                assert np.array_equal(raw,p["raw"]) and np.array_equal(strict,p["strict"])
                t=r["timestamp_us"].astype(float)/1e6;w=np.diff(np.r_[t[0],(t[1:]+t[:-1])/2,t[-1]])
                for h in (0,1):
                    for a in range(15):
                        row=j["rows"][h*15+a];mask=strict[:,h,a];ids=np.unique(ri[mask]);nq=len(np.unique(qi[mask]))
                        den=float(w@p["expected"][:,h,a]);cov=float(w[ids].sum()/den) if den else None
                        assert abs(den-row["expected_seconds"])<1e-9
                        assert cov is None or abs(cov-row["strict"]["coverage"])<1e-9
                        ready=bool(row["required"] and j["alignment_support"]>=.35 and cov is not None and cov>=.5 and min(nq,len(ids))>=8)
                        assert ready==row["strict"]["ready"];sumrows[condition][h*15+a]+=ready
                        if len(ids):
                            means=[np.abs(q["angle_deg"][qi[mask&(ri==k)],h,a]-r["angle_deg"][k,h,a]).mean() for k in ids]
                            mean=float(np.average(means,weights=w[ids]))
                            assert abs(mean-row["strict"]["mean_deg"])<1e-8
                counts["pair_results"]+=1
        for c in d.CONDITIONS:
            rows=d.read(U/"queries"/(c+"__"+cid+".json"))["rows"]
            for a,row in enumerate(rows):
                assert row["reference_count"]==int(sumrows[c][a])
                assert (row["status"]=="PARTIAL_COMPARISON")==bool(sumrows[c][a]>=5)
        counts["queries"]+=1
    d.write(U/"VERIFY.json",dict(status="PASS",counts=dict(counts),no_rest_path_exactly_preserved=not_changed,
        verification="same Codex separate group arithmetic, shortest-path objective, frame coverage and summary checks; no independent semantic annotator",
        protected=32))
    print("VERIFY",dict(counts),not_changed,flush=True)

def damage(q,side,indices):
    out={k:v.copy() for k,v in q.items()};h=("L","R").index(side)
    out["angle_state"][indices,h]=0;out["angle_deg"][indices,h]=np.nan
    start=0
    for g,(name,(width,_)) in enumerate(d.F.GROUPS.items()):
        if (name.endswith("_"+side) and not name.startswith("arm_")) or name=="bimanual":
            out["features"][indices,start:start+width]=np.nan;out["present"][indices,g]=False
        start+=width
    return out

def stress():
    f=d.read(U/"FROZEN.json");whole=[];short=[]
    for qr in f["rows"]:
        cid=qr["clip_id"];q=d.sample(qr);required=f["roles"].get(qr["word_code"],{}).get("required_sides",["L","R"])
        refs=[r for r in f["rows"] if r["word_code"]==qr["word_code"] and r["signer_id"]!=qr["signer_id"]]
        core=np.flatnonzero(q["core"]);mid=len(core)//2;window=core[max(0,mid-1):mid+2]
        for typ,side,indices in [("whole","L",np.arange(len(q["core"]))),("whole","R",np.arange(len(q["core"]))),("short",required[-1],window)]:
            h=("L","R").index(side);changed=0;covup=0;readyup=0;lost=0;compared=0
            z=damage(q,side,indices)
            for rr in refs:
                rid=rr["clip_id"];r=d.sample(rr);pid=cid+"__"+rid
                sch=d.load(R/"schedules"/(pid+".npz"));old=d.load(U/"pairs"/("rest_alignment__"+pid+".npz"))
                oldj=d.read(U/"pairs"/("rest_alignment__"+pid+".json"))
                path,support,mat=d.C.align(z,r,required,sch["rest"],d.F.GROUPS,d.legacy,d.dp)
                assert np.array_equal(mat["expected_group_weight"],old["expected_group_weight"])
                p=d.prev.candidate.pair(z,r,path,required,support>=.35,sch["expected"],d.adapter)
                assert np.array_equal(p["expected"],old["expected"])
                assert not p["strict"][np.isin(path[:,0],indices),h].any()
                changed+=not np.array_equal(path,old["path"])
                for a,b in zip(oldj["rows"][h*15:h*15+15],p["rows"][h*15:h*15+15]):
                    aa,bb=a["strict"],b["strict"]
                    covup+=aa["coverage"] is not None and bb["coverage"] is not None and bb["coverage"]>aa["coverage"]+1e-9
                    readyup+=(not aa["ready"]) and bb["ready"]
                    lost+=aa["ready"] and (not bb["ready"]);compared+=bb["ready"]
                if typ=="whole": assert not p["strict"][:,h].any() and compared==0
                np.savez_compressed(U/"stress"/(typ+"_"+side+"__"+pid+".npz"),
                    path=path,expected=p["expected"],strict=p["strict"],removed_query_rows=indices)
            row=dict(query=cid,side=side,removed_rows=indices.tolist(),changed_paths=changed,
                     reference_range_preserved=True,expected_group_weight_preserved=True,
                     coverage_increase_rows=int(covup),ready_increase_rows=int(readyup),ready_lost_rows=int(lost),
                     ready_pair_angles=int(compared),wrong_action=None)
            (whole if typ=="whole" else short).append(row)
        print("STRESS",cid,flush=True)
    d.write(U/"STRESS.json",dict(status="PASS_RANGE_AND_NO_ABSENT_HAND_SCORE",whole=whole,short=short,
        limit="Short deletion is 3 saved core frames, not annotated essential phase. Re-alignment may redirect surviving evidence and increase coverage; explicitly counted, no whole-action correctness verdict. Synthetic feature deletion, not RGB detector rerun."))
    print("STRESS DONE",len(whole),len(short),sum(x["coverage_increase_rows"] for x in short),flush=True)

if __name__=="__main__":
    try:
        if sys.argv[1]=="verify":verify()
        else:(U/"stress").mkdir();stress()
    except BaseException:
        p=U/"VERIFY_FIRST_ERROR.txt"
        if not p.exists():p.write_text(traceback.format_exc(),encoding="utf8")
        raise
