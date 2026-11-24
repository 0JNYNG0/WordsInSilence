"""Separate saved-array arithmetic and counterfactual checks. No fitting."""
import sys, traceback
from collections import Counter
import numpy as np
import study as d
U,S,A=d.U,d.S,d.A
read,load,write=d.read,d.load,d.write

def main():
    f=read(U/"FROZEN.json"); d.pipe.check(f["pins"]); d.pipe.check(f["protected"])
    count=Counter(); byid={r["clip_id"]:r for r in f["rows"]}
    for r in f["rows"]:
        z=load(U/"rechecked"/(r["clip_id"]+".npz"));obs=load(r["observation"])
        old=load(r["quality"]); xy=obs["xy_px"][:,33:].reshape(-1,2,21,2)
        usable=obs["usable"][:,33:].reshape(-1,2,21)
        assert np.array_equal(z["angle_state"][z["before_state"]!=2],z["before_state"][z["before_state"]!=2])
        # Independent cosine angles, not production atan2.
        chains=((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20))
        for a,(x,y,k) in enumerate((c[i-1],c[i],c[i+1]) for c in chains for i in (1,2,3)):
            v=xy[:,:,x].astype(float)-xy[:,:,y];w=xy[:,:,k].astype(float)-xy[:,:,y]
            norm=np.linalg.norm(v,axis=-1)*np.linalg.norm(w,axis=-1)
            valid=usable[:,:,x]&usable[:,:,y]&usable[:,:,k]&(norm>1e-12)&np.isfinite(norm)
            out=np.degrees(np.arccos(np.clip(np.divide((v*w).sum(-1),norm,out=np.zeros_like(norm),where=valid),-1,1)))
            assert np.allclose(out[valid],z["angle_deg"][:,:,a][valid],atol=1e-5)
        count["coordinate_values_preserved"]+=obs["xy_px"].size
        count["angle_states"]+=z["angle_state"].size
        count["clips"]+=1
    controls=[]
    for c in read(A/"VERIFY.json")["controls"]:
        z=load(U/"rechecked"/(c["clip_id"]+".npz"));i=c["index"]
        assert np.array_equal(z["before_state"][i],np.array(c["shape_evidence_states"]))
        controls.append(dict(**{k:c[k] for k in ("clip_id","index","group","label")},
            before_eligible=int((z["before_state"][i]==4).sum()),
            after_eligible=int((z["angle_state"][i]==4).sum()),coordinates_preserved=True))
        if c["group"]=="known_identity_error":
            assert np.array_equal(z["before_state"][i],z["angle_state"][i])
    primary=[r for r in f["rows"] if r["primary"]]
    schedule_details=[];expected_removed=Counter()
    for qr in primary:
        cid=qr["clip_id"]; refs=[r for r in primary if r["word_code"]==qr["word_code"] and r["signer_id"]!=qr["signer_id"]]
        for rr in refs:
            rid=rr["clip_id"]; target=load(U/"schedules"/(rid+"__source.npz"))
            rv=(target["rest"]&target["core"][:,None]).astype(int)
            av=(target["active"]&target["core"][:,None]).astype(int)
            for other in refs:
                oid=other["clip_id"]
                if oid==rid:continue
                src=load(U/"schedules"/(oid+"__source.npz")); path=load(S/"pairs"/(rid+"__"+oid+".npz"))["path"]
                for i in range(len(rv)):
                    js=[int(b) for a,b in path if a==i]
                    if not js:continue
                    for h in (0,1):
                        rv[i,h]+=all(bool(src["core"][j] and src["rest"][j,h]) for j in js)
                        av[i,h]+=any(bool(src["core"][j] and src["active"][j,h]) for j in js)
            expected=target["core"][:,None]&~(target["rest"]&(rv>=5)&(av==0))
            saved=load(U/"schedules"/(cid+"__"+rid+".npz"))
            assert np.array_equal(expected,saved["expected"])
            assert np.array_equal(rv,saved["rest_votes"]) and np.array_equal(av,saved["active_votes"])
            assert not (saved["rest"]&target["active"]).any()
            count["reference_schedules"]+=1
            for h,side in enumerate(("L","R")):
                ids=np.flatnonzero(saved["rest"][:,h]);expected_removed[(qr["label"],side)]+=len(ids)
                if len(ids):schedule_details.append(dict(query=cid,reference=rid,side=side,rows=ids.tolist()))
    for condition in f["conditions"]:
        for qr in primary:
            cid=qr["clip_id"]; quality=condition in ("recheck_only","combined")
            q=load(qr["prepared"])
            if quality:q.update(load(U/"rechecked"/(cid+".npz")))
            refs=[r for r in primary if r["word_code"]==qr["word_code"] and r["signer_id"]!=qr["signer_id"]]
            independently_ready=np.zeros(30,int);independent_values=[[] for _ in range(30)]
            for rr in refs:
                rid=rr["clip_id"];r=load(rr["prepared"])
                if quality:r.update(load(U/"rechecked"/(rid+".npz")))
                pid=cid+"__"+rid; p=load(U/"pairs"/(condition+"__"+pid+".npz")); j=read(U/"pairs"/(condition+"__"+pid+".json"))
                qi,ri=p["path"].T;t=r["timestamp_us"].astype(float)/1e6
                weights=np.diff(np.r_[t[0],(t[1:]+t[:-1])/2,t[-1]])
                expected_raw=p["expected"][ri]&np.isfinite(q["angle_deg"][qi])&np.isfinite(r["angle_deg"][ri])
                expected_strict=expected_raw&(q["angle_state"][qi]==4)&(r["angle_state"][ri]==4)
                assert np.array_equal(p["raw"],expected_raw) and np.array_equal(p["strict"],expected_strict)
                if condition=="baseline":
                    old=load(S/"pairs"/(pid+".npz"))
                    for k in ("expected","raw","strict","path"): assert np.array_equal(p[k],old[k]),(cid,k)
                for h in (0,1):
                    for a in range(15):
                        row=j["rows"][h*15+a];den=sum(float(weights[k]) for k in np.flatnonzero(p["expected"][:,h,a]))
                        assert abs(den-row["expected_seconds"])<1e-8
                        ok=p["strict"][:,h,a];ids=sorted(set(ri[ok].tolist()));qids=set(qi[ok].tolist())
                        num=sum(float(weights[k]) for k in ids);cov=num/den if den else None
                        if cov is not None:assert abs(cov-row["strict"]["coverage"])<1e-8
                        ready=bool(row["required"] and j["alignment_support"]>=.35 and cov is not None and cov>=.5 and min(len(ids),len(qids))>=8)
                        assert ready==row["strict"]["ready"]
                        if ready:
                            independently_ready[h*15+a]+=1
                            means=[np.abs(q["angle_deg"][qi[ok&(ri==k)],h,a]-r["angle_deg"][k,h,a]).mean() for k in ids]
                            mean=float(np.dot(means,weights[ids])/weights[ids].sum())
                            assert abs(mean-row["strict"]["mean_deg"])<1e-8
                            independent_values[h*15+a].append(mean)
                count["verified_pairs"]+=1
            rows=read(U/"queries"/(condition+"__"+cid+".json"))["rows"]
            for a,row in enumerate(rows):
                assert row["reference_count"]==int(independently_ready[a])
                if row["compared_median_deg"] is not None:
                    assert abs(row["compared_median_deg"]-float(np.median(independent_values[a])))<1e-8
    write(U/"VERIFY.json",dict(status="PASS",counts=dict(count),controls=controls,
        removed_reference_handframes=[dict(word=w,side=h,count=v) for (w,h),v in expected_removed.items()],
        lowered_intervals=schedule_details,independence="same Codex, separate arithmetic; not an independent semantic annotator"))
    print("VERIFY PASS",dict(count),flush=True)


def stress():
    f=read(U/"FROZEN.json");primary=[r for r in f["rows"] if r["primary"]];out=[];short=[]
    for qr in primary:
        cid=qr["clip_id"];q=load(qr["prepared"]);q.update(load(U/"rechecked"/(cid+".npz")))
        required=f["roles"].get(qr["word_code"],{}).get("required_sides",["L","R"])
        refs=[r for r in primary if r["word_code"]==qr["word_code"] and r["signer_id"]!=qr["signer_id"]]
        for h,side in enumerate(("L","R")):
            damaged={k:v.copy() for k,v in q.items()}
            damaged["angle_state"][:,h]=0;damaged["angle_deg"][:,h]=np.nan
            # Remove hand evidence used by alignment too; body elbow/arm stays observed.
            start=0
            for g,(name,(width,_)) in enumerate(d.s.F.GROUPS.items()):
                if (name.endswith("_"+side) and not name.startswith("arm_")) or name=="bimanual":
                    damaged["features"][:,start:start+width]=np.nan;damaged["present"][:,g]=False
                start+=width
            assert start==q["features"].shape[1]
            changed_paths=0
            for rr in refs:
                rid=rr["clip_id"];r=load(rr["prepared"]);r.update(load(U/"rechecked"/(rid+".npz")))
                sch=load(U/"schedules"/(cid+"__"+rid+".npz"));saved=load(U/"pairs"/("combined__"+cid+"__"+rid+".npz"))
                path,support=d.s.oldrun.alignment(damaged,r,required)
                changed_paths+=not np.array_equal(path,saved["path"])
                p=d.candidate.pair(damaged,r,path,required,support>=.35,sch["expected"],d.adapter)
                assert np.array_equal(p["expected"],saved["expected"])
                assert not p["strict"][:,h].any()
                assert all(not x["strict"]["ready"] for x in p["rows"][h*15:h*15+15])
            out.append(dict(clip_id=cid,side=side,paths_recomputed=8,paths_changed=changed_paths,
                            reference_range_preserved=True,missing_hand_available=False))
        # A single angle frame lost cannot delete neighboring angle evidence/range.
        ids=np.argwhere(q["angle_state"]==4)
        if len(ids):
            i,h,a=map(int,ids[len(ids)//2]);damaged={k:v.copy() for k,v in q.items()}
            damaged["angle_state"][i,h,a]=0;damaged["angle_deg"][i,h,a]=np.nan
            for rr in refs:
                rid=rr["clip_id"];r=load(rr["prepared"]);r.update(load(U/"rechecked"/(rid+".npz")))
                saved=load(U/"pairs"/("combined__"+cid+"__"+rid+".npz"))
                support=read(S/"pairs"/(cid+"__"+rid+".json"))["alignment_support"]
                sch=load(U/"schedules"/(cid+"__"+rid+".npz"))
                p=d.candidate.pair(damaged,r,saved["path"],required,support>=.35,sch["expected"],d.adapter)
                assert np.array_equal(p["expected"],saved["expected"])
                assert not (p["strict"]&~saved["strict"]).any()
                difference=p["strict"]!=saved["strict"]
                difference[:,h,a]=False;assert not difference.any()
            short.append(dict(clip_id=cid,index=i,side=h,angle=a,range_and_neighbors_preserved=True))
        print("STRESS",cid,flush=True)
    # Conflicts/unknown cannot be converted to reference rest, even if 7 agree.
    t=dict(core=np.ones(8,bool),rest=np.ones((8,2),bool),active=np.zeros((8,2),bool))
    pp=[dict(core=t["core"],rest=t["rest"],active=t["active"]) for _ in range(7)]
    identity=np.stack([np.arange(8),np.arange(8)],1)
    z=d.candidate.reference_schedule(t,pp,[identity]*7);assert z["rest"].all()
    pp[0]=dict(core=t["core"],rest=np.zeros((8,2),bool),active=np.ones((8,2),bool))
    assert not d.candidate.reference_schedule(t,pp,[identity]*7)["rest"].any()
    unknown=dict(core=t["core"],rest=np.zeros((8,2),bool),active=np.zeros((8,2),bool))
    assert not d.candidate.reference_schedule(unknown,pp,[identity]*7)["rest"].any()
    write(U/"STRESS.json",dict(status="PASS",whole_hand=out,single_angle=short,
        scope="Saved public hand-feature deletion; new DTW paths, fixed reference schedules. Not new RGB occlusion or semantic correctness.",
        synthetic_reference_conflict_and_unknown="PASS",training=0,user_data=0))
    print("STRESS PASS",len(out),len(short),flush=True)

if __name__=="__main__":
    try: main() if sys.argv[1]=="verify" else stress()
    except BaseException:
        p=U/"VERIFY_FIRST_ERROR.txt"
        if not p.exists():p.write_text(traceback.format_exc(),encoding="utf8")
        raise
