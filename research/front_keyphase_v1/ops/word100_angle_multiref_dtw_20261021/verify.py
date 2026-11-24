"""Separate saved-output arithmetic audit. Does not call adapter functions."""
import sys, json, hashlib, traceback
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent
O=U.parent
A=O/"word100_angle_evidence_20261021"
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open("x",encoding="utf8") as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def weights(t):
    t=np.asarray(t,float)/1e6
    return np.diff(np.r_[t[0],(t[1:]+t[:-1])/2,t[-1]])
def close(a,b):
    assert (a is None)==(b is None),(a,b)
    if a is not None:assert abs(a-b)<1e-9,(a,b)

def main():
    f=read(U/"FROZEN.json");assert not (U/"RUNNING.json").exists()
    for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
    assert read(U/"RUN_RESULT.json")["status"]=="COMPLETE_PENDING_VERIFY"
    source_rows={r["clip_id"]:r for r in f["rows"]}
    count=Counter();out_hash={}
    for word in sorted({r["word_code"] for r in f["rows"]}):
        rows=[r for r in f["rows"] if r["word_code"]==word]
        cache={r["clip_id"]:load(U/"prepared"/(r["clip_id"]+".npz")) for r in rows}
        for r in rows:
            s=load(r["observation"]);z=cache[r["clip_id"]]
            for k in ("xy_px","usable","timestamp_us","frame_id"):
                assert np.array_equal(s[k],z[k],equal_nan=True),(r["clip_id"],k)
            count["coordinate_values_preserved"]+=z["xy_px"].size
            count["all_bank_frames"]+=len(z["frame_id"])
            assert np.isfinite(z["angle_deg"]).sum()==np.isin(z["angle_state"],[2,3,4]).sum()
            if r["evidence"]:
                e=load(r["evidence"])
                for k in ("angle_deg","angle_state"):assert np.array_equal(e[k],z[k],equal_nan=True)
            else:
                assert not np.isin(z["angle_state"],[3,4]).any()
            count["bank_clips"]+=1
        for row in rows:
            if not row["query"]:continue
            cid=row["clip_id"];q=cache[cid];out=read(U/"queries"/(cid+".json"))
            qsave=load(U/"queries"/(cid+".npz"));v=np.zeros_like(qsave["votes"]);ev=np.zeros_like(v)
            per_angle=[[] for _ in range(30)];people=[]
            for pid in out["pairs"]:
                meta=read(U/"pairs"/(pid+".json"));p=load(U/"pairs"/(pid+".npz"))
                rr=source_rows[meta["reference"]];r=cache[meta["reference"]]
                assert rr["word_code"]==row["word_code"] and rr["signer_id"]!=row["signer_id"]
                people.append(rr["signer_id"])
                path=p["path"];qi,ri=path.T
                assert np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[len(q["frame_id"])-1,len(r["frame_id"])-1])
                assert all(tuple(x) in ((1,0),(0,1),(1,1)) for x in np.diff(path,axis=0))
                expect=np.repeat(r["core"][:,None,None],2,1);expect=np.repeat(expect,15,2)
                for h,side in enumerate(("L","R")):
                    if side not in out["required"]:expect[:,h]=False
                assert np.array_equal(expect,p["expected"])
                raw=expect[ri]&np.isfinite(q["angle_deg"][qi])&np.isfinite(r["angle_deg"][ri])
                strict=raw&(q["angle_state"][qi]==4)&(r["angle_state"][ri]==4)
                assert np.array_equal(raw,p["raw"]) and np.array_equal(strict,p["strict"])
                want=np.where(strict,abs(q["angle_deg"][qi]-r["angle_deg"][ri]),np.nan)
                assert np.array_equal(want,p["difference_deg"],equal_nan=True)
                # Independent loops project one vote per reference person, not one per DTW step.
                seen=np.zeros_like(v,bool);ex=np.zeros_like(v,bool)
                for k,(i,j) in enumerate(path):
                    if meta["alignment_support"]>=.35:seen[i]|=strict[k]
                    ex[i]|=expect[j]
                v+=seen;ev+=ex
                w=weights(r["timestamp_us"])
                for h in range(2):
                    for a in range(15):
                        rec=meta["rows"][h*15+a]
                        close(rec["expected_seconds"],float(w[expect[:,h,a]].sum()))
                        for name,mask in (("raw",raw),("strict",strict)):
                            chosen=mask[:,h,a];unique=sorted(set(ri[chosen]))
                            den=float(w[expect[:,h,a]].sum())
                            cov=sum(float(w[j]) for j in unique)/den if den else None
                            sums=0.;wt=0.
                            for j in unique:
                                vals=[abs(float(q["angle_deg"][i,h,a])-float(r["angle_deg"][j,h,a]))
                                      for i,jj in path[chosen] if jj==j]
                                sums+=sum(vals)/len(vals)*float(w[j]);wt+=float(w[j])
                            mean=sums/wt if wt else None
                            close(rec[name]["mean_deg"],mean);close(rec[name]["coverage"],cov)
                            ready=rec["required"] and meta["alignment_support"]>=.35 and cov is not None and cov>=.5 and min(len(set(qi[chosen])),len(unique))>=8
                            assert bool(ready)==rec[name]["ready"]
                        per_angle[h*15+a].append(rec)
                count["pairs"]+=1
                count["path_pairs"]+=len(path)
                count["strict_angle_pairs"]+=int(strict.sum());count["raw_angle_pairs"]+=int(raw.sum())
            assert len(people)==len(set(people))
            assert np.array_equal(v,qsave["votes"]) and np.array_equal(ev,qsave["expected_votes"])
            assert all(x<=len(people) for x in v.ravel())
            for k,result in enumerate(out["rows"]):
                h,a=divmod(k,15);side=("L","R")[h]
                for rawname,resultname in (("raw","raw_only_median_deg"),("strict","compared_median_deg")):
                    vals=[x[rawname]["mean_deg"] for x in per_angle[k] if x[rawname]["ready"]]
                    want=float(np.median(vals)) if len(vals)>=5 and side in out["required"] else None
                    close(result[resultname],want)
                strict_n=sum(x["strict"]["ready"] for x in per_angle[k])
                mask=q["core"]&(v[:,h,a]<5) if side in out["required"] else np.zeros(len(v),bool)
                covered=np.zeros(len(v),bool)
                for it in result["missing_intervals"]:covered[it["first_row"]:it["last_row"]+1]=True
                assert np.array_equal(covered,mask)
                target="NOT_REQUIRED" if side not in out["required"] else "INSUFFICIENT" if strict_n<5 else "PARTIAL" if mask.any() else "OBSERVED"
                assert target==result["status"]
                assert result["accepted"] is None and result["sign_error"] is None
                count["angle_"+target]+=1
            assert out["whole_hand_score"] is None
            count["query_clips"]+=1
        print("verified",word,flush=True)
    controls=read(A/"VERIFY.json")["controls"]
    control_counts=Counter()
    for r in controls:
        cid=r["clip_id"];z=load(U/"prepared"/(cid+".npz"));e=load(A/"outputs"/(cid+".npz"));i=r["index"]
        for k in ("xy_px","usable","angle_deg","angle_state"):
            assert np.array_equal(z[k][i],e[k][i],equal_nan=True),(cid,i,k)
        control_counts[r["group"]]+=1
    for sub in ("prepared","pairs","queries"):
        for p in (U/sub).iterdir():out_hash[str(p)]=sha(p)
    write(U/"VERIFY.json",dict(status="PASS",counts=dict(count),controls=dict(control_counts),
          protected=len(f["protected"]),implementation_and_checker_same_person=True,
          statement="independent stored-mask, time weights, median and interval arithmetic; not independent human sign annotation",
          outputs=out_hash))
    print(dict(count),dict(control_counts))
if __name__=="__main__":
    try:main()
    except BaseException:
        p=U/"VERIFY_FIRST_ERROR.txt"
        if not p.exists():p.write_text(traceback.format_exc(),encoding="utf8")
        raise
