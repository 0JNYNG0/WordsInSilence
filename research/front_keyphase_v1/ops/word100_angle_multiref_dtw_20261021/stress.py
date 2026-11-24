"""Public-derived counterfactual evidence omissions on the SAME saved alignment.

These isolate the reporting adapter. They are not new filmed signs, tracking tests,
recomputed alignment robustness, or recognition accuracy.
"""
import copy
from collections import Counter
import numpy as np
from run import U, read, load, write
from adapter import pair, aggregate

def main():
    counts=Counter();details=[]
    for ix in read(U/"INDEX.json"):
        cid=ix["clip_id"];q=load(U/"prepared"/(cid+".npz"));result=read(U/"queries"/(cid+".json"))
        original=[];refs=[];people=[]
        for pid in result["pairs"]:
            p=load(U/"pairs"/(pid+".npz"));meta=read(U/"pairs"/(pid+".json"))
            p["rows"]=meta["rows"];original.append(p)
            refs.append(load(U/"prepared"/(meta["reference"]+".npz")))
            people.append(int(meta["reference"].split("_REAL")[1][:2]))
        baseline=aggregate(q,original,result["required"],people)
        for h,side in enumerate(("L","R")):
            blank=copy.deepcopy(q)
            blank["angle_deg"][:,h]=np.nan;blank["angle_state"][:,h]=0
            pp=[pair(blank,r,p["path"],result["required"],p["rows"][0]["alignment_ok"])
                for r,p in zip(refs,original)]
            ag=aggregate(blank,pp,result["required"],people)
            for p,b in zip(pp,original):
                assert np.array_equal(p["expected"],b["expected"])
                assert not p["strict"][:,h].any()
                assert np.array_equal(p["strict"][:,1-h],b["strict"][:,1-h])
            assert ag["rows"][(1-h)*15:(2-h)*15]==baseline["rows"][(1-h)*15:(2-h)*15]
            for rr in ag["rows"][h*15:(h+1)*15]:
                assert rr["compared_median_deg"] is None
                assert rr["status"]==("INSUFFICIENT" if side in result["required"] else "NOT_REQUIRED")
                assert rr["sign_error"] is None
            counts["whole_hand_angle_evidence_removed"]+=1
        h=0 if "L" in result["required"] else 1
        core=np.flatnonzero(q["core"]);mid=int(core[len(core)//2]);a=7
        short=copy.deepcopy(q);short["angle_state"][mid,h,a]=0;short["angle_deg"][mid,h,a]=np.nan
        pp=[pair(short,r,p["path"],result["required"],p["rows"][0]["alignment_ok"]) for r,p in zip(refs,original)]
        ag=aggregate(short,pp,result["required"],people)
        rr=ag["rows"][h*15+a]
        assert any(t["first_row"]<=mid<=t["last_row"] for t in rr["reasons"]["query_missing"])
        assert rr["status"]!="OBSERVED"
        omitted_intervals=rr["reasons"]["query_missing"]
        for k in range(30):
            if k!=h*15+a:assert ag["rows"][k]==baseline["rows"][k]
        counts["single_saved_frame_single_angle_omission"]+=1
        # Slow timestamps preserve support/differences, only seconds on query timeline change.
        slow=copy.deepcopy(q);slow["timestamp_us"]*=2
        pp=[pair(slow,r,p["path"],result["required"],p["rows"][0]["alignment_ok"]) for r,p in zip(refs,original)]
        for p,b in zip(pp,original):assert p["rows"]==b["rows"]
        counts["timestamp_slowdown_fixed_path"]+=1
        # In public non-core regions, angle confidence cannot fabricate an internal action gap.
        boundary=copy.deepcopy(q);boundary["angle_state"][~q["core"]]=0
        pp=[pair(boundary,r,p["path"],result["required"],p["rows"][0]["alignment_ok"]) for r,p in zip(refs,original)]
        ag=aggregate(boundary,pp,result["required"],people)
        for rr in ag["rows"]:
            for t in rr["missing_intervals"]:
                assert q["core"][t["first_row"]:t["last_row"]+1].all()
            assert rr["sign_error"] is None
        counts["noncore_absence_not_action_error"]+=1
        details.append(dict(clip_id=cid,frame_id=int(q["frame_id"][mid]),
                            single_angle=f'{"L" if h==0 else "R"}:{a}',
                            interval=omitted_intervals,status="PASS"))
        print("stress",cid,flush=True)
    write(U/"STRESS.json",dict(status="PASS",counts=dict(counts),details=details,
          limitations="fixed saved DTW path; angle evidence removal only, not pixel occlusion or classifier test; normal repeats never judged wrong but linguistic repeat validity is not evaluated"))
    print(dict(counts))
if __name__=="__main__":main()
