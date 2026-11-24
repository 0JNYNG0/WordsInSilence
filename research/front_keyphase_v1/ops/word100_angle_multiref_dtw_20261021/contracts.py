"""Small scientific safety contracts; synthetic identities are NOT public evidence."""
import copy
import numpy as np
from adapter import pair, aggregate
from run import write, U, alignment

def main():
    n=30
    q=dict(angle_deg=np.full((n,2,15),40.),angle_state=np.full((n,2,15),4,np.uint8),
           timestamp_us=np.arange(n,dtype=np.int64)*33333,core=np.ones(n,bool))
    q["core"][:4]=False;q["core"][-4:]=False
    path=np.column_stack((np.arange(n),np.arange(n)))
    result=[]
    p=pair(q,q,path,["L","R"],True)
    ag=aggregate(q,[p]*5,["L","R"],[1,2,3,4,5])
    assert all(r["status"]=="OBSERVED" and r["compared_median_deg"]==0 for r in ag["rows"])
    result.append("exact observed 5-reference synthetic control: zero measured difference, no correctness verdict")
    blank=copy.deepcopy(q);blank["angle_state"][:,0]=0;blank["angle_deg"][:,0]=np.nan
    b=pair(blank,q,path,["L","R"],True)
    assert np.array_equal(b["expected"],p["expected"]) and not b["strict"][:,0].any()
    bb=aggregate(blank,[b]*5,["L","R"],[1,2,3,4,5])
    assert all(r["status"]=="INSUFFICIENT" and r["compared_median_deg"] is None for r in bb["rows"][:15])
    assert all(r["status"]=="OBSERVED" for r in bb["rows"][15:])
    result.append("entire missing hand stays unavailable; other hand unchanged")
    partial=copy.deepcopy(q);partial["angle_state"][14:16,0,7]=3
    pp=pair(partial,q,path,["L","R"],True)
    a=aggregate(partial,[pp]*5,["L","R"],[1,2,3,4,5])
    assert a["rows"][7]["status"]=="PARTIAL" and a["rows"][7]["compared_median_deg"]==0
    assert len(a["rows"][7]["reasons"]["query_conflict"])==1
    assert all(a["rows"][k]["status"]=="OBSERVED" for k in range(30) if k!=7)
    result.append("two conflicting angle frames: low number labeled PARTIAL; neighbors retained")
    unknown=copy.deepcopy(q);unknown["angle_state"][:]=2
    u=pair(q,unknown,path,["L","R"],True);a=aggregate(q,[u]*5,["L","R"],[1,2,3,4,5])
    assert all(r["compared_median_deg"] is None and r["raw_only_median_deg"]==0 for r in a["rows"])
    result.append("reference single-view evidence: raw zero cannot become supported zero")
    low=aggregate(q,[p]*4,["L","R"],[1,2,3,4])
    assert all(r["status"]=="INSUFFICIENT" for r in low["rows"])
    result.append("fewer than five references abstains")
    try:aggregate(q,[p]*5,["L","R"],[1,1,2,3,4])
    except ValueError:result.append("duplicate people rejected")
    else:raise AssertionError("duplicate people accepted")
    optional=pair(blank,q,path,["R"],True);ag=aggregate(blank,[optional]*5,["R"],[1,2,3,4,5])
    assert all(x["status"]=="NOT_REQUIRED" for x in ag["rows"][:15])
    assert all(x["status"]=="OBSERVED" for x in ag["rows"][15:])
    result.append("reference-declared unused hand not an action error")
    boundary=copy.deepcopy(q);boundary["angle_state"][~q["core"]]=0
    a=aggregate(boundary,[pair(boundary,q,path,["L","R"],True)]*5,["L","R"],[1,2,3,4,5])
    assert all(r["status"]=="OBSERVED" for r in a["rows"])
    result.append("public non-core leading/trailing absence does not create core gap")
    slow=copy.deepcopy(q);slow["timestamp_us"]*=2
    s=pair(slow,q,path,["L","R"],True)
    assert s["rows"]==p["rows"] and np.array_equal(s["strict"],p["strict"])
    result.append("timestamp-only speed difference keeps distances and coverage")
    repeated={k:np.repeat(v,2,axis=0) for k,v in q.items()}
    repeated["timestamp_us"]=np.arange(2*n,dtype=np.int64)*33333
    rp=np.column_stack((np.arange(2*n),np.arange(2*n)//2))
    x=pair(repeated,q,rp,["L","R"],True)
    assert all(r["strict"]["coverage"]==1 and r["strict"]["mean_deg"]==0 for r in x["rows"])
    result.append("repeated frames do not inflate unique-reference coverage")
    bad=pair(q,q,path,["L","R"],False);ag=aggregate(q,[bad]*5,["L","R"],[1,2,3,4,5])
    assert all(r["status"]=="INSUFFICIENT" for r in ag["rows"])
    result.append("unsupported temporal alignment cannot expose a supported summary")
    write(U/"CONTRACTS.json",dict(status="PASS",tests=result,synthetic_only=True))
    print(len(result),"PASS")
if __name__=="__main__":main()
