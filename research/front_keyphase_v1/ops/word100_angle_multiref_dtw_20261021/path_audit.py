"""Check representative saved path costs with a separate zero-based DP."""
import numpy as np
from run import U, read, load, write, F, frame_cost_matrix

results=[]
for item in read(U/"INDEX.json")[::12]:
    meta=read(U/"queries"/(item["clip_id"]+".json"))
    pid=meta["pairs"][0];p=load(U/"pairs"/(pid+".npz"));m=read(U/"pairs"/(pid+".json"))
    q=load(U/"prepared"/(m["query"]+".npz"));r=load(U/"prepared"/(m["reference"]+".npz"))
    required=m["required"]
    weights=None if len(required)==2 else {g:w if g.endswith("_"+required[0]) else 0. for g,(_,w) in F.GROUPS.items()}
    c,e=frame_cost_matrix(q,r,group_weights=weights,hand_participation=False,one_sided_mode="flat")
    n,k=c.shape;width=max(1,round(.2*max(n,k)));prev={}
    for i in range(n):
        center=i*(k-1)/(n-1);cur={}
        for j in range(max(0,int(np.floor(center-width))),min(k-1,int(np.ceil(center+width)))+1):
            options=[prev.get(j-1,np.inf),prev.get(j,np.inf),cur.get(j-1,np.inf)]
            if i==j==0:options.append(0.)
            cur[j]=min(options)+float(c[i,j])
        prev=cur
    expected=prev[k-1];actual=float(c[tuple(p["path"].T)].astype(float).sum())
    assert abs(expected-actual)<1e-7,(expected,actual)
    assert abs(float(e[tuple(p["path"].T)].mean())-m["alignment_support"])<1e-10
    results.append(dict(pair=pid,cost_error=abs(expected-actual)))
write(U/"PATH_AUDIT.json",dict(status="PASS",pairs=results,scope="every 12th query, first reference; same cost formula, independent optimal-sum DP"))
print(len(results),"PASS")
