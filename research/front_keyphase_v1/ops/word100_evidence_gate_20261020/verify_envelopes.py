"""Raw-time independent scalar integration, no criterion/coverage timeline call."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;B=U.parents[1];sys.path.insert(0,str(B/'src'));sys.dont_write_bytecode=True
from wis_lab.core.observation import Observation
from wis_lab.service.word100_observation_coverage import metric_flags
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def integral(ts,ok,pos):
    t=[(int(v)-int(ts[0]))/1e6 for v in ts];indices=[i for i,b in enumerate(pos) if b]
    if len(indices)<2:return None,None,None
    a,b=t[indices[0]],t[indices[-1]];spans=[];gaps=sorted(t[j+1]-t[j] for j in range(len(t)-1));m=len(gaps)//2
    median=gaps[m] if len(gaps)%2 else (gaps[m-1]+gaps[m])/2;rad=1.5*median
    for j,v in enumerate(ok):
        if not v:continue
        l=max(a,t[j]-rad,0 if j==0 else (t[j-1]+t[j])/2)
        r=min(b,t[j]+rad,t[-1] if j==len(t)-1 else (t[j]+t[j+1])/2)
        if r>l:spans.append((l,r))
    amount=sum(y-x for x,y in spans);cursor=a;longest=0.
    for x,y in spans:longest=max(longest,x-cursor);cursor=max(cursor,y)
    longest=max(longest,b-cursor)
    return 100*amount/(b-a),longest/(b-a),(a,b)
def run():
    idx={r['clip_id']:r for r in read(U.parent/'word100_multiref_observation_compare_20261019/FROZEN.json')['inputs']}
    rows=read(U/'BASELINE.json');cache={};maxerr=0
    for r in rows:
        key=(r['clip_id'],r['view'])
        if key not in cache:
            p=idx[key[0]]['inputs'][key[1]];o=Observation.load(p);cache[key]=(o.timestamp_us,metric_flags(o))
        ts,f=cache[key];c,g,b=integral(ts,f[r['side']][r['key']],f[r['side']]['location']);d=r['evidence']
        assert (c is None)==(d['coverage'] is None)
        if c is not None:
            maxerr=max(maxerr,abs(c-d['coverage']));assert abs(c-d['coverage'])<1e-6 and abs(g-d['gap_fraction'])<1e-6
            assert all(abs(x-y)<1e-7 for x,y in zip(b,d['bounds']))
    result=dict(status='PASS',raw_metric_hand_envelopes=len(rows),max_percent_error=maxerr,independent_scalar_time_integration=True)
    (U/'ENVELOPE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n','utf-8');print(result)
if __name__=='__main__':run()
