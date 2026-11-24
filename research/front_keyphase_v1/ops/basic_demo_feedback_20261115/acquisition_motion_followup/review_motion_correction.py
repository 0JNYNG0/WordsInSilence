"""Correct ambiguous development labels from full native RGB, retaining old evidence."""
from motion_model import *

def main():
    import sys
    sys.path.insert(0,str(U));from core import Features
    from basic_runtime import RelationObserver
    torch.set_num_threads(2)
    changes=[dict(person=11,old=[58,61],new=[61,64],label=1,
        reason='Native frames58–60 continue separating, then61–64 approach; old span crossed the turn.'),
        dict(person=12,old=[40,46],new=[42,46],label=2,
        reason='Native40–42 approach,42–46 separate; old span crossed the closest moment.')]
    assert not(D/'MOTION_CORRECTION.json').exists()
    write(D/'MOTION_CORRECTION.json',dict(changes=changes,review='failure_11.jpg/failure_12.jpg sequential original RGB, inspected by Codex',
        previous_comparison_valid=False,training_changed=False,
        reason='Two development windows were given a single direction despite containing a turn; fix source intervals rather than thresholds.',
        caveat='Earlier outputs retained; revised development is still development, never independent final evidence.'))
    rows=read(D/'MOTION_TRUTH.json')['rows'];sources={r['clip_id']:r for r in read(U/'INPUTS.json')['rows']}
    chs={k:dict(np.load(v['channels']))for k,v in sources.items()}
    factory=Features();base=RelationObserver();ss={}
    for k,ch in chs.items():
        f,q,_=factory(ch);ss[k]=base(f,q,ch['timestamp_us'])[0]
    obs={'linear':Observer('linear'),'mlp32':Observer('mlp32'),'edge_mlp':Observer('mlp32',D/'edge',True)}
    result=[]
    for old in rows:
        r=old.copy()
        for c in changes:
            if c['person']==r['person'] and c['old']==[r['start'],r['end']]:r.update(start=c['new'][0],end=c['new'][1])
        ch=chs[r['clip_id']];s=ss[r['clip_id']];a,b=s[r['start']],s[r['end']]
        pred={'static_state_proxy':1 if (a,b)==(1,2) else 2 if (a,b)==(2,1) else 0 if a==b and a>0 else -1}
        for k,o in obs.items():pred[k]=o.predict(feature(ch,r['start'],r['end'],o.edge))
        result.append(dict(**r,**pred))
    dev=[r for r in result if r['split']!='fit'];summary={k:metrics(dev,k) for k in ['static_state_proxy']+list(obs)}
    write(D/'MOTION_CORRECTED_COMPARISON.json',dict(summary=summary,rows=result,web_admitted=False,
        missing_capability='Windows containing direction reversal still need an explicit mixed/turn class. No runtime cycle fusion yet.'))
    print(json.dumps(summary))

if __name__=='__main__':main()
