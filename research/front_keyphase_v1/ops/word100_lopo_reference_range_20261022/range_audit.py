"""Post-outcome mechanism audit only; no range or rule changes."""
import numpy as np
import study as d
U=d.U;f=d.read(U/'FROZEN.json');res=d.read(U/'RESULT.json');es={r['clip_id']:d.prev.data(r)[1] for r in f['rows']}
cache={};out=[]
for meta in res['templates']:
    cid,rid=meta['query'],meta['anchor'];t=d.load(U/'templates'/(meta['name']+'.npz'));n=len(t['count']);restflags=[];outsideflags=[]
    for member in meta['contributors']:
        key=(member,rid)
        if key not in cache:
            e=es[member]
            path=np.column_stack([np.arange(n),np.arange(n)]) if member==rid else d.load(d.P/'cases'/('identity__'+member+'__'+rid+'.npz'))['path']
            rest=np.zeros((n,2),bool);outside=np.zeros(n,bool)
            for j in range(n):
                ids=path[path[:,1]==j,0];rest[j]=e['source_rest'][ids].any(0);outside[j]=(~e['core'][ids]).any()
            cache[key]=(rest,outside)
        rest,outside=cache[key];restflags.append(rest);outsideflags.append(outside)
    valid=t['member_complete'];dom=t['expected']&t['support']
    rest=np.array(restflags)[:,:,:,None];outside=np.array(outsideflags)[:,:,None,None]
    extreme=valid&((t['member_low']==t['low'][None])|(t['member_high']==t['high'][None]))
    rec=dict(name=meta['name'])
    for name,sl in [('angle',slice(0,15)),('palm',slice(15,17)),('elbow',slice(17,19)),('wrist',slice(19,21))]:
        target=dom[:,:,sl]
        rec[name]=dict(supported=int(target.sum()),
          rest_contributor=int((target&(valid&rest).any(0)[:,:,sl]).sum()),
          rest_extreme=int((target&(extreme&rest).any(0)[:,:,sl]).sum()),
          outside_core_extreme=int((target&(extreme&outside).any(0)[:,:,sl]).sum()),
          width_median=float(np.nanmedian((t['high']-t['low'])[:,:,sl][target])) if target.any() else None)
    out.append(rec)
summary={k:{s:sum(r[k][s] for r in out) for s in ['supported','rest_contributor','rest_extreme','outside_core_extreme']} for k in ('angle','palm','elbow','wrist')}
d.write(U/'RANGE_AUDIT.json',dict(at=d.now(),summary=summary,templates=out,
 interpretation='Positive resting or outside-annotated-core contributors can widen an active anchor range through existing alignment. These are review flags, not proven errors; different people can have different natural transitions. No ranges changed after outcomes. Counts are template time/hand/channel cells, not unique frames or independent trials.'))
print(summary,flush=True)
