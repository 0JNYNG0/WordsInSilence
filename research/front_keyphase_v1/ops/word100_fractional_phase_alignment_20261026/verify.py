"""Reconstruct phase/masks/interval arithmetic separately; no candidate import."""
from pathlib import Path
import sys,json,hashlib,datetime
from collections import Counter,defaultdict
import numpy as np
sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_point_sequence_split_20261022';I=O/'word100_cross_person_motion_support_20261022/inputs'
NORMAL={'fast075','slow15','pace','pause'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def channels(e):
    xy=np.concatenate((e['centre'],e['arm_xy'].reshape(-1,2,4)),axis=2)
    rad=np.concatenate((np.repeat(e['radius'][...,None],2,axis=2),np.zeros_like(xy[:,:,2:])),axis=2)
    v=np.concatenate((e['angle_valid'],np.repeat(e['palm_valid'][...,None],2,axis=2),np.repeat(e['arm_valid'],2,axis=2)),axis=2)
    lo=np.concatenate((e['angle_low'],xy-rad),axis=2);hi=np.concatenate((e['angle_high'],xy+rad),axis=2);v=v&np.isfinite(lo)&np.isfinite(hi)
    return lo,hi,v
def interp(c,ix):
    a=np.floor(ix).astype(int);b=np.ceil(ix).astype(int);w=(ix-a)[:,None,None];lo,hi,valid=c;ok=valid[a]&valid[b]
    return np.where(ok,lo[a]*(1-w)+lo[b]*w,np.nan),np.where(ok,hi[a]*(1-w)+hi[b]*w,np.nan),ok
def original_indices(path,n):
    ids=[path[path[:,1]==j,0] for j in range(n)]
    return np.array([np.median(x) for x in ids]),np.array([min(x) for x in ids]),np.array([max(x) for x in ids])
def independent_phase(x):
    y=x.copy()
    for j in range(2,len(x)-2):
        value=sum(weight*x[j+off] for off,weight in [(-2,1),(-1,4),(0,6),(1,4),(2,1)])/16
        y[j]=min(max(min(max(value,x[1]),x[-2]),x[j]-1),x[j]+1)
    return y
def describe(c,rest,mid,first,last,domain):
    n=len(mid);lo=np.full(domain.shape+(5,),np.nan);hi=lo.copy();valid=np.zeros(domain.shape,bool);sl,sh,sv=interp(c,mid)
    for j in range(2,n-2):
        frames=[j-2,j,j+2];a=int(first[j-2]);b=int(last[j+2]);good=domain[j-2:j+3].all(0)&c[2][a:b+1].all(0)&~rest[a:b+1].any(0)[:,None]&sv[frames].all(0)
        low=np.moveaxis(sl[frames],0,-1);high=np.moveaxis(sh[frames],0,-1)
        l=np.concatenate((low,np.stack((low[...,1]-high[...,0],low[...,2]-high[...,1]),-1)),-1)
        h=np.concatenate((high,np.stack((high[...,1]-low[...,0],high[...,2]-low[...,1]),-1)),-1)
        lo[j]=np.where(good[...,None],l,np.nan);hi[j]=np.where(good[...,None],h,np.nan);valid[j]=good
    return lo,hi,valid
def main():
    f=read(U/'FROZEN.json');r=read(U/'RESULT.json');assert not (U/'RUNNING.json').exists()
    for p,h in {**f['pins'],**f['protected'],**r['pins']}.items():assert sha(p)==h,p
    rows={x['clip_id']:x for x in f['rows']};es={cid:load(I/(cid+'.npz')) for cid in rows};cs={cid:channels(e) for cid,e in es.items()};dom={};anchors={}
    for cid,e in es.items():
        sides=f['roles'].get(rows[cid]['word_code'],{}).get('required_sides',['L','R'])
        d=np.broadcast_to((e['core'][:,None]&np.array([s in sides for s in ['L','R']])[None,:]&~e['source_rest'])[...,None],(len(e['core']),2,21)).copy();dom[cid]=d
        i=np.arange(len(d));anchors[cid]=describe(cs[cid],e['source_rest'],i,i,i,d)
    counts=Counter();summaries=[];detection=defaultdict(lambda:Counter());phase_metrics=defaultdict(list);coverage=Counter();warnings=defaultdict(Counter)
    for k,meta in enumerate(r['rows']):
        folder=meta['folder'];cid=meta['query'];rid=meta['reference'];old=load(meta['old']);new=load(meta['output']);ix=old['source_index'];e=es[cid]
        if folder=='missing':
            ch=[x.copy() for x in cs[cid]];h=['L','R'].index(meta['side']);sl=slice(17,21) if meta['kind']=='wholearm' else slice(0,17);ids=np.array(meta['removed'],int)
            ch[0][ids,h,sl]=np.nan;ch[1][ids,h,sl]=np.nan;ch[2][ids,h,sl]=False;rest=e['source_rest'].copy()
            if meta['kind']!='wholearm':rest[ids,h]=False
        else:ch=interp(cs[cid],ix);rest=e['source_rest'][np.floor(ix).astype(int)]&e['source_rest'][np.ceil(ix).astype(int)]
        m,a,b=original_indices(old['path'],len(dom[rid]));phase=independent_phase(m);first=np.minimum(a,np.floor(phase).astype(int));last=np.maximum(b,np.ceil(phase).astype(int))
        np.testing.assert_allclose(phase,new['index'],atol=1e-12,rtol=0);assert np.array_equal(first,new['first']) and np.array_equal(last,new['last'])
        assert (np.diff(phase)>=-1e-12).all() and np.max(abs(phase-m))<=1+1e-12 and (first<=a).all() and (last>=b).all()
        for name in ('expected','point_comparable','point_bad','native_expected','native_unknown'):assert np.array_equal(old[name],new[name]),name
        assert np.array_equal(new['expected'],dom[rid])
        lo,hi,valid=describe(ch,rest,phase,first,last,dom[rid]);rl,rh,rv=anchors[rid];cal=load(P/'calibration'/(cid+'__'+rid+'.npz'))
        ok=dom[rid]&valid&rv&cal['support'];gap=np.maximum(0,np.maximum(lo-rh,rl-hi));bad=ok[...,None]&(gap>cal['component_bound']+1e-6)
        assert np.array_equal(ok,new['sequence_comparable']) and np.array_equal(bad,new['component_bad'])
        assert not (ok&~old['sequence_comparable']).any();assert not (new['component_bad']&~ok[...,None]).any()
        counts['cases']+=1;counts['required_cells']+=int(dom[rid].sum());counts['score_cells_checked']+=bad.size;counts['point_evidence_preserved']+=int(old['point_comparable'].sum());counts['native_unknown_preserved']+=int(old['native_unknown'].sum());counts['shifted_reference_frames']+=int((abs(phase-m)>1e-12).sum())
        if folder=='missing':
            h=['L','R'].index(meta['side']);sl=slice(17,21) if meta['kind']=='wholearm' else slice(0,17)
            if meta['kind']!='shortgap':assert not ok[:,h,sl].any()
            summaries.append(dict(folder=folder,name=meta['name'],old_comparable=int(old['sequence_comparable'].sum()),new_comparable=int(ok.sum())))
        else:
            identname='identity__'+cid+('__'+rid if folder=='cross' else '')
            ob=load(P/folder/(identname+'.npz'));nb=load(U/'cases'/(folder+'__'+identname+'.npz'))
            common=old['sequence_comparable']&ok&ob['sequence_comparable']&nb['sequence_comparable']
            onew=old['component_bad']&~ob['component_bad']&common[...,None];nnew=bad&~nb['component_bad']&common[...,None]
            row=dict(folder=folder,name=meta['name'],kind=meta['kind'],query=cid,reference=rid,common=int(common.sum()),old_new_bad=int(onew.sum()),new_new_bad=int(nnew.sum()),old_comparable=int(old['sequence_comparable'].sum()),new_comparable=int(ok.sum()))
            if meta['kind'] in NORMAL:
                for group,sl in [('angle',slice(0,15)),('palm',slice(15,17)),('elbow',slice(17,19)),('wrist',slice(19,21))]:
                    ag=warnings[folder+'_'+group];ag['old']+=int(onew[:,:,sl].sum());ag['new']+=int(nnew[:,:,sl].sum());ag['common']+=int(common[:,:,sl].sum())
                coverage[folder+'_lost']+=int((old['sequence_comparable']&~ok).sum())
                rawphase=np.interp(m,np.arange(len(ix)),ix);newphase=np.interp(phase,np.arange(len(ix)),ix)
                if folder=='known':target=np.arange(len(dom[rid]));newtarget=target
                else:target=original_indices(ob['path'],len(dom[rid]))[0];newtarget=nb['index']
                core=dom[rid].any((1,2));oldmae=float(abs(rawphase-target)[core].mean());newmae=float(abs(newphase-newtarget)[core].mean())
                phase_metrics[folder+'_old'].append(oldmae);phase_metrics[folder+'_new'].append(newmae);row.update(old_phase_mae=oldmae,new_phase_mae=newmae)
            if meta['kind'].startswith('skip') and folder=='cross':
                assert np.allclose(ix,np.rint(ix));removed=sorted(set(range(len(e['core'])))-set(ix.astype(int)))
                refs=np.unique(ob['path'][np.isin(ob['path'][:,0],removed),1]);local=np.zeros_like(common);local[refs]=dom[rid][refs]
                # Matched support must be available in all four to credit a change.
                old_detect=bool((onew&local[...,None]).any());new_detect=bool((nnew&local[...,None]).any())
                d=detection[cid+'__'+meta['kind']];d['old_refs']+=old_detect;d['new_refs']+=new_detect;d['refs']+=1
                row.update(old_local_detection=old_detect,new_local_detection=new_detect,local_common=int((local&common).sum()))
            summaries.append(row)
        if (k+1)%500==0:print('VERIFIED',k+1,flush=True)
    omissions={}
    for kind in ('skip3','skip20pct'):
        ds=[x for key,x in detection.items() if ('__'+kind+'_') in key];assert len(ds)==81 and all(x['refs']==8 for x in ds)
        omissions[kind]=dict(variants=len(ds),old_at_least5=sum(x['old_refs']>=5 for x in ds),new_at_least5=sum(x['new_refs']>=5 for x in ds),gained=sum(x['old_refs']<5 and x['new_refs']>=5 for x in ds),lost=sum(x['old_refs']>=5 and x['new_refs']<5 for x in ds))
    pm={k:dict(cases=len(v),mean=float(np.mean(v)),minimum=float(min(v)),maximum=float(max(v))) for k,v in phase_metrics.items()}
    oldcross=sum(x['old'] for k,x in warnings.items() if k.startswith('cross_'));newcross=sum(x['new'] for k,x in warnings.items() if k.startswith('cross_'))
    gate=dict(known_timing_improves=pm['known_new']['mean']<pm['known_old']['mean'],cross_normal_new_warnings_decrease=newcross<oldcross,no_normal_coverage_loss=not sum(coverage.values()),short_omission_not_lost=omissions['skip3']['lost']==0,all_invariants_pass=True);passed=all(gate.values())
    for p,h in {**f['protected'],**r['pins']}.items():assert sha(p)==h,p
    out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS_NUMERIC_AND_PRESERVATION_NOT_SEMANTIC_TRUTH',counts=dict(counts),normal_phase=pm,normal_warnings={k:dict(v) for k,v in warnings.items()},normal_coverage_loss=dict(coverage),omissions=omissions,gate=gate,gate_passed=passed,rows=summaries,result_sha256=sha(U/'RESULT.json'),independence='Separate formula implementation by same Codex; not independent human labels.',operating_changed=False)
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('VERIFIED_COMPLETE',dict(counts),gate,omissions,flush=True)
if __name__=='__main__':main()
