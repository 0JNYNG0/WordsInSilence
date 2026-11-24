from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import sys,json,hashlib,datetime
sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent;O=U.parent;F=O/'word100_fractional_phase_alignment_20261026';P=O/'word100_point_sequence_split_20261022'
NORMAL={'fast075','slow15','pace','pause'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def med(path,n):return np.array([np.median(path[path[:,1]==j,0]) for j in range(n)])
def verify():
    f=read(U/'FROZEN.json');r=read(U/'RESULT.json');pf=read(F/'FROZEN.json');prior=read(P/'ANALYSIS.json')
    for p,h in {**f['pins'],**f['protected'],**r['pins']}.items():assert sha(p)==h,p
    counts=Counter();norm=defaultdict(Counter);detection=defaultdict(Counter);timing=[];rows=[];fallback_by_kind=Counter();crossing={}
    for num,meta in enumerate(f['cases']):
        folder=meta['folder'];cid=meta['query'];rid=meta['reference'];name=meta['name'];old=load(meta['old']);ff=load(meta['output']);z=load(U/'cases'/(folder+'__'+name+'.npz'))
        fresh=ff['sequence_comparable'];orig=old['sequence_comparable'];src=z['source'];b=z['component_bad']
        assert np.array_equal(src==1,fresh);assert np.array_equal(src==2,orig&~fresh);assert np.array_equal(src==0,~orig)
        assert np.array_equal(z['sequence_comparable'],orig);assert np.array_equal(z['refined_timing_unavailable'],orig&~fresh)
        assert np.array_equal(b[fresh],ff['component_bad'][fresh]);assert np.array_equal(b[orig&~fresh],old['component_bad'][orig&~fresh]);assert not b[~orig].any()
        for key in ('expected','point_comparable','point_bad','native_expected','native_unknown'):assert np.array_equal(z[key],old[key])
        counts['cases']+=1;counts['expected_cells']+=int(old['expected'].sum());counts['sequence_coverage_preserved']+=int(orig.sum());counts['fallback_cells']+=int((src==2).sum());counts['native_unknown_preserved']+=int(z['native_unknown'].sum())
        fallback_by_kind[folder+'_'+meta['kind']]+=int((src==2).sum())
        row=dict(folder=folder,name=name,kind=meta['kind'],query=cid,reference=rid,fallback_cells=int((src==2).sum()))
        if folder!='missing':
            ident='identity__'+cid+('__'+rid if folder=='cross' else '')
            ob=load(P/folder/(ident+'.npz'));nb=load(U/'cases'/(folder+'__'+ident+'.npz'));common=orig&ob['sequence_comparable']
            oc=common[...,None]&old['component_bad']&~ob['component_bad'];nc=common[...,None]&b&~nb['component_bad']
            oldany=common&old['component_bad'].any(-1)&~ob['component_bad'].any(-1);newany=common&b.any(-1)&~nb['component_bad'].any(-1)
            if meta['kind'] in NORMAL:
                for group,sl in [('angle',slice(0,15)),('palm',slice(15,17)),('elbow',slice(17,19)),('wrist',slice(19,21))]:
                    a=norm[folder+'_'+group];a['common']+=int(common[:,:,sl].sum());a['old_component']+=int(oc[:,:,sl].sum());a['new_component']+=int(nc[:,:,sl].sum());a['old_channel']+=int(oldany[:,:,sl].sum());a['new_channel']+=int(newany[:,:,sl].sum())
                if folder=='known':
                    n=len(src);m=med(old['path'],n);ix=old['source_index'];t=np.arange(n);oldphase=np.interp(m,np.arange(len(ix)),ix);newphase=np.interp(ff['index'],np.arange(len(ix)),ix)
                    # Source-specific timestamps for each supported whole descriptor.
                    oe=np.broadcast_to(abs(oldphase-t)[:,None,None],src.shape);ne=np.where(src==1,np.broadcast_to(abs(newphase-t)[:,None,None],src.shape),oe)
                    if orig.any():timing.append(dict(name=name,cells=int(orig.sum()),old_sum=float(oe[orig].sum()),new_sum=float(ne[orig].sum()),old_mae=float(oe[orig].mean()),new_mae=float(ne[orig].mean())))
            if meta['kind'].startswith('skip') and folder=='cross':
                ix=old['source_index'];removed=np.setdiff1d(np.arange(len(ob['source_index'])),ix.astype(int));refs=np.unique(ob['path'][np.isin(ob['path'][:,0],removed),1]);region=np.zeros_like(common);region[refs]=old['expected'][refs]
                a=bool((oldany&region).any());b1=bool((newany&region).any());d=detection[cid+'__'+meta['kind']];d['old']+=a;d['new']+=b1;d['refs']+=1;row.update(old_local_detect=a,new_local_detect=b1)
            if folder=='known' and cid=='NIA_SL_WORD0185_REAL03_F' and meta['kind']=='identity':
                ids=list(range(101,112))+[113];crossing=dict(point_equal=True,old_sequence=int(orig[ids].sum()),new_sequence=int(z['sequence_comparable'][ids].sum()),refined=int((src[ids]==1).sum()),original_only=int((src[ids]==2).sum()),not_semantic_shape_truth=True)
        else:
            h=('L','R').index(meta['side']);sl=slice(17,21) if meta['kind']=='wholearm' else slice(0,17)
            if meta['kind']!='shortgap':assert not z['sequence_comparable'][:,h,sl].any()
        rows.append(row)
        if (num+1)%700==0:print('VERIFIED',num+1,flush=True)
    omissions={}
    for kind,name in [('skip3','short'),('skip20pct','long')]:
        ds=[v for k,v in detection.items() if ('__'+kind+'_') in k];assert len(ds)==81 and all(x['refs']==8 for x in ds)
        old=sum(x['old']>=5 for x in ds);new=sum(x['new']>=5 for x in ds);assert old==prior['variants']['cross'][name]['counts']['split']['five']
        omissions[kind]=dict(variants=81,old=old,new=new,gained=sum(x['old']<5 and x['new']>=5 for x in ds),lost=sum(x['old']>=5 and x['new']<5 for x in ds))
    totalcells=sum(x['cells'] for x in timing);oldtime=sum(x['old_sum'] for x in timing)/totalcells;newtime=sum(x['new_sum'] for x in timing)/totalcells
    cross={k:sum(x[k] for name,x in norm.items() if name.startswith('cross_')) for k in ('old_component','new_component','old_channel','new_channel','common')}
    gate=dict(invariants=True,old_coverage_exact=True,known_supported_timing_improves=newtime<oldtime,cross_component_warnings_decrease=cross['new_component']<cross['old_component'],cross_channel_warnings_decrease=cross['new_channel']<cross['old_channel'],no_short_detected_variant_lost=omissions['skip3']['lost']==0)
    assert counts['cases']==2808 and crossing
    for p,h in f['protected'].items():assert sha(p)==h,p
    out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS_EVIDENCE_SELECTION_AND_CANONICAL_METRICS',counts=dict(counts),normal_groups={k:dict(v) for k,v in norm.items()},normal_cross=cross,omissions=omissions,known_effective_timing=dict(cases=len(timing),comparable_cells=totalcells,old_frame_mae=oldtime,new_frame_mae=newtime,rows=timing),crossing=crossing,fallback_by_kind=dict(fallback_by_kind),gate=gate,gate_passed=all(gate.values()),rows=rows,result_sha256=sha(U/'RESULT.json'),production_changed=False,independence='Same Codex separate arithmetic and exact comparison to original published aggregate; no independent person labels.')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print('VERIFIED',gate,cross,omissions,oldtime,newtime,flush=True)
if __name__=='__main__':verify()
