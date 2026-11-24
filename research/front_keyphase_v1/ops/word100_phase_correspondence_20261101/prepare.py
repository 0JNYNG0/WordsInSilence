"""Immutable local public inputs; no inference or outcome-dependent selection."""
from pathlib import Path
import json, hashlib, numpy as np
U=Path(__file__).resolve().parent; OPS=U.parent
OLD=OPS/'word100_relational_sequence_compare_20261031'
REC=OPS/'word100_observation_lineage_repair_20261101'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in ('x','mask','frame_id','timestamp_us')}
def main():
    assert not (U/'INPUTS.json').exists(),'Completed preparation must not be overwritten'
    for d in ['inputs','controls','results']: (U/d).mkdir(exist_ok=True)
    case=read(OPS/'word100_action_feedback_casebook_20261101/INPUTS.json')
    rows=case['rows']; arrays={}; inputs={}; pins={}; overlays={}
    def pin(p):
        p=Path(p).resolve();pins[str(p)]=sha(p);return str(p)
    pin(OPS/'word100_action_feedback_casebook_20261101/INPUTS.json')
    for a,row in sorted(rows.items()):
        src=Path(row['input']);assert sha(src)==row['input_sha256'];base=load(src)
        arrays[a]=base;inputs[a]={'original':pin(src),'role':row['role'],'word_id':row['word_id'],'label':row['label']}
        merged={k:v.copy() for k,v in base.items()};rp=REC/'recovery_outputs'/f'{a}.npz';extra=0
        if rp.exists():
            pin(rp)
            with np.load(rp,allow_pickle=False) as z:
                ids=z['source_index'].astype(int)
                assert np.array_equal(base['frame_id'][ids],z['frame_id'])
                assert np.array_equal(base['timestamp_us'][ids],z['timestamp_us'])
                assert np.array_equal(base['mask'][ids],z['baseline_mask'])
                assert np.array_equal(base['x'][ids],z['baseline_x'])
                old=base['mask'][ids];assert np.array_equal(z['x'][old],z['baseline_x'][old])
                assert np.all(z['mask'][old]);extra=int((z['mask']&~old).sum())
                merged['x'][ids]=z['x'];merged['mask'][ids]=z['mask']
        dest=U/'inputs'/f'{a}_merged.npz';np.savez_compressed(dest,**merged)
        inputs[a]['merged']=pin(dest);overlays[a]={'new_supported_cells':extra,'old_supported_exact':bool(np.array_equal(base['x'][base['mask']],merged['x'][base['mask']]))}
    pairs=[]
    for q,r in sorted({(d['query'],d['reference']) for d in case['natural']}):
        pairs.append({'query':q,'reference':r,'global_path':pin(OLD/'pairs'/f'B_same_path_{q}_{r}.json'),'same_cost_path':pin(OLD/'pairs'/f'C_relational_path_{q}_{r}.json')})
    controls=[]
    for j in sorted((OLD/'controls').glob('*.json')):
        d=read(j);a=d['alias'];npz=j.with_suffix('.npz')
        assert sha(npz)==d['array_sha256'];assert sha(inputs[a]['original'])==d['source_input_sha256']
        controls.append({'id':j.stem,'alias':a,'kind':d['control'],'input':pin(npz),'reference':inputs[a]['original'],'baseline':pin(j),'metadata':d['metadata'],'new':False})
    assert len(controls)==126 and len(pairs)==12 and len(inputs)==16
    # Fixed technical probes. The truth ledger is never passed into the matcher.
    for a,base in sorted(arrays.items()):
        n=len(base['x'])
        for kind,count in [('hold15',1),('repeat2_same',2),('repeat3_same',3)]:
            ri=np.tile(np.arange(n),count);ro=np.repeat(np.arange(count),n)
            if count==1:
                mid=n//2;qi=np.r_[np.arange(mid+1),np.repeat(mid,15),np.arange(mid+1,n)];qo=np.zeros(len(qi),int)
            else:
                slow=np.round(np.linspace(0,n-1,round(n/.75))).astype(int)
                qi=np.tile(slow,count);qo=np.repeat(np.arange(count),len(slow))
            def materialize(ids):
                z={k:v[ids].copy() for k,v in base.items()}
                z['frame_id']=np.arange(len(ids));z['timestamp_us']=np.arange(len(ids),dtype=np.int64)*33333
                return z
            name=f'{a}_{kind}';qp=U/'controls'/f'{name}_q.npz';rp=U/'controls'/f'{name}_r.npz'
            np.savez_compressed(qp,**materialize(qi));np.savez_compressed(rp,**materialize(ri))
            meta={'query_source_frame':qi.tolist(),'query_occurrence':qo.tolist(),'reference_source_frame':ri.tolist(),'reference_occurrence':ro.tolist(),'query_unrolled_index':(qi+qo*n).tolist(),'reference_unrolled_index':(ri+ro*n).tolist(),'original_length':n,'kind':'feature-space stress control, not a real repeated KSL performance','equal_repeat_count':count,'copy_seams_are_synthetic':count>1}
            controls.append({'id':name,'alias':a,'kind':kind,'input':pin(qp),'reference':pin(rp),'baseline':None,'metadata':meta,'new':True})
    assert len(controls)==174
    write(U/'INPUTS.json',{'inputs':inputs,'overlays':overlays,'natural':pairs,'parents':case['natural'],'controls':controls,'pins':pins,'scope':{'clips':16,'natural_pairs':12,'parent_intervals':39,'saved_controls':126,'new_controls':48},'no_personal_or_mentor_data':True,'no_new_observation_or_training':True})
    print(json.dumps({'status':'prepared','scope':{'clips':16,'pairs':12,'controls':174},'overlay_cells':sum(x['new_supported_cells'] for x in overlays.values())}))
if __name__=='__main__':main()
