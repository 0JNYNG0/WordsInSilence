"""Independent input arithmetic and stored score recount, no model/optimizer."""
from pathlib import Path
import json,hashlib,statistics
from collections import Counter,defaultdict
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
c=read(U/'FROZEN_DIAGNOSTIC.json');p=read(U/'PREPARED.json')
assert p['context_sha256']==sha(U/'FROZEN_DIAGNOSTIC.json')
for path,h in {**p['pins'],**c['pins']}.items():assert sha(path)==h,path
srcmap=[0,11,12,13,14,15,16,23,24]+list(range(33,75))
counts=Counter();maxerror=0;dtmax=0;bounds=Counter()
for j,row in enumerate(p['rows']):
    assert row['role'] in ['train_candidate','development_only']
    assert row['signer_id'] in (list(range(1,10)) if row['role']=='train_candidate' else [11,12,13])
    assert sha(row['path'])==row['sha256']
    with np.load(row['observation']) as z:
        lo,hi=row['original_whole_span'];xy=z['xy_px'][lo:hi+1][:,srcmap].copy()
        valid=z['usable'][lo:hi+1][:,srcmap].copy();ts=z['timestamp_us'][lo:hi+1];fid=z['frame_id'][lo:hi+1]
    xy[~valid]=0
    with np.load(row['path']) as z:
        for name,rate,phase in c['conditions']:
            if rate==30:idx=np.arange(len(ts))
            else:
                tt=(ts-ts[0])*2*rate
                targets=np.arange(phase*1000000,int(tt[-1])+1,2000000,dtype=np.int64)
                targets=targets[(targets>0)&(targets<tt[-1])]
                chosen=np.abs(tt[:,None]-targets[None,:]).argmin(axis=0)
                idx=np.unique(np.r_[0,chosen,len(ts)-1])
            assert np.array_equal(idx,z[name+'_index'])
            assert np.array_equal(ts[idx],z[name+'_timestamp_us'])
            assert np.array_equal(fid[idx],z[name+'_frame_id'])
            assert ts[idx[-1]]-ts[idx[0]]==ts[-1]-ts[0]
            points=xy[idx];ok=valid[idx];n=len(idx)
            x=z[name];assert x.shape==(n,280)
            shoulder=ok[:,1]&ok[:,2]
            scale=float(np.median(np.linalg.norm(points[shoulder,1]-points[shoulder,2],axis=1)))
            origins=(points[:,1]+points[:,2])/2
            bmask=ok&shoulder[:,None]
            body=np.where(bmask[:,:,None],(points-origins[:,None,:])/scale,0)
            local=np.zeros((n,42,2),np.float32);hmask=np.zeros((n,42),bool)
            for s,start in enumerate([9,30]):
                lengths=[]
                for mcp in [5,9,13,17]:
                    q=ok[:,start]&ok[:,start+mcp]
                    lengths.extend(np.linalg.norm(points[q,start+mcp]-points[q,start],axis=1).tolist())
                hs=statistics.median(lengths) if lengths else float('nan')
                if np.isfinite(hs) and hs>1e-6:
                    hm=ok[:,start,None]&ok[:,start:start+21]
                    local[:,s*21:(s+1)*21]=np.where(hm[:,:,None],(points[:,start:start+21]-points[:,start,None,:])/hs,0)
                    hmask[:,s*21:(s+1)*21]=hm
            expected=np.zeros_like(x);expected[:,:102]=body.reshape(n,102);expected[:,102:186]=local.reshape(n,84)
            expected[:,186:237]=bmask;expected[:,237:279]=hmask
            expected[1:,-1]=np.diff(ts[idx]).astype(np.float32)/1e6
            err=float(np.max(np.abs(expected-x)));maxerror=max(maxerror,err)
            assert err<1e-5,(row['clip_id'],name,err)
            counts[name]+=1;bounds[row['role']]+=1
    if (j+1)%400==0:print('inputs_verified',j+1,flush=True)
assert all(n==2398 for n in counts.values())
iv={'status':'PASS','scope':'separate formulas and index selection, no main builder import',
    'pins_verified':len({**p['pins'],**c['pins']}),'condition_views':dict(counts),'maximum280_absolute_error':maxerror,
    'all_elapsed_times_original_masks_and_frame_ids_preserved':True,
    'operator':'Codex root; separate code, not a different agent'}
new(U/'INPUTS_RECOUNT.json',iv)
# Persisted logits are the sole source of result metrics.
tc=read(U.parent/'word100_body_boundary_20261017/training_CONTEXT.json')
words=tc['word_codes'];groups=tc['missingness_groups']
def metrics(rows):
    return dict(n=len(rows),top1=sum(r['target_rank']==1 for r in rows),top5=sum(r['target_rank']<=5 for r in rows),
                top1_percent=100*sum(r['target_rank']==1 for r in rows)/len(rows),
                top5_percent=100*sum(r['target_rank']<=5 for r in rows)/len(rows))
tables={};allrows={};actualinferences=0;baselinereused=0
for view in c['views']:
    tables[view]={};allrows[view]={}
    for name,_,_ in c['conditions']:
        q=read(U/f'PRED_{view}_{name}.json');assert q['context_sha256']==p['context_sha256']
        assert len(q['rows'])==1199 and len({r['clip_id'] for r in q['rows']})==1199
        for r in q['rows']:
            order=sorted(range(100),key=lambda i:(-r['logits'][i],i))
            assert order.index(r['label_index'])+1==r['target_rank']
            assert r['top1']==order[0] and r['top5']==order[:5]
            actualinferences+=r['origin']=='new_frozen_inference'
            baselinereused+=r['origin']=='stored_dev_baseline'
        allrows[view][name]={r['clip_id']:r for r in q['rows']}
        tables[view][name]={}
        for role in ['train_candidate','development_only']:
            rows=[r for r in q['rows'] if r['role']==role]
            m=metrics(rows)
            m['by_signer']={str(s):metrics([r for r in rows if r['signer_id']==s]) for s in sorted({r['signer_id'] for r in rows})}
            m['by_word']={w:metrics([r for r in rows if r['word_code']==w]) for w in words}
            if role=='development_only':
                m['by_original_missing_group']={g:metrics([r for r in rows if groups[view][r['clip_id']]==g]) for g in sorted(set(groups[view].values()))}
            tables[view][name][role]=m
    for name,_,_ in c['conditions'][1:]:
        for role in ['train_candidate','development_only']:
            base=allrows[view]['fps30'];current=allrows[view][name]
            ids=[cid for cid,r in current.items() if r['role']==role]
            corrected=[cid for cid in ids if base[cid]['target_rank']!=1 and current[cid]['target_rank']==1]
            regressed=[cid for cid in ids if base[cid]['target_rank']==1 and current[cid]['target_rank']!=1]
            m=tables[view][name][role]
            m['paired']={'corrected':corrected,'regressed':regressed,'top1_changed':[cid for cid in ids if base[cid]['top1']!=current[cid]['top1']],
                        'delta_top1_pp':m['top1_percent']-tables[view]['fps30'][role]['top1_percent'],
                        'delta_top5_pp':m['top5_percent']-tables[view]['fps30'][role]['top5_percent']}
assert actualinferences==11390 and baselinereused==600
rgb=[]
for cid in c['rgb_check']['public_train_ids']:
    for rate in c['rgb_check']['rates']:
        r=read(U/'rgb'/f'{cid}_{rate}'/'RESULT.json');assert r['context_sha256']==p['context_sha256']
        assert r['source_video_sha256']==sha(r['video'])
        assert r['observation_sha256']==sha(r['observation'])
        with np.load(r['observation']) as z:assert np.array_equal(z['timestamp_us'],r['source_timestamps_us'])
        if r['prediction']:
            pr=r['prediction'];order=sorted(range(100),key=lambda i:(-pr['logits'][i],i))
            assert order.index(pr['label_index'])+1==pr['target_rank']
            r['top3_labels']=[tc['labels'][i] for i in order[:3]]
        rgb.append(r)
confusions={}
for word in c['rgb_check']['public_train_ids']:
    wc=word.split('_')[2];confusions[wc]={}
    for view in c['views']:
        confusions[wc][view]={}
        for name,_,_ in c['conditions']:
            rs=[r for r in allrows[view][name].values() if r['word_code']==wc and r['role']=='development_only']
            confusions[wc][view][name]=[{'clip_id':r['clip_id'],'rank':r['target_rank'],'top1':tc['labels'][r['top1']]} for r in rs]
result={'status':'VERIFIED_SAMPLING_DIAGNOSIS_COMPLETE_NO_TRAINING','tables':tables,'focus_words':confusions,
'rgb_checks':rgb,'new_inferences':actualinferences+sum(bool(r['prediction']) for r in rgb),'reused_dev30_predictions':baselinereused,
'new_detector_observations':len(rgb),'unique_public_clips':1199,'rgb_unique_train_clips':4,
'new_training':0,'user_recording_reads':0,'model_sha256':c['model']['sha256'],
'input_verification':iv,'frozen_context_sha256':p['context_sha256'],
'limitations':['Cached thinning is not fresh low-fps detection','RGB subset only four TRAIN clips, no population accuracy',
'20fps target rounded onto30fps source; endpoints forced, intervals alternate33/67ms',
'280 features renormalized on selected observations; effect includes scale estimate changes',
'Original whole boundaries held constant in cached arm; end-to-end RGB arm reruns active_span',
'Repeated DEV used in prior model selection; single frozen web C seed only',
'No user causal attribution or improved-candidate performance claim']}
new(U/'RESULT.json',result)
print(json.dumps({'status':result['status'],'new_inferences':result['new_inferences'],'summary':{v:{n:{k:{x:m[x] for x in ['n','top1','top5','top1_percent']} for k,m in rs.items()} for n,rs in ns.items()} for v,ns in tables.items()}},ensure_ascii=False,indent=2))

