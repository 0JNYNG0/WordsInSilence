"""Frozen C/T3F public TRAIN100 sequence sensitivity and input coverage audit."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys,hashlib,math,time,traceback
from collections import Counter
import numpy as np
import torch
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
B=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019';PREV=O/'word100_believe_rearing_sequence_20261019'
sys.path.insert(0,str(ROOT/'research/sign_model_pilot_v1/src'));sys.path.insert(0,str(T))
from frozen_temporal import FrozenCTemporal
from pilot.model_masked_v1 import MaskedSignTCN
CONDITIONS=('original','prefix_half','suffix_half','reverse_thirds','double_sequence')
VIEWS=('masked','sensor')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);assert not p.exists(),p;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def statehash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def variant(x,condition):
    n=len(x)
    if condition=='original':a=x.copy()
    elif condition=='prefix_half':a=x[:math.ceil(n/2)].copy()
    elif condition=='suffix_half':a=x[n//2:].copy();a[0,279]=0
    elif condition=='reverse_thirds':
        a=x.copy();a[:,:279]=x[np.concatenate(np.array_split(np.arange(n),3)[::-1]),:279]
        assert np.array_equal(a[:,279],x[:,279])
    elif condition=='double_sequence':a=np.concatenate([x,x]);a[n,279]=np.median(x[1:,279])
    else:raise ValueError(condition)
    assert np.isfinite(a).all() and a.shape[1]==280 and a[0,279]==0 and np.all(a[1:,279]>0)
    return a
def score(logit,y,labels):
    z=np.asarray(logit,dtype=np.float64);e=np.exp(z-z.max());p=e/e.sum();order=np.argsort(-z,kind='stable');other=next(int(i) for i in order if i!=y)
    return {'logits':z.tolist(),'target_rank':int(np.where(order==y)[0][0])+1,'target_score':float(100*p[y]),
            'target_margin':float(z[y]-z[other]),'top1_index':int(order[0]),'top1_label':labels[order[0]]}
def runs(mask,ts,fid):
    result=[];start=None
    for i,val in enumerate(list(mask)+[False]):
        if val and start is None:start=i
        elif not val and start is not None:
            end=i-1;result.append({'start_index':start,'end_index':end,'first_frame':int(fid[start]),'last_frame':int(fid[end]),'frames':end-start+1,'seconds':float((ts[end]-ts[start])/1e6)});start=None
    return result
def prepare():
    assert not (U/'SCOPE.json').exists()
    ctx=read(B/'training_CONTEXT.json');prep=read(B/'PREP_CONTEXT.json');ready=read(B/'READY.json')
    rows=sorted([r for r in ctx['ordered_rows'] if r['role']=='train_candidate'],key=lambda r:r['clip_id'])
    assert len(rows)==899 and len({r['word_code'] for r in rows})==100 and set(r['signer_id'] for r in rows)==set(range(1,10))
    files=[B/'training_CONTEXT.json',B/'PREP_CONTEXT.json',B/'READY.json',B/'READINESS_VISUAL_REVIEW.json',B/'actual_input_triage_RESULT.json',B/'FULL_INPUTS_INDEPENDENT.json',
      B/'training_C_20261013_best.pt',T/'T3F_20261013_best.pt',T/'temporal.py',T/'frozen_temporal.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py',
      ROOT/'research/front_keyphase_v1/src/wis_lab/service/word100_candidate.py',ROOT/'research/front_keyphase_v1/src/wis_lab/service/word100_t3f.py',
      PREV/'ANALYSIS.json',PREV/'ORDER_CLOCK_CORRECTION.json',PREV/'INDEPENDENT.json',ROOT/'research/front_keyphase_v1/config/word100_analysis_references.json',Path(__file__)]
    pins={str(p):sha(p) for p in files};assert ready['status']=='PASS' and sha(ready['visual_review'])==ready['visual_review_sha256']
    old=read(PREV/'ANALYSIS.json')
    for p,h in old['pins'].items():assert sha(p)==h,p
    for r in rows:
        p=B/'data/records'/f"{r['clip_id']}.json";pins[str(p)]=sha(p);rec=read(p)
        assert pins[str(p)]==ready['record_sha256'][r['clip_id']]
        for view in VIEWS:
            for key in ('channel_file','observation'):
                p=Path(rec['views'][view][key]);pins[str(p)]=sha(p)
            assert pins[rec['views'][view]['channel_file']]==r['views'][view]['sha256']
    write(U/'SCOPE.json',{'at':now(),'status':'FROZEN_PUBLIC_DIAGNOSIS_NO_TRAINING','training_authorized':False,
      'user_request':'공개100단어의 순서 민감도·구별동작 관절입력 보존 검증, 기존C/T3F보존, 사용자자료학습/튜닝금지, 추가학습전보고',
      'rows':rows,'model_seed':20261013,'views':VIEWS,'conditions':CONDITIONS,
      'reuse':'Prior18 TRAIN clips36views×5conditions=180pairs reused from hash-verified stored results (corrected order condition only). No completed DEV/heldout re-evaluation.',
      'input_audit':'All899TRAIN annotation interval coverage, internal chronology, hand masks and early/middle/late support. Reuse previous full numeric and flagged visual gate with explicit limitations.',
      'visual_sample':'One fixed public reference per100word from existing config;8 equally spaced frames over annotation interval, with original RGB and saved sensor observation overlay. Additional8risk clips selected before scores:4 largest core trim,4 longest both-hand loss, unique and TRAIN only. Not semantic ground truth.',
      'clock':'Reverse only first279channels; original dt remains bitwise equal. All transformed real frames retained, padding excluded.',
      'metrics':'Original-label retention, conditional on originally top1, perword count, probability and raw logit margin change. Counterfactuals are NOT labeled incorrect signs, not new accuracy/generalization.',
      'forbidden':['training','new model/threshold selection','user/mentor files','DEV or heldout new inference','new detector','production changes','downloads','Git'],
      'pins':pins})
    print(json.dumps({'stage':'FROZEN','train':len(rows),'pins':len(pins)},ensure_ascii=False),flush=True)

def main():
    if not (U/'SCOPE.json').exists():prepare()
    scope=read(U/'SCOPE.json');assert sha(__file__)==scope['pins'][str(Path(__file__))]
    assert not (U/'COMPLETE.json').exists()
    for p,h in scope['pins'].items():assert sha(p)==h,p
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ctx=read(B/'training_CONTEXT.json');prep={r['clip_id']:r for r in read(B/'PREP_CONTEXT.json')['ordered_rows'] if r['role']=='train_candidate'}
    ck=torch.load(T/'T3F_20261013_best.pt',map_location='cpu',weights_only=True);cp=torch.load(B/'training_C_20261013_best.pt',map_location='cpu',weights_only=True)
    m=FrozenCTemporal(**ck['model_config']);m.load_state_dict(ck['state_dict']);m.eval();m.assert_frozen()
    c=MaskedSignTCN(**ck['model_config']);c.load_state_dict(cp['state_dict']);c.eval()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in c.state_dict().items())
    before=[statehash(c),statehash(m)];labels=ctx['labels']
    old=read(PREV/'ANALYSIS.json');corr=read(PREV/'ORDER_CLOCK_CORRECTION.json')
    prior={(r['clip_id'],r['view'],r['condition']):r for r in old['predictions'] if r['condition'] in CONDITIONS and r['condition']!='reverse_thirds'}
    prior.update({(r['clip_id'],r['view'],r['condition']):r for r in corr['predictions']})
    started=time.monotonic();count=0;reused=0;predictions=0
    for row in scope['rows']:
        cid=row['clip_id'];path=U/'records'/f'{cid}.json'
        if path.exists():
            existing=read(path);assert existing['scope_sha256']==sha(U/'SCOPE.json');count+=1;continue
        rec=read(B/'data/records'/f'{cid}.json');rawrow=prep[cid];assert len(rawrow['intervals'])==1
        interval=rawrow['intervals'][0];rr=[];stats=[]
        for view in VIEWS:
            z=load(rec['views'][view]['channel_file']);obs=load(rec['views'][view]['observation']);x=z['x_whole'];n=len(x)
            fid=obs['frame_id'];ts=obs['timestamp_us'];used=obs['usable']
            idx=np.flatnonzero(np.isin(fid,z['whole_frame_id']));assert len(idx)==n
            assert np.array_equal(fid[idx],z['whole_frame_id']) and np.array_equal(ts[idx],z['whole_ts'])
            assert np.array_equal(idx,np.arange(idx[0],idx[-1]+1)),'internal frame deletion'
            assert np.array_equal(x[1:,279],np.diff(z['whole_ts']).astype(np.float32)/1e6)
            core=(ts>=round(interval[0]*1e6))&(ts<=round(interval[1]*1e6));coreidx=np.flatnonzero(core);assert len(coreidx)>0
            retained=np.zeros(len(ts),bool);retained[idx]=True
            # Compare final body-relative hand validity to saved observations; actual input support may depend on shoulder origin.
            handmask=x[:,186:237][:,9:51].reshape(n,2,21)>0
            localmask=x[:,237:279].reshape(n,2,21)>0
            support=np.zeros((len(ts),2),bool);support[idx]=handmask.any(2)
            local=np.zeros((len(ts),2),bool);local[idx]=localmask.any(2)
            stages=[]
            for ids in np.array_split(coreidx,3):
                stages.append({'frames':len(ids),'in_input':int(retained[ids].sum()),'any_hand':int(support[ids].any(1).sum()),
                  'left_any':int(support[ids,0].sum()),'right_any':int(support[ids,1].sum()),'left_local':int(local[ids,0].sum()),'right_local':int(local[ids,1].sum())})
            nohands=runs(core&~support.any(1),ts,fid);cutruns=runs(core&~retained,ts,fid)
            full={'clip_id':cid,'word_code':row['word_code'],'label':labels[row['label_index']],'signer_id':row['signer_id'],'view':view,
              'annotation_seconds':interval,'core_frames':len(coreidx),'core_retained':int((core&retained).sum()),
              'core_trimmed_frames':int((core&~retained).sum()),'trimmed_runs':cutruns,
              'active_frames':n,'active_first_last_frame':[int(z['whole_frame_id'][0]),int(z['whole_frame_id'][-1])],
              'core_any_hand':int(support[core].any(1).sum()),'core_left_any':int(support[core,0].sum()),'core_right_any':int(support[core,1].sum()),
              'core_left_local':int(local[core,0].sum()),'core_right_local':int(local[core,1].sum()),'no_hand_runs':nohands,
              'longest_no_hand_seconds':max((r['seconds'] for r in nohands),default=0.),'stages':stages,
              'internal_frames_deleted':0,'source_times_retained':True,
              'core_observed_but_input_left_unavailable':int((core&used[:,33:54].any(1)&~support[:,0]).sum()),
              'core_observed_but_input_right_unavailable':int((core&used[:,54:75].any(1)&~support[:,1]).sum())}
            stats.append(full)
            for condition in CONDITIONS:
                key=(cid,view,condition);oldr=prior.get(key)
                if oldr:
                    out={model:score(oldr[model]['logits'],row['label_index'],labels) for model in ('C','T3F')};source='prior_corrected_diagnosis_reuse';reused+=1
                else:
                    data=variant(x,condition);xx=torch.from_numpy(data[None]);pad=torch.ones(1,len(data))
                    with torch.inference_mode():zz=c(xx,pad)[0].numpy();tt=m(xx,pad)[0].numpy()
                    out={'C':score(zz,row['label_index'],labels),'T3F':score(tt,row['label_index'],labels)};source='new_public_TRAIN_diagnostic';predictions+=1
                rr.append({'view':view,'condition':condition,'source':source,**out})
        write(path,{'scope_sha256':sha(U/'SCOPE.json'),'clip_id':cid,'label_index':row['label_index'],'label':labels[row['label_index']],
          'word_code':row['word_code'],'signer_id':row['signer_id'],'input_stats':stats,'predictions':rr})
        count+=1
        if count%100==0:print(json.dumps({'stage':'DIAGNOSING','clips':count,'of':899,'seconds':round(time.monotonic()-started,1)}),flush=True)
    assert count==899 and before==[statehash(c),statehash(m)]
    for p,h in scope['pins'].items():assert sha(p)==h,p
    write(U/'COMPLETE.json',{'status':'COMPLETE_PUBLIC_DIAGNOSIS_PENDING_RECOUNT_VISUAL','at':now(),'clips':count,'words':100,
       'model_state_unchanged':True,'source_pins_unchanged':len(scope['pins']),'new_pairs_this_execution':predictions,'reused_pairs_this_execution':reused,
       'scope_sha256':sha(U/'SCOPE.json'),'record_hashes':{p.stem:sha(p) for p in sorted((U/'records').glob('*.json'))},
       'seconds':time.monotonic()-started,'training_runs':0,'user_recordings_read':0})
    print('COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'FIRST_ERROR.json'
        if not p.exists():write(p,{'at':now(),'traceback':traceback.format_exc()})
        raise
