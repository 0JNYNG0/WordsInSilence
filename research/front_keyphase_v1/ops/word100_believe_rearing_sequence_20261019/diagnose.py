"""Public TRAIN sequence diagnosis only. No training or user-recording access."""
from pathlib import Path
import json, sys, hashlib, zipfile, math
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parent
ROOT = O.parents[2]
B = O / 'word100_body_boundary_20261017'
T = O / 'word100_frozen_c_temporal_20261019'
sys.path.insert(0, str(ROOT / 'research/sign_model_pilot_v1/src'))
sys.path.insert(0, str(T))
from frozen_temporal import FrozenCTemporal
from temporal import phase_weights, summarize
from pilot.model_masked_v1 import MaskedSignTCN

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x):
    assert not p.exists(), p
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}
def statehash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()): h.update(k.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()
def predict(logits, label, other, labels):
    a=np.asarray(logits,dtype=np.float64); e=np.exp(a-a.max()); p=e/e.sum(); order=np.argsort(-a,kind='stable')
    return {'logits':a.tolist(), 'target_rank':int(np.where(order==label)[0][0])+1,
            'other_rank':int(np.where(order==other)[0][0])+1,
            'target_score':float(p[label]*100), 'other_score':float(p[other]*100),
            'target_minus_other_logit':float(a[label]-a[other]),
            'top3':[{'label':labels[i], 'score':float(p[i]*100)} for i in order[:3]]}

def main():
    assert not (U/'ANALYSIS.json').exists()
    ctx=read(B/'training_CONTEXT.json')
    rows=sorted([r for r in ctx['ordered_rows'] if r['role']=='train_candidate' and r['word_code'] in ('WORD1139','WORD0196')],key=lambda r:r['clip_id'])
    assert len(rows)==18 and set(r['signer_id'] for r in rows)==set(range(1,10))
    assert all(r['clip_id'].startswith('NIA_SL_') for r in rows)
    protected=[B/'training_C_20261013_best.pt', T/'T3F_20261013_best.pt',T/'temporal.py',T/'frozen_temporal.py',
               ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py', B/'training_CONTEXT.json']
    web=ROOT/'research/front_keyphase_v1/src/wis_lab/service'
    protected += [web/'word100_candidate.py',web/'word100_t3f.py']
    pins={str(p):sha(p) for p in protected}
    for r in rows:
        rp=B/'data/records'/f"{r['clip_id']}.json"; pins[str(rp)]=sha(rp)
        record=read(rp)
        for view in ('masked','sensor'):
            for key in ('channel_file','observation'):
                p=Path(record['views'][view][key]);pins[str(p)]=sha(p)
            assert pins[record['views'][view]['channel_file']]==r['views'][view]['sha256']
    protocol={'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
      'task':'믿다–사육장 공개 자료의 길이·단계·순서 진단 및 다음 계획',
      'authorized':'Public-only diagnosis and planning. No additional training before user review.',
      'train_clips':[r['clip_id'] for r in rows], 'views':['masked','sensor'], 'model_seed':20261013,
      'counterfactuals':['original','prefix_half','suffix_half','reverse_thirds','double_sequence','dt_x2','dt_x0.5'],
      'counterfactual_limit':'Deliberately altered inputs, not valid sign labels, accuracy tests or training examples. No word-specific edits, coefficient search or model selection.',
      'visual_sample':'TRAIN signers 1,5,9 for both words, 16 equally spaced active-span frames and full local videos.',
      'saved_DEV':'Read existing C/T3F three-seed predictions for these two words only; no re-inference, no new holdout.',
      'forbidden':['training','user/mentor recording reads','model/threshold selection','ID14–18 raw inputs','production edits','new MediaPipe extraction','downloads','Git'],
      'implementation_and_verification':'Codex local diagnosis plus separate saved-output arithmetic; not independent human/agent sign annotation.',
      'pins':pins}
    write(U/'SCOPE.json',protocol)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ck=torch.load(T/'T3F_20261013_best.pt',map_location='cpu',weights_only=True)
    cp=torch.load(B/'training_C_20261013_best.pt',map_location='cpu',weights_only=True)
    m=FrozenCTemporal(**ck['model_config']);m.load_state_dict(ck['state_dict']);m.eval();m.assert_frozen()
    c=MaskedSignTCN(**ck['model_config']);c.load_state_dict(cp['state_dict']);c.eval()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in c.state_dict().items())
    before=[statehash(c),statehash(m)]
    labels=ctx['labels']; pair=[ctx['word_codes'].index(w) for w in ('WORD1139','WORD0196')]
    output=[]; decomposition=[]; meta=[]; features={}; audit_max=0.
    for r in rows:
        record=read(B/'data/records'/f"{r['clip_id']}.json")
        label=r['label_index']; other=next(i for i in pair if i!=label)
        for view in ('masked','sensor'):
            z=load(record['views'][view]['channel_file']);obs=load(record['views'][view]['observation']);x=z['x_whole'];n=len(x)
            active=np.isin(obs['frame_id'],z['whole_frame_id'])
            used=obs['usable'][active]
            meta.append({'clip_id':r['clip_id'],'label':labels[label],'signer':r['signer_id'],'view':view,
              'full_seconds':float((obs['timestamp_us'][-1]-obs['timestamp_us'][0])/1e6),
              'active_seconds':float((z['whole_ts'][-1]-z['whole_ts'][0])/1e6),
              'context_seconds':float((z['context_ts'][-1]-z['context_ts'][0])/1e6),
              'active_frames':n,'active_frame_ids':[int(z['whole_frame_id'][0]),int(z['whole_frame_id'][-1])],
              'left_any':int(used[:,33:54].any(1).sum()),'right_any':int(used[:,54:75].any(1).sum()),
              'left_all':int(used[:,33:54].all(1).sum()),'right_all':int(used[:,54:75].all(1).sum()),
              'both_none':int((~used[:,33:75].any(1)).sum())})
            cuts=np.array_split(np.arange(n),3)
            variants={'original':x.copy(),'prefix_half':x[:math.ceil(n/2)].copy(),
                      'suffix_half':x[n//2:].copy(),'reverse_thirds':x[np.concatenate(cuts[::-1])].copy(),
                      'double_sequence':np.concatenate([x,x]).copy(),'dt_x2':x.copy(),'dt_x0.5':x.copy()}
            variants['dt_x2'][:,279]*=2;variants['dt_x0.5'][:,279]*=.5
            for name,a in variants.items():
                if name in ('suffix_half','reverse_thirds'):a[0,279]=0
                if name=='double_sequence':a[n,279]=np.median(x[1:,279])
                xx=torch.from_numpy(a[None]);pad=torch.ones(1,len(a))
                with torch.inference_mode():
                    cc=c(xx,pad); tt=m(xx,pad)
                    if name=='original':
                        mask=pad.unsqueeze(1);h=m.proj(xx.transpose(1,2)*mask)*mask
                        for block in m.blocks:h=block(h,mask)
                        weights=phase_weights(xx,pad);g,d=summarize(h,pad,weights)
                        delta=torch.einsum('kc,kc->k',(m.phase_weight[label]-m.phase_weight[other]).reshape(3,32),d[0].reshape(3,32))
                        bm=cc[0,label]-cc[0,other];recon=bm+delta.sum();actual=tt[0,label]-tt[0,other]
                        err=float(abs(recon-actual));audit_max=max(audit_max,err);assert err<1e-4
                        decomposition.append({'clip_id':r['clip_id'],'view':view,'label':labels[label],
                          'base_margin':float(bm),'early_middle_late_delta':delta.numpy().tolist(),'final_margin':float(actual),'error':err})
                        features[r['clip_id']+'_'+view+'_h']=h[0].numpy()
                        features[r['clip_id']+'_'+view+'_weights']=weights[0].numpy()
                        features[r['clip_id']+'_'+view+'_contrast']=d[0].numpy()
                output.append({'clip_id':r['clip_id'],'label_index':label,'other_index':other,'label':labels[label],
                  'view':view,'condition':name,'frames':len(a),
                  'C':predict(cc[0].numpy(),label,other,labels),'T3F':predict(tt[0].numpy(),label,other,labels)})
    assert before==[statehash(c),statehash(m)]
    saved=[]
    for seed in (20261013,20261014,20261015):
        for name,p in [('C',B/f'training_C_{seed}_RESULT.json'),('T3F',T/f'T3F_{seed}_RESULT.json')]:
            pins[str(p)]=sha(p)
            data=read(p)
            for view in ('masked','sensor'):
                for row in data['predictions'][view]:
                    if row['label_index'] in pair:
                        label=row['label_index'];other=next(i for i in pair if i!=label)
                        saved.append({'model':name,'seed':seed,'view':view,'clip_id':row['clip_id'],'label':labels[label],
                          'label_index':label,'other_index':other,**predict(row['logits'],label,other,labels)})
    np.savez_compressed(U/'hidden_summaries.npz',**features,
                        pair_phase_weights=(m.phase_weight[pair[0]]-m.phase_weight[pair[1]]).detach().numpy())
    for p,h in pins.items():assert sha(p)==h,p
    write(U/'ANALYSIS.json',{'status':'COMPLETE_PUBLIC_TRAIN_DIAGNOSIS_NOT_NEW_ACCURACY','metadata':meta,
      'predictions':output,'phase_contributions':decomposition,'saved_DEV':saved,
      'decomposition_max_error':audit_max,'protected_hashes_unchanged':True,'model_state_unchanged':True,
      'pins':pins,'training_runs':0,'user_recordings_read':0,'new_mediapipe_runs':0})
    print(json.dumps({'train_clips':len(rows),'public_inputs':len(meta),'diagnostic_pairs':len(output),'saved_DEV_rows':len(saved),'decomposition_error':audit_max}))

if __name__=='__main__':main()
