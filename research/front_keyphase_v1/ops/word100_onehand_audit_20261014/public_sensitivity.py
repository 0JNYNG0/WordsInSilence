"""Frozen PUBLIC-only intervention, fixed original active span, all 36 controls."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;LAB=U.parents[1];ROOT=LAB.parents[1]
sys.path[:0]=[str(LAB/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]
from wis_lab.core.observation import Observation
from wis_lab.service import word100,neural
from pilot.web_input import to_native_like,active_span
from pilot.data import build_channels
import torch

def main():
    output=U/'PUBLIC_SENSITIVITY.json';assert not output.exists(),'Preserve first result'
    audit=json.loads((U/'AUDIT.json').read_text('utf-8'))
    cases=[p for p in audit['public_profiles'] if p['role']=='development']
    assert len(cases)==36 and set(p['signer'] for p in cases)=={11,12,13}
    config=word100.config();torch.set_num_threads(4);model=word100._load();rows=[]
    saved=json.loads((LAB/'runs/word100_tcn_v1/whole_context/PREDICTIONS.json').read_text('utf-8'))
    if isinstance(saved,dict):
        print('saved keys',saved.keys());raise RuntimeError('Inspect saved prediction schema')
    saved={r['clip_id']:r for r in saved}
    for p in cases:
        path=Path(p['observation']);assert hashlib.sha256(path.read_bytes()).hexdigest()==p['sha256']
        obs=Observation.load(str(path));nat,present=to_native_like(obs.xy_px,obs.usable)
        lo,hi=active_span(present);nat=nat[lo:hi+1];present=present[lo:hi+1];ts=obs.timestamp_us[lo:hi+1]
        control=build_channels(nat,present,ts)['x']
        service,why,_=neural._channels(obs);assert why is None and np.array_equal(control,service['x'])
        target=config['word_codes'].index(p['word_id'])
        variants={}
        for name,sl in [('original',None),('remove_left',slice(25,46)),('remove_right',slice(46,67))]:
            xy=nat.copy();pr=present.copy()
            if sl is not None:xy[:,sl]=np.nan;pr[:,sl]=False
            x=build_channels(xy,pr,ts)['x']
            assert x.shape==control.shape and np.array_equal(x[:,-1],control[:,-1])
            if sl is not None:
                # Exact channel ranges affected by just one 21-point hand; BODY_25 untouched.
                hi_hand=0 if name=='remove_left' else 1
                changed=np.concatenate([np.arange(18+hi_hand*42,60+hi_hand*42),np.arange(102+hi_hand*42,144+hi_hand*42),np.arange(195+hi_hand*21,216+hi_hand*21),np.arange(237+hi_hand*21,258+hi_hand*21)])
                keep=np.setdiff1d(np.arange(280),changed)
                assert np.array_equal(x[:,keep],control[:,keep]),'Unaffected channels changed'
                assert np.count_nonzero(x[:,changed])==0,'Removed hand did not become masked zero'
            with torch.inference_mode():logits=model(torch.from_numpy(x[None]),torch.ones((1,len(x))))[0].numpy()
            order=np.argsort(-logits,kind='stable');probs=np.exp(logits.astype(float)-logits.max());probs/=probs.sum()
            rec=dict(target_rank=int(np.where(order==target)[0][0])+1,target_score_percent=float(probs[target]*100),top1=config['labels'][order[0]],logits=logits.tolist(),input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
            if name=='original':
                delta=float(np.max(np.abs(logits-np.array(saved[p['clip_id']]['logits']))));assert delta<1e-4,delta
                rec['saved_control_logit_max_abs_diff']=delta
            variants[name]=rec
        rows.append(dict(clip_id=p['clip_id'],word_id=p['word_id'],label=p['label'],signer=p['signer'],span=[int(lo),int(hi)],variants=variants))
        print(p['label'],p['signer'],[(k,v['target_rank']) for k,v in variants.items()],flush=True)
    output.write_text(json.dumps(dict(schema='public_frozen_missing_hand_sensitivity_v1',checkpoint_sha256=word100.PIN,case_count=36,forward_count=108,fixed_span=True,no_training=True,no_user_inference=True,rows=rows),ensure_ascii=False,indent=2)+'\n','utf-8')
if __name__=='__main__':main()
