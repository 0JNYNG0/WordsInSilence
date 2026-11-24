"""Fixed alternate-wait diagnostic after training. No weight/threshold selection."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
U=Path('C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1/ops/word100_boundary_aux_20261026')
sys.path.insert(0,str(U));import run_aux as a

def varied(xy,pr,ts,kind,seconds):
 step=int(np.median(np.diff(ts)));n=max(1,round(seconds*1e6/step))
 left=np.repeat(xy[:1],n,0);right=np.repeat(xy[-1:],n,0);lp=np.repeat(pr[:1],n,0);rp=np.repeat(pr[-1:],n,0)
 left[:,25:67]=0;right[:,25:67]=0;lp[:,25:67]=False;rp[:,25:67]=False
 for active,arr,mask,index,source in [(kind in ('start_spike','both_spikes'),left,lp,0,0),(kind in ('end_spike','both_spikes'),right,rp,-1,-1)]:
  if active:
   available=np.flatnonzero(pr[source,25:67])+25;j=next((k for k in (25,46) if pr[source,k]),int(available[0]))
   arr[index,j]=xy[source,j];mask[index,j]=True
 xx=np.concatenate([left,xy,right]);pp=np.concatenate([lp,pr,rp]);tt=np.concatenate([ts[0]-step*np.arange(n,0,-1),ts,ts[-1]+step*np.arange(1,n+1)])
 lo,hi=a.b.active_span(pp);assert lo<=n and hi>=n+len(xy)-1
 assert np.array_equal(xx[n:n+len(xy)],xy) and np.array_equal(pp[n:n+len(xy)],pr)
 return a.b.build_channels(xx[lo:hi+1],pp[lo:hi+1],tt[lo:hi+1])['x']

def main():
 a.checks();assert a.b.read(U/'VERIFIED.json')['status']=='PASS'
 pins={str(p):a.b.sha(p) for s in a.SEEDS for p in (U/f'BA_{s}_best.pt',U/f'CONTROL_{s}_best.pt')}
 a.write('TRANSFER_FROZEN.json',dict(at=time.time(),seconds=[.5,1.5],kinds=a.KINDS[1:],views=a.VIEWS,
   scope='same DEV300, different synthetic wait durations; development stress transfer only, not fresh people or natural webcam test',
   action='fixed originalT3F vs clean continuation vs augmented head; no selection or tuning from these scores',
   script_sha256=a.b.sha(__file__),pins=pins))
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 ctx,tr,dv=a.load_source_rows();conds=[(v,sec,k) for v in a.VIEWS for sec in (.5,1.5) for k in a.KINDS[1:]]
 for seed in a.SEEDS:
  ck=torch.load(a.b.T/f'T3F_{seed}_best.pt',map_location='cpu',weights_only=True)
  m=a.FrozenCTemporal(**a.MODEL);m.load_state_dict(ck['state_dict']);m.eval();before=a.statehash(m.state_dict())
  ws=[ck['state_dict']['phase_weight'].numpy()]+[torch.load(U/f'{prefix}_{seed}_best.pt',map_location='cpu',weights_only=True)['state_dict']['phase_weight'].numpy() for prefix in ('CONTROL','BA')]
  scores=np.zeros((300,len(conds),3,100),np.float32)
  with torch.inference_mode():
   for ri,r in enumerate(dv):
    rec=a.b.read(a.b.B/'data/records'/(r['clip_id']+'.json'));orig={}
    for v in a.VIEWS:orig[v]=a.b.prepare_native(a.b.arrays(rec['views'][v]['observation']))[:3]
    xs=[varied(*orig[v],k,sec) for v,sec,k in conds]
    if ri==0:
     for v in a.VIEWS:
      for k in a.KINDS[1:]:assert np.array_equal(varied(*orig[v],k,1.),a.b.transform(*orig[v],k)[0]['x'])
    x,p=a.padded_arrays(xs);base,d=m.base_and_contrast(torch.from_numpy(x),torch.from_numpy(p));base=base.numpy();d=d.numpy()
    for wi,w in enumerate(ws):scores[ri,:,wi]=base+d@w.T
  assert before==a.statehash(m.state_dict());a.checks()
  file=U/f'TRANSFER_{seed}_LOGITS.npz';assert not file.exists();np.savez_compressed(file,logits=scores)
  labels=np.array([r['label_index'] for r in dv]);counts={}
  for ci,(v,sec,k) in enumerate(conds):
   pred=scores[:,ci].argmax(-1);correct=pred==labels[:,None]
   counts[f'{v}/{sec}/{k}']=dict(zip(('original','control','augmented'),correct.sum(0).tolist()))
  a.write(f'TRANSFER_{seed}.json',dict(seed=seed,logits_sha256=a.b.sha(file),conditions=conds,
      ids=[r['clip_id'] for r in dv],labels=labels.tolist(),counts=counts,protocol_sha256=a.b.sha(U/'TRANSFER_FROZEN.json')))
  print('TRANSFER',seed,counts,flush=True)
 for p,h in pins.items():assert a.b.sha(p)==h
if __name__=='__main__':main()
