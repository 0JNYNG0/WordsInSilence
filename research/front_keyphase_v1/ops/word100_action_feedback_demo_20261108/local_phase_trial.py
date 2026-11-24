"""One predeclared structural follow-up, retaining the first completed trial."""
from task import *
import numpy as np,torch,random,traceback
from local_phase import LocalPhase
from model import EvidenceHeads
from evidence import assess
from evaluate import controls,infer
from runtime_candidate import Candidate
from prepare_feedback import normalize

def batch(rows,stats):
 T=max(len(r['raw_x']) for r in rows);x=np.zeros((len(rows),T,280),np.float32);y=np.full((len(rows),T),-1,np.int64);w=np.zeros((len(rows),T),np.float32)
 for i,r in enumerate(rows):
  n=len(r['raw_x']);x[i,:n]=normalize(r['raw_x'],stats);y[i,:n]=r['stage_y'];w[i,:n]=r['stage_w']
 return torch.from_numpy(x),torch.tensor([r['word'] for r in rows]),torch.arange(T)[None]<torch.tensor([len(r['raw_x']) for r in rows])[:,None],torch.from_numpy(y),torch.from_numpy(w)

def main():
 assert not (U/'LOCAL_PHASE_COMPLETE.json').exists();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 prep=read(U/'PREPARED.json');base=Candidate();cards=read(U/'CARDS.json')['words'];f=read(U/'FROZEN.json');meta={r['clip_id']:r for r in read(G/'INPUT_MANIFEST.json')['rows']}
 plan=dict(hypothesis='Whole-sequence latent can retain a familiar stage despite removed local evidence. Test a local coordinate-only phase branch plus FIT-only partial-sequence examples.',architecture='280 relative coordinates/masks -> Conv5 64 -> Conv5 64 -> 14 phase classes; receptive field9 frames',seeds=[42,43,44],epochs=100,lr=.001,batch=8,normalization='existing FIT only',partial_examples='Remove the last required physical group and following main-stage frames using FIT labels; retain original rest; labels remain at source frames. No resynthesized joint points.',development_used_to_select=False,known_dev_exposure=True,selection='No replacement unless fixed gates pass all three seeds; otherwise demonstration only.',source_pins={str(U/n):sha(U/n) for n in ['local_phase.py','local_phase_trial.py','CARDS.json','EVALUATION.json']})
 write(U/'LOCAL_PHASE_FROZEN.json',plan);rows=[];fit=[]
 for r in prep['rows']:
  z=np.load(r['path']);rr={**r,**{k:z[k] for k in z.files}};rows.append(rr)
  if r['role']!='fit':continue
  fit.append(rr);word=prep['words'][r['word']];last=[c for c in cards[word] if c.get('required',True)][-1];group=last['groups'][-1];ids=np.flatnonzero(np.isin(rr['stage_y'],np.array(group)+1));main=np.flatnonzero(rr['stage_y']>0)
  if len(ids) and len(main):
   mapping=np.r_[np.arange(ids[0]),np.arange(main[-1]+1,len(rr['raw_x']))];cut={**rr}
   for k in ['raw_x','stage_y','stage_w']:cut[k]=rr[k][mapping].copy()
   cut['raw_x'][:,-1]=1/30;fit.append(cut)
 done=[];evaluation=[]
 for seed in plan['seeds']:
  d=U/'local_runs'/str(seed);d.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);m=LocalPhase(prep['guide_words']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);t=time.perf_counter()
  for epoch in range(100):
   order=np.random.default_rng(seed+epoch).permutation(len(fit));m.train()
   for j in range(0,len(order),8):
    b=batch([fit[k] for k in order[j:j+8]],base.stats);ph=m(*b[:3]);ce=torch.nn.functional.cross_entropy(ph.transpose(1,2),b[3],ignore_index=-1,reduction='none');loss=(ce*b[4]).sum()/b[4].sum().clamp_min(1);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
  m.eval();torch.save(m.state_dict(),d/'model.pt');seconds=time.perf_counter()-t;head=EvidenceHeads(prep['guide_words']);head.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True));head.eval()
  for r in [r for r in rows if r['role']=='development']:
   word=prep['words'][r['word']];saved=np.load(U/'runs'/str(seed)/'predictions'/f"{r['clip_id']}.npz")
   cases=[('normal',r['raw_x'],r['timestamp_us'],r['frame_id'],np.arange(len(r['raw_x'])),[])]+controls(r,r,meta[r['clip_id']])
   for kind,x,ts,fid,mapping,removed in cases:
    with torch.inference_mode():phase=m(torch.from_numpy(normalize(x,base.stats)[None]),torch.tensor([r['word']]),torch.ones((1,len(x)),dtype=torch.bool))[0].softmax(-1).numpy()
    if kind=='normal':d0,d1=saved['different_if0'],saved['different_if1']
    else:_,d0,d1=infer(head,base,x,word)
    out=assess(word,phase,d0,d1,x,ts,fid);p=d/'outputs'/f"{r['clip_id']}_{kind}.json";write(p,out);evaluation.append(dict(seed=seed,clip_id=r['clip_id'],word=word,kind=kind,statuses=[c['status'] for c in out['cards']],required=[c.get('required',True) for c in out['cards']],output=str(p),sha256=sha(p),mapping=mapping.tolist(),removed=removed))
  done.append(dict(seed=seed,seconds=seconds,parameters=sum(p.numel() for p in m.parameters()),sha256=sha(d/'model.pt')));event('local_phase_trial',**done[-1])
 summary={}
 for seed in plan['seeds']:
  summary[str(seed)]={}
  for kind in ['normal']+f['evaluation']['controls']:
   rs=[r for r in evaluation if r['seed']==seed and r['kind']==kind];summary[str(seed)][kind]=dict(clips=len(rs),missing=sum('missing_candidate' in r['statuses'] for r in rs),difference=sum('difference_candidate' in r['statuses'] for r in rs),unknown=sum('unknown' in r['statuses'] for r in rs),supported=sum(all(s=='supported' for s,q in zip(r['statuses'],r['required']) if q) for r in rs))
 write(U/'LOCAL_PHASE_COMPLETE.json',dict(runs=done,rows=evaluation,summary=summary,fit_examples=len(fit),real_error_gold=False,completed_first_trial_retained=True,development_not_heldout=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'LOCAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
