from task import *
import numpy as np,torch,traceback
from model import EvidenceHeads
from evidence import assess
from runtime_candidate import Candidate
from prepare_feedback import normalize

def infer(model,base,x,word):
 wi=base.f['feedback']['words'].index(word);z=normalize(x,base.stats)
 with torch.inference_mode():
  b=base.feedback(torch.from_numpy(z[None]),torch.tensor([len(z)]),torch.tensor([wi]));h=b['latent'];p=b['attributes'].sigmoid();valid=torch.ones((1,len(z)),dtype=torch.bool);ph,d0=model(h,p,torch.tensor([wi]),valid,torch.zeros_like(p));_,d1=model(h,p,torch.tensor([wi]),valid,torch.ones_like(p))
 return ph[0].softmax(-1).numpy(),d0[0].sigmoid().numpy(),d1[0].sigmoid().numpy()

def controls(row,z,meta):
 x=z['raw_x'];ts=z['timestamp_us'];fid=z['frame_id'];n=len(x);dt=int(np.median(np.diff(ts)));lo,hi=meta['intervals'][0];a=int(np.searchsorted(ts,lo*1e6));b=int(np.searchsorted(ts,hi*1e6))
 cases=[]
 for kind in ['speed075','speed125','omit_last_main_then_rest','remove_middle_event','hand_gap_2','hand_gap_30']:
  mapping=np.arange(n);remove=[];y=x.copy();t=ts.copy()
  if kind.startswith('speed'):
   speed=.75 if kind=='speed075' else 1.25;mapping=np.rint(np.linspace(0,n-1,round(n/speed))).astype(int);y=x[mapping].copy();t=np.arange(len(y))*dt
  elif kind=='omit_last_main_then_rest':
   start=a+max(2,(b-a)//2);end=min(n,b+2);remove=list(range(start,end));mapping=np.delete(mapping,remove);y=x[mapping].copy();t=np.arange(len(y))*dt
  elif kind=='remove_middle_event':
   start=a+max(1,(b-a)//3);end=min(n,a+max(2,2*(b-a)//3));remove=list(range(start,end));mapping=np.delete(mapping,remove);y=x[mapping].copy();t=np.arange(len(y))*dt
  else:
   size=2 if kind=='hand_gap_2' else 30;start=max(0,min(n-size,(a+b)//2-size//2));end=start+size;remove=list(range(start,end));y[start:end,18:186]=0;y[start:end,195:279]=0
  y[:,-1]=np.r_[0,np.diff(t)/1e6];cases.append((kind,y,t,fid[mapping],mapping,remove))
 return cases

def main():
 assert not (U/'EVALUATION.json').exists();torch.set_num_threads(4);prep=read(U/'PREPARED.json');base=Candidate();meta={r['clip_id']:r for r in read(G/'INPUT_MANIFEST.json')['rows']};f=read(U/'FROZEN.json');allrows=[];features=[]
 write(U/'EVALUATION_CODE_PINS.json',{str(p):sha(p) for p in [U/'evidence.py',U/'evaluate.py',U/'CARDS.json',U/'GUIDE_REVIEW.json',U/'FROZEN.json']})
 for seed in f['new_heads']['seeds']:
  model=EvidenceHeads(prep['guide_words']);model.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',map_location='cpu',weights_only=True));model.eval()
  for row in [r for r in prep['rows'] if r['role']=='development']:
   z=np.load(row['path']);word=prep['words'][row['word']];saved=np.load(U/'runs'/str(seed)/'predictions'/f"{row['clip_id']}.npz")
   normal=assess(word,saved['phase'],saved['different_if0'],saved['different_if1'],z['raw_x'],z['timestamp_us'],z['frame_id']);p=U/'evaluated'/str(seed)/f"{row['clip_id']}_normal.json";write(p,normal)
   allrows.append(dict(seed=seed,clip_id=row['clip_id'],word=word,kind='normal',statuses=[c['status'] for c in normal['cards']],required=[c.get('required',True) for c in normal['cards']],output=str(p),sha256=sha(p)))
   for i,j in zip(*np.where(z['ym']>0)):
    for expected in [0,1]:
     pr=float(saved['different_if1' if expected else 'different_if0'][i,j]);truth=int((z['y'][i,j]>.5)!=bool(expected));features.append(dict(seed=seed,clip_id=row['clip_id'],frame=int(z['frame_id'][i]),feature=int(j),expected=expected,truth=truth,probability=pr))
   for kind,x,ts,fid,mapping,removed in controls(row,z,meta[row['clip_id']]):
    ph,d0,d1=infer(model,base,x,word);out=assess(word,ph,d0,d1,x,ts,fid);p=U/'evaluated'/str(seed)/f"{row['clip_id']}_{kind}.json";write(p,out)
    allrows.append(dict(seed=seed,clip_id=row['clip_id'],word=word,kind=kind,statuses=[c['status'] for c in out['cards']],required=[c.get('required',True) for c in out['cards']],output=str(p),sha256=sha(p),mapping=mapping.tolist(),removed=removed,synthetic_not_real_learner_error=True))
 summary={}
 for seed in f['new_heads']['seeds']:
  summary[str(seed)]={}
  for kind in ['normal']+f['evaluation']['controls']:
   rs=[r for r in allrows if r['seed']==seed and r['kind']==kind];summary[str(seed)][kind]=dict(clips=len(rs),missing=sum('missing_candidate' in r['statuses'] for r in rs),differences=sum('difference_candidate' in r['statuses'] for r in rs),order=sum('order_candidate' in r['statuses'] for r in rs),unknown=sum('unknown' in r['statuses'] for r in rs),all_required_supported=sum(all(s=='supported' for s,req in zip(r['statuses'],r['required']) if req) for r in rs))
  fs=[r for r in features if r['seed']==seed];summary[str(seed)]['feature_pair_accuracy']=sum((r['probability']>=.5)==bool(r['truth']) for r in fs)/len(fs)
 gates=[]
 for seed in f['new_heads']['seeds']:
  s=summary[str(seed)];gate=s['normal']['missing']==0 and s['normal']['all_required_supported']>=9 and s['omit_last_main_then_rest']['missing']>=9
  gates.append(dict(seed=seed,pass_for_action_omission_claim=gate))
 write(U/'EVALUATION.json',dict(status='FIXED_DEVELOPMENT_AND_ENGINEERING_CONTROLS_COMPLETE',summary=summary,rows=allrows,feature_pairs=features,gates=gates,used_to_tune=False,real_learner_error_gold=False));event('evaluation_complete',summary=summary,gates=gates)
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
