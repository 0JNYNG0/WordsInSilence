"""One fixed evaluation per checkpoint/view. No optimizer or checkpoint selection."""
from common import *
import torch,copy
sys.path[:0]=[str(ROOT/'research/sign_model_pilot_v1/src'),str(T),str(B)]
from pilot.model_masked_v1 import MaskedSignTCN
from frozen_temporal import FrozenCTemporal
from training_core import MODEL,prediction,padded_arrays

def statehash(state):
 h=hashlib.sha256()
 for k,t in sorted(state.items()):
  h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def model(arm,seed,ctx):
 info=ctx['models'][arm][str(seed)];assert sha(info['path'])==info['sha256']
 ck=torch.load(info['path'],map_location='cpu',weights_only=True)
 m=MaskedSignTCN(**MODEL) if arm=='C' else FrozenCTemporal(**MODEL)
 m.load_state_dict(ck['state_dict'],strict=True);m.eval()
 for p in m.parameters():p.requires_grad_(False)
 # T3-F assert_frozen expects its auxiliary parameter marked trainable, while no_grad globally prevents learning.
 if arm=='T3F':m.phase_weight.requires_grad_(True);m.assert_frozen()
 return m,ck

def preflight():
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins']);torch.set_num_threads(4)
 prior=read(B/'training_CONTEXT.json');row=next(r for r in prior['ordered_rows'] if r['role']=='train_candidate')
 x=load(row['views']['sensor']['channel_file'])['x_whole'];xx,pp=padded_arrays([x,x]);xx=torch.from_numpy(xx);pp=torch.from_numpy(pp)
 tests=[]
 for seed in SEEDS:
  c,cc=model('C',seed,ctx);f,fc=model('T3F',seed,ctx)
  assert all(torch.equal(v,fc['state_dict'][k]) for k,v in cc['state_dict'].items())
  hashes=[statehash(m.state_dict()) for m in [c,f]]
  with torch.inference_mode():
   a=c(xx,pp);base,d=f.base_and_contrast(xx,pp);full=f(xx,pp)
  assert torch.equal(a,base) and torch.isfinite(full).all()
  assert hashes==[statehash(m.state_dict()) for m in [c,f]]
  tests.append({'seed':seed,'C_equals_T3F_base':True,'finite':True,'unchanged':True,'selected_epoch_C':cc['epoch'],'selected_epoch_T3F':fc['epoch']})
 write(U/'PREFLIGHT.json',{'status':'PASS','at':now(),'tests':tests,'public_train_probe':row['clip_id'],'heldout_inference':0,'training':0,'model_checkpoint_pins':ctx['models']},new=True)
 print('PREFLIGHT PASS',flush=True)

def main():
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins']);verified=read(U/'INPUTS_VERIFIED.json');assert verified['status']=='PASS'
 check(verified['record_pins']);assert not (U/'PREP_RUNNING.json').exists() and not (U/'EVAL_RUNNING.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 rows=[]
 for r in ctx['rows']:
  rec=read(DATA/'records'/f"{r['clip_id']}.json");q=copy.deepcopy(r);q['arrays']={};q['reasons']={}
  for v in VIEWS:
   cp=rec['views'][v]['channel_file'];q['arrays'][v]=load(cp)['x_whole'] if cp else None;q['reasons'][v]=rec['views'][v]['feature_info']['reason']
  rows.append(q)
 write(U/'EVAL_RUNNING.json',{'pid':os.getpid(),'at':now()},new=True)
 for arm in ['C','T3F']:
  for seed in SEEDS:
   out=U/f'{arm}_{seed}_SCORES.json'
   if out.exists():
    old=read(out);assert old['context_sha256']==sha(U/'EVAL_CONTEXT.json');continue
   started=U/f'{arm}_{seed}_STARTED.json';write(started,{'at':now(),'checkpoint':ctx['models'][arm][str(seed)]},new=True)
   m,ck=model(arm,seed,ctx);before=statehash(m.state_dict());pred={}
   with torch.inference_mode():
    for v in VIEWS:
     good=[r for r in rows if r['arrays'][v] is not None]
     scores=[prediction(r,reason=r['reasons'][v]) for r in rows if r['arrays'][v] is None]
     for i in range(0,len(good),32):
      batch=good[i:i+32];x,p=padded_arrays([r['arrays'][v] for r in batch]);logits=m(torch.from_numpy(x),torch.from_numpy(p));assert torch.isfinite(logits).all()
      scores.extend(prediction(r,z) for r,z in zip(batch,logits.numpy()))
     pred[v]=sorted(scores,key=lambda r:r['clip_id']);assert len(pred[v])==200
   assert before==statehash(m.state_dict());check(ctx['pins'])
   write(out,{'at':now(),'status':'FROZEN_HELDOUT_EVALUATION','arm':arm,'seed':seed,'context_sha256':sha(U/'EVAL_CONTEXT.json'),'checkpoint':ctx['models'][arm][str(seed)],'model_state_before':before,'model_state_after':statehash(m.state_dict()),'epoch':ck['epoch'],'predictions':pred,'input_groups':verified['groups'],'training':0,'checkpoint_reselected':False},new=True)
   event(stage='EVALUATION_MODEL_SAVED',arm=arm,seed=seed)
 (U/'EVAL_RUNNING.json').unlink();event(stage='EVALUATION_COMPLETE_PENDING_RECOUNT',models=6,predictions=2400)

if __name__=='__main__':preflight() if '--preflight' in sys.argv else main()
