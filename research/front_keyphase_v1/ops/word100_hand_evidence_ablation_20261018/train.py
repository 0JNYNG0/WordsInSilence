"""Frozen paired C0/C1. Runs only after independent input and visual READY."""
from common import *
import copy, math, random, time, traceback
import numpy as np
import torch
sys.path.insert(0,str(ROOT/'research/sign_model_pilot_v1/src'))
sys.path.insert(0,str(BASE))
from pilot.model_masked_v1 import MaskedSignTCN
from training_core import SEEDS,MODEL,TRAIN,schedule,prediction,detailed_metrics,aggregate,paired
VIEWS=('masked','sensor')

def statehash(state):
 h=hashlib.sha256()
 for k,t in sorted(state.items()):
  h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def savept(p,obj):
 t=p.with_name(p.name+'.writing');torch.save(obj,t);os.replace(t,p)
def event(**kw):
 x={'at':now(),**kw};write(U/'STATUS.json',x)
 with (U/'TRAIN_EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps(x,ensure_ascii=False),flush=True)
def initialize(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 base=MaskedSignTCN(**MODEL);rng=torch.get_rng_state();bs=base.state_dict()
 expanded=MaskedSignTCN(**{**MODEL,'in_channels':310});es=expanded.state_dict()
 for k in es:
  if k=='proj.weight':es[k].zero_();es[k][:,:280]=bs[k]
  else:es[k].copy_(bs[k])
 expanded.load_state_dict(es);torch.set_rng_state(rng)
 assert base.parameter_count()==37636 and expanded.parameter_count()==38596
 return base,expanded
def batch(rows,arm,epoch=None,view=None):
 choices=[(view,'x_whole') if view else schedule(r['clip_id'],epoch,'C') for r in rows]
 xs=[r['arrays'][v][t] for r,(v,t) in zip(rows,choices)];n=max(map(len,xs))
 x=np.zeros((len(xs),n,310),np.float32);pad=np.zeros((len(xs),n),np.float32)
 for i,a in enumerate(xs):x[i,:len(a)]=a;pad[i,:len(a)]=1
 off,size=GROUPS[arm.removeprefix('minus_')]
 for side in (0,1):x[:,:,280+side*15+off:280+side*15+off+size]=0
 return torch.from_numpy(x),torch.from_numpy(pad),torch.tensor([r['label_index'] for r in rows],dtype=torch.long),choices
def loadrows(context):
 rows=[]
 for r in context['records']:
  m=read(SRC/'inputs_v2'/f"{r['clip_id']}.json");checkpins(m['outputs'])
  q=copy.deepcopy(r);q['arrays']={v:loadnp(m['views'][v]) for v in VIEWS};rows.append(q)
 return [r for r in rows if r['role']=='train_candidate'],[r for r in rows if r['role']=='development_only']
def scores(model,dev,arm):
 before=statehash(model.state_dict());model.eval();out={}
 with torch.no_grad():
  for v in VIEWS:
   result=[]
   for j in range(0,len(dev),32):
    rows=dev[j:j+32];x,p,_,_=batch(rows,arm,view=v);a=model(x,p)
    assert torch.isfinite(a).all()
    result.extend(prediction(r,z) for r,z in zip(rows,a.numpy()))
   out[v]=sorted(result,key=lambda r:r['clip_id'])
 assert before==statehash(model.state_dict());return out
def initial_check(train):
 rows=train[:4];report=[]
 for seed in SEEDS:
  b,m=initialize(seed);b.eval();m.eval();x,p,_,_=batch(rows,ARMS[0],epoch=1)
  with torch.no_grad():d=float((b(x[:,:,:280],p)-m(x,p)).abs().max())
  assert d<=1e-5,(seed,d)
  old=read(BASE/f'training_C_{seed}_RESULT.json')
  assert statehash(b.state_dict())==old['initial_state_sha256']
  report.append({'seed':seed,'max_initial_logit_difference':d,'original_initial_state_matches':True})
 write(U/'INITIAL_EQUIVALENCE.json',{'status':'PASS','cases':report,'trained_model_inference':0},new=True)
def run_one(context,train,dev,arm,seed):
 stem=f'{arm}_{seed}';rp=U/f'{stem}_RESULT.json';lp=U/f'{stem}_last.pt';contextsha=sha(U/'TRAIN_CONTEXT.json')
 if rp.exists():
  x=read(rp);assert x['context_sha256']==contextsha;assert sha(U/f'{stem}_best.pt')==x['checkpoint_sha256'];return x
 _,m=initialize(seed);init=statehash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
 trace_dir=U/'traces'/stem;trace_dir.mkdir(parents=True,exist_ok=True)
 if lp.exists():
  last=torch.load(lp,map_location='cpu',weights_only=True);assert last['context_sha256']==contextsha and last['initial_sha256']==init
  m.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);torch.set_rng_state(last['rng'])
 else:last={'epoch':0,'best':None,'history':[],'seconds':0.,'initial_sha256':init,'context_sha256':contextsha}
 start=time.monotonic();previous_seconds=last['seconds']
 for epoch in range(last['epoch']+1,81):
  if last['best'] and last['epoch']-last['best']['epoch']>=12:break
  m.train();order=np.random.default_rng(seed+epoch).permutation(899);tr=[];loss_sum=0.
  for step,j in enumerate(range(0,899,32),1):
   rr=[train[i] for i in order[j:j+32]];x,p,y,choices=batch(rr,arm,epoch=epoch)
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x,p),y);assert torch.isfinite(loss)
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step();loss_sum+=float(loss.item())*len(rr)
   tr.append({'step':step,'clips':[r['clip_id'] for r in rr],'choices':choices})
  pred=scores(m,dev,arm);metrics={v:detailed_metrics(pred[v],context['missingness_groups'][v]) for v in VIEWS};ce=(metrics['masked']['mean_ce']+metrics['sensor']['mean_ce'])/2
  if last['best'] is None or ce<last['best']['ce']-1e-6:last['best']={'epoch':epoch,'ce':ce,'state_dict':copy.deepcopy(m.state_dict()),'predictions':pred,'metrics':metrics}
  tp=trace_dir/f'{epoch:03}.json'
  if tp.exists():raise RuntimeError('Orphan uncommitted epoch trace: preserve and review exact checkpoint recovery')
  write(tp,{'arm':arm,'seed':seed,'epoch':epoch,'batches':tr},new=True)
  h={'epoch':epoch,'train_ce':loss_sum/899,'selection_ce':ce,'best_epoch':last['best']['epoch'],'sensor_top1':metrics['sensor']['top1_correct'],'masked_top1':metrics['masked']['top1_correct'],'trace_sha256':sha(tp)}
  last['history'].append(h);last.update(epoch=epoch,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state(),seconds=previous_seconds+time.monotonic()-start)
  savept(lp,last);write(U/f'{stem}_history.json',last['history'])
  event(stage='TRAINING',arm=arm,seed=seed,**h)
  assert last['seconds']<3600,'Frozen per-run safety limit reached'
 assert last['epoch']==80 or last['epoch']-last['best']['epoch']>=12
 best=last['best'];ck={'state_dict':best['state_dict'],'in_channels':310,'classes':context['labels'],'word_codes':context['word_codes'],'epoch':best['epoch'],'arm':arm,'seed':seed,'context_sha256':contextsha,'feature_schema':'hand_evidence_v2_310','model_config':{**MODEL,'in_channels':310}}
 savept(U/f'{stem}_best.pt',ck)
 result={'status':'COMPLETE_DEVELOPMENT_ONLY','arm':arm,'seed':seed,'context_sha256':contextsha,'initial_sha256':init,'selected_epoch':best['epoch'],'completed_epochs':last['epoch'],'seconds':last['seconds'],'selected_ce':best['ce'],'checkpoint_sha256':sha(U/f'{stem}_best.pt'),'metrics':best['metrics'],'predictions':best['predictions'],'selected_scores_reinferred':False,'actual_batches_logged':True,'user_data_used':False,'production_changed':False}
 write(rp,result,new=True);event(stage='RUN_COMPLETE',arm=arm,seed=seed,selected_epoch=best['epoch'],completed_epochs=last['epoch']);return result
def main():
 c=read(SRC/'PREP_CONTEXT.json');ready=read(U/'READY.json');ctx=read(U/'TRAIN_CONTEXT.json')
 assert ready['status']=='PASS' and ready['training_allowed'] and not ready['blockers']
 checkpins(ctx['pins']);checkpins(c['pins'])
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 train,dev=loadrows(c);assert len(train)==899 and len(dev)==300
 lock=U/'TRAIN_RUNNING.json';write(lock,{'pid':os.getpid(),'at':now(),'context_sha256':sha(U/'TRAIN_CONTEXT.json')},new=True)
 try:
  for seed in SEEDS:
   for arm in ARMS:run_one(c,train,dev,arm,seed)
  checkpins(c['pins']);checkpins(ctx['pins']);lock.unlink();event(stage='TRAINING_COMPLETE_PENDING_INDEPENDENT_RECOUNT',runs=12)
 except BaseException:
  if not (U/'FIRST_TRAIN_ERROR.json').exists():write(U/'FIRST_TRAIN_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
if __name__=='__main__':main()
