"""Balanced minibatch paired runs. Original C remains read-only."""
from common import *
import copy,random,time,traceback
import torch
sys.path.insert(0,str(ROOT/'research/sign_model_pilot_v1/src'))
sys.path.insert(0,str(BASE))
from pilot.model_masked_v1 import MaskedSignTCN
from training_core import SEEDS,MODEL,TRAIN,schedule,prediction,detailed_metrics
from balanced import balanced_batches
ARMS=('BAL31',);VIEWS=('masked','sensor')

def statehash(state):
 h=hashlib.sha256()
 for k,t in sorted(state.items()):
  h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def savept(p,obj):
 t=p.with_name(p.name+'.writing');torch.save(obj,t);os.replace(t,p)

def initialize(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 m=MaskedSignTCN(**MODEL);assert m.parameter_count()==37636
 assert statehash(m.state_dict())==read(BASE/f'training_C_{seed}_RESULT.json')['initial_state_sha256']
 return m

def batch(rows,arm,epoch=None,view=None):
 choices=[(view,'x_whole') if view else schedule(r['clip_id'],epoch,'C') for r in rows]
 xs=[r['arrays'][arm][v][t] for r,(v,t) in zip(rows,choices)];n=max(map(len,xs))
 x=np.zeros((len(xs),n,280),np.float32);pad=np.zeros((len(xs),n),np.float32)
 for i,a in enumerate(xs):x[i,:len(a)]=a;pad[i,:len(a)]=1
 return torch.from_numpy(x),torch.from_numpy(pad),torch.tensor([r['label_index'] for r in rows],dtype=torch.long),choices

def loadrows(context):
 original=read(BASE/'training_CONTEXT.json');assert original['ordered_rows']==context['rows']
 rows=[]
 for r in context['rows']:
  q=copy.deepcopy(r);q['arrays']={'BAL31':{}}
  for v in VIEWS:
   info=r['views'][v];path=info['channel_file']
   if path is None: q['arrays']['BAL31'][v]=None;assert r['role']!='train_candidate'
   else:
    assert sha(path)==info['sha256'];z=load(path)
    keys=('x_whole','x_context') if r['role']=='train_candidate' else ('x_whole',)
    q['arrays']['BAL31'][v]={k:z[k] for k in keys}
  rows.append(q)
 return [r for r in rows if r['role']=='train_candidate'],[r for r in rows if r['role']=='development_only']

def scores(model,dev,arm):
 before=statehash(model.state_dict());model.eval();out={}
 with torch.no_grad():
  for v in VIEWS:
   valid=[r for r in dev if r['arrays'][arm][v] is not None]
   result=[prediction(r,reason='frozen_input_unavailable') for r in dev if r['arrays'][arm][v] is None]
   for j in range(0,len(valid),32):
    rows=valid[j:j+32];x,p,_,_=batch(rows,arm,view=v);a=model(x,p);assert torch.isfinite(a).all()
    result.extend(prediction(r,z) for r,z in zip(rows,a.numpy()))
   out[v]=sorted(result,key=lambda r:r['clip_id'])
 assert before==statehash(model.state_dict());return out

def run_one(context,train,dev,arm,seed):
 stem=f'{arm}_{seed}';rp=U/f'{stem}_RESULT.json';lp=U/f'{stem}_last.pt';contextsha=sha(U/'TRAIN_CONTEXT.json')
 if rp.exists():
  x=read(rp);assert x['context_sha256']==contextsha;assert sha(U/f'{stem}_best.pt')==x['checkpoint_sha256'];return x
 m=initialize(seed);init=statehash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])
 trace_dir=U/'traces'/stem;trace_dir.mkdir(parents=True,exist_ok=True)
 if lp.exists():
  last=torch.load(lp,map_location='cpu',weights_only=True);assert last['context_sha256']==contextsha and last['initial_sha256']==init
  m.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);torch.set_rng_state(last['rng'])
 else:last={'epoch':0,'best':None,'history':[],'seconds':0.,'initial_sha256':init,'context_sha256':contextsha}
 start=time.monotonic();previous_seconds=last['seconds']
 for epoch in range(last['epoch']+1,81):
  if last['best'] and last['epoch']-last['best']['epoch']>=12:break
  m.train();order=np.random.default_rng(seed+epoch).permutation(899);tr=[];loss_sum=0.
  for step,indices in enumerate(balanced_batches(order,TRAIN['batch_size']),1):
   rr=[train[i] for i in indices];x,p,y,choices=batch(rr,arm,epoch=epoch)
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x,p),y);assert torch.isfinite(loss)
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step();loss_sum+=float(loss.item())*len(rr)
   tr.append({'step':step,'clips':[r['clip_id'] for r in rr],'choices':choices})
  pred=scores(m,dev,arm);metrics={v:detailed_metrics(pred[v],context['missingness_groups'][v]) for v in VIEWS};ce=(metrics['masked']['mean_ce']+metrics['sensor']['mean_ce'])/2
  if last['best'] is None or ce<last['best']['ce']-1e-6:last['best']={'epoch':epoch,'ce':ce,'state_dict':copy.deepcopy(m.state_dict()),'predictions':pred,'metrics':metrics}
  tp=trace_dir/f'{epoch:03}.json';assert not tp.exists(),'Orphan epoch trace: preserve and review before recovery'
  write(tp,{'arm':arm,'seed':seed,'epoch':epoch,'batches':tr},new=True)
  h={'epoch':epoch,'train_ce':loss_sum/899,'selection_ce':ce,'best_epoch':last['best']['epoch'],'sensor_top1':metrics['sensor']['top1_correct'],'masked_top1':metrics['masked']['top1_correct'],'sensor_ce':metrics['sensor']['mean_ce'],'masked_ce':metrics['masked']['mean_ce'],'trace_sha256':sha(tp)}
  last['history'].append(h);last.update(epoch=epoch,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state(),seconds=previous_seconds+time.monotonic()-start)
  savept(lp,last);write(U/f'{stem}_history.json',last['history']);event(stage='TRAINING',arm=arm,seed=seed,**h)
  assert last['seconds']<3600,'Frozen per-run safety limit reached'
 assert last['epoch']==80 or last['epoch']-last['best']['epoch']>=12
 best=last['best'];ck={'state_dict':best['state_dict'],'in_channels':280,'classes':context['labels'],'word_codes':context['word_codes'],'epoch':best['epoch'],'arm':arm,'seed':seed,'context_sha256':contextsha,'feature_schema':'original_C_280_balanced_batches','model_config':MODEL}
 savept(U/f'{stem}_best.pt',ck)
 result={'status':'COMPLETE_DEVELOPMENT_ONLY','arm':arm,'seed':seed,'context_sha256':contextsha,'initial_sha256':init,'selected_epoch':best['epoch'],'completed_epochs':last['epoch'],'seconds':last['seconds'],'selected_ce':best['ce'],'checkpoint_sha256':sha(U/f'{stem}_best.pt'),'metrics':best['metrics'],'predictions':best['predictions'],'selected_scores_reinferred':False,'actual_batches_logged':True,'user_data_used':False,'production_changed':False}
 write(rp,result,new=True);event(stage='RUN_COMPLETE',arm=arm,seed=seed,selected_epoch=best['epoch'],completed_epochs=last['epoch']);return result

def main():
 c=read(U/'PREP_CONTEXT.json');ready=read(U/'READY.json');assert ready['status']=='PASS' and ready['training_allowed'] and not ready['blockers']
 check(c['pins']);check(ready['pins']);assert not (U/'TRAIN_RUNNING.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 train,dev=loadrows(c);assert len(train)==899 and len(dev)==300
 original=read(BASE/'training_CONTEXT.json');ctx=U/'TRAIN_CONTEXT.json'
 if not ctx.exists():
  initial=[{'seed':s,'state_sha256':statehash(initialize(s).state_dict())} for s in SEEDS]
  for r in train:
   for a in ARMS:
    for v in VIEWS:
     for temporal,x in r['arrays'][a][v].items():assert x.shape[1]==280 and np.isfinite(x).all()
  pins={str(U/n):sha(U/n) for n in ['READY.json','PREP_CONTEXT.json','train.py','common.py','balanced.py','FROZEN_PROTOCOL.json']}
  for name in ['training_core.py'] :pins[str(BASE/name)]=sha(BASE/name)
  pins[str(ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py')]=sha(ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py')
  write(ctx,{'at':now(),'pins':pins,'arms':ARMS,'seeds':SEEDS,'model':MODEL,'training':TRAIN,'initial_states':initial,'labels':original['labels'],'word_codes':original['word_codes'],'missingness_groups':original['missingness_groups'],'ordered_train_ids':[r['clip_id'] for r in train],'runtime':{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__},'user_data_used':False},new=True)
 context=read(ctx);check(context['pins']);lock=U/'TRAIN_RUNNING.json';write(lock,{'pid':os.getpid(),'at':now(),'context_sha256':sha(ctx)},new=True)
 try:
  for seed in SEEDS:
   for arm in ARMS:run_one(context,train,dev,arm,seed)
  check(c['pins']);check(context['pins']);lock.unlink();event(stage='TRAINING_COMPLETE_PENDING_INDEPENDENT_RECOUNT',runs=3)
 except BaseException:
  if not (U/'FIRST_TRAIN_ERROR.json').exists():write(U/'FIRST_TRAIN_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise

if __name__=='__main__':main()
