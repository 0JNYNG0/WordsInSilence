"""Approved three-seed order auxiliary experiment; old C/T3F are read-only."""
from common import *
import copy,random,traceback,argparse
import torch
from collections import Counter
sys.path.insert(0,str(BASE));sys.path.insert(0,str(P))
from training_core import MODEL,TRAIN,schedule,padded_arrays,prediction,detailed_metrics,validate_channels
from frozen_order_aux import FrozenOrderAux
NEW_NAMES=['memory.weight_ih_l0','memory.weight_hh_l0','memory.bias_ih_l0','memory.bias_hh_l0','residual.weight']

def initialize(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 result=read(T/f'T3F_{seed}_RESULT.json');path=T/f'T3F_{seed}_best.pt';assert sha(path)==result['checkpoint_sha256']
 ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['model_config']==MODEL
 m=FrozenOrderAux(ck['model_config'],ck['state_dict']);m.assert_frozen()
 assert [n for n,p in m.named_parameters() if p.requires_grad]==NEW_NAMES
 assert sum(p.numel() for p in m.parameters() if p.requires_grad)==4000
 assert preflight_hash(m)==read(P/'RESULT.json')['initial_state_hashes'][str(seed)]
 assert statehash(m.base.state_dict())==statehash(ck['state_dict'])
 return m

def loadrows(context):
 rows=[]
 for row in context['rows']:
  q=copy.deepcopy(row);q['arrays']={}
  for view in VIEWS:
   info=row['views'][view];path=info['channel_file']
   if path is None:
    assert row['role']=='development_only';q['arrays'][view]=None;continue
   assert sha(path)==info['sha256']
   with np.load(path,allow_pickle=False) as z:
    keys=('x_whole','x_context') if row['role']=='train_candidate' else ('x_whole',)
    q['arrays'][view]={k:z[k] for k in keys}
   for a in q['arrays'][view].values():validate_channels(a)
  rows.append(q)
 train=[r for r in rows if r['role']=='train_candidate'];dev=[r for r in rows if r['role']=='development_only']
 assert len(train)==899 and len(dev)==300 and len(rows)==1199
 assert {r['signer_id'] for r in train}==set(range(1,10))
 assert Counter(r['signer_id'] for r in dev)=={11:100,12:100,13:100}
 assert not ({r['clip_id'] for r in train}&{r['clip_id'] for r in dev})
 return train,dev

def batch(rows,epoch=None,view=None):
 choices=[(view,'x_whole') if view else schedule(r['clip_id'],epoch,'C') for r in rows]
 xs=[r['arrays'][v][t] for r,(v,t) in zip(rows,choices)];x,p=padded_arrays(xs)
 return torch.from_numpy(x),torch.from_numpy(p),torch.tensor([r['label_index'] for r in rows],dtype=torch.long),choices

def prepare():
 assert not (U/'TRAIN_CONTEXT.json').exists()
 pre=read(P/'FINAL_AUDIT.json');assert pre['status']=='PASS'
 for name,h in pre['files'].items():assert sha(P/name)==h,name
 assert read(P/'INDEPENDENT.json')['status']=='PASS'
 check(read(P/'SCOPE.json')['pins'])
 authorization={'at':now(),'user_request':'그 계획대로 진행해줘.','approved_plan':str(P/'NEXT_PLAN.json'),'scope':'F-G16 auxiliary4000 only, three seeds, sameTRAIN899/DEV300, C/T3F saved comparison and independent verification','overrides_old_preflight_wait':True,'no_user_data':True,'no_production_change':True,'implementer':'Codex local','verification':'Separate saved-score/tensor/trace verifier by Codex, no independent agent'}
 write(U/'AUTHORIZATION.json',authorization,new=True)
 original=read(BASE/'training_CONTEXT.json');plan=read(P/'NEXT_PLAN.json')
 assert list(SEEDS)==plan['seeds']
 protocol={'at':now(),'training':TRAIN,'model':MODEL,'seeds':SEEDS,'new_parameters':4000,'frozen_parameters':47236,'selection':'mean DEV masked/sensor CE, strict improvement >1e-6; early stop after12epochs without best, max80',
 'input':'Original C280channels, same schedule, masks/dt, no dropping missing real frames; only trailing padding excluded',
 'baseline':'Each matching T3F seed checkpoint fully frozen. Reuse C and T3F saved scores without re-inference.',
 'gate_baseline':'T3F primary, C secondary','gate':'Sensor mean top1 higher and >=2/3seeds higher; masked top1, both views top5, signer and missingness mean counts no regression; input failures no increase.',
 'late_window':[61,80],'partial_late_window':'Explicit unavailable if early stop before61; no claim of stability from absent epochs',
 'counterfactual_inference':'Not run in this unit; normal classification comparison first. Not a completion or invalid-sign evaluation.',
 'forbidden':['user/mentor footage','ID14-18 raw data','model selection by personal footage','changes to detector/ROI/normalization','retune based on comparison','production/web/DTW replacement','Git','external transfer/download'],
 'publication':'all3seeds, mean/range and %p, gains/regressions and development limitations; no best-seed-only report'}
 write(U/'FROZEN_PROTOCOL.json',protocol,new=True)
 rows=original['ordered_rows'];train,dev=loadrows({'rows':rows});pins={}
 for p in [U/'AUTHORIZATION.json',U/'FROZEN_PROTOCOL.json',U/'train.py',U/'common.py',BASE/'training_core.py',BASE/'training_CONTEXT.json',P/'FINAL_AUDIT.json',P/'NEXT_PLAN.json',P/'RESULT.json',P/'INDEPENDENT.json',P/'frozen_order_aux.py',T/'temporal.py',T/'frozen_temporal.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py',ROOT/'research/front_keyphase_v1/src/wis_lab/service/word100_candidate.py',ROOT/'research/front_keyphase_v1/src/wis_lab/service/word100_t3f.py']:
  pins[str(p)]=sha(p)
 for seed in SEEDS:
  for folder,stem in [(BASE,f'training_C_{seed}'),(T,f'T3F_{seed}')]:
   for suffix in ('_best.pt','_RESULT.json','_history.json'):
    p=folder/(stem+suffix);pins[str(p)]=sha(p)
 for row in rows:
  for view in VIEWS:
   info=row['views'][view]
   if info['channel_file'] is not None:
    p=Path(info['channel_file']);assert sha(p)==info['sha256'];pins[str(p)]=sha(p)
 initial=[]
 for seed in SEEDS:
  m=initialize(seed);x,p,_,_=batch(train[:8],epoch=1)
  m.train();m.assert_frozen()
  with torch.no_grad():base,delta,_,_,_=m.components(x,p);assert torch.count_nonzero(delta)==0
  initial.append({'seed':seed,'full_state_sha256':statehash(m.state_dict()),'preflight_state_sha256':preflight_hash(m),'frozen_state_sha256':statehash(m.base.state_dict()),'initial_delta_zero':True})
 context={'at':now(),'arms':[ARM],'seeds':SEEDS,'model':MODEL,'training':TRAIN,'pins':pins,'rows':rows,'initial_states':initial,'labels':original['labels'],'word_codes':original['word_codes'],'missingness_groups':original['missingness_groups'],'ordered_train_ids':[r['clip_id'] for r in train],'trainable_parameter_names':NEW_NAMES,'runtime':{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__},'user_recordings_read':0}
 write(U/'TRAIN_CONTEXT.json',context,new=True)
 write(U/'READY.json',{'status':'PASS','training_allowed':True,'authorization':str(U/'AUTHORIZATION.json'),'context_sha256':sha(U/'TRAIN_CONTEXT.json'),'prior_preflight_reused':True,'input_train':899,'input_dev':300,'initial_states':initial,'blockers':[]},new=True)
 event(stage='READY_APPROVED_FOR_THREE_AUXILIARY_RUNS',train=899,dev=300)

def scores(m,dev):
 before=statehash(m.state_dict());m.eval();out={}
 with torch.no_grad():
  for view in VIEWS:
   valid=[r for r in dev if r['arrays'][view] is not None]
   result=[prediction(r,reason='frozen_input_unavailable') for r in dev if r['arrays'][view] is None]
   for j in range(0,len(valid),32):
    rows=valid[j:j+32];x,p,_,_=batch(rows,view=view);logits=m(x,p);assert torch.isfinite(logits).all()
    result.extend(prediction(r,z) for r,z in zip(rows,logits.numpy()))
   out[view]=sorted(result,key=lambda r:r['clip_id'])
 assert statehash(m.state_dict())==before
 return out

def savept(p,x):
 tmp=p.with_name(p.name+'.writing');torch.save(x,tmp);os.replace(tmp,p)

def run_one(context,train,dev,seed):
 stem=f'{ARM}_{seed}';rp=U/(stem+'_RESULT.json');lp=U/(stem+'_last.pt');cs=sha(U/'TRAIN_CONTEXT.json')
 if rp.exists():
  r=read(rp);assert r['context_sha256']==cs and sha(U/(stem+'_best.pt'))==r['checkpoint_sha256'];return
 assert not lp.exists(),'Partial run must be explicitly verified before recovery; do not silently restart'
 m=initialize(seed);initial=statehash(m.state_dict());frozen=statehash(m.base.state_dict())
 assert initial==next(s['full_state_sha256'] for s in context['initial_states'] if s['seed']==seed)
 params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])
 assert len(params)==5 and sum(p.numel() for p in params)==4000
 probe_x,probe_p,_,_=batch(train[:8],epoch=1)
 with torch.no_grad():probe=m.components(probe_x,probe_p)[0].clone()
 last={'epoch':0,'best':None,'history':[],'context_sha256':cs,'initial_sha256':initial};start=time.monotonic()
 event(stage='TRAIN_START',seed=seed,arm=ARM,initial_sha256=initial,frozen_state_sha256=frozen)
 for epoch in range(1,81):
  if last['best'] and last['epoch']-last['best']['epoch']>=12:break
  m.train();order=np.random.default_rng(seed+epoch).permutation(899);trace=[];loss_sum=0.
  for step,j in enumerate(range(0,899,32),1):
   rr=[train[i] for i in order[j:j+32]];x,p,y,choices=batch(rr,epoch=epoch)
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x,p),y);assert torch.isfinite(loss)
   loss.backward();assert all(p.grad is None for p in m.base.parameters())
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
   norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);opt.step()
   m.assert_frozen();assert statehash(m.base.state_dict())==frozen
   loss_sum+=float(loss.detach())*len(rr)
   trace.append({'step':step,'clips':[r['clip_id'] for r in rr],'choices':choices,'frozen_state_sha256':frozen,'frozen_modules_eval':True,'gradient_norm':float(norm),'optimizer_total_steps':(epoch-1)*29+step})
  with torch.no_grad():assert torch.equal(m.components(probe_x,probe_p)[0],probe)
  pred=scores(m,dev);metrics={v:detailed_metrics(pred[v],context['missingness_groups'][v]) for v in VIEWS};ce=sum(metrics[v]['mean_ce'] for v in VIEWS)/2
  if last['best'] is None or ce<last['best']['ce']-1e-6:
   last['best']={'epoch':epoch,'ce':ce,'state_dict':copy.deepcopy(m.state_dict()),'predictions':pred,'metrics':metrics}
  tp=U/'traces'/stem/f'{epoch:03}.json';write(tp,{'seed':seed,'arm':ARM,'epoch':epoch,'batches':trace},new=True)
  h={'epoch':epoch,'train_ce':loss_sum/899,'selection_ce':ce,'best_epoch':last['best']['epoch'],'sensor_top1':metrics['sensor']['top1_correct'],'masked_top1':metrics['masked']['top1_correct'],'sensor_ce':metrics['sensor']['mean_ce'],'masked_ce':metrics['masked']['mean_ce'],'trace_sha256':sha(tp),'frozen_state_sha256':frozen,'base_probe_max_error':0.0}
  last['history'].append(h);last.update(epoch=epoch,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state(),seconds=time.monotonic()-start)
  savept(lp,last);write(U/(stem+'_history.json'),last['history']);event(stage='TRAINING',arm=ARM,seed=seed,seconds=round(last['seconds'],1),**h)
  assert last['seconds']<3600,'Frozen per-run wall safety limit'
 assert last['epoch']==80 or last['epoch']-last['best']['epoch']>=12
 best=last['best'];ck={'state_dict':best['state_dict'],'classes':context['labels'],'word_codes':context['word_codes'],'epoch':best['epoch'],'arm':ARM,'seed':seed,'context_sha256':cs,'base_checkpoint_sha256':sha(T/f'T3F_{seed}_best.pt'),'frozen_state_sha256':frozen,'trainable_parameters':4000,'parameters':51236,'model_config':MODEL,'model_type':m.model_type,'model_version':m.model_version,'feature_schema':'original_C_280_frozen_T3F_GRU16'}
 savept(U/(stem+'_best.pt'),ck)
 result={'status':'COMPLETE_DEVELOPMENT_ONLY','arm':ARM,'seed':seed,'context_sha256':cs,'initial_sha256':initial,'selected_epoch':best['epoch'],'completed_epochs':last['epoch'],'seconds':last['seconds'],'selected_ce':best['ce'],'checkpoint_sha256':sha(U/(stem+'_best.pt')),'metrics':best['metrics'],'predictions':best['predictions'],'selected_scores_reinferred':False,'actual_batches_logged':True,'frozen_state_sha256':frozen,'base_checkpoint_sha256':ck['base_checkpoint_sha256'],'trainable_parameters':4000,'all_steps_frozen_checked':True,'base_probe_max_error':0,'user_data_used':False,'production_changed':False}
 write(rp,result,new=True);event(stage='RUN_COMPLETE',seed=seed,selected_epoch=best['epoch'],completed_epochs=last['epoch'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);action=parser.parse_args().action
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 if action=='prepare':prepare();return
 context=read(U/'TRAIN_CONTEXT.json');ready=read(U/'READY.json');assert ready['status']=='PASS' and ready['training_allowed'] and ready['context_sha256']==sha(U/'TRAIN_CONTEXT.json')
 check(context['pins']);assert not (U/'TRAIN_RUNNING.json').exists()
 train,dev=loadrows(context);lock=U/'TRAIN_RUNNING.json';write(lock,{'pid':os.getpid(),'at':now(),'context_sha256':sha(U/'TRAIN_CONTEXT.json')},new=True)
 try:
  for seed in SEEDS:run_one(context,train,dev,seed)
  check(context['pins']);lock.unlink();write(U/'EXIT.json',{'exit_code':0,'at':now(),'runs':3},new=True);event(stage='THREE_RUNS_COMPLETE_PENDING_VERIFICATION')
 except BaseException:
  if not (U/'FIRST_TRAIN_ERROR.json').exists():write(U/'FIRST_TRAIN_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
if __name__=='__main__':main()
