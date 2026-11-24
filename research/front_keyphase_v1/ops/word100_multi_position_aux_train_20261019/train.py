"""Approved three-seed order auxiliary experiment; old C/T3F are read-only."""
from common import *
import copy,random,traceback,argparse
import torch
from collections import Counter
from objective import objective, LAMBDA_AUX
sys.path.insert(0,str(BASE));sys.path.insert(0,str(P))
from training_core import MODEL,TRAIN,schedule,padded_arrays,prediction,detailed_metrics,validate_channels
from multi_position_aux import MultiPositionAux
NEW_NAMES=['memory.weight_ih_l0','memory.weight_hh_l0','memory.bias_ih_l0','memory.bias_hh_l0','residual.weight']

def initialize(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 result=read(T/f'T3F_{seed}_RESULT.json');path=T/f'T3F_{seed}_best.pt';assert sha(path)==result['checkpoint_sha256']
 ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['model_config']==MODEL
 m=MultiPositionAux(ck['model_config'],ck['state_dict']);m.assert_frozen()
 assert [n for n,p in m.named_parameters() if p.requires_grad]==NEW_NAMES
 assert sum(p.numel() for p in m.parameters() if p.requires_grad)==7200
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
 from preparation import prepare_experiment
 prepare_experiment(initialize,loadrows,batch,MODEL,TRAIN,NEW_NAMES)

def scores(m,dev):
 before=statehash(m.state_dict());m.eval();out={};auxout={};baseout={}
 with torch.no_grad():
  for view in VIEWS:
   valid=[r for r in dev if r['arrays'][view] is not None]
   absent=[prediction(r,reason='frozen_input_unavailable') for r in dev if r['arrays'][view] is None]
   result=list(absent);ar=list(absent);br=list(absent)
   for j in range(0,len(valid),32):
    rows=valid[j:j+32];x,p,_,_=batch(rows,view=view)
    b,a,_,_,_=m.components(x,p);z=b+a
    assert torch.isfinite(z).all() and torch.isfinite(a).all()
    result.extend(prediction(r,q) for r,q in zip(rows,z.numpy()))
    ar.extend(prediction(r,q) for r,q in zip(rows,a.numpy()))
    br.extend(prediction(r,q) for r,q in zip(rows,b.numpy()))
   out[view]=sorted(result,key=lambda r:r['clip_id']);auxout[view]=sorted(ar,key=lambda r:r['clip_id']);baseout[view]=sorted(br,key=lambda r:r['clip_id'])
 assert statehash(m.state_dict())==before
 return out,auxout,baseout

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
 assert len(params)==5 and sum(p.numel() for p in params)==7200
 probe_x,probe_p,_,_=batch(train[:8],epoch=1)
 with torch.no_grad():probe=m.components(probe_x,probe_p)[0].clone()
 last={'epoch':0,'best':None,'history':[],'context_sha256':cs,'initial_sha256':initial};start=time.monotonic()
 event(stage='TRAIN_START',seed=seed,arm=ARM,initial_sha256=initial,frozen_state_sha256=frozen)
 for epoch in range(1,81):
  if last['best'] and last['epoch']-last['best']['epoch']>=12:break
  m.train();order=np.random.default_rng(seed+epoch).permutation(899);trace=[];loss_sum=0.;aux_sum=0.;total_sum=0.;aux_correct=0
  for step,j in enumerate(range(0,899,32),1):
   rr=[train[i] for i in order[j:j+32]];x,p,y,choices=batch(rr,epoch=epoch)
   opt.zero_grad(set_to_none=True);b,a,_,_,_=m.components(x,p)
   loss,final_ce,aux_ce=objective(b,a,y)
   loss.backward();assert all(q.grad is None for q in m.base.parameters())
   assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in params)
   grads={n:float(q.grad.norm()) for n,q in m.named_parameters() if q.requires_grad}
   prior=[q.detach().clone() for q in params]
   norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);opt.step()
   update=float(torch.sqrt(sum(((q.detach()-v)**2).sum() for q,v in zip(params,prior))))
   m.assert_frozen();assert statehash(m.base.state_dict())==frozen
   fc=float(final_ce.detach());ac=float(aux_ce.detach());lc=float(loss.detach())
   loss_sum+=fc*len(rr);aux_sum+=ac*len(rr);total_sum+=lc*len(rr)
   hits=int((a.detach().argmax(1)==y).sum());aux_correct+=hits
   trace.append({'step':step,'clips':[r['clip_id'] for r in rr],'choices':choices,'frozen_state_sha256':frozen,'frozen_modules_eval':True,'gradient_norm':float(norm),'optimizer_total_steps':(epoch-1)*29+step,
    'final_ce':fc,'aux_ce':ac,'total_loss':lc,'lambda_aux':LAMBDA_AUX,'aux_correct_preupdate':hits,'gradient_norm_by_tensor':grads,'update_norm':update,
    'mean_abs_aux_logits':float(a.detach().abs().mean()),'max_abs_aux_logits':float(a.detach().abs().max())})
  with torch.no_grad():assert torch.equal(m.components(probe_x,probe_p)[0],probe)
  pred,auxpred,basepred=scores(m,dev);metrics={v:detailed_metrics(pred[v],context['missingness_groups'][v]) for v in VIEWS};ce=sum(metrics[v]['mean_ce'] for v in VIEWS)/2
  auxmetrics={v:detailed_metrics(auxpred[v],context['missingness_groups'][v]) for v in VIEWS}
  if last['best'] is None or ce<last['best']['ce']-1e-6:
   last['best']={'epoch':epoch,'ce':ce,'state_dict':copy.deepcopy(m.state_dict()),'predictions':pred,'metrics':metrics,'aux_predictions':auxpred,'base_predictions':basepred,'aux_metrics':auxmetrics}
  tp=U/'traces'/stem/f'{epoch:03}.json';write(tp,{'seed':seed,'arm':ARM,'epoch':epoch,'batches':trace},new=True)
  h={'epoch':epoch,'train_ce':loss_sum/899,'train_aux_ce':aux_sum/899,'train_total_loss':total_sum/899,'train_aux_correct_online':aux_correct,'aux_sensor_top1':auxmetrics['sensor']['top1_correct'],'aux_masked_top1':auxmetrics['masked']['top1_correct'],'aux_sensor_ce':auxmetrics['sensor']['mean_ce'],'aux_masked_ce':auxmetrics['masked']['mean_ce'],'selection_ce':ce,'best_epoch':last['best']['epoch'],'sensor_top1':metrics['sensor']['top1_correct'],'masked_top1':metrics['masked']['top1_correct'],'sensor_ce':metrics['sensor']['mean_ce'],'masked_ce':metrics['masked']['mean_ce'],'trace_sha256':sha(tp),'frozen_state_sha256':frozen,'base_probe_max_error':0.0}
  last['history'].append(h);last.update(epoch=epoch,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state(),seconds=time.monotonic()-start)
  savept(lp,last);write(U/(stem+'_history.json'),last['history']);event(stage='TRAINING',arm=ARM,seed=seed,seconds=round(last['seconds'],1),**h)
  assert last['seconds']<3600,'Frozen per-run wall safety limit'
 assert last['epoch']==80 or last['epoch']-last['best']['epoch']>=12
 best=last['best'];ck={'state_dict':best['state_dict'],'classes':context['labels'],'word_codes':context['word_codes'],'epoch':best['epoch'],'arm':ARM,'seed':seed,'context_sha256':cs,'base_checkpoint_sha256':sha(T/f'T3F_{seed}_best.pt'),'frozen_state_sha256':frozen,'trainable_parameters':7200,'parameters':54436,'model_config':MODEL,'model_type':m.model_type,'model_version':m.model_version,'feature_schema':'original_C_280_frozen_T3F_GRU16_elapsed3','training_objective':'final_CE_plus_0.1_aux_CE','lambda_aux':LAMBDA_AUX}
 savept(U/(stem+'_best.pt'),ck)
 result={'status':'COMPLETE_DEVELOPMENT_ONLY','arm':ARM,'seed':seed,'context_sha256':cs,'initial_sha256':initial,'selected_epoch':best['epoch'],'completed_epochs':last['epoch'],'seconds':last['seconds'],'selected_ce':best['ce'],'checkpoint_sha256':sha(U/(stem+'_best.pt')),'metrics':best['metrics'],'predictions':best['predictions'],'aux_predictions':best['aux_predictions'],'base_predictions':best['base_predictions'],'aux_metrics':best['aux_metrics'],'lambda_aux':LAMBDA_AUX,'selected_scores_reinferred':False,'actual_batches_logged':True,'frozen_state_sha256':frozen,'base_checkpoint_sha256':ck['base_checkpoint_sha256'],'trainable_parameters':7200,'all_steps_frozen_checked':True,'base_probe_max_error':0,'user_data_used':False,'production_changed':False}
 write(rp,result,new=True);event(stage='RUN_COMPLETE',seed=seed,selected_epoch=best['epoch'],completed_epochs=last['epoch'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);action=parser.parse_args().action
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 if action=='prepare':prepare();return
 context=read(U/'TRAIN_CONTEXT.json');ready=read(U/'READY.json');assert ready['status']=='PASS' and ready['training_allowed'] and ready['context_sha256']==sha(U/'TRAIN_CONTEXT.json')
 check(context['pins']);assert not (U/'TRAIN_RUNNING.json').exists()
 train,dev=loadrows(context);lock=U/'TRAIN_RUNNING.json';write(lock,{'pid':os.getpid(),'at':now(),'context_sha256':sha(U/'TRAIN_CONTEXT.json')},new=True)
 write(U/'CURRENT_EXECUTION.json',{'pid':os.getpid(),'status':'RUNNING','running':True,'at':now(),'command':'train.py run','context_sha256':sha(U/'TRAIN_CONTEXT.json')},new=True)
 try:
  for seed in SEEDS:run_one(context,train,dev,seed)
  check(context['pins']);current=read(U/'CURRENT_EXECUTION.json');current.update(status='COMPLETE',running=False,exit_code=0,completed_at=now());write(U/'CURRENT_EXECUTION.json',current);lock.unlink();write(U/'EXIT.json',{'exit_code':0,'at':now(),'runs':3},new=True);event(stage='THREE_RUNS_COMPLETE_PENDING_VERIFICATION')
 except BaseException:
  if not (U/'FIRST_TRAIN_ERROR.json').exists():write(U/'FIRST_TRAIN_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
if __name__=='__main__':main()
