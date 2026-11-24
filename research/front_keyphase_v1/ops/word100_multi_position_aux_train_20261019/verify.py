"""Saved logit arithmetic, actual trace, and frozen tensor audit. No inference."""
from common import *
import math,statistics,torch
from collections import Counter

def recount(scores,groups):
 assert len(scores)==len({p['clip_id'] for p in scores})==300
 assert Counter(p['signer_id'] for p in scores)=={11:100,12:100,13:100}
 assert Counter(p['label_index'] for p in scores)=={i:3 for i in range(100)}
 rr=[]
 for p in scores:
  if p['status']=='OK':
   z=p['logits'];assert len(z)==100 and all(math.isfinite(x) for x in z)
   rank=sorted(range(100),key=lambda i:(-z[i],i));yes=rank[0]==p['label_index'];top5=p['label_index'] in rank[:5]
   mx=max(z);ce=mx+math.log(sum(math.exp(x-mx) for x in z))-z[p['label_index']]
   assert p['prediction']==rank[0] and p['top5']==rank[:5]
  else:
   assert p['logits'] is None and p['prediction'] is None;yes=top5=False;ce=math.log(100)
  assert p['correct']==yes and p['top5_correct']==top5 and abs(p['cross_entropy']-ce)<1e-10
  rr.append({'cid':p['clip_id'],'signer':p['signer_id'],'correct':yes,'top5':top5,'ce':ce,'failure':p['status']!='OK','group':groups[p['clip_id']]})
 return {'correct':sum(x['correct'] for x in rr),'top5':sum(x['top5'] for x in rr),'mean_ce':sum(x['ce'] for x in rr)/300,'failures':sum(x['failure'] for x in rr),
  'signers':{str(s):sum(x['correct'] for x in rr if x['signer']==s) for s in [11,12,13]},
  'groups':{g:{'total':sum(x['group']==g for x in rr),'correct':sum(x['correct'] for x in rr if x['group']==g)} for g in sorted(set(groups.values()))}}

def trace(result,ctx):
 seed=result['seed'];stem=f'{ARM}_{seed}';hist=read(U/(stem+'_history.json'))
 last=torch.load(U/(stem+'_last.pt'),map_location='cpu',weights_only=True);best=torch.load(U/(stem+'_best.pt'),map_location='cpu',weights_only=True)
 original=torch.load(T/f'T3F_{seed}_best.pt',map_location='cpu',weights_only=True);fh=statehash(original['state_dict'])
 assert result['context_sha256']==best['context_sha256']==last['context_sha256']==sha(U/'TRAIN_CONTEXT.json')
 assert sha(U/(stem+'_best.pt'))==result['checkpoint_sha256']
 assert result['initial_sha256']==next(x['full_state_sha256'] for x in ctx['initial_states'] if x['seed']==seed)==last['initial_sha256']
 assert best['classes']==ctx['labels'] and best['word_codes']==ctx['word_codes']
 assert result['frozen_state_sha256']==best['frozen_state_sha256']==fh
 assert result['base_checkpoint_sha256']==best['base_checkpoint_sha256']==sha(T/f'T3F_{seed}_best.pt')
 assert statehash(best['state_dict'])==statehash(last['best']['state_dict'])
 assert result['lambda_aux']==best['lambda_aux']==0.1
 assert result['aux_predictions']==last['best']['aux_predictions'] and result['base_predictions']==last['best']['base_predictions']
 assert result['predictions']==last['best']['predictions'] and result['metrics']==last['best']['metrics']
 assert hist==last['history'] and len(hist)==last['epoch']==result['completed_epochs']
 opt=last['optimizer'];assert len(opt['param_groups'])==1 and len(opt['param_groups'][0]['params'])==5 and len(opt['state'])==5
 assert opt['param_groups'][0]['lr']==.001 and opt['param_groups'][0]['weight_decay']==.001
 assert sum(x['exp_avg'].numel() for x in opt['state'].values())==7200
 assert all(int(x['step'])==29*len(hist) for x in opt['state'].values())
 for weights in [best['state_dict'],last['model'],last['best']['state_dict']]:
  assert set(weights)=={'base.'+k for k in original['state_dict']}|set(ctx['trainable_parameter_names'])
  assert all(torch.equal(weights['base.'+k],value) for k,value in original['state_dict'].items())
  assert all(torch.isfinite(weights[k]).all() for k in ctx['trainable_parameter_names'])
  assert torch.count_nonzero(weights['residual.weight'])>0
 selected=None;count=0
 for epoch,h in enumerate(hist,1):
  assert h['epoch']==epoch and abs(h['selection_ce']-(h['sensor_ce']+h['masked_ce'])/2)<1e-12
  if selected is None or h['selection_ce']<selected['selection_ce']-1e-6:selected=h
  assert h['best_epoch']==selected['epoch'] and h['frozen_state_sha256']==fh and h['base_probe_max_error']==0
  if epoch<len(hist):assert epoch-selected['epoch']<12
  tp=U/'traces'/stem/f'{epoch:03}.json';assert sha(tp)==h['trace_sha256'];tr=read(tp)
  assert tr['seed']==seed and tr['epoch']==epoch and tr['arm']==ARM and len(tr['batches'])==29
  ids=[ctx['ordered_train_ids'][i] for i in np.random.default_rng(seed+epoch).permutation(899)]
  seen=[cid for b in tr['batches'] for cid in b['clips']];assert seen==ids and len(set(seen))==899
  for field,tracefield in [('train_ce','final_ce'),('train_aux_ce','aux_ce'),('train_total_loss','total_loss')]:assert abs(h[field]-sum(b[tracefield]*len(b['clips']) for b in tr['batches'])/899)<1e-10
  assert h['train_aux_correct_online']==sum(b['aux_correct_preupdate'] for b in tr['batches'])
  for step,b in enumerate(tr['batches'],1):
   assert b['lambda_aux']==0.1 and abs(b['total_loss']-(b['final_ce']+0.1*b['aux_ce']))<1e-6
   assert 0<=b['aux_correct_preupdate']<=len(b['clips']) and b['update_norm']>=0
   assert set(b['gradient_norm_by_tensor'])==set(ctx['trainable_parameter_names']) and all(math.isfinite(n) and n>=0 for n in b['gradient_norm_by_tensor'].values())
   assert b['step']==step and len(b['clips'])==(32 if step<29 else 3)
   assert b['optimizer_total_steps']==29*(epoch-1)+step
   assert b['frozen_state_sha256']==fh and b['frozen_modules_eval'] and math.isfinite(b['gradient_norm'])
   assert len(b['choices'])==len(b['clips'])
   for cid,ch in zip(b['clips'],b['choices']):
    a=int(hashlib.sha256(cid.encode()).hexdigest(),16)%2
    bpar=int(hashlib.sha256(('sensor_view:'+cid).encode()).hexdigest(),16)%2
    assert ch==['sensor' if ((epoch-1)//2+bpar)%2 else 'masked','x_context' if (epoch+a)%2 else 'x_whole']
   count+=1
 assert selected['epoch']==result['selected_epoch']==best['epoch']==last['best']['epoch']
 assert abs(selected['selection_ce']-result['selected_ce'])<1e-12
 assert len(hist)==80 or len(hist)-selected['epoch']>=12
 return {'epochs':len(hist),'selected_epoch':selected['epoch'],'actual_batches':count,'actual_source_exposures':len(hist)*899,'frozen_T3F_tensors_equal':True,'optimizer_new_tensors':5,'optimizer_new_parameters':7200,'all_batch_ids_verified':True,'all_steps_frozen':True,'base_probe_max_error':0}

def late(hist):
 h=[x for x in hist if 61<=x['epoch']<=80]
 assert [x['epoch'] for x in h]==list(range(61,61+len(h)))
 out={'window':[61,80],'epochs_available':len(h),'full_window':len(h)==20}
 for v in VIEWS:
  counts=[x[v+'_top1'] for x in h];ces=[x[v+'_ce'] for x in h];changes=[abs(b-a)/3 for a,b in zip(counts,counts[1:])]
  out[v]={'counts':counts,'mean_accuracy_percent':statistics.mean(counts)/3 if counts else None,'accuracy_range_percent':[min(counts)/3,max(counts)/3] if counts else None,'accuracy_sd_pp':statistics.pstdev(counts)/3 if counts else None,'adjacent_abs_mean_pp':statistics.mean(changes) if changes else None,'adjacent_abs_max_pp':max(changes) if changes else None,'mean_ce':statistics.mean(ces) if ces else None}
 return out

def gate(new,old):
 checks={'sensor_mean_higher':sum(x['correct'] for x in new['sensor'])>sum(x['correct'] for x in old['sensor']),
 'sensor_better_in_two_seeds':sum(a['correct']>b['correct'] for a,b in zip(new['sensor'],old['sensor']))>=2,
 'masked_top1_preserved':sum(x['correct'] for x in new['masked'])>=sum(x['correct'] for x in old['masked'])}
 for v in VIEWS:
  checks[v+'_top5_preserved']=sum(x['top5'] for x in new[v])>=sum(x['top5'] for x in old[v])
  checks[v+'_failures_preserved']=sum(x['failures'] for x in new[v])<=sum(x['failures'] for x in old[v])
  for s in ['11','12','13']:checks[v+'_signer'+s]=sum(x['signers'][s] for x in new[v])>=sum(x['signers'][s] for x in old[v])
  for g in new[v][0]['groups']:checks[v+'_group_'+g]=sum(x['groups'][g]['correct'] for x in new[v])>=sum(x['groups'][g]['correct'] for x in old[v])
 return {'pass':all(checks.values()),'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'statistical_proof':False,'automatic_adoption':False}

def main():
 assert not (U/'COMPARISON_INDEPENDENT.json').exists() and not (U/'TRAIN_RUNNING.json').exists()
 ctx=read(U/'TRAIN_CONTEXT.json');check(ctx['pins']);assert read(U/'EXIT.json')['exit_code']==0
 models=['C','T3F','FG16','FG16AUX01',ARM];metrics={m:{v:[] for v in VIEWS} for m in models};stored={};traces={};stability={m:[] for m in models};pins={}
 for model in models:
  for seed in SEEDS:
   folder=BASE if model=='C' else T if model=='T3F' else F if model=='FG16' else A if model=='FG16AUX01' else U;stem=f'training_C_{seed}' if model=='C' else f'{model}_{seed}'
   rp=folder/(stem+'_RESULT.json');hp=folder/(stem+'_history.json');r=read(rp);pins[str(rp)]=sha(rp);pins[str(hp)]=sha(hp);stored[model,seed]=r
   assert sha(folder/(stem+'_best.pt'))==r['checkpoint_sha256']
   for v in VIEWS:
    m=recount(r['predictions'][v],ctx['missingness_groups'][v]);old=r['metrics'][v]
    assert m['correct']==old['top1_correct'] and m['top5']==old['top5_correct'] and m['failures']==old['input_failures'] and abs(m['mean_ce']-old['mean_ce'])<1e-10
    assert all(n==old['by_signer'][s]['top1_correct'] for s,n in m['signers'].items())
    for g,gg in m['groups'].items():assert gg['correct']==old['by_observation_pattern'][g]['top1_correct'] and gg['total']==old['by_observation_pattern'][g]['total']
    metrics[model][v].append(m)
   assert abs(r['selected_ce']-sum(metrics[model][v][-1]['mean_ce'] for v in VIEWS)/2)<1e-10
   if model==ARM:traces[str(seed)]=trace(r,ctx)
   stability[model].append({'seed':seed,**late(read(hp))})
 summary={m:{v:{'top1_counts':[x['correct'] for x in mm],'top5_counts':[x['top5'] for x in mm],'accuracy_mean':sum(x['correct'] for x in mm)/9,'accuracy_range':[min(x['correct'] for x in mm)/3,max(x['correct'] for x in mm)/3],'top5_mean':sum(x['top5'] for x in mm)/9,'ce_mean':sum(x['mean_ce'] for x in mm)/3} for v,mm in views.items()} for m,views in metrics.items()}
 contrasts={};paired={}
 for baseline in ['C','T3F','FG16','FG16AUX01']:
  contrasts[baseline]={};paired[baseline]={}
  for v in VIEWS:
   events=[]
   for seed in SEEDS:
    before={x['clip_id']:x for x in stored[baseline,seed]['predictions'][v]};after={x['clip_id']:x for x in stored[ARM,seed]['predictions'][v]};assert before.keys()==after.keys()
    assert all(before[c]['label_index']==after[c]['label_index'] and before[c]['signer_id']==after[c]['signer_id'] for c in before)
    gains=[c for c in before if not before[c]['correct'] and after[c]['correct']];losses=[c for c in before if before[c]['correct'] and not after[c]['correct']]
    events.append({'seed':seed,'gains':gains,'losses':losses,'delta_pp':(len(gains)-len(losses))/3})
   paired[baseline][v]=events;contrasts[baseline][v]={'per_seed_pp':[x['delta_pp'] for x in events],'mean_pp':sum(x['delta_pp'] for x in events)/3}
 classes=[]
 for label,name in enumerate(ctx['labels']):
  row={'label':name,'index':label}
  for v in VIEWS:row[v]={m:sum(p['correct'] for seed in SEEDS for p in stored[m,seed]['predictions'][v] if p['label_index']==label) for m in models}
  row[ARM+'_sensor_errors']=[{'seed':seed,'clip_id':p['clip_id'],'rank':sorted(range(100),key=lambda i:(-p['logits'][i],i)).index(label)+1 if p['status']=='OK' else None,'predicted':ctx['labels'][p['prediction']] if p['status']=='OK' else None} for seed in SEEDS for p in stored[ARM,seed]['predictions']['sensor'] if p['label_index']==label and not p['correct']]
  classes.append(row)
 from verify_aux import audit_aux
 aux_verification=audit_aux(ctx,stored,recount)
 out={'auxiliary_verification':aux_verification,'status':'PASS_SAVED_SCORE_TRACE_AND_FROZEN_T3F_VERIFICATION','at':now(),'summary':summary,'metrics':metrics,'contrasts':contrasts,'paired':paired,'all_classes':classes,'late_per_seed':stability,
 'primary_review_gate_vs_T3F':gate(metrics[ARM],metrics['T3F']),'primary_review_gate_vs_FG16':gate(metrics[ARM],metrics['FG16']),'primary_review_gate_vs_AUX01':gate(metrics[ARM],metrics['FG16AUX01']),'secondary_gate_vs_C':gate(metrics[ARM],metrics['C']),
 'traces':traces,'selected_epochs':{m:[stored[m,s]['selected_epoch'] for s in SEEDS] for m in models},'completed_epochs':{m:[stored[m,s]['epochs_completed' if m=='C' else 'completed_epochs'] for s in SEEDS] for m in models},
 'pins':pins,'code_sha256':sha(__file__),'saved_predictions_verified':9000,'new_inference':0,'user_data_used':False,'production_changed':False,'automatic_adoption':False,
 'verification':'Codex separate arithmetic + actual batch trace + frozen tensor equality; no independent agent/expert.',
 'interpretation':'Repeated DEV development comparison, not new-user or signing-completeness accuracy. Versus AUX01 only multi-position readout including3200extra head parameters changes; semantic stage/order-only causality not established.'}
 write(U/'COMPARISON_INDEPENDENT.json',out,new=True)
 print(json.dumps({'summary':summary,'contrasts':contrasts,'gate':out['primary_review_gate_vs_T3F'],'traces':traces},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
