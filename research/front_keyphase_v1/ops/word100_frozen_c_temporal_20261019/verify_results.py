"""Saved-logit recount and actual batch/selection/checkpoint audit. No inference."""
from common import *
import math,argparse,torch
from collections import Counter
SEEDS=[20261013,20261014,20261015];VIEWS=['masked','sensor']

def statehash(state):
 h=hashlib.sha256()
 for k,t in sorted(state.items()):
  h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.cpu().numpy().tobytes())
 return h.hexdigest()

def recount(scores,groups):
 assert len(scores)==300 and len({r['clip_id'] for r in scores})==300
 assert Counter(r['signer_id'] for r in scores)=={11:100,12:100,13:100}
 assert Counter(r['label_index'] for r in scores)=={k:3 for k in range(100)}
 out=[]
 for r in scores:
  if r['status']=='OK':
   logits=r['logits'];assert len(logits)==100 and all(math.isfinite(x) for x in logits)
   ranking=sorted(range(100),key=lambda i:(-logits[i],i));correct=ranking[0]==r['label_index'];top5=r['label_index'] in ranking[:5]
   maximum=max(logits);ce=maximum+math.log(sum(math.exp(x-maximum) for x in logits))-logits[r['label_index']]
   assert r['prediction']==ranking[0] and r['top5']==ranking[:5]
  else:
   assert r['logits'] is None and r['prediction'] is None;correct=top5=False;ce=math.log(100)
  assert r['correct']==correct and r['top5_correct']==top5 and abs(r['cross_entropy']-ce)<1e-10
  out.append({'clip_id':r['clip_id'],'signer':r['signer_id'],'correct':correct,'top5':top5,'ce':ce,'failure':r['status']!='OK','group':groups[r['clip_id']]})
 return {'correct':sum(x['correct'] for x in out),'top5':sum(x['top5'] for x in out),'mean_ce':sum(x['ce'] for x in out)/300,'failures':sum(x['failure'] for x in out),'signers':{str(s):sum(x['correct'] for x in out if x['signer']==s) for s in [11,12,13]},'groups':{g:{'total':sum(x['group']==g for x in out),'correct':sum(x['correct'] for x in out if x['group']==g)} for g in sorted(set(groups.values()))}}

def trace_check(result,context):
 stem=f"{result['arm']}_{result['seed']}";hist=read(U/(stem+'_history.json'));last=torch.load(U/(stem+'_last.pt'),map_location='cpu',weights_only=True)
 checkpoint=torch.load(U/(stem+'_best.pt'),map_location='cpu',weights_only=True);assert sha(U/(stem+'_best.pt'))==result['checkpoint_sha256']
 assert result['context_sha256']==sha(U/'TRAIN_CONTEXT.json')==checkpoint['context_sha256']==last['context_sha256']
 assert result['initial_sha256']==read(U/'READY.json')['initial_states'][str(result['seed'])]['full_state_sha256']
 assert statehash(checkpoint['state_dict'])==statehash(last['best']['state_dict']) and result['predictions']==last['best']['predictions']
 assert checkpoint['classes']==context['labels'] and checkpoint['word_codes']==context['word_codes']
 assert all(g['lr']==.001 and g['weight_decay']==.001 for g in last['optimizer']['param_groups'])
 assert all(int(st['step'])==29*last['epoch'] for st in last['optimizer']['state'].values())
 assert hist==last['history'] and len(hist)==result['completed_epochs']==last['epoch']
 best=None;clips=context['ordered_train_ids'];batches=0
 for epoch,h in enumerate(hist,1):
  assert h['epoch']==epoch
  if best is None or h['selection_ce']<best['selection_ce']-1e-6:best=h
  assert h['best_epoch']==best['epoch']
  if epoch<len(hist):assert epoch-best['epoch']<12
  path=U/'traces'/stem/f'{epoch:03}.json';assert sha(path)==h['trace_sha256'];tr=read(path)
  assert tr['arm']==result['arm'] and tr['seed']==result['seed'] and tr['epoch']==epoch
  expected=[clips[i] for i in np.random.default_rng(result['seed']+epoch).permutation(899)]
  actual=[cid for b in tr['batches'] for cid in b['clips']];assert actual==expected and len(set(actual))==899
  assert len(tr['batches'])==29
  for step,b in enumerate(tr['batches'],1):
   expected_size=32 if step<29 else 3
   assert b['step']==step and len(b['clips'])==expected_size and len(b['choices'])==expected_size
   for cid,ch in zip(b['clips'],b['choices']):
    parity=int(hashlib.sha256(cid.encode()).hexdigest(),16)%2
    sp=int(hashlib.sha256(('sensor_view:'+cid).encode()).hexdigest(),16)%2
    assert ch==['sensor' if ((epoch-1)//2+sp)%2 else 'masked','x_context' if (epoch+parity)%2 else 'x_whole']
   batches+=1
 assert best['epoch']==result['selected_epoch']==checkpoint['epoch'] and abs(best['selection_ce']-result['selected_ce'])<1e-12
 assert len(hist)==80 or len(hist)-best['epoch']>=12
 original=torch.load(BASE/f"training_C_{result['seed']}_best.pt",map_location='cpu',weights_only=True)
 frozen_hash=statehash(original['state_dict'])
 assert result['frozen_state_sha256']==checkpoint['frozen_state_sha256']==frozen_hash
 assert result['base_checkpoint_sha256']==checkpoint['base_checkpoint_sha256']==sha(BASE/f"training_C_{result['seed']}_best.pt")
 for state in [checkpoint['state_dict'],last['model'],last['best']['state_dict']]:
  assert set(state)==set(original['state_dict'])|{'phase_weight'}
  assert statehash({k:v for k,v in state.items() if k!='phase_weight'})==frozen_hash
  for k,v in original['state_dict'].items():assert torch.equal(state[k],v)
  assert torch.isfinite(state['phase_weight']).all() and torch.count_nonzero(state['phase_weight'])>0
 assert len(last['optimizer']['state'])==1 and len(last['optimizer']['param_groups'])==1 and len(last['optimizer']['param_groups'][0]['params'])==1
 for h in hist:
  assert h['frozen_state_sha256']==frozen_hash and h['base_probe_max_error']==0
  tr=read(U/'traces'/stem/f"{h['epoch']:03}.json")
  assert all(b['frozen_state_sha256']==frozen_hash and b['frozen_modules_eval'] for b in tr['batches'])
 return {'epochs':len(hist),'batches':batches,'actual_source_exposures':899*len(hist),'selected_epoch':best['epoch'],'all_batch_frozen_state_matches_C':True,'all_epoch_base_probe_error':0,'frozen_state_sha256':frozen_hash,'one_parameter_optimizer_verified':True,'selected_phase_weight_norm':float(checkpoint['state_dict']['phase_weight'].norm())}


def gate(new,old):
 nv=new['sensor'];ov=old['sensor'];checks={
  'sensor_mean_higher':sum(x['correct'] for x in nv)>sum(x['correct'] for x in ov),
  'sensor_better_in_two_seeds':sum(x['correct']>y['correct'] for x,y in zip(nv,ov))>=2,
  'masked_mean_preserved':sum(x['correct'] for x in new['masked'])>=sum(x['correct'] for x in old['masked']),
  'sensor_top5_preserved':sum(x['top5'] for x in nv)>=sum(x['top5'] for x in ov)}
 for v in VIEWS:
  checks[v+'_failures_preserved']=sum(x['failures'] for x in new[v])<=sum(x['failures'] for x in old[v])
  for s in ['11','12','13']:checks[v+'_signer_'+s]=sum(x['signers'][s] for x in new[v])>=sum(x['signers'][s] for x in old[v])
  for g in new[v][0]['groups']:checks[v+'_group_'+g]=sum(x['groups'][g]['correct'] for x in new[v])>=sum(x['groups'][g]['correct'] for x in old[v])
 return {'pass':all(checks.values()),'checks':checks,'statistical_proof':False,'automatic_adoption':False}

def late(history):
 import statistics
 h=[x for x in history if 61<=x['epoch']<=80]
 assert [x['epoch'] for x in h]==list(range(61,61+len(h)))
 out={'window':[61,80],'epochs_available':len(h),'full_window':len(h)==20}
 for view in VIEWS:
  counts=[x[view+'_top1'] for x in h];ces=[x[view+'_ce'] for x in h]
  changes=[abs(b-a) for a,b in zip(counts,counts[1:])]
  cechanges=[abs(b-a) for a,b in zip(ces,ces[1:])]
  out[view]={'counts':counts,'mean_accuracy_percent':statistics.mean(counts)/3 if counts else None,
   'accuracy_range_percent':[min(counts)/3,max(counts)/3] if counts else None,
   'accuracy_sd_pp':statistics.pstdev(counts)/3 if counts else None,
   'adjacent_abs_mean_pp':statistics.mean(changes)/3 if changes else None,
   'adjacent_abs_max_pp':max(changes)/3 if changes else None,
   'mean_ce':statistics.mean(ces) if ces else None,
   'adjacent_ce_abs_mean':statistics.mean(cechanges) if cechanges else None,
   'adjacent_ce_abs_max':max(cechanges) if cechanges else None}
  if counts:
   assert abs(out[view]['mean_accuracy_percent']-float(np.mean(counts))/3)<1e-12
   assert abs(out[view]['accuracy_sd_pp']-float(np.std(counts))/3)<1e-12
 return out

def main():
 assert not (U/'TRAIN_RUNNING.json').exists()
 context=read(U/'TRAIN_CONTEXT.json');check(context['pins']);check(read(U/'READY.json')['pins']);check(read(U/'PREP_CONTEXT.json')['pins'])
 T=O/'word100_temporal_phase_training_20261019'
 metrics={a:{v:[] for v in VIEWS} for a in ['C','T3','T3F']};stored={};traces={};pins={};stability={a:[] for a in metrics}
 for arm in metrics:
  for seed in SEEDS:
   folder=BASE if arm=='C' else T if arm=='T3' else U;stem=f'training_C_{seed}' if arm=='C' else f'{arm}_{seed}'
   path=folder/(stem+'_RESULT.json');hp=folder/(stem+'_history.json');r=read(path);pins[str(path)]=sha(path);pins[str(hp)]=sha(hp);stored[arm,seed]=r
   assert sha(folder/(stem+'_best.pt'))==r['checkpoint_sha256']
   for v in VIEWS:
    m=recount(r['predictions'][v],context['missingness_groups'][v]);old=r['metrics'][v]
    assert m['correct']==old['top1_correct'] and m['top5']==old['top5_correct'] and m['failures']==old['input_failures'] and abs(m['mean_ce']-old['mean_ce'])<1e-10
    for signer,count in m['signers'].items():assert count==old['by_signer'][signer]['top1_correct']
    for group,g in m['groups'].items():assert g['correct']==old['by_observation_pattern'][group]['top1_correct'] and g['total']==old['by_observation_pattern'][group]['total']
    metrics[arm][v].append(m)
   assert abs(r['selected_ce']-(metrics[arm]['sensor'][-1]['mean_ce']+metrics[arm]['masked'][-1]['mean_ce'])/2)<1e-10
   if arm=='T3F':traces[f'{arm}_{seed}']=trace_check(r,context)
   stability[arm].append({'seed':seed,**late(read(hp))})
 summary={a:{v:{'top1_counts':[x['correct'] for x in mm],'top5_counts':[x['top5'] for x in mm],'accuracy_mean':sum(x['correct'] for x in mm)/9,'accuracy_range':[min(x['correct'] for x in mm)/3,max(x['correct'] for x in mm)/3],'ce_mean':sum(x['mean_ce'] for x in mm)/3} for v,mm in vv.items()} for a,vv in metrics.items()}
 paired={};contrasts={};allclass=[]
 for baseline in ['C','T3']:
  paired[baseline]={};contrasts[baseline]={}
  for v in VIEWS:
   events=[];deltas=[]
   for i,seed in enumerate(SEEDS):
    before={x['clip_id']:x for x in stored[baseline,seed]['predictions'][v]};after={x['clip_id']:x for x in stored['T3F',seed]['predictions'][v]};assert before.keys()==after.keys()
    assert all(before[c]['label_index']==after[c]['label_index'] and before[c]['signer_id']==after[c]['signer_id'] for c in before)
    gains=[cid for cid in before if not before[cid]['correct'] and after[cid]['correct']];losses=[cid for cid in before if before[cid]['correct'] and not after[cid]['correct']]
    delta=(len(gains)-len(losses))/3;deltas.append(delta);events.append({'seed':seed,'gains':gains,'losses':losses,'delta_pp':delta})
   paired[baseline][v]=events;contrasts[baseline][v]={'per_seed_pp':deltas,'mean_pp':sum(deltas)/3}
 for label,name in enumerate(context['labels']):
  row={'label':name,'index':label}
  for v in VIEWS:row[v]={arm:sum(p['correct'] for seed in SEEDS for p in stored[arm,seed]['predictions'][v] if p['label_index']==label) for arm in metrics}
  allclass.append(row)
 result={'status':'PASS_SAVED_SCORES_ACTUAL_TRACE_AND_FROZEN_C','at':now(),'summary':summary,'metrics':metrics,'paired':paired,'contrasts':contrasts,'all_classes':allclass,'late_per_seed':stability,'review_gate':gate(metrics['T3F'],metrics['C']),'traces':traces,'selected_epochs':{a:[stored[a,s]['selected_epoch'] for s in SEEDS] for a in metrics},'completed_epochs':{a:[len(read((BASE if a=='C' else T if a=='T3' else U)/(('training_C' if a=='C' else a)+f'_{s}_history.json'))) for s in SEEDS] for a in metrics},'pins':pins,'code_sha256':sha(__file__),'saved_predictions_verified':5400,'new_inference':0,'user_data_used':False,'production_changed':False,'automatic_adoption':False,'verification':'same Codex, separate saved-logit arithmetic + actual trace + tensor-level frozen C equality. Not independent person/agent.','interpretation':'Already trained C frozen, auxiliary head only; not fresh-training/order-only causal control. Reused DEV not new-user result.'}
 write(U/'COMPARISON_INDEPENDENT.json',result,new=True)
 print(json.dumps({'summary':summary,'contrasts':contrasts,'gate':result['review_gate'],'traces':traces},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
