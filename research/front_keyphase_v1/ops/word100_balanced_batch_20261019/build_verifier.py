"""Reuse separate saved-score arithmetic; adapt only this unit's contract."""
from pathlib import Path
U=Path(__file__).resolve().parent
s=(U.parent/'word100_assignment_scale_factorial_20261019/verify_results.py').read_text(encoding='utf-8')
s=s[:s.index('\ndef main():')]
s=s.replace("len(b['clips'])==(32 if step<29 else 3)","len(b['clips'])==31 and len(b['choices'])==31")
s=s.replace("assert hist==last['history']", "assert checkpoint['classes']==context['labels'] and checkpoint['word_codes']==context['word_codes']\n assert all(g['lr']==.001 and g['weight_decay']==.001 for g in last['optimizer']['param_groups'])\n assert all(int(st['step'])==29*last['epoch'] for st in last['optimizer']['state'].values())\n assert hist==last['history']")
s+='''
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
 metrics={a:{v:[] for v in VIEWS} for a in ['C','BAL31']};stored={};traces={};pins={};stability={a:[] for a in metrics}
 for arm in metrics:
  for seed in SEEDS:
   path=BASE/f'training_C_{seed}_RESULT.json' if arm=='C' else U/f'{arm}_{seed}_RESULT.json'
   histpath=BASE/f'training_C_{seed}_history.json' if arm=='C' else U/f'{arm}_{seed}_history.json'
   r=read(path);pins[str(path)]=sha(path);pins[str(histpath)]=sha(histpath);stored[arm,seed]=r
   for v in VIEWS:
    m=recount(r['predictions'][v],context['missingness_groups'][v]);old=r['metrics'][v]
    assert m['correct']==old['top1_correct'] and m['top5']==old['top5_correct'] and m['failures']==old['input_failures'] and abs(m['mean_ce']-old['mean_ce'])<1e-10
    for signer,count in m['signers'].items():assert count==old['by_signer'][signer]['top1_correct']
    for group,g in m['groups'].items():assert g['correct']==old['by_observation_pattern'][group]['top1_correct'] and g['total']==old['by_observation_pattern'][group]['total']
    metrics[arm][v].append(m)
   assert abs(r['selected_ce']-(metrics[arm]['sensor'][-1]['mean_ce']+metrics[arm]['masked'][-1]['mean_ce'])/2)<1e-10
   if arm=='BAL31':traces[f'{arm}_{seed}']=trace_check(r,context)
   stability[arm].append({'seed':seed,**late(read(histpath))})
 summary={a:{v:{'top1_counts':[x['correct'] for x in mm],'top5_counts':[x['top5'] for x in mm],'accuracy_mean':sum(x['correct'] for x in mm)/9,'accuracy_range':[min(x['correct'] for x in mm)/3,max(x['correct'] for x in mm)/3], 'ce_mean':sum(x['mean_ce'] for x in mm)/3} for v,mm in vv.items()} for a,vv in metrics.items()}
 paired={};contrasts={}
 for v in VIEWS:
  events=[];deltas=[];ce=[]
  for i,seed in enumerate(SEEDS):
   before={x['clip_id']:x for x in stored['C',seed]['predictions'][v]};after={x['clip_id']:x for x in stored['BAL31',seed]['predictions'][v]};assert before.keys()==after.keys()
   assert all(before[c]['label_index']==after[c]['label_index'] and before[c]['signer_id']==after[c]['signer_id'] for c in before)
   gains=[cid for cid in before if not before[cid]['correct'] and after[cid]['correct']];losses=[cid for cid in before if before[cid]['correct'] and not after[cid]['correct']]
   deltas.append((len(gains)-len(losses))/3);ce.append(metrics['BAL31'][v][i]['mean_ce']-metrics['C'][v][i]['mean_ce'])
   events.append({'seed':seed,'gains':gains,'losses':losses,'delta_pp':deltas[-1]})
  paired[v]=events;contrasts[v]={'per_seed_pp':deltas,'mean_pp':sum(deltas)/3,'per_seed_ce':ce,'mean_ce_delta':sum(ce)/3}
 late_summary={}
 keys=['mean_accuracy_percent','accuracy_sd_pp','adjacent_abs_mean_pp','adjacent_abs_max_pp','mean_ce','adjacent_ce_abs_mean','adjacent_ce_abs_max']
 for arm,rr in stability.items():
  late_summary[arm]={v:{key:sum(r[v][key] for r in rr)/3 if all(r[v][key] is not None for r in rr) else None for key in keys} for v in VIEWS}
  late_summary[arm]['full_window_all_seeds']=all(r['full_window'] for r in rr)
 result={'status':'PASS_SAVED_SCORES_AND_ACTUAL_TRACES','at':now(),'summary':summary,'metrics':metrics,'paired':paired,'contrasts':contrasts,'late_per_seed':stability,'late_summary':late_summary,
  'review_gate':gate(metrics['BAL31'],metrics['C']),'traces':traces,'selected_epochs':{a:[stored[a,s]['selected_epoch'] for s in SEEDS] for a in metrics},'pins':pins,'code_sha256':sha(__file__),'new_inference':0,'user_data_used':False,'production_changed':False,'automatic_adoption':False,
  'verification':'Separate saved-logit arithmetic and actual batch trace/checkpoint audit by same Codex operator; not an independent agent/expert. Late summaries independently recomputed from pinned saved epoch counts/CE; epoch logits unavailable.',
  'interpretation':'Batch partition intervention including gradient composition/padding/dropout/BN; not BN-only causal isolation. Reused public DEV, not blind or webcam performance.'}
 write(U/'COMPARISON_INDEPENDENT.json',result,new=True);print(json.dumps({'summary':summary,'contrasts':contrasts,'late_summary':late_summary,'gate':result['review_gate']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
'''
p=U/'verify_results.py';assert not p.exists();p.write_text(s,encoding='utf-8')
print('Verifier written; no scores read or inference performed.')
