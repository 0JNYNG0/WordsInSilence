from pathlib import Path
U=Path(__file__).resolve().parent;T=U.parent/'word100_temporal_phase_training_20261019'
s=(T/'verify_results.py').read_text(encoding='utf8');s=s[:s.index('\ndef main():')]
target=" return {'epochs':len(hist),'batches':batches,'actual_source_exposures':899*len(hist),'selected_epoch':best['epoch']}"
assert s.count(target)==1
s=s.replace(target,""" original=torch.load(BASE/f\"training_C_{result['seed']}_best.pt\",map_location='cpu',weights_only=True)
 frozen_hash=statehash(original['state_dict'])
 assert result['frozen_state_sha256']==checkpoint['frozen_state_sha256']==frozen_hash
 assert result['base_checkpoint_sha256']==checkpoint['base_checkpoint_sha256']==sha(BASE/f\"training_C_{result['seed']}_best.pt\")
 for state in [checkpoint['state_dict'],last['model'],last['best']['state_dict']]:
  assert set(state)==set(original['state_dict'])|{'phase_weight'}
  assert statehash({k:v for k,v in state.items() if k!='phase_weight'})==frozen_hash
  for k,v in original['state_dict'].items():assert torch.equal(state[k],v)
  assert torch.isfinite(state['phase_weight']).all() and torch.count_nonzero(state['phase_weight'])>0
 assert len(last['optimizer']['state'])==1 and len(last['optimizer']['param_groups'])==1 and len(last['optimizer']['param_groups'][0]['params'])==1
 for h in hist:
  assert h['frozen_state_sha256']==frozen_hash and h['base_probe_max_error']==0
  tr=read(U/'traces'/stem/f\"{h['epoch']:03}.json\")
  assert all(b['frozen_state_sha256']==frozen_hash and b['frozen_modules_eval'] for b in tr['batches'])
 return {'epochs':len(hist),'batches':batches,'actual_source_exposures':899*len(hist),'selected_epoch':best['epoch'],'all_batch_frozen_state_matches_C':True,'all_epoch_base_probe_error':0,'frozen_state_sha256':frozen_hash,'one_parameter_optimizer_verified':True,'selected_phase_weight_norm':float(checkpoint['state_dict']['phase_weight'].norm())}
""")
main='''
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
'''
p=U/'verify_results.py';assert not p.exists();compile(s+main,str(p),'exec');p.write_text(s+main,encoding='utf8')
print('Verifier prepared')
