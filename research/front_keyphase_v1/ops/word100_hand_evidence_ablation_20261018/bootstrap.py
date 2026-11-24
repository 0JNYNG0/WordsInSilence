"""Create isolated leave-one-group-out code. Never modifies previous experiments."""
import json, pathlib, shutil
U=pathlib.Path(__file__).resolve().parent
SRC=U.parent/'word100_hand_evidence_training_20261018'

def put(name,text):
 p=U/name
 assert not p.exists(),p
 p.write_text(text,encoding='utf-8')

common=(SRC/'common.py').read_text(encoding='utf-8')
common += "\nSRC=O/'word100_hand_evidence_training_20261018'\nDIAG=O/'word100_hand_evidence_attribution_20261018'\nGROUPS={'position':(0,6),'extent':(6,3),'intersection':(9,3),'proximity':(12,3)}\nARMS=tuple('minus_'+g for g in GROUPS)\n"
put('common.py',common)
train=(SRC/'train.py').read_text(encoding='utf-8')
train=train.replace("ARMS=('C0','C1');VIEWS=('masked','sensor')","VIEWS=('masked','sensor')")
train=train.replace(" if arm=='C0':x[:,:,280:]=0", " off,size=GROUPS[arm.removeprefix('minus_')]\n for side in (0,1):x[:,:,280+side*15+off:280+side*15+off+size]=0")
train=train.replace("read(U/'inputs_v2'", "read(SRC/'inputs_v2'")
train=train.replace("batch(rows,'C1',epoch=1)","batch(rows,ARMS[0],epoch=1)")
train=train[:train.index('def main():')]+'''def main():
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
'''
put('train.py',train)
v=(SRC/'verify_and_compare.py').read_text(encoding='utf-8')
v=v[:v.index('def main():')]
v=v.replace(" if arm=='C0':assert torch.count_nonzero(extra).item()==0\n else:assert torch.count_nonzero(extra).item()>0", " off,size=GROUPS[arm.removeprefix('minus_')]\n dropped=[side*15+k for side in (0,1) for k in range(off,off+size)]\n kept=[k for k in range(30) if k not in dropped]\n assert torch.count_nonzero(extra[:,dropped]).item()==0\n assert torch.count_nonzero(extra[:,kept]).item()>0")
v+='''def main():
 assert not (U/'TRAIN_RUNNING.json').exists()
 assert read(U/'STATUS.json')['stage']=='TRAINING_COMPLETE_PENDING_INDEPENDENT_RECOUNT'
 c=read(SRC/'PREP_CONTEXT.json');checkpins(c['pins']);checkpins(read(U/'TRAIN_CONTEXT.json')['pins'])
 results={};metrics={};traces=[];pins={}
 for arm in ('C','C0','C1')+ARMS:
  results[arm]=[];metrics[arm]=[]
  for seed in SEEDS:
   f=BASE/f'training_C_{seed}_RESULT.json' if arm=='C' else (SRC if arm in ('C0','C1') else U)/f'{arm}_{seed}_RESULT.json'
   r=read(f);pins[str(f)]=sha(f);results[arm].append(r);metrics[arm].append(recount(r,c))
   if arm in ARMS:traces.append(tracecheck(arm,seed,r,c))
 for i in range(3):
  assert len({results[a][i]['initial_sha256'] for a in ('C0','C1')+ARMS})==1
 sums={a:summarize(metrics[a]) for a in results};gates={};pairs={}
 for arm in ARMS:
  new=sums[arm];checks={}
  for b in ('C','C0'):
   old=sums[b]
   checks[f'sensor_Top1_mean_above_{b}']=new['sensor']['top1_correct']['sum']>old['sensor']['top1_correct']['sum']
   checks[f'masked_Top1_no_regression_vs_{b}']=new['masked']['top1_correct']['sum']>=old['masked']['top1_correct']['sum']
   for view in VIEWS:
    checks[f'{view}_Top5_no_regression_vs_{b}']=new[view]['top5_correct']['sum']>=old[view]['top5_correct']['sum']
    checks[f'{view}_input_failures_no_increase_vs_{b}']=all(x<=y for x,y in zip(new[view]['input_failures'],old[view]['input_failures']))
    for signer in ('11','12','13'):
     checks[f'{view}_signer{signer}_no_regression_vs_{b}']=new[view]['by_signer'][signer]['top1_sum']>=old[view]['by_signer'][signer]['top1_sum']
   for g in new['sensor']['by_observation_pattern']:
    checks[f'sensor_{g}_no_regression_vs_{b}']=new['sensor']['by_observation_pattern'][g]['top1_sum']>=old['sensor']['by_observation_pattern'][g]['top1_sum']
  checks['sensor_improves_fullC1_at_least2seeds']=sum(x>y for x,y in zip(new['sensor']['top1_correct']['counts'],sums['C1']['sensor']['top1_correct']['counts']))>=2
  gates[arm]={'pass_to_next_review':all(checks.values()),'checks':checks,'automatic_deployment':False}
  for b in ('C','C0','C1'):
   pairs[f'{b}_to_{arm}']={str(seed):changes(results[b][i],results[arm][i],c) for i,seed in enumerate(SEEDS)}
 report={'status':'COMPLETE_VERIFIED_DEVELOPMENT_COMPARISON','at':now(),'summary':sums,'paired_changes':pairs,'gates':gates,'trace':traces,'pins':pins,'score_rows_recounted':12600,'new_runs':12,'reference_runs_reused':9,'reinference':0,'user_mentor_heldout_used':False,'production_changed':False,'limitations':['Same DEV300 used repeatedly for epoch selection and exploratory four-arm comparison, not final blind test.','Three seeds are repeat fits of same split, not new people.','Public sensor crop is a webcam proxy.','Removal estimates group contribution conditional on remaining groups and fixed protocol; interactions/optimization remain.']}
 write(U/'COMPARISON.json',report,new=True)
 write(U/'SCORES_TRACE_INDEPENDENT.json',{'status':'PASS','at':now(),'score_rows':12600,'runs_trace_checked':12,'batches':sum(t['batches'] for t in traces),'logged_video_exposures':sum(t['actual_logged_video_exposures'] for t in traces),'model_inference':0,'operator':'Codex local independent arithmetic code, not separate reviewer','comparison_sha256':sha(U/'COMPARISON.json')},new=True)
 print(json.dumps({'status':'PASS','top1':{a:{v:q[v]['top1_correct'] for v in VIEWS} for a,q in sums.items()},'gates':{a:g['pass_to_next_review'] for a,g in gates.items()}},ensure_ascii=False))
if __name__=='__main__':main()
'''
put('verify.py',v)
print('Created isolated common/train/verify; no training executed.')
