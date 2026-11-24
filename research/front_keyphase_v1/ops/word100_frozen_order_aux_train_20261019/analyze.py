"""Interpret saved runs only. No inference, training, or tuning."""
from common import *
import statistics,torch,math
r=read(U/'COMPARISON_INDEPENDENT.json');assert r['status'].startswith('PASS')
assert not (U/'ANALYSIS.json').exists()
rows=[]
for seed in SEEDS:
 old=read(T/f'T3F_{seed}_RESULT.json');new=read(U/f'FG16_{seed}_RESULT.json');hist=read(U/f'FG16_{seed}_history.json')
 grads=[b['gradient_norm'] for h in hist for b in read(U/f'traces/FG16_{seed}/{h["epoch"]:03}.json')['batches']]
 row={'seed':seed,'epochs':len(hist),'selected_epoch':new['selected_epoch'],'seconds':new['seconds'],
      'train_ce_first':hist[0]['train_ce'],'train_ce_last':hist[-1]['train_ce'],'gradient_norm_median':statistics.median(grads),'gradient_norm_max':max(grads),
      'early_stop_reason':'12 epochs with no strict mean DEV CE improvement','views':{}}
 for v in VIEWS:
  before={p['clip_id']:p for p in old['predictions'][v]};after={p['clip_id']:p for p in new['predictions'][v]}
  delta=[];rank_changes=[];predchanges=[]
  for cid in before:
   x=before[cid];y=after[cid];assert x['status']==y['status']
   if x['status']!='OK':continue
   delta.extend(abs(a-b) for a,b in zip(x['logits'],y['logits']))
   if x['prediction']!=y['prediction']:predchanges.append(cid)
   rankx=sorted(range(100),key=lambda i:(-x['logits'][i],i)).index(x['label_index'])+1
   ranky=sorted(range(100),key=lambda i:(-y['logits'][i],i)).index(y['label_index'])+1
   if rankx!=ranky:rank_changes.append({'clip_id':cid,'old_rank':rankx,'new_rank':ranky})
  counts=[h[v+'_top1'] for h in hist]
  row['views'][v]={'mean_absolute_saved_logit_change':statistics.mean(delta),'max_absolute_saved_logit_change':max(delta),'top1_prediction_changes':predchanges,'target_rank_changes':rank_changes,'observed_epoch_count_range':[min(counts),max(counts)],'selected_ce_delta':new['metrics'][v]['mean_ce']-old['metrics'][v]['mean_ce']}
 ck=torch.load(U/f'FG16_{seed}_best.pt',weights_only=True,map_location='cpu')
 row['residual_head_norm']=float(ck['state_dict']['residual.weight'].norm())
 rows.append(row)
errors=[c for c in r['all_classes'] if c['FG16_sensor_errors']]
focus=[c for c in r['all_classes'] if c['label'] in ('믿다','사육장')]
assert all(not p['gains'] and not p['losses'] for vv in r['paired']['T3F'].values() for p in vv)
out={'status':'COMPLETE_NO_EXTRA_TOP1_GAIN','source_comparison_sha256':sha(U/'COMPARISON_INDEPENDENT.json'),'rows':rows,
 'primary_result':'FG16 matches T3F top1/top5 in all3 seeds and both views. No corrected/newly-wrong clips at selected checkpoints.',
 'facts':['Early training CE about1.6e-5 to3.1e-5: the existing model already predicts TRAIN labels with very low loss.','Nonzero head updates and gradient logs show the new part did train.','Only small saved-logit/CE changes, no new correct DEV classifications.','All runs stopped at13/14/38 before prespecified late window61-80; cannot claim improved late stability.'],
 'inference':'Existing correct confident logits can leave little corrective signal for a residual-only classification objective. This is a plausible explanation supported by tiny losses/gradients, not an isolated causal proof. It does not show GRU or temporal order is useless.',
 'limitations':['DEV near ceiling and repeatedly reused; not new-user measurement.','No explicit stage/order correctness labels.','No missing hand reconstruction.','No capacity-matched order ablation in this experiment.'],
 'sensor_error_classes':errors,'believe_rearing_public_dev':focus,
 'unique_sensor_error_clips':len({p['clip_id'] for c in errors for p in c['FG16_sensor_errors']}),
 'sensor_error_events_across_seeds':sum(len(c['FG16_sensor_errors']) for c in errors),
 'training_seconds_sum':sum(x['seconds'] for x in rows),'training_runs':3,'new_inference':0,'user_recordings_read':0}
write(U/'ANALYSIS.json',out,new=True)
plan={'status':'KEEP_C_T3F_FG16_NOT_PROMOTED_WAIT_REVIEW','automatic_further_training':False,
 'reason':'New FG16 did not improve T3F accuracy. C +0.56pp is inherited T3F improvement, not a new GRU gain.',
 'next_single_proposal':'공개 자료만 사용해 새 순서 보조 부분에도 독립적인 단어 학습 신호를 주는 손실 설계를 먼저 검토한다. 현재는 기존 T3-F와 합친 점수만 정답에 맞춰 새 부분이 배울 신호가 매우 작다. 기존점수는고정하고 보조출력 자체도 정답을 구분하도록 하는 방안을 검토하며, 계수·실행조건은 결과를 함께 확인한 뒤 학습 전에 고정한다.',
 'mechanism_to_test':'Keep frozen base, current input and memory architecture. Consider auxiliary-only classification loss in addition to final residual classification; not yet implemented or trained. It may strengthen learning but does not guarantee learning required sign stages.',
 'before_next_training':['Explain and freeze objective/weight, initialization and comparison before scores.','Validate existing model and BN frozen, same data/sampling and no private data.','Use same all-seed/reporting gates and retain adverse results.'],
 'not_justified':['Increasing GRU size or epochs based on these scores without a new approved plan.','Calling unchanged top1 a success because CE declined slightly.','Claiming remembered stages or correct/incorrect signing from class labels alone.','Reusing personal tests to fit the candidate.'],
 'longer_term':'필수동작을완수했는지판정하려면전문가가확인한공개정상·생략·순서오류자료와별도검증이필요. 임의순서변형을모두잘못된수어라고라벨링하지않음.',
 'protected':'C/T3F/web/DTW unchanged, user/mentor footage not read or used, ID14-16 reserved.'}
write(U/'NEXT_PLAN.json',plan,new=True)
print(json.dumps({'seed_summaries':rows,'focus':focus,'unique_error_clips':out['unique_sensor_error_clips'],'error_events':out['sensor_error_events_across_seeds'],'training_seconds_sum':out['training_seconds_sum']},ensure_ascii=False))
