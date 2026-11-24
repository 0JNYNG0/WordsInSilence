"""Run the fixed comparison once; all scores are new-task annotation-scope scores."""
from common import *
import time,traceback
from collections import defaultdict

def learn_point(samples,plan):
 fit=[s for s in samples if s['record']['role']=='fit' and s['mask'] and s['usable']]
 X=torch.from_numpy(np.stack([s['pose'] for s in fit]));Z=torch.from_numpy(np.stack([s['rgb'] for s in fit]))
 y=torch.tensor([s['y'] for s in fit],dtype=torch.float32);j=torch.tensor([POINT.index(s['record']['target']) for s in fit])
 weights=grouped_weights(fit,POINT);torch.manual_seed(plan['seed']);model=PointNet()
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=plan['weight_decay'])
 history=[];start=time.perf_counter()
 for epoch in range(plan['epochs']):
  model.train();logits=model(X,Z)[torch.arange(len(fit)),j]
  loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).sum()
  assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
  if epoch%20==0 or epoch==plan['epochs']-1:history.append(dict(epoch=epoch+1,loss=float(loss.detach())))
 model.eval();return model,dict(seconds=time.perf_counter()-start,history=history,train_records=len(fit))

def learn_temporal(samples,condition,plan):
 fit=[s for s in samples if s['record']['role']=='fit' and s['mask'] and s['variant'] in plan['train_variants']]
 # Missing coordinates remain explicit inputs. A known supervised event does
 # not cease to be in the denominator when the sensor is difficult.
 x,l=batch(fit);y=torch.tensor([s['y'] for s in fit],dtype=torch.float32)
 j=torch.tensor([TEMP.index(s['record']['target']) for s in fit]);weights=grouped_weights(fit,TEMP)
 torch.manual_seed(plan['seed']);model=TemporalNet(condition)
 optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay']);history=[];start=time.perf_counter()
 for epoch in range(plan['epochs']):
  model.train();logits=model(x,l)[torch.arange(len(fit)),j]
  loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).sum()
  assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
  if epoch%20==0 or epoch==plan['epochs']-1:
   history.append(dict(epoch=epoch+1,loss=float(loss.detach())))
   print(condition,epoch+1,float(loss.detach()),flush=True)
 model.eval();return model,dict(seconds=time.perf_counter()-start,history=history,train_records=len(fit))

def status(p,usable):
 if not usable:return 'INSUFFICIENT_OBSERVATION'
 return 'COMPONENT_OBSERVED' if p>=.8 else 'CONTRARY_COMPONENT_CANDIDATE' if p<=.2 else 'UNCERTAIN'

def predict_point(model,samples):
 result=[]
 with torch.inference_mode():
  for s in samples:
   t=time.perf_counter();pp=model(torch.from_numpy(s['pose'][None]),torch.from_numpy(s['rgb'][None])).sigmoid().numpy()[0]
   elapsed=time.perf_counter()-t;j=POINT.index(s['record']['target']);p=float(pp[j]);r=s['record']
   result.append(dict(**metadata(s),probability=p,class_prediction=int(p>=.5),status=status(p,s['usable']),
    component_head_seconds=elapsed,diagnostic_only=r['target']==POINT[0],whole_sign_correctness=None))
 return result

def predict_temporal(model,samples):
 result=[]
 with torch.inference_mode():
  for s in samples:
   x,l=batch([s]);t=time.perf_counter();pp=model(x,l).sigmoid().numpy()[0];elapsed=time.perf_counter()-t
   p=float(pp[TEMP.index(s['record']['target'])])
   result.append(dict(**metadata(s),probability=p,class_prediction=int(p>=.5),status=status(p,s['usable']),
    component_head_seconds=elapsed,whole_sign_correctness=None))
 return result

def metrics(rows):
 out={}
 groups=defaultdict(list)
 for r in rows:
  if r['record']['role']!='development':continue
  key=(r['record']['target'],r.get('variant','native'));groups[key].append(r)
 for (target,variant),ss in groups.items():
  known=[s for s in ss if s['mask']];pos=[s for s in known if s['y']==1];neg=[s for s in known if s['y']==0]
  def rate(rr,correct):
   clips=defaultdict(list)
   for s in rr:clips[s['record']['clip_id']].append(float(correct(s)))
   return float(np.mean([np.mean(v) for v in clips.values()])) if clips else None
  out[target+'::'+variant]=dict(records=len(ss),known=len(known),unknown=len(ss)-len(known),
   clips=len({s['record']['clip_id'] for s in ss}),positive=len(pos),negative=len(neg),
   TP=sum(s['class_prediction']==1 for s in pos),TN=sum(s['class_prediction']==0 for s in neg),
   recall=rate(pos,lambda s:s['class_prediction']==1),specificity=rate(neg,lambda s:s['class_prediction']==0),
   usable_positive_recall=rate(pos,lambda s:s['usable'] and s['class_prediction']==1),
   usable_negative_specificity=rate(neg,lambda s:s['usable'] and s['class_prediction']==0),
   confirmed_positive_recall=rate(pos,lambda s:s['status']=='COMPONENT_OBSERVED'),
   confirmed_negative_specificity=rate(neg,lambda s:s['status']=='CONTRARY_COMPONENT_CANDIDATE'),
   confident_wrong=sum(s['status']=='CONTRARY_COMPONENT_CANDIDATE' for s in pos)+sum(s['status']=='COMPONENT_OBSERVED' for s in neg),
   held=sum(s['status']=='INSUFFICIENT_OBSERVATION' for s in ss),uncertain=sum(s['status']=='UNCERTAIN' for s in ss),
   unsupported_not_dropped=True)
 return out

def gates(point,pooled,ordered):
 checks={}
 po=point[POINT[1]+'::native'];checks['oblique_recall']=po['usable_positive_recall'] is not None and po['usable_positive_recall']>=.8
 checks['oblique_specificity']=po['usable_negative_specificity'] is not None and po['usable_negative_specificity']>=.8
 for t in TEMP:
  a=ordered[t+'::native'];b=pooled[t+'::native']
  checks[t+':native_recall']=a['usable_positive_recall'] is not None and a['usable_positive_recall']>=.8
  checks[t+':recall_not_below_pool']=a['usable_positive_recall']>=b['usable_positive_recall']
  a=ordered[t+'::still'];checks[t+':still_specificity']=a['usable_negative_specificity'] is not None and a['usable_negative_specificity']>=.8
  g=ordered[t+'::image_gap'];checks[t+':gap_all_held']=g['held']==g['records']
 for t in TEMP[:2]:checks[t+':reverse_specificity']=ordered[t+'::reverse']['usable_negative_specificity']>=.8
 checks['alternation_reverse_recall']=ordered[TEMP[2]+'::reverse']['usable_positive_recall']>=.8
 checks['traversal_natural_specificity']=ordered[TEMP[1]+'::native']['usable_negative_specificity']>=.8
 checks['direction_reverse_better_than_pool']=sum(ordered[t+'::reverse']['TN'] for t in TEMP[:2])>sum(pooled[t+'::reverse']['TN'] for t in TEMP[:2])
 # Raw .5 discrimination alone must never turn an all-uncertain classifier
 # into a successful feedback component. At least half of each supported
 # natural class must also cross the frozen .8/.2 reporting thresholds.
 checks['oblique_confirmed_recall']=po['confirmed_positive_recall'] is not None and po['confirmed_positive_recall']>=.5
 checks['oblique_confirmed_specificity']=po['confirmed_negative_specificity'] is not None and po['confirmed_negative_specificity']>=.5
 checks['oblique_confident_wrong_zero']=po['confident_wrong']==0
 for t in TEMP:
  a=ordered[t+'::native'];checks[t+':confirmed_recall']=a['confirmed_positive_recall'] is not None and a['confirmed_positive_recall']>=.5
  checks[t+':natural_confident_wrong_zero']=a['confident_wrong']==0
  checks[t+':confirmed_still_specificity']=ordered[t+'::still']['confirmed_negative_specificity']>=.5
 for t in TEMP[:2]:checks[t+':confirmed_reverse_specificity']=ordered[t+'::reverse']['confirmed_negative_specificity']>=.5
 checks['alternation_confirmed_reverse_recall']=ordered[TEMP[2]+'::reverse']['confirmed_positive_recall']>=.5
 checks['traversal_confirmed_natural_specificity']=ordered[TEMP[1]+'::native']['confirmed_negative_specificity']>=.5
 return dict(passed=all(checks.values()),checks=checks,
  scope='Annotation-scope geometry order diagnostic and natural point readouts; not actual RGB edits, short-event error detection or end-to-end adoption',
  thumb_diagnostic_only=True,whole_goal_complete=False)

def main():
 torch.set_num_threads(4);plan=read(U/'FROZEN.json')
 for p,h in plan['pins'].items():assert sha(p)==h,p
 rows,meta,arrays=load();ts=build_temporal(rows,arrays,read(U/'NORMALIZATION.json'));ps=point_samples(rows,arrays)
 root=U/'runs';assert not root.exists(),'Inspect existing execution; never duplicate a completed/partial run'
 root.mkdir();save('RUNNING.json',dict(pid=__import__('os').getpid(),start=time.time(),status='TRAINING_FIXED_COMPONENT_COMPARISON'))
 p,ph=learn_point(ps,plan);torch.save(p.state_dict(),root/'point.pt');pr=predict_point(p,ps)
 save('POINT_RESULT.json',dict(history=ph,rows=pr,metrics=metrics(pr),checkpoint_sha256=sha(root/'point.pt')))
 collected={}
 for condition in plan['conditions']:
  model,h=learn_temporal(ts,condition,plan);path=root/(condition+'.pt');torch.save(model.state_dict(),path)
  results=predict_temporal(model,ts);m=metrics(results);collected[condition]=m
  save(condition.upper()+'_RESULT.json',dict(condition=condition,history=h,rows=results,metrics=m,checkpoint_sha256=sha(path)))
  del model
 report=gates(metrics(pr),collected['pooled'],collected['ordered'])
 for p,h in plan['pins'].items():assert sha(p)==h,p
 report.update(status='COMPARISON_PASSED_SCOPED_GATES' if report['passed'] else 'COMPARISON_GATES_NOT_MET',
  old_models_and_labels_unchanged=True,new_point_runs=1,new_temporal_runs=2,
  no_old_model_inference=True,private_or_holdout_used=False,full_latency_measured=False)
 save('COMPARISON.json',report);save('COMPLETE.json',dict(status=report['status'],time=time.time(),whole_goal_complete=False))
 print(report,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
