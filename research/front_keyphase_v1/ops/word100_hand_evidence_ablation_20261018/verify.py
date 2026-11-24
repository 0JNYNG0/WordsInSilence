"""Independent saved-score/trace recount. No model construction or inference."""
from common import *
import math
from collections import Counter
import numpy as np
import torch
SEEDS=(20261013,20261014,20261015)
VIEWS=('masked','sensor')

def calc(rows):
 n=len(rows)
 return {'total':n,'top1_correct':sum(r['correct'] for r in rows),'top5_correct':sum(r['top5_correct'] for r in rows),'mean_ce':sum(r['cross_entropy'] for r in rows)/n,'input_failures':sum(r['status']!='OK' for r in rows)}

def assert_metrics(actual,expected):
 for k,x in actual.items():
  if k=='mean_ce':assert abs(x-expected[k])<1e-10,(k,x,expected[k])
  else:assert x==expected[k],(k,x,expected[k])
 if 'top1_percent' in expected:assert abs(100*actual['top1_correct']/actual['total']-expected['top1_percent'])<1e-10
 if 'top5_percent' in expected:assert abs(100*actual['top5_correct']/actual['total']-expected['top5_percent'])<1e-10

def recount(result,context):
 expected={r['clip_id']:r for r in context['records'] if r['role']=='development_only'}
 out={}
 for view in VIEWS:
  rows=result['predictions'][view];assert len(rows)==300 and {r['clip_id'] for r in rows}==set(expected)
  for r in rows:
   e=expected[r['clip_id']];assert (r['label_index'],r['signer_id'])==(e['label_index'],e['signer_id'])
   assert r['status']=='OK' and len(r['logits'])==100 and all(math.isfinite(x) for x in r['logits'])
   order=sorted(range(100),key=lambda i:(-r['logits'][i],i));m=max(r['logits'])
   ce=m+math.log(math.fsum(math.exp(x-m) for x in r['logits']))-r['logits'][e['label_index']]
   assert r['prediction']==order[0] and r['top5']==order[:5]
   assert r['correct']==(order[0]==e['label_index']) and r['top5_correct']==(e['label_index'] in order[:5])
   assert abs(ce-r['cross_entropy'])<1e-10
  a=calc(rows);assert_metrics(a,result['metrics'][view])
  a['by_signer']={str(s):calc([r for r in rows if r['signer_id']==s]) for s in (11,12,13)}
  a['by_class']={str(k):calc([r for r in rows if r['label_index']==k]) for k in range(100)}
  groups=context['missingness_groups'][view]
  a['by_observation_pattern']={g:calc([r for r in rows if groups[r['clip_id']]==g]) for g in sorted(set(groups.values()))}
  for field in ('by_signer','by_class','by_observation_pattern'):
   for k,x in a[field].items():assert_metrics(x,result['metrics'][view][field][k])
  out[view]=a
 assert abs((out['masked']['mean_ce']+out['sensor']['mean_ce'])/2-result['selected_ce'])<1e-10
 return out

def tensor_equal(a,b):
 assert set(a)==set(b)
 assert all(torch.equal(a[k],b[k]) for k in a)

def tracecheck(arm,seed,result,context):
 stem=f'{arm}_{seed}';hist=read(U/f'{stem}_history.json');last=torch.load(U/f'{stem}_last.pt',map_location='cpu',weights_only=True);ck=torch.load(U/f'{stem}_best.pt',map_location='cpu',weights_only=True)
 assert hist==last['history'] and len(hist)==last['epoch']==result['completed_epochs']
 assert result['context_sha256']==last['context_sha256']==ck['context_sha256']==sha(U/'TRAIN_CONTEXT.json')
 assert result['checkpoint_sha256']==sha(U/f'{stem}_best.pt')
 train=[r['clip_id'] for r in context['records'] if r['role']=='train_candidate'];assert len(train)==899
 best=None;exposure=0;steps=0
 for ep,h in enumerate(hist,1):
  assert h['epoch']==ep and math.isfinite(h['selection_ce']) and math.isfinite(h['train_ce'])
  if best is None or h['selection_ce']<best['selection_ce']-1e-6:best=h
  assert h['best_epoch']==best['epoch']
  if ep<len(hist):assert ep-best['epoch']<12
  f=U/'traces'/stem/f'{ep:03}.json';assert sha(f)==h['trace_sha256'];t=read(f)
  assert (t['arm'],t['seed'],t['epoch'])==(arm,seed,ep)
  ids=[x for b in t['batches'] for x in b['clips']]
  expected=[train[i] for i in np.random.default_rng(seed+ep).permutation(len(train))]
  assert ids==expected and Counter(ids)==Counter(train) and len(t['batches'])==29
  for step,b in enumerate(t['batches'],1):
   assert b['step']==step and len(b['clips'])==(32 if step<29 else 3)
   for cid,choice in zip(b['clips'],b['choices'],strict=True):
    tp=int.from_bytes(hashlib.sha256(cid.encode()).digest(),'big')%2
    vp=int.from_bytes(hashlib.sha256(('sensor_view:'+cid).encode()).digest(),'big')%2
    expected_choice=['sensor' if ((ep-1)//2+vp)%2 else 'masked','x_context' if (ep+tp)%2 else 'x_whole']
    assert choice==expected_choice
   steps+=1
  exposure+=len(ids)
 assert len(hist)==80 or len(hist)-best['epoch']>=12
 assert best['epoch']==ck['epoch']==last['best']['epoch']==result['selected_epoch']
 assert abs(best['selection_ce']-result['selected_ce'])<1e-10
 assert last['best']['predictions']==result['predictions'] and last['best']['metrics']==result['metrics']
 tensor_equal(ck['state_dict'],last['best']['state_dict'])
 for k,t in ck['state_dict'].items():
  assert torch.isfinite(t).all()
  if k.endswith('num_batches_tracked'):assert t.item()==29*best['epoch']
 assert ck['classes']==context['labels'] and ck['word_codes']==context['word_codes']
 assert ck['in_channels']==310 and ck['seed']==seed and ck['arm']==arm
 extra=ck['state_dict']['proj.weight'][:,280:]
 off,size=GROUPS[arm.removeprefix('minus_')]
 dropped=[side*15+k for side in (0,1) for k in range(off,off+size)]
 kept=[k for k in range(30) if k not in dropped]
 assert torch.count_nonzero(extra[:,dropped]).item()==0
 assert torch.count_nonzero(extra[:,kept]).item()>0
 return {'arm':arm,'seed':seed,'epochs':len(hist),'selected_epoch':best['epoch'],'batches':steps,'actual_logged_video_exposures':exposure,'best_checkpoint_link_verified':True,'extra_projection_nonzero':int(torch.count_nonzero(extra))}

def summarize(ms):
 out={}
 for view in VIEWS:
  a=[r[view] for r in ms];q={}
  for metric in ('top1_correct','top5_correct'):
   vals=[r[metric] for r in a];q[metric]={'counts':vals,'sum':sum(vals),'percent_mean':100*sum(vals)/900,'percent_range':[min(vals)/3,max(vals)/3]}
  q['mean_ce']=sum(r['mean_ce'] for r in a)/3
  q['input_failures']=[r['input_failures'] for r in a]
  for field in ('by_signer','by_observation_pattern'):
   q[field]={k:{'total_per_seed':a[0][field][k]['total'],'top1_counts':[r[field][k]['top1_correct'] for r in a],'top1_sum':sum(r[field][k]['top1_correct'] for r in a),'top1_percent_mean':100*sum(r[field][k]['top1_correct'] for r in a)/(3*a[0][field][k]['total'])} for k in a[0][field]}
  out[view]=q
 return out

def changes(before,after,context):
 byid={r['clip_id']:r for r in context['records']};out={}
 for view in VIEWS:
  aa={r['clip_id']:r for r in before['predictions'][view]};bb={r['clip_id']:r for r in after['predictions'][view]}
  q={'gained':[],'regressed':[]}
  for cid in sorted(aa):
   if aa[cid]['correct']==bb[cid]['correct']:continue
   key='gained' if bb[cid]['correct'] else 'regressed'
   q[key].append({'clip_id':cid,'label':byid[cid]['label'],'before_prediction':context['labels'][aa[cid]['prediction']],'after_prediction':context['labels'][bb[cid]['prediction']]})
  q['net_correct']=len(q['gained'])-len(q['regressed']);q['pp']=q['net_correct']/3;out[view]=q
 return out

def main():
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
