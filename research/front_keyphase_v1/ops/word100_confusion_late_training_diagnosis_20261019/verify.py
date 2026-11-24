"""Separate arithmetic verification, no model execution or training."""
from common import *
from collections import Counter
import statistics,math

def main():
 a=read(U/'SAVED_ANALYSIS.json');q=read(U/'INPUT_QUALITY.json');m=read(U/'LATE_MECHANISM.json');c=read(U/'CONTRACT.json');visual=read(U/'VISUAL_REVIEW.json')
 for k,h in {**c['protected_pins'],**c['prior_result_pins'],**a['pins'],**q['pins'],**m['pins']}.items():assert sha(k)==h,('changed',k)
 count=Counter();by_signer=Counter();r2=r5=0;epochs=0
 for run in a['runs']:
  rr=read(resultpath(run['arm'],run['seed']));hist=read(historypath(run['arm'],run['seed']));epochs+=len(hist)
  for x in rr['predictions']['sensor']:
   scores=x['logits'];best=max(range(len(scores)),key=lambda i:(scores[i],-i));rank=1+sum((v>scores[x['label_index']]) or (v==scores[x['label_index']] and i<x['label_index']) for i,v in enumerate(scores))
   wrong=best!=x['label_index'];count[x['clip_id']]+=wrong;by_signer[str(x['signer_id'])]+=wrong;r2+=wrong and rank==2;r5+=wrong and rank<=5
  late=[h for h in hist if 61<=h['epoch']<=80];prev=[h for h in hist if 41<=h['epoch']<=60]
  assert run['late']['sensor_range']==[min(h['sensor_top1'] for h in late),max(h['sensor_top1'] for h in late)]
  assert run['late']['sensor_mean']==statistics.mean(h['sensor_top1'] for h in late)
  assert run['late_adjacent_max_change']==max(abs(x['sensor_top1']-y['sensor_top1']) for x,y in zip(late,late[1:]))
  assert statistics.mean(h['sensor_top1'] for h in late)>statistics.mean(h['sensor_top1'] for h in prev)
  assert run['selected_epoch']==rr['selected_epoch']
 assert sum(count.values())==a['total_wrong_repeated']==62 and len([v for v in count.values() if v])==17 and epochs==960
 assert a['focus']==sorted(count,key=lambda k:(-count[k],k))[:5] and r2==a['wrong_rank_le2']==37 and r5==a['wrong_rank_le5']==52
 assert dict(by_signer)==m['wrong_by_signer']
 assert len(q['quality'])==345 and len(q['sheets'])==25 and len(visual['cases'])==5
 rowmap={r['clip_id']:r for r in read(FACT/'PREP_CONTEXT.json')['rows']}
 for cid in a['focus']:
  row=rowmap[cid];rec=read(BASE/'data/records'/f'{cid}.json')
  for view in ['sensor','masked']:
   z=load(rec['views'][view]['observation']);ch=load(rec['views'][view]['channel_file']);times=z['timestamp_us']/1e6
   indices=[i for i,t in enumerate(times) if any(lo<=t<=hi for lo,hi in row['intervals'])]
   check=q['quality'][cid]['views'][view]['annotation_core'];assert check['frames']==len(indices)
   left=sum(sum(z['usable'][i,33:54])>=3 for i in indices)/len(indices);right=sum(sum(z['usable'][i,54:75])>=3 for i in indices)/len(indices)
   assert abs(check['left_observed_fraction']-left)<1e-12 and abs(check['right_observed_fraction']-right)<1e-12
   covered=sum(z['frame_id'][i] in ch['whole_frame_id'] for i in indices)/len(indices);assert covered==q['quality'][cid]['views'][view]['core_covered_by_model_fraction']
 for x in q['sheets']:
  assert Path(x['sheet']).is_file() and Path(x['video']).is_file();assert len(x['frames'])==8
 for x in m['correlations']:
  first=next(r for r in a['runs'] if r['arm']=='C' and r['seed']==x['seed']);other=next(r for r in a['runs'] if r['arm']==x['arm'] and r['seed']==x['seed'])
  aa=[v['sensor'] for v in first['late_series']];bb=[v['sensor'] for v in other['late_series']];am=statistics.mean(aa);bm=statistics.mean(bb)
  corr=sum((v-am)*(w-bm) for v,w in zip(aa,bb))/math.sqrt(sum((v-am)**2 for v in aa)*sum((w-bm)**2 for w in bb));assert abs(corr-x['late_top1_r_with_C'])<1e-12
 assert sum([32]*28+[3])==sum([31]*29)==899 and len([32]*28+[3])==len([31]*29)==29
 nextplan=read(U/'NEXT_PLAN.json');assert not nextplan['training_authorized_now']
 write(U/'VERIFICATION.json',{'at':now(),'status':'PASS_SAVED_ARITHMETIC_SOURCE_PINS_AND_SELECTED_QUALITY','saved_predictions':3600,'history_epochs':epochs,'quality_records':345,'sheets':25,'reviewed_sample_frames':200,'original_checkpoints_preserved':len(c['protected_pins']),
  'author':'Codex separate arithmetic; not independent person/agent','training':0,'inference':0,'detector_runs':0,'user_data_used':False,'production_changed':False,
  'pins':{n:sha(U/n) for n in ['SAVED_ANALYSIS.json','INPUT_QUALITY.json','LATE_MECHANISM.json','VISUAL_REVIEW.json','NEXT_PLAN.json']},'code_sha256':sha(__file__)})
 print('PASS_SAVED_ARITHMETIC_SOURCE_PINS_AND_SELECTED_QUALITY')
if __name__=='__main__':main()
