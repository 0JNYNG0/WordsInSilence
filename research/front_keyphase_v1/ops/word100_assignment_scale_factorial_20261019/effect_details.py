"""Explain verified stored comparisons without training or new inference."""
from common import *
from collections import Counter

def main():
 r=read(U/'COMPARISON_INDEPENDENT.json');ctx=read(U/'TRAIN_CONTEXT.json')
 labels=ctx['labels'];seeds=ctx['seeds'];data={}
 for a in ['C','Q','R10','R01']:
  data[a]=[]
  for seed in seeds:
   p=BASE/f'training_C_{seed}_RESULT.json' if a=='C' else PREV/f'Q_{seed}_RESULT.json' if a=='Q' else U/f'{a}_{seed}_RESULT.json'
   result=read(p);rows=result['predictions']['sensor'];incorrect=[]
   for x in rows:
    if not x['correct']:
     rank=next((i+1 for i,k in enumerate(sorted(range(100),key=lambda i:(-x['logits'][i],i))) if k==x['label_index']),None) if x['status']=='OK' else None
     incorrect.append({'clip':x['clip_id'],'target':labels[x['label_index']],'predicted':labels[x['prediction']] if x['prediction'] is not None else None,'target_rank':rank})
   data[a].append({'seed':seed,'selected_epoch':result['selected_epoch'],'incorrect':incorrect})
 changes={}
 for pair,views in r['paired'].items():
  entries=views['sensor'];gains=Counter(cid for e in entries for cid in e['gains']);losses=Counter(cid for e in entries for cid in e['losses'])
  changes[pair]={'gains_total':sum(gains.values()),'losses_total':sum(losses.values()),'gains':dict(gains),'losses':dict(losses)}
 grouped={}
 for a in data:
  ms=r['metrics'][a]['sensor'];grouped[a]={}
  for g in ms[0]['groups']:
   correct=sum(m['groups'][g]['correct'] for m in ms);total=sum(m['groups'][g]['total'] for m in ms)
   grouped[a][g]={'correct_repeated':correct,'total_repeated':total,'accuracy_percent':100*correct/total}
 scale=read(U/'PREP_RESULT.json')
 changed_dev=sorted({x['clip'] for x in scale['changed'] if x['role']=='development_only'})
 changed_train=sorted({x['clip'] for x in scale['changed'] if x['role']=='train_candidate'})
 assert changed_dev==['NIA_SL_WORD2029_REAL11_F']
 # Prior pinned audit establishes Q-vs-C also changes only this DEV source.
 old=read(PLAN/'SAVED_SCORE_DIAGNOSIS.json')
 out={'at':now(),'status':'STORED_COMPARISON_DESCRIPTION','arms':data,'paired_sensor':changes,'sensor_groups':grouped,
      'finite_scale_changed_train_sources':changed_train,'finite_scale_changed_dev_sources':changed_dev,
      'scale_changed_dev_correct_all_cells':all(all(next(x for x in (read(BASE/f'training_C_{seed}_RESULT.json') if a=='C' else read(PREV/f'Q_{seed}_RESULT.json') if a=='Q' else read(U/f'{a}_{seed}_RESULT.json'))['predictions']['sensor'] if x['clip_id']==changed_dev[0])['correct'] for seed in seeds) for a in data),
      'prior_observation_audit':{'path':str(PLAN/'SAVED_SCORE_DIAGNOSIS.json'),'sha256':sha(PLAN/'SAVED_SCORE_DIAGNOSIS.json')},
      'pins':{n:sha(U/n) for n in ['COMPARISON_INDEPENDENT.json','TRAIN_CONTEXT.json','PREP_RESULT.json']},
      'new_inference':0,'user_data_used':False,'code_sha256':sha(__file__)}
 write(U/'EFFECT_DETAILS.json',out,new=True)
 print(json.dumps({'paired':changes,'groups':grouped,'changed_train':len(changed_train),'scale_dev_all_correct':out['scale_changed_dev_correct_all_cells']},ensure_ascii=False))
if __name__=='__main__':main()
