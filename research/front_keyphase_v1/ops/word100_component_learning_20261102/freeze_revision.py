from common import *
def main():
 old=(U/'FROZEN.json').read_bytes()
 assert not (U/'runs').exists(),'Revision only before any candidate training/predictions'
 with (U/'FROZEN_pre_review.json').open('xb') as f:f.write(old)
 p=json.loads(old);p['revision']=2
 p['gates'].update(confirmed_natural_recall_each_min=.5,confirmed_negative_specificity_each_min=.5,
  natural_confident_wrong_zero=True,reporting_probability_thresholds_unchanged=[.2,.8])
 for source in [U/'common.py',U/'train_evaluate.py',U/'runtime.py',V/'relation_model.py',M/'motion_model.py',Q/'SUPERVISION.json']:
  p['pins'][str(source)]=sha(source)
 (U/'FROZEN.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
 save('PRETRAINING_CORRECTION.json',dict(before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=sha(U/'FROZEN.json'),
  reasons=['A .51/.49 all-uncertain classifier must not count as successful feedback; add confidence coverage and confident-error gates.',
   'Pin actual imported feature/model code and authoritative source manifest read at execution.'],
  training_already_run=False,candidate_predictions_seen=False,old_plan_preserved=True))
 # Synthetic boundary check of the verdict contract; no real model outputs.
 from train_evaluate import gates
 pm={POINT[1]+'::native':dict(usable_positive_recall=1.,usable_negative_specificity=1.,confirmed_positive_recall=0.,confirmed_negative_specificity=0.,confident_wrong=0)}
 ms={}
 for t in TEMP:
  for v in ['native','still','image_gap','reverse']:
   ms[t+'::'+v]=dict(usable_positive_recall=1.,usable_negative_specificity=1.,confirmed_positive_recall=0.,confirmed_negative_specificity=0.,held=3,records=3,confident_wrong=0,TN=3)
 out=gates(pm,ms,ms);assert not out['passed'] and not out['checks']['oblique_confirmed_recall']
 save('CONFIDENCE_GATE_PREFLIGHT.json',dict(all_uncertain_rejected=True,old_probability_thresholds_unchanged=True,new_model_run=False))
 print('Pretraining revision2 frozen; all-uncertain pseudo-output rejected')
if __name__=='__main__':main()
