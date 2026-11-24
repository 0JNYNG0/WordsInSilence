from experiment import *
def main():
 assert not (U/'runs').exists()
 old=(U/'FROZEN.json').read_bytes()
 with (U/'FROZEN_pre_review.json').open('xb') as f:f.write(old)
 p=json.loads(old);p['revision']=2
 for q in [U/'experiment.py',BASE/'POOLED_RESULT.json',BASE/'COMPARISON.json']:
  p['pins'][str(q)]=sha(q)
 p['upstream_pins_checked_recursively']=True
 p['matched_additional_gates']=['Retain native negative specificity and confirmed negative coverage',
  'At least one known native/reverse/still discrimination or confirmed-coverage metric strictly improves over the matched output-only control']
 (U/'FROZEN.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
 save('PRETRAINING_CORRECTION.json',dict(before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=sha(U/'FROZEN.json'),
  candidate_predictions_seen=False,training_started=False,old_plan_preserved=True,
  reason='Separate review identified missing transitive dependency checks and missing native negative retention/added-contribution gate.'))
 for q,h in p['pins'].items():assert sha(q)==h,q
 for q,h in read(BASE/'FROZEN.json')['pins'].items():assert sha(q)==h,q
 print('Revised transitive pins and matched negative/gain criteria before training')
if __name__=='__main__':main()
