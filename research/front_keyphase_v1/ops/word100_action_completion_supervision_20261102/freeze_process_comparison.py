"""Freeze one bounded comparison only after the independent readiness review.

This is a single research candidate, not permission to replace any live model.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 dest=U/'PROCESS_COMPARISON_FROZEN.json';assert not dest.exists()
 assert not (U/'PROCESS_TRAIN_RESULT.json').exists()
 ready=read(U/'PROCESS_FINAL_READINESS_REVIEW.json')
 assert ready['status']=='READY_FOR_ONE_FIXED_TRAINING_RUN' and ready['training_allowed'] is True
 for p,h in ready['pins'].items():assert sha(p)==h,p
 fit=read(U/'PROCESS_VIEW_TARGETS.json')['rows'];dev=read(U/'DEV_PROCESS_VIEW_TARGETS.json')['rows']
 assert Counter(str(r['observed_process_evidence']) for r in fit)=={'1':79,'0':128,'None':10}
 assert Counter(str(r['observed_process_evidence']) for r in dev)=={'1':30,'0':45,'None':5}
 assert {r['recorded_person_id'] for r in fit}==set(range(1,10))
 assert {r['recorded_person_id'] for r in dev}=={11,12,13}
 assert len({r['query_id'] for r in dev})==15 and len({r['clip_id'] for r in dev})==12
 for n in ['PROCESS_ACTUAL_INPUT_PREFLIGHT.json','PROCESS_INPUTS_VERIFIED.json','DEV_PROCESS_INPUTS_VERIFIED.json',
           'PROCESS_VIEW_TARGETS_INDEPENDENT.json','DEV_PROCESS_VIEW_TARGETS_VERIFIED.json','PROCESS_AUTO_CONTRACT_VERIFIED.json']:
  assert read(U/n)['status'].startswith('PASS'),n
 required=['process_head.py','train_process_head.py','evaluate_process_head.py','verify_process_evaluation_saved.py',
  'process_auto_proposal.py','process_qualification.py','prepare_process_auto_native.py','evaluate_process_auto.py',
  'check_process_auto_contract.py','PROCESS_AUTO_NATIVE_MANIFEST.json','PROCESS_AUTO_PROPOSAL_PLAN.json',
  'PROCESS_AUTO_CONTRACT_VERIFIED.json','PROCESS_VIEW_TARGETS.json','DEV_PROCESS_VIEW_TARGETS.json',
  'PROCESS_TARGETS.json','DEV_PROCESS_TARGETS.json',
  'PROCESS_INPUT_MANIFEST.json','DEV_PROCESS_INPUT_MANIFEST.json','PROCESS_TRAINING_DESIGN_REVIEW.json',
  'PROCESS_HEAD_CODE_REVIEW.json','PROCESS_EVALUATION_CODE_REVIEW.json','PROCESS_EVALUATOR_REVIEW_ADJUSTMENT.txt',
  'PROCESS_ACTUAL_INPUT_PREFLIGHT.json','PROCESS_INPUTS_VERIFIED.json','DEV_PROCESS_INPUTS_VERIFIED.json',
  'PROCESS_VIEW_TARGETS_INDEPENDENT.json','DEV_PROCESS_VIEW_TARGETS_VERIFIED.json',
  'PROCESS_FINAL_READINESS_REVIEW.json','freeze_process_comparison.py']
 pins=dict(ready['pins'])
 for fn in ['PROCESS_EXTRACTION_FROZEN.json','DEV_PROCESS_EXTRACTION_FROZEN.json','PROCESS_AUTO_NATIVE_MANIFEST.json']:
  frozen=read(U/fn)
  for p,h in frozen['pins'].items():assert sha(p)==h,p
  pins.update(frozen['pins']);pins[str(U/fn)]=sha(U/fn)
 pins.update({str(U/n):sha(U/n) for n in required})
 for manifest in ['PROCESS_INPUT_MANIFEST.json','DEV_PROCESS_INPUT_MANIFEST.json']:
  for r in read(U/manifest)['rows']:
   assert sha(r['input'])==r['input_sha256'];pins[r['input']]=r['input_sha256']
 c=dict(at=datetime.now(timezone.utc).isoformat(),status='READY_FOR_ONE_FIXED_TRAINING_RUN',
  seed=42,epochs=80,candidates=1,checkpoint='final_only',pins=pins,
  optimization=dict(optimizer='AdamW',lr=.0005,weight_decay=.01,gradient_clip=2,batch_source_queries=12,
   loss='L_view + 0.25 L_point + 0.25 L_interval',known_only=True,unknown_gradient=0,
   balancing='equal original person/query groups; within-view known class balance; inverse multiplicity sparse point/interval'),
  normalization='FIT216 unique actual views, visible RGB frames only, std floor .001; no DEV, not person-balanced statistics',
  inputs='Frozen RGB768 + 59 body-relative geometry +59 validity +1 image availability +typed physical query',
  fixed_gates=dict(raw_macro_query_balanced_accuracy_min=.8,known_positive_confident_required=24,known_positive_total=30,
   known_negative_confident_required=36,known_negative_total=45,known0_confident_false_positive_max=0,
   native1_confident_false_negative_max=0,native_positive_per_query_required=2,native_positive_per_query_total=3,
   automatic_native_positive_required=12,automatic_native_positive_total=15,automatic_per_query_required=2,
   qualified_positive_required=24,qualified_positive_total=30,qualified_known0_false_confirmation_max=0,
   qualified_unknown_false_confirmation_max=0,old_output_changes_max=0,
   confidence_positive=.8,confidence_negative=.2,raw_binary=.5),
  additional_gates=['No confident reversal of FIT alternate-intact repetitions/splices and positive speed/trim controls',
   'Unknown observations remain unknown; an actually intact separate repetition may still support the query',
   'Manual bounded-window success alone does not authorize runtime integration',
   'Existing C/T3F/observation/order/source fields and all old model files unchanged'],
  evaluation_order=['Fixed final candidate training once','Raw FIT217/DEV80 predictions once',
   'Saved-prediction arithmetic verification','Edited-view qualification from saved point/process predictions; bounded new interval scores only',
   'One saved-prototype automatic proposal per full-native FIT45/DEV15 query; no old model rerun',
   'Separate saved-result verification before any integration decision'],
  no_selection=['No second seed/checkpoint/threshold/weight/epoch search','Failure preserved, never relabeled to improve a score'],
  interpretation=['Physical observed-process evidence, not whole-word correctness, mandatory repetition or learner error',
   'Known0 says supported connected process absent from this shown scope; it does not prove the performer made a language error',
   'DEV repeatedly used; not independent final population evaluation',
   'Same-Codex physical annotations, not independent human/KSL expert labels',
   'Retimed identical pixels do not establish natural execution speed robustness',
   'Source-seam metadata qualification in edited tests is oracle-aware and separate from actual native runtime'],
  runtime_replacement_authorized=False,new_downloads=False,private_or_mentor_data_allowed=False,whole_goal_complete=False)
 dest.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print('Frozen one fixed candidate with',len(pins),'pins; no training executed by freeze')
if __name__=='__main__':main()
