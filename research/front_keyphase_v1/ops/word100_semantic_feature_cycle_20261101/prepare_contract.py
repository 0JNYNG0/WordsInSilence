"""Prospective supported-feature contract, separate from failed micro-motion task."""
from pathlib import Path
import sys,json,hashlib,datetime,collections
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; OPS=U.parent
SRC=OPS/'word100_core_feature_split_20261101'; OLD=OPS/'word100_stage_feedback_cycle_20261031'
TARGETS=['R_index_only','R_open_four','L_open_four','R_closed_four','L_closed_four','R_higher','L_higher']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def prepare():
 source=read(SRC/'DATASET.json');inputs=read(SRC/'INPUTS.json');rule=read(SRC/'RUBRIC.json');sel=read(SRC/'SELECTION.json')
 audit=read(U/'review/DATA_AUDIT.json')
 for p,h in {**source['pins'],**sel['protected']}.items():assert sha(p)==h,p
 ix=[source['targets'].index(k) for k in TARGETS]
 rows=[]
 for r in source['rows']:
  q=dict(r);q['y']=[r['y'][i] for i in ix];q['target_mask']=[r['target_mask'][i] for i in ix]
  q['reasons']={k:r['reasons'][k] for k in TARGETS};q['scope']='Direct observed center-frame physical feature, 13 native frames of input context; not wholeword phase or semantic correctness'
  rows.append(q)
 assert len(rows)==288 and len({r['clip_id'] for r in rows})==96
 roles={role:sorted({r['person'] for r in rows if r['role']==role}) for role in {r['role'] for r in rows}}
 assert roles=={'fit':[1,2,3],'development':[4],'new_feature_expression_evaluation':[7,8]}
 supports={}
 for role in sorted(roles):
  supports[role]={}
  for j,k in enumerate(TARGETS):
   supports[role][k]={}
   for val,name in [(1,'positive'),(0,'negative')]:
    rr=[r for r in rows if r['role']==role and r['target_mask'][j] and r['y'][j]==val]
    supports[role][k][name]=dict(windows=len(rr),clips=len({r['clip_id'] for r in rr}),words=len({r['word_id'] for r in rr}),people=len({r['person'] for r in rr}))
 gate=rule['class_gate']
 assert all(supports['fit'][k][sign][metric]>=gate[f'fit_{sign}_{metric}_min'] for k in TARGETS for sign in ['positive','negative'] for metric in ['windows','clips','words','people'])
 numerical_checks=0
 import numpy as np
 for r in inputs['rows']:
  assert sha(r['input'])==r['input_sha256']
  with np.load(r['input'],allow_pickle=False) as a:
   assert a['x'].shape==(len(a['frame_id']),69) and a['rgb'].shape==(len(a['frame_id']),768)
   assert a['mask'].dtype==bool and np.isfinite(a['x'][a['mask']]).all() and np.isfinite(a['rgb']).all()
   assert np.all(np.diff(a['frame_id'])>0) and np.all(np.diff(a['timestamp_us'])>0)
   lookup={int(v):i for i,v in enumerate(a['frame_id'])}
   for q in [v for v in rows if v['clip_id']==r['clip_id']]:
    ii=[lookup[v] for v in q['frame_ids']]
    assert a['timestamp_us'][ii].tolist()==q['timestamps_us'] and q['center_frame']==q['frame_ids'][6]
    numerical_checks+=1
 def pin(paths):return {str(p):sha(p) for p in paths}
 paths=[SRC/'DATASET.json',SRC/'INPUTS.json',SRC/'RUBRIC.json',SRC/'LABEL_AUDIT.json',SRC/'SELECTION.json',U/'review/DATA_AUDIT.json',OPS/'word100_semantic_target_review_20261101/PLAN_KO.txt']
 write(U/'AUTHORIZATION.json',dict(at=now(),user='오케이 그럼 너가 지금 제안한 다음 작업을 4번까지 모두 전부 진행해줘.',scope='Four steps: common contract, supported public labels, fixed six-run feature-head comparison, separate feedback checks',private_data_access=False,new_external_download=False,production_change=False,original_failure_preserved=True))
 write(U/'FEATURE_CONTRACT.json',dict(status='SUPPORTED_FEATURES_SEPARATE_FROM_UNRESOLVED_SEMANTICS',goal='한국수어 설명과 여러 공개 참고를 연결하여 어느 손의 어떤 모양·관계·움직임·순서가 보이는지 구분하고 중요한 차이와 허용 가능한 차이를 분리한다.',
  learned_now=TARGETS,definitions=dict(shape=rule['shape_definitions'],relation=rule['relation_definition']),
  evidence_types=['direct_RGB_observation','temporal_support_not_direct_truth','source_description_not_frame_truth','unknown'],
  stages=['preparation','main_action','transition_between_observed_states','ending','unknown'],stage_labels_not_created_from_word_names=True,
  current_temporal_scope='13-native-frame contextual classifier at labeled center; retained dense sequence annotations support separate technical sequence tests, not wholeword semantic training',
  deferred={'L_index_only':'FIT positive words3 below unchanged minimum4','support_hand_vs_active_hand':'Whole-main-action roles have no validated per-time training truth yet','body_contact_and_region':'Need same-expression guide link and actual visible contact; generic higher labels are not body contact','rotation_alternation_repeat_order':'Sequence checking only where observed; no learned universal mandatory-repeat/phase labels'},
  micro_drift='Preserved as observation, not obligatory independent action/error. No smoothing or automatic unknown-to-still replacement.',
  guide_policy='Guide selects relevant comparison features after exact form link; never used to force observed handshape. Requiredness/tolerance remain unknown without evidence.',
  sobok='No verified exact lexical guide or educational tolerance. User interpretation is not training truth.',
  sign_language_word='Source describes index/relative placement/alternating rotation; exact two cycles not mandated.',
  source_pins=pin(paths)))
 write(U/'DATASET.json',dict(rows=rows,targets=TARGETS,selected_targets=TARGETS,selected_indices=list(range(7)),feature_names=source['feature_names'],pins={**source['pins'],**pin(paths)},label_provenance='Reused same-Codex public RGB research labels, not independent human/KSL truth'))
 write(U/'PREPARATION.json',dict(status='PASS_SUPPORTED_TARGETS_INPUTS_AND_SPLITS',at=now(),supports=supports,class_gate=gate,roles=roles,windows=288,clips=96,exact_frame_time_checks=numerical_checks,known_cells=sum(sum(r['target_mask']) for r in rows),unknown_cells=sum(7-sum(r['target_mask']) for r in rows),old_motion_labels_used=0,new_labels=0,protected=sel['protected'],evaluation_history='All clips originate historical TCN TRAIN; feature-expression evaluation is held out for this head, not globally unseen project data.',old_failure_unchanged=True))
 print('Prepared7 supported targets /96 clips /288 exact center labels. No training yet.')
def freeze():
 info=read(U/'DATASET.json');prep=read(U/'PREPARATION.json');tests=read(U/'TEST_IMPLEMENTATION.json');review=read(U/'review/CONTRACT_REVIEW.json')
 assert tests['comparison']['status'].startswith('PASS') and tests['driver']['status'].startswith('PASS') and review['status'].startswith('PASS')
 source_pins=dict(info['pins'])
 for seed in [42,43,44]:
  for name in ['RUN.json','head.pt','probabilities.npy']:
   p=OLD/'runs'/f'pose_plus_signrep_{seed}'/name;source_pins[str(p)]=sha(p)
 for name in ['NORMALIZATION.json','LEARNING_INPUTS.json','learning.py']:
  p=OLD/name;source_pins[str(p)]=sha(p)
 paths=[U/n for n in ['AUTHORIZATION.json','FEATURE_CONTRACT.json','DATASET.json','PREPARATION.json','prepare_contract.py','comparison.py','experiment.py','TEST_IMPLEMENTATION.json','FROZEN_EVALUATION.json','review/CONTRACT_REVIEW.json','review/CONTRACT_REVIEW_FINAL.json','review/SUBGROUPS.json']]
 write(U/'FROZEN.json',dict(status='FROZEN_BEFORE_NEW_SCORES',at=now(),conditions=['joint','protected_shape_split_features'],seeds=[42,43,44],epochs=160,lr=.001,weight_decay=.01,threshold=.5,targets=TARGETS,fit_samples=[r['sample_id'] for r in info['rows'] if r['role']=='fit'],
  selection='Final epoch160; no score-dependent selection/threshold/target changes',
  comparison='Joint7 new output vs protected original3 +new4. Unequal prior exposure/capacity: system comparison, not causal test of architecture.',
  primary='Per-target and family balanced accuracy on DEV and feature-held expressions; retain negatives/unknown denominators and separate old3/new4',
  success={'protected_shape':'Exactly equal saved original3 probabilities, not a claimed accuracy gain','new4':'Both splits mean balanced accuracy >0.55 and every target mean >0.50; prior-constant balanced accuracy0.50 only a trivial floor, not prior capable system','joint':'Any common3 regression is explicitly reported and prevents claim of universally improved replacement','feedback':'No increase in missing-evidence false matches; technical order/omission gains must retain normal speed/shape-state coverage; no educational-error accuracy claim'},
  actual_word_accuracy_not_measured=True,production_adoption=False,protected=prep['protected'],pins={str(p):sha(p) for p in paths},source_pins=source_pins))
 write(U/'READY.json',dict(status='PASS_SUPPORTED_SEVEN_NOT_ALL_SEMANTICS',allow_training=True,frozen_sha256=sha(U/'FROZEN.json'),dataset_sha256=sha(U/'DATASET.json'),independent_review=str(U/'review/CONTRACT_REVIEW.json'),no_linguistic_validity_claim=True))
 print('FROZEN and READY; six fixed runs authorized.')
if __name__=='__main__':{'prepare':prepare,'freeze':freeze}[sys.argv[1]]()
