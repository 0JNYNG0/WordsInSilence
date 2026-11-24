from common import *
import shutil
assert not (U/'FROZEN.json').exists()
auth=dict(at=now(),user_request='인식률을 개선할 때까지 스스로 검증하며 자율적으로 최선의 방법을 찾아 계속 진행',scope='Local research candidates, public TRAIN/DEV-only development, frozen independent public check, speed/integrity audit; ordinary gates resolved autonomously. Preserve C/T3F and production; user/mentor captures excluded. No success guarantee or unlimited score fishing.',supersedes='Previous per-stage approval waiting for isolated local research; scientific holdout rules and production replacement restrictions remain.')
write(U/'AUTHORIZATION.json',auth)
pins={};models={};scores={'dev':{},'known200':{}}
for s in SEEDS:
 models[str(s)]=dict(path=str(T/f'T3F_{s}_best.pt'),sha256=sha(T/f'T3F_{s}_best.pt'))
 for split,path in [('dev',T/f'T3F_{s}_RESULT.json'),('known200',E/f'T3F_{s}_SCORES.json')]:
  scores[split][str(s)]={'path':str(path),'sha256':sha(path)};pins[str(path)]=sha(path)
 pins[models[str(s)]['path']]=models[str(s)]['sha256']
for p in [B/'training_C_20261013_RESULT.json',E/'C_20261013_SCORES.json',T/'TRAIN_CONTEXT.json',B/'training_CONTEXT.json',E/'EVAL_CONTEXT.json',U/'common.py',U/'compare_saved.py']:
 pins[str(p)]=sha(p)
protected=read(O/'word100_mano_position_shape_split_20261026/FROZEN.json')['protected']
for p,h in protected.items():assert sha(p)==h
f=dict(at=now(),status='FROZEN_BEFORE_ENSEMBLE_SCORES',candidate='E3 uniform arithmetic mean of three existing T3F softmax probability vectors',seeds=list(SEEDS),weights=[1/3]*3,temperature=1.,tie_policy='lowest fixed class index',baseline='Deployed opt-in T3F seed20261013; C seed20261013 secondary. Also show all individual seeds, no bestseed selection.',models=models,saved_scores=scores,pins=pins,protected=protected,
 rationale='Training initializations already give different public validation predictions. Equal-weight averaging can reduce individual variability without fabricated joints, new labels, training or selecting weights. Complementarity is a hypothesis, not guaranteed.',
 development_gate='E3 sensor correct count strictly greater than T3F20261013; masked correct count and both top5 nondecreasing; no signer correct count regression in either view. Otherwise retain result and stop this candidate without weight/seed/temperature search.',
 verification_gate='Separate scalar probability/rank/paired-count verification plus source/checkpoint integrity and measured same-input inference timing. Existing input extraction and class mapping unchanged.',
 known200_policy='Previously exposed REAL17/18 saved scores only as regression test. Never call these fresh independent evaluation. No re-inference or selecting weights on this result.',
 fresh_gate='Only after development gate and integrity PASS: freeze complete eligible REAL14-16 cohort (299 plus explicit unresolved metadata exclusion), audit exposure, prepare exact unchanged C views, verify channels, evaluate fixed C13/T3F13/E3 once. Do not tune on fresh cohort. If actually prior-exposed, report accurately and do not call blind.',
 fresh_success='Sensor top1 improves over T3F13 and masked/top5 nondecreasing. Report gains/losses per person and paired uncertainty; tiny gain not broad proof. If baseline ceiling or no improvement, say not demonstrated rather than chase fresh scores.',
 wrist_route='First define current RGB wrist/forearm connection evidence, reuse previous reviewed uncertainty regions. No ambiguous HaMeR coordinate changes before reliable reference evidence. Classifier candidate leaves all joints unchanged.',
 bounds='One predeclared ensemble candidate, no parameter sweep; fresh cohort one use only. Further hypotheses must use TRAIN/DEV, never user recordings or the new holdout for tuning. No auto deployment.',training=0,user_data_access=0,production_change=False)
write(U/'FROZEN.json',f)
write(U/'WORK_ITEMS.json',dict(items=[dict(id='wrist_criterion',status='pending'),dict(id='fixed_E3_development',status='frozen'),dict(id='scalar_verification_and_timing',status='pending'),dict(id='conditional_fresh_public_evaluation',status='pending'),dict(id='report_and_handoff',status='pending')]))
print('Frozen E3 single candidate; no score aggregation performed.')
