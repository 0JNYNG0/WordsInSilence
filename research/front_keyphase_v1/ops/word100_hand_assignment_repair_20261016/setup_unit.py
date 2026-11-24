"""Initialize a separate, non-training repair unit. Never rewrite prior evidence."""
from pathlib import Path
import json, hashlib, shutil
U=Path(__file__).resolve().parent
P=U.with_name('word100_hand_assignment_20261016')
ROOT=U.parents[3]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
scope=json.loads((P/'SCOPE.json').read_text('utf-8'))
scope.update(unit=U.name, authorization='User: 정상 손 제외·중복 검출 선택 보완 후 보고, 함께 확인한 다음 학습.',
 status='FROZEN_REPAIR_PLAN_BEFORE_REPLAY',input_scope='Reuse frozen public TRAIN REAL01 12 clips / native+crop24 raw caches. No detector execution or user/dev/heldout arrays.',
 candidate='Separate anatomical-side evidence from competing detections for one side. Strong nearby pose is not vetoed by raw handedness. Same-side competition uses current wrist proximity, not temporal residual as a shape-quality metric.',
 previous_unit=str(P),
 repairs=[
  'Bounded pose side clarity compares opposite anatomical side separately from competing raw detections.',
  'Two detections with same clear pose side compete for that side only, unless independently clear distinct tracks support two hands. Select nearest current pose wrist. Exact distance tie uses raw handedness score only as deterministic fallback, then coordinate key; no shape-quality claim.',
  'Remove high-handedness hard veto on clear nearby pose when no clear contrary track exists. Keep temporal priority and consecutive current-conflict confirmation.',
  'Close palms alone do not prove a duplicate: preserve independently distinct pose or track identities; unresolved coincident identities remain unassigned.',
  'Palm admission uses existing output contract bounds margin (5%) instead of strict image rectangle. All output point validity remains existing per-point mask; no coordinate synthesis.'
 ],
 acceptance=[
  'Preserve all nine visually verified side corrections at seven prior frames and true extra left hand at believe crop49.',
  'Restore airport native63-65 and crop78 visible right hand; restore previous pose-supported raw selection at like crop47.',
  'Behavior tests: wrong high label/close pose, same-side competition, order invariance, real crossing and overlapping hands, ambiguity, absence, reset, partial bounds.',
  'Audit all24 conditions against production and previous candidate; review every new side change or loss and known selector cases.',
  'Independent conservation and protected-file hashes, source coordinates/time/masks, empty detection frames, adapter test.',
  'No recognition accuracy or unseen-user generalization claim; stop for user review before training.'
 ])
scope['parameter_basis']='All previous numerical association parameters unchanged. Bounds margin and minimum score read from current observation contract. Structural bug fixes; no coefficient sweep or word-specific branch.'
write(U/'SCOPE.json',scope)
contract=json.loads((U.parents[1]/'config/contract.json').read_text('utf-8'))
write(U/'POINT_CONTRACT.json',contract['usable_rule'])
for name in ['assignment.py','research_landmarker.py','replay.py']:
    shutil.copyfile(P/name,U/name)
(U/'before').mkdir()
for rel in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md','research/front_keyphase_v1/ops/STATE.json','Docs/WEBCAM_IMPROVEMENT_TODO.md']:
    shutil.copyfile(ROOT/rel,U/'before'/Path(rel).name)
pins={str(p):sha(p) for p in P.rglob('*') if p.is_file()}
oldctx=json.loads((P/'RAW_CAPTURE_CONTEXT.json').read_text('utf-8'))
pins.update(oldctx['pins'])
write(U/'PROTECTED_FILES.json',pins)
print('initialized',U,'protected files',len(pins))
