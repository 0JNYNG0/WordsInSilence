from pathlib import Path
import json,hashlib,shutil
U=Path(__file__).resolve().parent;P=U.with_name('word100_hand_assignment_repair_20261016');ROOT=U.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
scope=json.loads((P/'SCOPE.json').read_text(encoding='utf-8'))
scope.update(unit=U.name,authorization='User: remaining believe/coldcoffee withheld visible hands must be repaired with common rules before training; report all improvements/problems.',
 status='PLAN_FROZEN_BEFORE_CANDIDATE_REPLAY',previous_unit=str(P),
 input_scope='Existing TRAIN REAL01 12clips24conditions plus deterministic separate TRAIN02/03 eight other words16conditions. No user/dev/heldout.',
 candidate='Conservative joint-hand evidence after an independently strong anchor; short initial-velocity-unknown time-aware association with real opposite-track exclusion.',
 repairs=[
 'Preserve previous strong single-hand assignments. For unresolved detection only, anchor must have temporal_and_pose agreement; the remaining anatomical side must have its own bounded pose AND track evidence. Both complete pair costs must beat swapped pair by existing ambiguity_ratio. No same-side-suppressed or close-palms candidate is promoted.',
 'Track explicitly records whether velocity was estimated from two observations. Only unresolved initial-velocity-unknown tracks, at most two frame intervals and100ms, may use max(existing track_gate, existing max_speed*actual_dt) reachability. Existing opposite-side live track must exclude that identity by existing ratio, raw label must agree, and clear contrary pose must be absent. No extension for stale single tracks.',
 'No new coordinates, future-frame lookahead, change to point masks, or per-word code. Missing frames remain present.'
 ],
 acceptance=[
 'Recover existing raw left detections at believe native71/crop56 and right at coldcoffee crop52 without forcing every missing hand.',
 'Keep prior ten visual side corrections, added real/partial L, airport and raw selector repairs; independently audit new changes and regressions.',
 'Synthetic safeguards: own opposite-side evidence required, label-only anchor rejected, duplicate hypotheses not blindly filling opposite slot, true crossing/support hands, bootstrap elapsed time, stale/one-sided track, raw order, zero frames/reset/masks.',
 'Evaluate frozen candidate on additional8TRAIN words from other two training performers; no post-evaluation retuning; inspect all material new differences.',
 'No claim of recognition/generalization or guaranteed point correctness. Report and stop before training/production adoption.'
 ])
scope['parameters']['bootstrap_max_gap_s']=0.1
scope['parameters']['bootstrap_max_frame_gap']=2
scope['parameter_basis']='Previous eight numeric association constants unchanged. New100ms/two-frame bootstrap horizon fixed engineering limit to restrict unknown-velocity reachability to a brief initial omission; not fit by sweeping observed results. Opposite identity evidence mandatory; no relaxed raw-label fallback threshold.'
scope['prohibited']=[x for x in scope['prohibited'] if x!='bulk899 re-extraction']+['bulk899 re-extraction','further dataset/model expansion','post-result coefficient search']
write(U/'SCOPE.json',scope)
for name in ['assignment.py','research_landmarker.py','POINT_CONTRACT.json']:shutil.copyfile(P/name,U/name)
(U/'before').mkdir()
for rel in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md','research/front_keyphase_v1/ops/STATE.json','Docs/WEBCAM_IMPROVEMENT_TODO.md']:shutil.copyfile(ROOT/rel,U/'before'/Path(rel).name)
pins=json.loads((P/'PROTECTED_FILES.json').read_text(encoding='utf-8'))
pins.update({str(p):sha(p) for p in P.rglob('*') if p.is_file()})
write(U/'PROTECTED_FILES.json',pins)
print('Initialized',U.name,'protected',len(pins))
