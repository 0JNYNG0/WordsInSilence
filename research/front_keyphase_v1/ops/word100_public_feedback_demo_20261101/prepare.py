from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (U/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 assert not (U/'FROZEN.json').exists()
 case=read(O/'word100_action_feedback_casebook_20261101/INPUTS.json')
 data=read(O/'word100_semantic_feature_cycle_20261101/DATASET.json')
 paths=[O/'word100_action_feedback_casebook_20261101/INPUTS.json',O/'word100_semantic_feature_cycle_20261101/DATASET.json',
 O/'word100_semantic_feature_cycle_20261101/NORMALIZATION.json',O/'word100_semantic_feature_cycle_20261101/comparison.py',
 O/'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json',O/'word100_stage_feedback_cycle_20261031/NORMALIZATION.json',
 O/'word100_description_evidence_feedback_20261101/REPORT_DATA.json',
 O/'word100_feedback_causal_pilot_20261101/TRAINING_DECISION.json',O/'word100_feedback_causal_pilot_20261101/action_evidence.py',
 O/'word100_multimodal_evidence_route_20261101/FROZEN.json']
 for r in data['rows']:
  assert sha(r['input'])==r['input_sha256'];paths.append(Path(r['input']))
 for r in case['rows'].values():
  assert sha(r['input'])==r['input_sha256'];paths += [Path(r['observation']),Path(r['input'])]
  media=O/'word100_core_stage_evidence_20261031/media'/f"{r['alias']}.webm"
  meta=read(media.with_suffix('.json'));assert sha(media)==meta['output_sha256'];paths += [media,media.with_suffix('.json')]
 for seed in [42,43,44]:
  paths += [O/f'word100_semantic_feature_cycle_20261101/runs/protected_shape_split_features_{seed}'/n for n in ['probabilities.npy','head.pt','COMPLETE.json']]
  paths += [O/f'word100_stage_feedback_cycle_20261031/runs/pose_plus_signrep_{seed}'/n for n in ['probabilities.npy','head.pt','RUN.json']]
 before=U/'before';before.mkdir(exist_ok=False)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
  shutil.copyfile(p,before/p.name)
 save('FROZEN.json',dict(at=datetime.now().astimezone().isoformat(),authorization='2026-11-01 user approves bounded public feedback demo, conditional supervision and history recording',
 steps=['public missing-observation stratum assessment','reuse or conditionally train small head','guide-linked comparison replay','demo and regression verification'],
 audit=dict(dataset='existing 96 public clips / 288 exact-center labels; no new labels',model='protected_shape_split_features',
   seeds=[42,43,44],threshold=.5,partitions='original fit/development/new_feature_expression_evaluation; recorded persons disjoint; historically reused, not fresh final test',
   strata=['all_required_geometry_present','geometry_gate_failed'],
   missing='target-specific saved wrist+10/14 angle support or both palm center support absent; not automatically hand invisibility',
   report='known labels only; per-feature/role/stratum denominators, positives, negatives, correct, confusion; no threshold/seed selection',
   balanced_accuracy='only when positive and negative truth both exist; sparse stratum not a calibrated safety guarantee'),
 training='Reuse already supervised fixed model first. No new fit labels or demonstrated source/target defect yet; do not train on development failures or diagnostic labels merely to enable demo.',
 demo=dict(queries=['P036','P048','P066','P072'],references='all three same-expression recorded IDs1/2/3 retained separately',
   seed=42,targets=data['selected_targets'],previous_three='exact stored native predictions',new_four='saved exact centers reused; only previously unevaluated native centers with 13 real frames get one inference',
   boundaries='no padding or copied predictions outside 6..T-7',
   alignment='read-only existing DTW paths; no correct-stage/mandatory-repeat claim',
   guides='reuse source-linked descriptions; unlinked physical descriptions explicitly distinguished',
   output='comparison/model estimate/geometry support/conflict/stage uncertainty separated; no automatic educational verdict',
   service='localhost read-only public-video replay; no camera/upload/private file endpoint',
   latency='measure cached-input processing separately; cannot represent raw video to feedback latency'),
 forbidden=['user/mentor recordings','operational C/T3-F/DTW replacement','new download/upload/Git','diagnostic labels used for training','word-specific threshold exceptions','completed center inference rerun'],
 pins={str(p.resolve()):sha(p) for p in sorted(set(paths))}))
 hist=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
 with hist.open('a',encoding='utf8') as f:
  f.write('\n\n2026-11-01 — 소규모 공개 피드백 데모 연결 승인·착수\n'
  '사용자는 관절 누락 검증 후 실제 비교 데모까지 진행하고 연구 흐름을 계속 기록하도록 요청했다. 목표는 모든 좌표를 완벽히 복원하는 것이 아니라 지원 표현의 중요한 손모양·관계·순서를 근거와 함께 비교하는 것이다.\n'
  '이번 고정 범위: 기존96영상288시점의 저장 예측을 관절지원 유무로 분리 평가하고,공개4표현16영상의 기존판정부/DTW/설명연결을 로컬 재생 데모로 통합한다. 이미 학습한 판정부를 우선 재사용하고 실제 추가학습 필요근거가 있을 때만 학습한다. 새로운 교육정답/실제 학습자오류 성능을 만들어내지 않는다.\n'
  '착수 시점이며 결과는 아직 미확인. 근거: '+str(U/'FROZEN.json')+'\n')
 state=read(O/'STATE.json');state['current_stage']='PUBLIC_FEEDBACK_DEMO_IN_PROGRESS'
 state['next_single_action']='고정 공개 자료의 관절지원별 저장성적 진단 후4표현 연구 데모 연결·검증. 기존학습 재사용 우선,운영/사용자자료보존.'
 state[U.name]=dict(status='FROZEN_PRE_EXECUTION',unit=str(U),frozen_sha256=sha(U/'FROZEN.json'),active_session=None)
 (O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 save('CURRENT_EXECUTION.json',dict(status='FROZEN_PRE_EXECUTION',frozen_sha256=sha(U/'FROZEN.json')))
 print('FROZEN: 288 stored labeled centers; 4 expressions / 16 public replay clips.')
if __name__=='__main__':main()
