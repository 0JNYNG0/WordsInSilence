"""Finalize evidence and current state; never launches training or adopts models."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
v=read(U/'INDEPENDENT_VERIFICATION.json');a=read(U/'CHANGE_AUDIT.json');r=read(U/'REPLAY_RESULT.json')
assert v['independent_behavior_tests']==47 and v['failures']==v['errors']==0
assert a['all_target_checks_pass'] and a['previous_repairs_all_preserved']
assert a['known24_only_three_target_detections_changed'] and not a['added16_new_differences_vs_v2']
assert r['counts_vs_previous']=={'unchanged':6717,'newly_assigned':3}
assert v['raw_order_invariance']['conditions']==40 and v['raw_order_invariance']['changed_physical_assignments']==0
for name in ['RESULT_KO.txt','EMPIRICAL_REVIEW.md','ADDITIONAL_VISUAL_REVIEW.md','CODE_REVIEW.md']:
    assert (U/name).is_file(),name
for p,h in {**read(U/'PROTECTED_FILES.json'),**read(U/'REPLAY_CONTEXT.json')['pins'],**read(U/'CANDIDATE_FREEZE.json')['pins']}.items():
    assert sha(Path(p))==h,p
cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';sp=ROOT/'research/front_keyphase_v1/ops/STATE.json'
tp=ROOT/'Docs/WEBCAM_IMPROVEMENT_TODO.md';hp=ROOT/'NEW_CHAT_HANDOFF.md'
for p in [cp,sp,tp,hp]:assert sha(p)==sha(U/'before'/p.name),f'Concurrent document change: {p}'
summary=('믿다 원본71·상반신56의 L, 냉커피 상반신52의 R 보류를 공통 양손 대응·초기 시간 처리로 회복. '
 '공개 TRAIN20영상40조건(기존12+다른단어8/촬영자02·03),5,072프레임·6,720검출 비교. '
 '직전 후보 대비6,717불변+3회복,새 제외/좌우변경0;이전10교정·2추가관측·기존복구 유지. '
 '독립47/47검사·3591파일 해시·40조건 raw순서반전 불변 통과. '
 '추가자료에서 이전 후보의 소복57 정상R 중복후보동률 손실,응접실46·팔십구43 초기손 보류가 확인됨. '
 '성남4건은 잘못된 중복L 제거,믿다/냉커피의 다른 프레임은 원검출 부재도 남음. '
 '따라서 목표3건 수정 검증 완료이나 전체 손 문제 해결·학습 준비 완료 아님. 학습/분류추론/운영·웹·DTW 교체0,상호검토 대기.')
next_action=('사용자와 이번 회복3건 및 남은 문제를 검토한다. 다음 보완은 pose 손목 미검출 시 같은 손 후보의 동률로 정상 손을 모두 버리는 문제와 '
 '화면 진입 초기 손의 근거 부족 정책을 하나의 입력 안정화 단위로 정리하는 것이다. 단어별 예외·무조건 채움·단순 임계값 하향 금지. '
 '새 범위·반례·통과 기준을 고정하기 전 동결 후보 재수정이나899영상 재추출·학습·운영 교체를 시작하지 않는다.')
decision=dict(status='TARGET_THREE_RECOVERIES_VERIFIED_OTHER_INPUT_LIMITS_USER_REVIEW',updated_kst=now,
    summary=summary,next_single_action=next_action,known_target_repairs_pass=True,all_visible_hands_preserved=False,
    training_ready=False,production_adopted=False,user_review='PENDING',training=0,classifier_inference=0,
    new_detector_conditions=16,new_source_clips=8,raw_candidate_replay_conditions=40,
    running_processes=[],implementation='Codex root',independent_reviewers=['assignment_verify','raw_capture','assignment_review'],
    candidate_sha256=sha(U/'assignment.py'),
    first_failure_and_repair='Synthetic stale opposite prediction could steal current hand; first counterexample preserved; fixed before real replay by both observed/predicted exclusion and recent opposite track; no real-result retuning.',
    remaining_issues=['Pose-absent duplicate same-side tie can discard visible real hand (소복57).',
       'Initial partial/weak hand with no pose or history remains withheld (믿다46/47,응접실46,팔십구43,사계절34).',
       'Exit boundary palm may be outside mask contract (즐겁다94).',
       'MediaPipe raw missing despite visible hand, notably believe/coldcoffee neighboring frames.',
       'Assignment correctness is not 21-point accuracy, classifier accuracy, or DTW validity.'],
    prohibitions_preserved=['user data tuning','dev/heldout use','camera','downloads','external transmission','Git mutation','production changes'])
new(U/'DECISION.json',decision)
entry={**decision,'unit':'ops/'+U.name,'report':'ops/'+U.name+'/RESULT_KO.txt',
       'conditions':40,'frames':5072,'raw_hand_detections':6720,'counts_vs_previous':r['counts_vs_previous'],
       'counts_vs_production':{k:r['counts'][k] for k in ['unchanged','newly_assigned','side_changed','dropped']},
       'behavior_tests_passed':47,'protected_file_hashes':3591,'completion':'ops/'+U.name+'/COMPLETION.json'}
state=read(sp);state['updated_date_kst']=now
state['current_stage']='WORD100_HAND_ASSIGNMENT_JOINT_VERIFIED_REMAINING_ISSUES_USER_REVIEW'
state['next_single_action']=next_action
out={}
for k,val in state.items():
    if k==U.name:continue
    out[k]=val
    if k=='current_stage':out[U.name]=entry
sp.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=cp.read_text('utf-8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert text.count(h8)==text.count(h9)==1
text=text.replace(h8,h8+'\n### 2026-10-16 양손 대응·초기 추적 보완 검증 — 세 장면 회복, 추가 잔여 결함 상호 검토\n'+summary+'\n근거 `ops/'+U.name+'/RESULT_KO.txt`, `DECISION.json`, `COMPLETION.json`. 원본/이전후보/모델 보존.\n',1)
text=text.replace(h9,h9+'\n### 2026-10-16 다음 — 회복 결과와 몸 손목 없는 후보 경쟁·진입 손 정책 검토\n'+next_action+'\n사용자의 단계별 결과 검토 후 다음 단계 진행 원칙 유지.\n',1)
cp.write_text(text,encoding='utf-8')
hp.write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}. 근거리 웹캠 상반신/화면 밖 손 조건에 맞춘 입력 안정화 중.

## 사용자 진행 방식
기능별 작업 → 결과·문제 보고 → 함께 검토 → 다음 단계. 최신 요청: 믿다·냉커피 보류를 단어 예외 없이 보완하고 다른 문제까지 보고, 학습 전 검토.

## 마지막 완료
{summary}
단위 `ops/{U.name}`. 구현 root, 독립 검증 assignment_verify/raw_capture/assignment_review.
새 규칙은 strong temporal+pose anchor와 남은 손 자체 근거의 공동 대응, 속도 미확정 초기100ms/2프레임의 제한적 시간 처리다. 실제 영상 결과 전 stale 반대손 예측 반례를 고쳤다. 반대손도 최근 관측이어야 하고 마지막 실제 위치/예측 모두에서 제외돼야 한다. 없는 좌표 보간/생성0. 기존8수치상수 유지, 단어 이름/촬영자 분기0.
기존12영상의 운영 대비 제외8→5는 인식률이 아니다. 추가8영상16조건은 이전후보와 동일하지만 운영 대비 좌우2/새배정4/제외8이며 소복 정상손 제외 등 기존한계가 있다. 전체40조건6,720검출의 새변경은 목표3회복뿐.
믿다 native72/crop55·57, 냉커피crop51·53 등의 반대손은 raw 자체가 없다. 배정 복구와 MediaPipe 검출/손가락 품질을 혼동하지 않는다.
근거: RESULT_KO.txt, INDEPENDENT_VERIFICATION.json, CHANGE_AUDIT.json, EMPIRICAL_REVIEW.md, ADDITIONAL_VISUAL_REVIEW.md, DECISION.json, COMPLETION.json.

## 다음 한 작업
{next_action}
현재 단위는 검증 완료 후 사용자 검토 대기. 새학습/899개 재추출 미실행. 원본과 이번 결과를 덮어쓰지 않는다. 사용자자료 학습·튜닝·후보선택·임계값 조정 금지.

## 작업 목록과 보존 기준
`Docs/WEBCAM_IMPROVEMENT_TODO.md`: 1단계 완료,2단계 목표수정완료/추가잔여문제 검토,3단계 전체자료준비·4단계 학습 대기. DTW 필수손·웹재시험 별도.
TCN `runs/word100_tcn_v1/whole_context/best.pt`, SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 학습9명899/개발3명300,96.67%는 개발검증. 보류10/14–18 미사용. 사용자12회 Top1 6/12,Top5 8/12는 개발참여자1인 실사용.
이번 TRAIN8원본부재는 W:ZIP에서8멤버36.002MB만 새unit에복사,16조건원검출86.66초로완료. 기존native배열8/8exact. 추출프로세스종료,중복실행불필요. 학습/분류추론/카메라/장치/다운로드/외부전송/Git변경/운영웹·모델교체0. 발표PDF상태유지.
''',encoding='utf-8')
todo=tp.read_text('utf-8').replace('| 후속 수정·검증 완료 / 잔여 보류 상호 검토 대기 |',
    '| 목표3건 보완·검증 완료 / 추가 표본 잔여 결함 상호 검토 대기 |',1)
todo+='\n\n## 2026-10-16 2단계 양손 대응·초기 추적 후속 결과\n\n'+summary+'\n\n다음: '+next_action+'\n\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`.\n'
tp.write_text(todo,encoding='utf-8')
artifacts=[p for p in U.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'before' not in p.parts]
new(U/'COMPLETION.json',{**decision,'artifacts_sha256':{str(p):sha(p) for p in artifacts},
    'documents_sha256':{str(p):sha(p) for p in [cp,sp,tp,hp]}})
for p,h in read(U/'PROTECTED_FILES.json').items():assert sha(Path(p))==h,p
for p in [sp,U/'DECISION.json',U/'COMPLETION.json']:read(p)
print('Recorded evidence and four state documents:',now)
