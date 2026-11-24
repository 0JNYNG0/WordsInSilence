"""Record completed evidence and the next single action; preserve historical entries."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

OPS=Path(__file__).resolve().parent
WEB=OPS.parents[1]
ROOT=WEB.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
now=datetime.now(timezone(timedelta(hours=9)))
stamp=now.strftime('%Y-%m-%d %H:%M KST')
native=read(OPS/'native_pair_result_audit.json')
followup=read(OPS/'followup_validation_LOCAL_ONLY.json')
freeze=read(OPS/'real_checkpoint_freeze_audit.json')
assert native['passed']==native['total']==45
assert followup['passed']==followup['total']==18
assert freeze['passed']==freeze['total']==11
next_action=('Reuse completed head-only pair and all audits. Freeze the existing visibility parent and both new head-only checkpoints. '
 'Prepare a concrete fixed self-participant fresh-capture comparison using the existing web capture path: all18 words, predeclared repetitions/order, '
 'same new observations scored by all three frozen models, project18 primary and full93 secondary, all failures retained. '
 'Separate the +9 within-pair effect from preservation versus the parent. Do not start camera or claim fresh captures until the user is ready. '
 'Do not retrain/search seeds, epochs or thresholds on the same development records. User recordings stay evaluation-only; other participants deferred. '
 'No promotion, new augmentation, heldout reads, per-word rules, downloads, inquiry, deletion, backup, Unity builds or schedules. '
 'Remaining multi-person WORD data gaps stay explicit, not a universal raw-video-download prerequisite.')
state_path=WEB/'ops/STATE.json'
state=read(state_path)
state['updated_date_kst']=stamp
state['updated_at']=now.isoformat()
state['current_stage']='HEAD_ONLY_PAIR_COMPLETE_PARTIAL_RETENTION_NO_PROMOTION'
state['next_single_action']=next_action
state['participant_evaluation_clarification_20260919'].update({
    'user_clarification':'User records not direct training inputs; observed failures inform development of common methods. Includes preprocessing, public-data training treatments and training strategy, not preprocessing alone.',
    'common_method_intent_is_not_multisigner_effect_proof':True})
stage={'status':'COMPLETE_MIXED_EFFECT_NO_PROMOTION','report':str((OPS/'RESULT.md').relative_to(ROOT)).replace('\\','/'),
 'parent':'union93_visibility_v1','training_runs':2,'steps_per_arm':660,'execution_session':54049,'execution_exit_code':0,
 'claude_execution_exit_code':0,'synthetic_checks':[23,23],'native_audit':[45,45],'followup_audit':[18,18],'saved_encoder_freeze_audit':[11,11],
 'frozen_encoder_parameters':34,'frozen_encoder_buffers':24,'changed_parameter_keys':['head.weight','head.bias'],
 'native699_correct_control_treatment':[689,692],'native_source696_correct_control_treatment':[689,689],
 'native_apology3_correct_control_treatment':[0,3],'target18_restricted33_correct_control_treatment':[30,32],
 'target18_actual_gt_classes':6,'target18_restricted_regression':'SEN0355 REAL11 thanks -> apology',
 'web_project18_correct_control_treatment':[16,17],'web_full93_correct_control_treatment':[15,16],
 'reviewed_masked5_correct_control_treatment':[5,5],'intact_public18_correct_control_treatment':[18,18],
 'unchanged_parent_same_web_correct_project18_full93':[17,17],
 'checkpoints':native['checkpoints'],'model_promoted':False,'user_records_trained':False,'heldout_used':False,'new_captures':0,
 'conclusion':'Reviewed mask retention recovered under frozen encoder, and added apology improves matched development inputs; parent superiority not established, restricted native thanks regression and web NA/full93 JADA errors remain.'}
state['stage_results']={'head_only_native_pair_20260919':stage,**state['stage_results']}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
text=context.read_text(encoding='utf-8')
current=f'''### {stamp} 최신 — 특징 추출부 고정 두 군 학습·독립 검증 완료

- Claude 실제 실행 종료0로 새 독립 head-only 연구 함수를 받고 합성23/23 후 두 군 각660step 학습했다(세션54049 종료0). 저장 체크포인트에서 encoder 파라미터34+버퍼24개 완전 동일, head.weight/bias만 변경11/11 확인. 기존 동결 러너/production 변경·새 증강·사용자 학습0.
- 제공 개발699:689→692, 기존696:689→689(예측 동일), 사과3:0→3. 목표18 제한33건은30→32이며 실제GT6종이다. 감사합니다1건→죄송하다 회귀가 있으므로 전체93 결과만으로 회귀 없음이라 하지 않는다.
- 본인 개발18: 제한18은16→17, 전체93은15→16. 죄송하다 개선, 나 오답 유지/전체93 자다 오답 유지. 기존 visibility 동일입력 저장 결과17/18 양 범위에 대한 우월성은 없어 승격0. 공개원본18/18·기존 누락5조건5/5는 양 군 유지됐다. 이 부분 보완 유지와 다인 일반화·수행 판정을 구분한다.
- 새 결과45/45+후속18/18 검증 완료. 사용자 오류를 범용 처리 개발에 참고하는 것은 맞지만 전부 전처리는 아니며 공개자료 학습 처치/학습전략도 포함된다. 사용자는 학습 미참여자1명의 유효한 사례, 기존 기록은 개발 노출 자료라는 구분을 유지한다. 상세 `research/front_keyphase_v1/ops/head_only_native_pair_20260919/RESULT.md`.

'''
anchor='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+current,1)
next_text=f'''### {stamp} 다음 단일 작업 — 고정 후보의 본인 새 촬영 비교 절차 준비

- 이번 head-only 두 후보와 기존 visibility 부모를 동결하고 기존 웹 촬영 경로로 전체18단어·고정 반복 수/순서의 새 촬영 비교 절차를 구체화한다. 같은 새 입력을 세 모델에 적용해 사과9건 추가 효과와 부모 대비 유지 여부를 분리한다. 기본 모델 교체 없이 가능한 평가 경로를 사용한다. 사용자 준비 전 카메라 활성화/촬영 완료를 주장하지 않는다.
- 기존 사용자 기록에서 seed/step/임계값을 탐색하며 맞추는 후속 학습을 반복하지 않는다. 현재 합성23/실제45/후속18/고정11 검사와 저장 결과를 재사용한다. 본인 새 촬영은 같은 사람의 새로운 수행에 대한 재현성이며 다른 사람 시험은 나중이라는 사용자 결정을 유지한다.
- 사용자 기록 비학습·촬영자 분리·보류 보호·전체18목표·Claude 구현 역할을 유지한다. 단어별 예외·새 증강·기본모델 승격·추가 대형다운로드·신청/문의·삭제·백업·Unity빌드·예약0. 다인 자료 부족은 별도 확보 과제로 유지하며 원본 전체 확보를 다시 일률적 조건으로 두지 않는다.

'''
anchor='## 9. 다음 단일 작업\n\n'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+next_text,1)
context.write_text(text,encoding='utf-8')

handover=ROOT/'research/HANDOVER_2026-09-18.md'
text=handover.read_text(encoding='utf-8')
anchor='<!-- current-continuation-20260919 -->\n'
assert text.count(anchor)==1
update=f'> {stamp} 최우선 후속: Claude 새 head-only 함수 실제 작성·합성23/23 후 같은 자료 두 군660step 완료(54049 종료0). encoder 파라미터34+버퍼24 불변/분류층2개만 변경11/11. 제공699:689→692, 기존696 동일689, 사과0→3; 목표18 제한30→32에는 감사합니다 회귀1건. 본인 제한18:16→17/전체93:15→16, 누락5조건은 양 군5/5 유지. 부모 웹17/18 양 범위에 대한 우월성은 없어 승격0. 결과45/45+후속18/18 재사용. 다음은 기존 visibility 부모/이번 두 후보를 고정한 본인 새 촬영 비교 절차 준비이며 같은 기록에 맞춘 재학습/탐색은 하지 않는다. 사용자 준비 전 새 촬영/카메라 실행0. 상세 ops/head_only_native_pair_20260919/RESULT.md 및 실제STATE/루트8·9절 우선.\n\n'
handover.write_text(text.replace(anchor,anchor+update,1),encoding='utf-8')
assert read(state_path)['current_stage']=='HEAD_ONLY_PAIR_COMPLETE_PARTIAL_RETENTION_NO_PROMOTION'
assert current in context.read_text(encoding='utf-8') and update in handover.read_text(encoding='utf-8')
print(json.dumps({'updated_at_kst':stamp,'state_context_handover_synchronized':True,'new_training_or_inference_in_update':False}))
