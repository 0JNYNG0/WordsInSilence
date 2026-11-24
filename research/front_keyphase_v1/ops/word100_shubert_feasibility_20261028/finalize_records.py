from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,shutil
U=Path(__file__).resolve().parent
root=U.parents[3]
assert root.name=='WordsInSilence'
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
evidence='research/front_keyphase_v1/ops/word100_shubert_feasibility_20261028'
status='SHUBERT_TECHNICAL_FEASIBILITY_COMPLETE_KSL_FEEDBACK_UNVALIDATED'
current=('사용자 승인 후 공식 SHuBERT+손/얼굴 DINO 3가중치 약1.283GB 확보, 기존 CPU환경에서 공개4표현/3기록수행자/12영상 1438프레임 특징 추출을 완료했다. '
 '공식 코드와 별도 계산 검산11건(실제V01포함) 최대차이0, 저장된 특징·시점·출처·해시 검산PASS. '
 '별도 CPU지연3영상은14.475~17.873초(중앙16.454), 대부분3DINO처리이며 원영상decode/MP검출/DTW/피드백표시는 제외했다. '
 '같은프로세스반복차이0이나 별도실행 최종특징최대차이4.20e-5. 2876손프레임 중112이전crop복사는 현재관측아님으로보존. '
 '상반신masked자료는보존만하고 이번모델입력은native30fps라 웹과동일한가림조건비교가아니다. '
 '연구우선후보유지, CPU경로즉시제품채택보류. KSL속성/단계교정정확도미측정·학습0·운영변경0·개인/멘토자료접근0. '
 '사용자는 재생중분석을허용했으므로3초하드탈락기준은해제했지만 실제추가대기는미검증이다.')
next_action=('현재 공식파일과저장특징을재사용해 실제보이는KSL손모양·관계·변화순서와관측부족을구분하는작은평가기준을확인한다. '
 'AI초안/사전대표속성을영상의독립정답으로복사하지않는다. '
 '웹상반신사용범위와결측보류, 원30fps/논문사전학습약15fps조건을미리고정해짧은변화보존을비교한다. '
 '검증된관측정답이있을때만 작은속성/단계판정부를학습하여기존관절기준과비교; 교육적허용경계는별도확인한다. '
 '품질이득조건에대해Intel가속/이미지특징계산/촬영·재생중분산처리를검토하고촬영종료부터실제피드백표시까지측정한다. '
 '기존다운로드/완료추론재실행금지(새기준비교/검산은별도동결). 새모델자동순회금지, C/T3F/운영/개인자료보존.')
report=U/'RESULT_KO.html'
assert report.exists() and (U/'FEATURE_VERIFICATION.json').exists()
before=U/'before_final_docs';before.mkdir(exist_ok=True)
paths=[root/'WordsInSilence_PROJECT_CONTEXT.md',root/'NEW_CHAT_HANDOFF.md',root/'research/front_keyphase_v1/ops/STATE.json']
for p in paths:
    target=before/p.name
    if not target.exists():shutil.copy2(p,target)
context=paths[0].read_text(encoding='utf-8')
title='### 2026-10-28 SHuBERT 공개 실행 완료 — CPU 병목·한국수어 교정 미검증'
if title not in context:
    for sec,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',current),('## 9. 다음 단일 작업',next_action)]:
        assert context.count(sec)==1
        context=context.replace(sec+'\n',sec+f'\n\n{title}\n{now}\n{body}\n근거: {evidence}/RESULT_KO.txt, FEATURE_VERIFICATION.json, LATENCY_ISOLATED.json, FEASIBILITY_REVIEW.txt.\n',1)
    paths[0].write_text(context,encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

{now}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 사용자가 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고, 중요한 차이와 허용 가능한 차이를 구분해 알려준다.

현재 단위: {evidence}
현재 상태: {status}

{current}

사용자 최신 승인: 여러 번 검토해 SHuBERT가 적합하면 진행. 촬영후 사용자 스켈레톤 영상과 참고 재생 중 분석 가능;3초 엄격탈락 아님. 이 승인으로 공식SHuBERT 파일 확보/격리실행을 완료했다. 과거 Qwen/SHuBERT 다운로드대기 지시로 되돌리지 않는다. SignRep/Qwen/MiniCPM은 받지 않았다. 새 라이브러리설치0, 학습0.

사양: Samsung960XHA,CoreUltra7 258V,32GB,Arc140V통합GPU,NPU. 현재Torch2.12CPU;GPU가없다는뜻아님. Arc/NPU미시험.

완료 세션: acquisition74505 / publicfeatures30774 / adapteractual71386 / isolatedlatency93863 모두exit0. 입력준비와검산도완료. 진행중추론/학습/다운로드없음;FEATURE_RUNNING없고FEATURE_FINISHED있음. CURRENT_EXECUTION.json완료상태. 재다운로드나완료결과조용한재채점금지.

가중치: weights/checkpoint_836_400000.pt (SHA9087f47602401c07c44a9c70164ca1387373938ac4642149126a43e61c20cd7d), face_dinov2_checkpoint.pth, hands_dinov2_checkpoint.pth. 공식코드핀 vendor/shubert cc1929326075bfbad7ad73159b2acf84356059bb;vendor/dinov2 7764ea0f912e53c92e82eb78a2a1631e92725fc8. torch weights_only검사/strict키shape로딩완료. 상업배포가중치조건은확정하지않음.

입력: inputs/V01~V12는멀다/믿다/감사/별,기록ID1/2/3,1438연속프레임. 모든기존개발자료이고새독립시험아님. 공식과의차이=YOLO사람crop생략,기존native좌우배정/facecache사용,30fps보존. masked상반신조건은배열보존만,모델에는nativeRGB/body입력. source_kind2=직전손crop복사112건;현재관측정답으로쓰지않음. 모델자체는source_kind마스크를입력받지않음.

범위: 공식adapter별도검산11PASS와저장산술PASS는기술검증. AI관측초안은독립사람정답아님. 역순/중간제거로특징변화는수어오류이해가아님. timestamp-only속도시험은실제빠른동작검증아님. 정상손/교차보존은원입력보존이지검출정확도개선아님.

다음 단일 작업: {next_action}

구현/검산: root=공식확보/DINO/runner/결과검산/보고,shubert_fit_audit=CPUadapter/반대검토,shubert_adapter_verifier=공식소스독립검산,shubert_public_probe_design=공개입력/독립지연. 같은Codex별도검산을한국수어전문가정답으로혼동하지않는다.

근거: RESULT_KO.txt/html,AUTHORIZATION,FROZEN_FEASIBILITY,PUBLIC_INPUT_PROTOCOL,inputs/MANIFEST와VERIFY,ADAPTER_INDEPENDENT,PUBLIC_FEATURE_RESULT,FEATURE_VERIFICATION,LATENCY_ISOLATED,FEASIBILITY_REVIEW. 첫경로/FPS검사실패·네트워크실패는보존됐으며원인확인후복구.
'''
paths[1].write_text(handoff,encoding='utf-8')
state=json.loads(paths[2].read_text(encoding='utf-8'))
state['updated_date_kst']=now;state['current_stage']=status;state['next_single_action']=next_action
state['word100_shubert_feasibility_20261028']={
 'at':now,'status':status,'unit':str(U),'authorization':str(U/'AUTHORIZATION.json'),
 'report':str(report),'current_summary':current,'next_single_action':next_action,
 'public_primary_clips':12,'public_frames':1438,'training_runs':0,'operating_changes':0,'private_footage_access':0,
 'feature_verification':'PASS','adapter_independent_tests':11,'independent_human_semantic_ground_truth':False,
 'isolated_cpu_cached_feature_seconds':{'min':14.475,'median':16.454,'max':17.873},
 'latency_is_end_to_end':False,'runtime_installs':0,'downloaded_weight_bytes':1283198698,
 'held_or_reused_hand_crops':112,'current_hand_crops':2764,'model_input_roi_applied':False,
 'input_fps':30,'published_pretrain_average_fps':14.89,'active_execution':None,
 'evidence':['RESULT_KO.txt','ADAPTER_INDEPENDENT.json','FEATURE_VERIFICATION.json','LATENCY_ISOLATED.json','FEASIBILITY_REVIEW.txt']}
paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
execution=json.loads((U/'CURRENT_EXECUTION.json').read_text(encoding='utf-8'))
execution.update(at=now,stage=status,feature_session=None,feature_exit_code=0,active_sessions=[],
 completed_sessions={'acquisition':74505,'public_features':30774,'adapter_public_verification':71386,'isolated_latency':93863},training_runs=0)
(U/'CURRENT_EXECUTION.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2),encoding='utf-8')
print('Synced context sections 8/9, latest handoff, current STATE, CURRENT_EXECUTION')
