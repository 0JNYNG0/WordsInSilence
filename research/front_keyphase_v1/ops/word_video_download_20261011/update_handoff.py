import datetime as dt
import json
from pathlib import Path
import shutil

UNIT=Path(__file__).resolve().parent
PROJECT=UNIT.parents[3]
now=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
names=['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md','research/front_keyphase_v1/ops/STATE.json']
for name in names:
    src=PROJECT/name
    backup=UNIT/'before'/src.name
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists(): shutil.copy2(src,backup)

context=PROJECT/names[0]
s=context.read_text(encoding='utf-8-sig')
entry='''### 2026-10-11 WORD 원본 전체 다운로드 승인 — 시간 제한 해제·로그인 대기

사용자가 시간 제한을 명시적으로 해제하고 필요한 단어 영상 전체 수신을 승인했다. 기존 20시 중단 규칙은 이번 승인으로 대체된다. `ops/word_video_download_20261011/ACQUISITION_PLAN.json`·PREFLIGHT·RESULT_KO.txt. REAL/WORD ZIP33개/1.181TB 중 기존1개32.55GB를 W:에 복사·해시/CRC검증 중이며, 신규32개1.149TB는 로그인 대기다. 공식 큐와 파일ID/경로/크기 전부 일치, 가용 공간 및 최대파일3배 임시공간 확보 확인. 문장/SYN/CROWD/OpenPose 전체 제외, WORD 주석2개 재사용.
Chrome 공식 다운로드 클릭 후 로그인 요구 확인, 사용자 로그인 질문 전달. 새 영상 네트워크 수신은 아직0. 기존 ZIP복사 PID52200, 진행 `LOCAL_COPY_PROGRESS.json`, 완료는 `LOCAL_COPY_RESULT.json`을 확인한다. 원본 C: 보존. 기존소용량/시간제한으로 다운로드 승인을 재요청하지 않는다.
AIHub 영상→동일MediaPipe 추출로 학습조건 통일 제안. 성능 향상 미검증, 기존 사람분할 보존. NIKL18은 과거 학습/개발 노출이 있으므로 미사용 최종시험으로 재분류하지 않고 교차출처 참고로 사용한다. 기존NIKL노출체크포인트를 이어받고 AIHub-only라고 부르지 않는다. 새 모델학습/운영교체 없음.

'''
marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'
if entry.splitlines()[0] not in s: s=s.replace(marker,marker+entry,1)
next_entry='''### 2026-10-11 다음 — 로그인 후 승인된 WORD32개 순차 다운로드

사용자 최신 지시로 시간 제한 해제, REAL/WORD 원본 전체 확보 승인. `ops/word_video_download_20261011`의 큐/저장경로/용량 검산 완료. Chrome AIHub 로그인 및 데이터셋 접근 상태 확인 후 신규32개1.149TB를 W:에 수신한다. 기존1개 로컬 복사는 실제 PID/LOCAL_COPY_PROGRESS/RESULT 먼저 확인하여 중복 실행하지 않는다. 한 파일씩 임시저장·크기/ZIP/CRC 확인 후 완료, 기존 원본·불완전 수신 증거 보존. 과거의20시중단 문구는 이 최신승인에 적용하지 않는다. 다운로드 완료 후 정면 원본과 주석·사람 역할을 대조하고 동일MediaPipe/RGB 준비를 이어간다. 새모델/국립18시험역할/최종어휘 자동확정 없음.

'''
marker='## 9. 다음 단일 작업\n\n'
if next_entry.splitlines()[0] not in s: s=s.replace(marker,marker+next_entry,1)
context.write_text(s,encoding='utf-8')

statefile=PROJECT/names[2]
state=json.loads(statefile.read_text(encoding='utf-8-sig'))
state['updated_date_kst']=now
state['current_stage']='WORD_BULK_DOWNLOAD_AUTHORIZED_LOGIN_REQUIRED_LOCAL_COPY_RUNNING'
state['next_single_action']='User removed time cutoff; finish verified local copy, then authenticate and download approved remaining32 REAL/WORD video archives to W:. No training or model replacement.'
state['word_video_download_20261011']={'status':'QUEUE_READY_LOGIN_REQUIRED_LOCAL_COPY_RUNNING','at_kst':now,'unit':'ops/word_video_download_20261011','plan':'ops/word_video_download_20261011/ACQUISITION_PLAN.json','preflight':'ops/word_video_download_20261011/PREFLIGHT.json','time_cutoff':None,'new_download_count':32,'new_download_bytes':1148671327122,'network_video_bytes':0,'local_copy_pid':52200,'local_copy_progress':'ops/word_video_download_20261011/LOCAL_COPY_PROGRESS.json','local_copy_result':'ops/word_video_download_20261011/LOCAL_COPY_RESULT.json','signin_required':True,'browser_tab':'199984604','training':0,'production_replacement':False,'nikl18_role':'previously exposed cross-source reference; not untouched final test'}
statefile.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

handoff='''# WordsInSilence 최신 인수인계
갱신: NOW. 시작: AGENTS.md, PROJECT_CONTEXT 최신 §3·§8·§9, WORKFLOW_HANDOFF.md.

## 현재 작업 — AIHub 독립 단어 영상 전체 확보
사용자가 시간 제한을 명시적으로 해제하고 필요한 단어 자료 전부 다운로드를 승인했다.
과거 20시 이후 중단/대용량 미승인 문구는 현재 지시가 아니다. 같은 승인을 재요청하지 않는다.
단위 `research/front_keyphase_v1/ops/word_video_download_20261011`.
ACQUISITION_PLAN.json / PREFLIGHT.json / RESULT_KO.txt. 공식큐33개 크기/ID/경로 검산 PASS.
REAL/WORD 원본 ZIP33개1.181TB, 기존1개32.55GB는 C: 원본을 보존하며 W:로 복사 중.
LOCAL_COPY_PROGRESS.json/PID52200 확인, LOCAL_COPY_RESULT.json 있어야 복사·SHA256·CRC 완료.
신규32개1.149TB는 AI Hub 로그인이 필요하여 아직 수신0. Chrome탭199984604 로그인 화면 유지,
사용자에게 로그인/보안문자 직접 입력 후 알려달라고 질문했다. API 키/비밀번호를 채팅에 요구하지 않았다.
공식 aihubshell 소스만 확보/검토, 실행0. 실제 다운로드 방식은 로그인/승인 확인 후 결정한다.
저장: W:/WordsInSilence/data/AIHub_103/videos, 큐 사본 downloads_20261011/ACQUISITION_PLAN.json.
가용3.000TB 확인, 전체 압축본+최대파일3배 임시 공간 확보. 파일별순차 수신/검증, 미완료 파일 보존.
문장SEN/합성SYN/CROWD/OpenPose 전체 제외. 이미 받은 WORD 주석2개 재사용.

## 다음 한 작업
로그인 답변 및 실제복사 PID/진행/완료를 확인하고 중복 실행 없이32개 다운로드로 이어간다.
파일별 크기/ZIP/CRC 확인 전 완료 주장 금지. 계정 접근/사이트 이용조건은 별도 확인한다.
수신이 시작되면 실제 프로세스·로그·파일 증가를 확인하고 이 문서와 STATE·§8/9를 갱신한다.
완료 후 정면 영상과 주석·사람역할 대응 확인, 같은 MediaPipe/RGB 추출 준비.

## 자료·평가 조건
AIHub 영상에서 웹과 동일 MediaPipe로 추출하는 학습 통일 방향은 타당한 비교안이며 정확도향상 미검증.
NIKL18은 과거학습/반복개발노출: 교차출처 참고/회귀확인, 깨끗한 최종시험으로 재분류 금지.
NIKL에 노출된 기존체크포인트에서 학습을 이어가면서 AIHub-only라고 부르지 않는다.
사람01~09학습/11~13개발/14~16보류/17~18공식Validation 유지,10미배정.
사용자학습·튜닝·후보선택/자동TOP100/운영교체/카메라/Git변경 없음.

## 완료된 자료 조사 — 반복하지 않음
`ops/word_distribution_20261011`: 코드3000×18정면주석54000, 보류83후53917.
코드별18건2917개/17건83개. 원문표기2965는 의미클래스확정 아님, 같은수형자동통합 없음.
18ID+코드내동일표기2659코드. 전체순위 W:/WordsInSilence/data/AIHub_103/distribution_20261011.
`ops/word_acquisition_inventory_20261011`: 기존ZIP에 REAL01/WORD1501~3000/5각도7500개.
정면1500중주석통과1499. 고정12개54.32MB W: pilot12에 추출,1444프레임디코딩/시각/해시검증완료.
이12개는 입력기술점검용이며 추천단어·새사람시험 아님. 관절/RGB 추출은 아직 안 함.
원본전체시간열 보존, 주석구간 별도 인덱스. 대표백영상 손올림/내림확인, 전수경계의미검증 아님.
과거 'AIHub 전체가 문장전용' 설명은 잘못이었고 정정완료. WORD와SEN 모두존재.

## 보존·기준
WIS_DATA W: GPT/NTFS3TB. 재포맷 금지. 두번째HDD 보류. 모두의말뭉치 실제자료 승인/수신 미확인.
현재TCN832·웹기본 모델 불변. RGB후보미채택, 정확도향상 아직 없음. 기존 실험재실행 없음.
Codex가 승인범위로컬구현·실행·검증·문서 가능. Claude불가시 역할승인 반복질문 금지.
st-gcn 과거 자동화완료/PAUSED 유지. 현재 다운로드 자동화는 아직 생성하지 않았다.
'''.replace('NOW',now)
(PROJECT/names[1]).write_text(handoff,encoding='utf-8')
print(json.dumps({'updated':names,'at_kst':now,'new_video_download_started':False}))
