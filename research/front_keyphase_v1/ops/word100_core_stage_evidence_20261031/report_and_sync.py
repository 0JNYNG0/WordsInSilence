from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,urllib.request
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
ui=read(U/'UI_VERIFICATION.json');verify=read(U/'review/PHASE_VERIFICATION.json');protected=read(U/'review/PROTECTED_CHECK.json')
assert ui['status']=='PASS' and protected['status']=='PASS' and protected['old_report_unchanged']
assert verify['status']=='PASS_STORED_ARITHMETIC_AND_BOUNDED_CONTRACT' and verify['checks']['failed']==0,verify.get('status')
assert ui['report_sha256']==sha(U/'index.html')
server=json.load(urllib.request.urlopen('http://127.0.0.1:8791/health',timeout=3));assert server['read_only'] and not server['model_inference']
summary=read(U/'PHASE_DATA.json')['summary'];projection=read(U/'STAGE_PROJECTION.json')['summary']
report='''이번 작업: 영상 화면 복구 + 준비/핵심 후보/마무리의 시간 근거와 설명 연결

화면: http://127.0.0.1:8791/
이전 비교 화면: http://127.0.0.1:8791/previous.html

지금 분석하는 것
1) MediaPipe가 원본 전체 화면에서 손·몸 관절을 찾는다. 그 뒤 일정한 상반신 범위의 관절만 모델 입력에 사용한다.
2) 어깨를 기준으로 손목·손바닥·팔꿈치 위치와 시간에 따른 변화량을 만든다. 못 찾은 점은 관측 없음으로 유지한다.
3) 공개 RGB 표본을 직접 확인해 움직임·손 모양·팔과 몸의 관계를 임시 답안으로 기록한다. 가림·경계·접촉 불확실성은 함께 남긴다.
4) 이전 작은 실험은 오른손 상승/하강/기타를 분류했다. 31개 지원되는 학습 구간, 41개 숫자 입력, 160epoch×3seed였다. 새 모델은 알려준 구간에서 7/16, 단순 기하 방식은 8/16이므로 채택하지 않았다.
5) 설명 문장 전체나 RGB 픽셀을 이 작은 모델에 학습시킨 것은 아니다. 설명은 별도의 연결 단계에서 비교했다. 기존 C/T3-F 단어 인식은 유지한다.

이번에 실제 완료한 것
- 공개4표현16영상(참고12/개발4), 전체2084프레임을 보존했다. 영상별48표본=768시점을 검토했고 필요한 원본 장면은 확대 확인했다. 전 원본프레임의 사람 정답은 아니다.
- 110개 물리 구간 초안: 대기32/준비16/핵심 후보45/복귀16/불확실1. 각 core는 표본이 뒷받침하는 안쪽 범위이며 확정된 언어적 필수 구간이 아니다.
- 설명11항목의 출처를 확인하고, 19개 구간에 48개의 구간-설명 연결을 기록했다. 실제 확인한 anchor 시점에 한정하며 구간 전체의 정답으로 확대하지 않는다.
- 좌표 존재 비율과 RGB에서 보이는 특징을 분리했다. 위치가 검출돼도 손가락 모양·접촉을 확인했다고 주장하지 않는다.
- 기존 저장 DTW 12쌍을 재사용해 세 참고의 단계 역할이 같은지 확인했다. 624프레임 모두 유지: 470합의/154보류. 양쪽 초안 역할이 알려진444중442가 같은 큰 역할이다. 이것은 정확도나 새 평가가 아니다.
- 핵심 후보179프레임만 따로 보면135동일/1차이/43보류다. 역할이 같아도 세부 순서·모양·중요 동작 정체성이 맞는지 아직 별도로 확인해야 한다.

발견/해결한 문제
- 순종 초기 얼굴 앞 펼친 손이 연결한 사전의 후반 주먹·옆팔 동작과 다르다. 초기 동작을 준비라고 지우지 않고 설명되지 않은 동작으로 보존했다. 손목 상승과 팔꿈치 상승도 구분했다.
- 검산 중 불확실 역할 1프레임이 비교 가능한 답안으로 잘못 집계되는 오류를 찾아 수정했다. 최초 결과/코드를 보존했고 3개 고유 참고 검사도 추가했다.
- RGB 주석의 가림·접촉 한계가 파생 자료에서 빠지지 않도록 원본 제한을 모두 보존했다.
- 영상은 공개 원본의 모든 프레임을 유지한 960×540 WebM 표시본으로 제공한다. 모델 입력·원본은 그대로다. 16개의 프레임 수와 시간 오차≤1ms를 검증했다. 최초1개 표시본 시간오차16.7ms는 별도보존 후 원본PTS를 고정해 수정했다.
- 글자·배경·입력창·그래프 색을 명시했고 같은 로컬주소로 영상을 제공한다. 26개 Chrome 확인에서16영상+이전4쌍 로딩/탐색,양쪽재생,밝고어두운테마,좁은화면을 통과했다. 실제 앱 자동화는 런타임 초기화 오류로 실패했으므로 그 앱에서의 원래 고장 원인까지 확정했다고 하지는 않는다.

현재 한계와 다음 단일 작업
새 학습0회, C/T3-F/운영웹 변경0, 사용자/멘토자료 접근0. 단위 작업은 완료했으나 전체 교육 피드백AI는 완료가 아니다.
다음은 지금 근거가 있는 구간에서 공통 손 모양·팔/몸 관계와 앞뒤 순서를 하나의 기록으로 연결하는 것이다. 예: 반대팔에 접근→팔을 따라 이동→복귀. 참고12에서 공통 항목·관측 제한을 고정하고 개발4에서 중요한 단계가 바뀌거나 빠졌을 때의 비교를 진단한다. 단어별 예외코드로 정답을 강요하거나 이번 DEV 결과에 맞춰 경계를 바꾸지 않는다. 새 판정부 학습은 근거가 있는 항목만 대상으로 하며 설명에 없는 동작을 정답으로 만들지 않는다. 출시의 교육적 정오 검토와 현재의 개발 검증은 구분한다.

구현: root. RGB 초안: rgb_labels_a/b. 출처 연결: guide_stage_map. 별도 산술검산: rgb_labels_b.
동일 Codex의 표본 검토/별도 검산은 독립된 한국수어 전문가 정답이 아니다.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
completion=dict(status='COMPLETE_BOUNDED_PHASE_EVIDENCE_AND_VIEWER_REPAIR',at=now,summary=summary,projection=projection,new_training=0,new_inference=0,new_dtw_search=0,production_changed=False,personal_data_used=False,report_url='http://127.0.0.1:8791/',report_server=dict(pid=server['pid'],port=8791,purpose='read-only public presentation; intentionally retained for user'),verification_files=['UI_VERIFICATION.json','review/PHASE_VERIFICATION.json','review/PROTECTED_CHECK.json','DISPLAY_MEDIA.json'],pins={n:sha(U/n) for n in ['PHASE_DATA.json','STAGE_PROJECTION.json','SOURCE_PHASE_CONTRACTS.json','integrate_evidence.py','index.html','previous_fixed.html','RESULT_KO.txt','AUTHORIZATION.json']})
write(U/'COMPLETION.json',completion)
write(U/'CURRENT_EXECUTION.json',dict(status=completion['status'],active=None,training_started=False,read_only_report_server=completion['report_server'],url=completion['report_url']))
statepath=ROOT/'research/front_keyphase_v1/ops/STATE.json';state=read(statepath);state['updated_date_kst']=now;state['current_stage']='CORE_STAGE_EVIDENCE_AND_VIEWER_REPAIR_COMPLETE';state[U.name]=completion;write(statepath,state)
section8='''### 2026-10-31 動作段階の根拠と表示修正完了 — 新規学習なし

- `word100_core_stage_evidence_20261031` 완료. 공개4표현16영상2084원본프레임보존. RGB48표본×16/필요원본확대검토로110구간(핵심후보45)초안. 설명11항목→19구간/48구간-설명연결,anchor시점한정/언어필수성X.
- 기존저장DTW12쌍역할투영: 개발624중470합의154보류,양쪽알려진초안444중442동일. 핵심후보179중135동일1차이43보류. 같은큰역할이며denseGT/세부단계정확도/새평가X. 불확실1프레임known분모오류최초보존후수정,rawunknown제한보존. 별도검산2900검사불일치0.
- 순종초기얼굴앞펼침은연결사전후반주먹·옆팔과미일치,source-uncovered로보존/준비자동삭제X. 손목상하와팔꿈치상하구분. 지금작은모델은이전31창41수치→R상승하강기타160epoch×3seed이며글/RGB픽셀학습X. 이번새학습0.
- 화면동일originHTTP8791,16개WebM표시본원본프레임/PTS≤1ms검증,명시밝은배경/짙은글자. Chrome26확인PASS/원본1프레임버튼/표시용DTW대응·원본범위보존. CUA앱runtime초기화실패로원래앱고장원인미확정. 기존보고/보호32해시보존. 서버는사용자열람용읽기전용으로의도적유지(PIDはCURRENT_EXECUTION参照).

'''.replace('動作段階の根拠と表示修正完了 — 新規学習なし','동작 단계 근거·영상 화면 수정 완료 — 새 학습 없음').replace('PIDはCURRENT_EXECUTION参照','PID는 CURRENT_EXECUTION 참조')
section9='''### 2026-10-31 다음 단일 작업 — 손 모양·몸 관계·순서의 공통 근거 연결

완료단위 `word100_core_stage_evidence_20261031`의 PHASE_DATA/SOURCE_PHASE_CONTRACTS를재사용한다. 참고12에서관측지원되는손모양·반대팔/몸관계·앞뒤순서의공통항목을정리하고,개발4는사후진단만한다. 핵심후보전체를정답으로보간하지말고직접확인anchor/가림제한을유지한다. 순종source-uncovered앞선동작은삭제·의미강제X. 단어특례/DEV점수튜닝/새모델순회/개인자료금지. 이번단계전체학습완료로오해금지. 새판정부학습전항목지원검증과공통규칙동결,현재개발을전문가출시검토대기로멈추지않는다. HTTP8791읽기전용열람서버유지중이므로중복bind/운영서버교체금지. 기존C/T3F/운영/과거동결결과유지.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf-8-sig')
head8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';head9='## 9. 다음 단일 작업'
assert head8 in text and head9 in text
if section8.splitlines()[0] not in text:text=text.replace(head8,head8+'\n\n'+section8,1)
if section9.splitlines()[0] not in text:text=text.replace(head9,head9+'\n\n'+section9,1)
context.write_text(text,encoding='utf8')
handoff='# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고 중요한 차이와 허용 가능한 차이를 구분한다.\n\n'+section8+section9+'\n현재단위: '+U.name+'; 분석 active=null, 로컬 보고서 서버는 열람용 실행 중. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 서버 exec9048/PID'+str(server['pid'])+', http://127.0.0.1:8791/ . 프로세스 미관측만으로 중복실행하지 말고 health 확인. 운영8765/8767미수정. 신규download/camera/external/Git/private0. 모델재학습/재추론/DTW재탐색0.\n\n근거: COMPLETION.json, RESULT_KO.txt/html, PHASE_DATA.json, SOURCE_PHASE_CONTRACTS.json, STAGE_PROJECTION.json, UI_VERIFICATION.json, review/PHASE_VERIFICATION.json. 원본·자료분리·32보호해시유지. 같은Codex의별도검산을독립사람수어정답으로쓰지않음. UI실패/시간PTS오류/unknown분모오류는첫흔적보존후검증복구.\n'
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
print('COMPLETE synced',U.name,'server',server['pid'])
