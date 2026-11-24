"""Sync measured completion and next bounded task; preserve prior history."""
from pathlib import Path
import json,shutil,hashlib
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;OPS=U.parent;ROOT=OPS.parent.parent.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    now=datetime.now(timezone(timedelta(hours=9))).isoformat()
    verify=read(U/'VERIFICATION.json');ind=read(U/'INDEPENDENT_REVIEW.json');ui=read(U/'REPORT_UI_VERIFICATION.json');execution=read(U/'EXECUTION_COMPLETE.json')
    assert verify['status'].startswith('PASS') and ind['status'].startswith('PASS') and ui['status']=='PASS' and execution['active'] is None
    assert not (U/'RUNNING.json').exists()
    current=dict(status='COMPLETE_ISOLATED_WEB_REPLAY_NOT_PRODUCTION',at=now,active=None,session_id=91452,exit_code=0,server_closed=True,chrome_closed=True)
    save(U/'CURRENT_EXECUTION.json',current)
    completion=dict(**current,frames=1256,runs=12,unique_public_clips=3,repeat_per_mode=2,
      median_display_ack_seconds=dict(after_end=4.51343605,during_recording=.41136695),
      verified='Both modes exact observations/assignment/probabilities/DTW payload; original PTS/reference denominators preserved',
      evidence=verify['status'],independent_checks=ind['checks'],independent_truth='Separate Codex code and saved-array audit, not human KSL correctness',
      ui=ui['status'],not_included=['real webcam','SignRep video inference','C/T3F word inference','full-body educational correction','video ready time at every ack'],
      protected_models_unchanged=True,personal_data_used=False,training=False,production_changed=False,
      residual='Fixed old support gate retained for equivalence; finer evidence gate and false-correction/semantic phase validation still needed',
      first_failure='CDP fixed port risk fixed pre-run; nonfatal video transfer cancellation preserved, safe_server added separately')
    save(U/'COMPLETION.json',completion)
    paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']
    before=U/'before';before.mkdir(exist_ok=False)
    for p in paths:shutil.copy2(p,before/p.name)
    state=read(OPS/'STATE.json')
    state['updated_date_kst']=now;state['current_stage']='ISOLATED_STREAMING_WEB_REPLAY_COMPLETE_NOT_PRODUCTION'
    next_action='고정 관절 기반 비교에 최신 특징별 관측지원 기준을 연결하고 fit/dev 공개자료의 정상 손모양·손내림·교차·미관측에서 잘못된 차이 후보가 줄어드는지 검증. 의미단계/필수반복은 확인된 주석 범위만. 완료12속도실행·소모평가 재실행/튜닝금지. 실제웹캠·운영교체 별도.'
    state['next_single_action']=next_action
    state[U.name]=dict(completion,unit=str(U).replace('\\','/'),report=str(U/'RESULT_KO.html').replace('\\','/'),next_single_action=next_action)
    save(OPS/'STATE.json',state)
    status='''### 2026-10-31 격리 웹 재생 연결 완료 — 같은 비교결과, 표시 확인 대기 단축

- 현재 단위 `word100_streaming_web_replay_20261031`. 사용자 범위질문에 현재10개 물리특징(검지/네손가락펴기,손/얼굴겹침,손이동방향)이며 전체수어/전신교정 아님을명시. 공개 P005수어/P043멀다/P079별3개×2회×종료후/촬영중2조건=12회1256프레임. 두조건단일decode/fullframe MediaPipe+기존joint배정+고정ROI+기존pose head seed42+fit ID1/2/3 개별참고3개DTW+HTTP/Chrome표시. SignRep/TCN추론/새학습 제외.
- 종료→비교표/부족안내2rAF후응답수신 중앙4.513→0.411초(촬영중범위0.341~0.559). 서버계산4.400→0.239초. 동일서버시계의보수적표시확인상한이며영상준비/물리모니터지연/실웹캠측정아님. 앱/참고/검출기준비는촬영전. 이전1.134초와서로빼서추가효과주장금지(범위/부하차이).
- 조건6쌍 관절/배정배열·head값·전체DTW/UIpayload동일,이전완료관측과도동일. 원본PTS/전체프레임/참고분모유지,allmissing통제distance없음/근거부족. 별도Codex검산9034/636배열불일치0/보호32hashPASS. 사람/교육자정답검증아님. 결과보고서공개3영상쌍/시간슬라이더/스켈레톤/부분근거표시/좁은화면PASS.
- 사전검토고정CDP포트충돌위험을새프로필임시포트로수정후동결. 영상src전환시WinError10053취소로그보존,동결server유지하고safe_server.py 별도취소처리추가. 12작업실패없음. 원래generic지원gate는동등성검사용고정이며최신손가락별근거gate미통합. 특징차이후보는교육오류아님. 기존C/T3F/운영웹/DTW/개인자료보존. exec91452 exit0·서버Chrome종료·active=null.

'''
    following='''### 2026-10-31 웹 재생 연결 이후 — 속도와 피드백 신뢰성 구분

`word100_streaming_web_replay_20261031`의고정12회웹재생/출력동일성/저장검산/보고서완료. 재실행불필요.0.411초는CPU관절+고정head+DTW비교표의공개파일재생측정이며실제품/SignRep포함속도로보고금지. 현재10물리특징만검증,전체몸/수어교정미완료.
다음단일우선순위는고정관절기반비교에이미준비한특징별관측지원기준을별도연결하고,fit/dev공개자료의정상손모양·손내림·교차·미관측에서잘못된차이후보를검토하는것이다. reference분모는유지하고소모held/fresh점수로임계값튜닝금지. 의미단계·필수반복·얼굴부위접촉은확인된설명/시간주석범위에서만다루며TCN정답을근거정답으로쓰지않는다. 속도개선만으로운영교정출시/후보교체하지않는다. 실웹캠/새다운로드/개인자료/운영교체로자동확대금지.

'''
    context=(ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf8')
    for marker,content in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',status),('## 9. 다음 단일 작업',following)]:
        assert context.count(marker)==1
        context=context.replace(marker+'\n',marker+'\n\n'+content,1)
    (ROOT/'WordsInSilence_PROJECT_CONTEXT.md').write_text(context,encoding='utf8')
    handoff=f'''# WordsInSilence 최신 인수인계

{now}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고 중요한 차이와 허용가능한 차이를 구분한다.

현재단위: {U.name}. 상태 COMPLETE_ISOLATED_WEB_REPLAY_NOT_PRODUCTION. exec91452 exit0, 서버/Chrome종료, active=null. 자동화없음.

{status}
{following}
## 현재 검증 범위

10개: R_index_only/R_open_four/L_open_four/R_face_overlap/hands_overlap/R_motion_up/down/left/right/L_motion_down. 앞5개손모양·투영겹침,뒤5개이동. 어깨팔꿈치손목은관측/정규화입력,전신교육교정아님. 얼굴표정부재. 얼굴특정부위접촉/정확한3D손바닥/필수반복/의미단계의정오아직미검증.
이전새shared+interval후보의known80.78/new73.25BA와회귀는그대로미채택. 이번속도시험은그새후보를채택하지않고이전pose_capacity_matched_42 고정연구head사용. C/T3F운영값은안바뀜. 과거fresh32/held32는소모평가이므로튜닝금지,이번3영상은시간/동등성재생뿐품질재점수없음.

## 실행·검증 및 접근

pipeline.py/FROZEN.json/AUTHORIZATION.json에계약. 결과RESULT.json/VERIFICATION.json/INDEPENDENT_REVIEW.json(9034체크)/COMPLETION.json. RESULT_KO.html는서버없이직접열고공개3영상선택/정렬막대/스켈레톤확인가능. REPORT_UI_VERIFICATION.json PASS. 원래index.html는측정앱,완료job재실행을막음. safe_server.py는동결server.py의영상취소예외만처리하는별도보완,원래실험코드/결과수정없음. NONFATAL_TRANSPORT_FIRST_FAILURE.json및FROZEN_PRE_REVIEW/browser_test_PRE_REVIEW보존.

실제웹캠없음. 공개파일원본PTS를실시간처럼공급,ROI/목표word는미리알려진조건. 2rAF→서버ack는텍스트/표/안내렌더상한이며영상준비완료시간아님. 마지막두영상의웹로드seek와최종file보고서3쌍을별도검증. 같은Codex별도검산을독립사람정답과혼동금지. root구현/rgb_labels_b독립검산(이전stream모듈작성자였던이력명시).

환경 Core Ultra7 258V/32GB/Arc140V. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. MediaPipe현재Windowswheel CPU만지원,SignRep OpenVINO IntelGPU기존경로존재하지만이번속도경로미포함. PYTHONDONTWRITEBYTECODE=1 유지. 원본/W:/sources/과거동결읽기전용,사용자/멘토자료접근·학습·튜닝/카메라/다운로드/외부전송/Git/운영기본교체금지.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
    artifacts={str(p.relative_to(U)):hashlib.sha256(p.read_bytes()).hexdigest() for p in U.iterdir() if p.is_file() and p.name!='FINAL_ARTIFACTS.json'}
    save(U/'FINAL_ARTIFACTS.json',dict(at=now,sha256=artifacts))
    for p in [OPS/'STATE.json',U/'CURRENT_EXECUTION.json',U/'COMPLETION.json']:read(p)
    assert U.name in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
    print('completion and four state documents synced')
if __name__=='__main__':main()
