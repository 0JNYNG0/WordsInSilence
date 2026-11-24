"""Saved evidence finalization only; no inference, DTW execution or training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,html,shutil
U=Path(__file__).resolve().parent; B=U.parents[1]; ROOT=B.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
replay=read(U/'USER_REPLAY.json');ready=read(U/'FROZEN_READY.json');http=read(U/'HTTP_CHECK.json')
prior=read(B/'ops/word100_c_user_paired_20261017/RESULT.json')
assert read(U/'PUBLIC_VALIDATION.json')['status']==read(U/'BANK_INDEPENDENT_RECOMPUTE.json')['status']=='PASS'
assert sha(U/'ROLE_BANK.json')==ready['bank_sha256']
for p,h in ready['code_sha256'].items():assert sha(p)==h,p
for p,h in prior['source_hashes'].items():assert sha(B/p)==h,p
assert http['health']['camera'] is None and http['health']['camera_session_id'] is None
assert http['default_model']['model_key']=='word100_tcn_context_v1'
assert all(x['model_status']=='READY' for x in http['models'])
assert (U/'WEB_DTW.png').is_file()
single=[t for t in replay['trials'] if len(t['required_sides'])==1]
score=lambda m:100*m['ratio']/(1+m['ratio'])
rows='\n'.join(f"{t['label']}: {t['old_status']} → {t['new_status']} / 굽힘 {score(t['metrics'][0]):.1f}, 위치 {score(t['metrics'][1]):.1f}, 궤적 {score(t['metrics'][2]):.1f}점" for t in single)
report=f"""마음에 들다 진단 · 필요한 손 중심 DTW 보완
갱신 {stamp}

한 손 표현의 DTW를 웹에 적용했다. TCN 모델은 보존했다. 사용자 촬영은 진단·평가에만 사용했으며 학습·튜닝·후보 선택에 사용하지 않았다.

1. 마음에 들다: 원인 범위를 좁힌 결과
첫 촬영은 M0 2위, C 8위였고, 재시도는 모두 2위였다. 확인 저장을 누르지 않은 첫 실패도 자동 저장 원본으로 포함했다.
입력 처리와 모델의 영향을 나누기 위해 저장된 두 입력을 고정된 두 모델에 교차 입력했다. 두 촬영×두 입력×두 모델 = 진단용 forward 8회이며, 원래 결과를 덮어쓰지 않았다.

첫 촬영의 목표 단어 순위
입력                 기존 M0 모델       C 모델
기존 입력                2위            10위
C 입력                   2위             8위
재시도: 네 조합 모두 2위.

첫 촬영에서 C는 두 입력 모두 ‘수어’를 1위로 골랐다. M0는 두 입력 모두 ‘가평군수어통역센터’를 1위로 골랐다. 따라서 첫 실패를 손 검출/입력 문제 하나로 설명할 수 없다. 이번 수행에서 C의 학습된 분류가 다른 단어에 치우치는 현상이 남았다. 기존 모델도 1위 정답이 아니므로 되돌리면 해결된다고 말할 수 없다.
두 촬영 모두 손이 처음~마지막 보인 구간에서 적어도 한 손은 계속 검출됐다. C의 비교 가능한 손-팔 손목 대응 구간에는 반대편 손목에 더 가까운 경우가 0회였다. 이는 기하 점검이며 가림 중 오배정·관절 오차가 없다는 보장은 아니다.
공개 DEV 저장 점수에서는 같은 단어를 3명 모두 맞혔다(같은 3명의 두 입력 보기에서 각각 3/3). 이번 사용자 수행과 차이가 있어 공개 촬영 환경의 높은 점수를 웹캠 성능으로 확대할 수 없다.
손모양·위치·순서·속도 중 정확히 무엇이 원인인지는 두 촬영만으로 분리되지 않았다. 사용자의 수행이 틀렸다고 확정하거나 ‘천천히 하면 된다’고 처방하지 않는다.

2. 필요한 손 중심 DTW
기존에는 양손 모두 관측돼야 종합 비교가 가능했다. 이제 목표 단어의 공개 참고 동작에서 필요한 손만 시간 정렬과 차이 계산에 사용한다. 쉬는 손은 ‘비교 제외’/None으로 남기고 0점으로 넣지 않는다. 필요한 손이 보이지 않으면 수치를 보류한다.
정지해 있어도 다른 손을 받치거나 의미 있는 모양을 만드는 손은 필요하다. 사용자의 화면에 안 보인다는 이유로 자동 제외하지 않는다.
현재 연습 12단어 중 멀다·공항·별은 공개 TRAIN 9명씩에서 오른손 단독 사용을 확인했다. 마음에 들다는 양손이 필요하며 다른 9단어의 양손 요구를 유지했다. 단어마다 점수 예외를 만든 것이 아니라 공통 비교 코드와 별도 역할 정보로 구현했다.
역할 근거는 공개 원본 36개(세 단어27개+마음에들다9개)의 전체 시간대에서 각8장면을 표본 관찰한 것이다. 모든 연속 프레임 검토나 수어 전문가 인증은 아니다. 100단어 자동 역할 판별과 왼손 주도 변형 대응은 미검증이다.

3. 저장 촬영의 변화
세 항목을 모두 계산 가능한 촬영: 10/13 → 13/13.
{rows}
세 촬영 모두 오른손 기준이다. 다른 양손 촬영 10회의 거리·상대비율은 기존 값과 정확히 같았다. 13회 TCN 저장 순위도 그대로다.
첫12회 C 11/12(91.67%)는 이전 분류 결과이며, 이번 DTW 변경으로 분류 정확도가 오른 것은 아니다.
0~100 차이 점수는 낮을수록 비교한 특징의 차이가 작다는 표시다. 50점은 공개 학습자끼리의 대표 차이와 같은 크기다. 틀린 비율이 아니다. 참고 상한은 각 학습자를 나머지8명과 비교한 차이 중 최댓값이며 수어 전체의 합격선으로 검증되지 않았다.
새 세 촬영의 아홉 항목은 모두 해당 참고 상한을 넘었다. 실제 수행 차이뿐 아니라 검출 오차·촬영 시점·준비/마무리·시간 정렬 영향도 포함될 수 있다. 이것을 곧바로 잘못 수행했다는 확정으로 해석하지 않으며 점수를 낮추려고 사용자 결과에 맞춰 기준을 바꾸지 않았다.

4. 검증
공개 TRAIN으로 역할과 기준을 만들고 공개 검증을 통과해 동결한 뒤 사용자13회를 평가했다.
- TRAIN: 3단어×9명×나머지8명 = 216쌍. 별도 표준 라이브러리로81개 항목 집계와9개 분포 검산 PASS.
- 기본18검사: 쉬는 손 특징 제거 시 불변, 필요한 손 제거 시 보류, 같은 입력끼리 차이0, 양손 계산 기존 일치.
- DEV36개: 한 손9개에서 쉬는 손 제거 후 값 불변. 필요한 손 제거9조건과 양손27개의 좌/우 제거54조건은 모두 보류. 정상 양손27개 결과·기준은 기존과 정확히 같음.
- 특징 제거 검사는 저장 특징에 가한 검사이며, 영상을 가린 뒤 MediaPipe를 다시 실행한 검사는 아님.
- Node 화면 검사: 비교 제외/관측 부족 구분, 필요한 손 부재 보류, 양손 유지, 저장 전체 결과 표시 PASS. 기존 M0 기본/C 명시 선택 검사도 PASS.
- 실제 웹에서 멀다 기록의 오른손 그래프 세 개와 왼손 비교 제외 표시 확인. 두 모델 READY, 카메라 꺼짐, PID38708/8765.
작성·실행·별도 계산 검산은 같은 Codex 담당이며 별도 에이전트·외부 전문가 독립 검증은 아니다.
첫 화면 모의 검사는 테스트 환경의 crypto 스텁 누락으로 실패했고 스텁 추가 후 통과했다. 제품 계산 조건 변경은 없다. 제한환경의 로컬 소켓/CIM 접근은 불가해 지침에 따라 일반 실행환경에서 유휴 서버 확인 후 재시작했다.

5. 남은 문제와 후속 방향
마음에 들다의 TCN 1위 실패는 해결되지 않았다. 후속 분류 보완은 공개 TRAIN의 해당 표현·혼동 단어에 대해 구간·손 접촉·라벨을 감사하고 사전 고정한 공개 자료 비교 계획으로 진행할 필요가 있다. 사용자 두 촬영을 맞추는 학습이나 임계값 조정은 하지 않는다.
필요한 손의 겹침·검출 누락, 왼손 주도 변형, 전체100단어 손 역할, 정상/오류 수행을 가르는 기준은 남은 과제다.
DTW는 기존 관측과 첫~마지막 관측 구간·샘플링을 유지한다. 준비/마무리 영향까지 제거했다고 보장하지 않는다. TCN 가중치 내부를 읽는 것이 아니라 공개 학습 참고 동작과 비교한다.

6. 사용·보존
http://127.0.0.1:8765/practice.html?model=c 에서 새로고침 후 촬영 기록을 선택하면 기존 촬영에도 새 DTW를 볼 수 있다. 다시 촬영할 필요 없다. 카메라는 사용자가 직접 연결할 때만 켜진다.
원본 촬영·기존 결과·모델·기존 DTW 은행은 해시 보존 확인. 새 DTW는 별도 버전 파일에 저장했다. 학습·튜닝·모델 교체 0회. 진단용 고정 TCN forward8회만 수행했다.
"""
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
sections=''.join('<section><pre>'+html.escape(b)+'</pre></section>' for b in report.split('\n\n') if b.strip())
(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>한 손 DTW 보완 결과</title><style>body{font-family:system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;background:#f5f6f8;color:#21313a;line-height:1.75}section{background:white;border:1px solid #dde3e8;border-radius:12px;padding:18px;margin:14px 0}pre{font:inherit;white-space:pre-wrap;margin:0}img{max-width:527px;width:100%}</style><h1>마음에 들다 진단 · 한 손 DTW 보완</h1>'+sections+'<h2>실제 웹 화면</h2><img src="WEB_DTW.png" alt="오른손 그래프와 왼손 비교 제외"></html>',encoding='utf-8')
changed=[B/'src/wis_lab/service/word100_role_dtw.py',B/'src/wis_lab/service/word100_web_analysis.py',B/'web/practice.js',B/'web/practice.html',B/'config/word100_role_dtw.json']
result=dict(status='COMPLETE_WEB_APPLIED_FIXED_MODELS',updated_kst=stamp,role_bank_sha256=ready['bank_sha256'],
 mind=dict(first_M0_weights_ranks=[2,2],first_C_weights_ranks=[10,8],retry_all_four_rank=2,precise_motion_cause_unresolved=True,diagnostic_forward_calls=8),
 dtw=dict(old_fully_computable=10,new_fully_computable=13,total=13,recovered_words=[t['label'] for t in single],both_hand_numeric_parity=10,classification_improvement_this_unit=False),
 validation=dict(public_train_pairs=216,public_DEV=36,required_missing_conditions=63,optional_hand_invariance=9,both_DEV_exact=27,unit_checks=18,independent_aggregates=81,independent_scales=9,frontend='PASS',old_model_selection='PASS',HTTP='HTTP_CHECK.json',browser='WEB_DTW.png',independent_agent_review=False,operator='Codex root and separate arithmetic code'),
 execution=dict(training=0,user_learning_tuning_selection=False,TCN_diagnostic_forwards=8,new_captures=0,original_sources_and_models_preserved=True,server_pid=38708,default_model='word100_tcn_context_v1'),
 changed_files_sha256={str(p):sha(p) for p in changed},
 evidence={n:sha(U/n) for n in ['SCOPE.json','HAND_ROLES.json','FROZEN_READY.json','PUBLIC_VALIDATION.json','BANK_INDEPENDENT_RECOMPUTE.json','MIND_DIAGNOSIS.json','USER_REPLAY.json','HTTP_CHECK.json','WEB_DTW.png','RESULT_KO.txt']},
 limitations=['No validated sign correctness threshold','TRAIN roles sampled8frames/clip, not expert certification','Left dominant variants and all100 roles unvalidated','Precise mind failure cause unresolved','Legacy observations and sampling retained'],
 next_single_action='사용자와 새 DTW 결과 확인. 마음에들다 분류 후속은 공개 자료 감사·사전 조건 계획부터 범위 확인. 사용자자료 학습/튜닝/후보선택 금지.')
(U/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ['NEW_CHAT_HANDOFF.md','WordsInSilence_PROJECT_CONTEXT.md']:
 dest=U/'before'/name
 if not dest.exists():shutil.copy2(ROOT/name,dest)
state_path=B/'ops/STATE.json'
if not (U/'before/STATE.json').exists():shutil.copy2(state_path,U/'before/STATE.json')
state=read(state_path)
state.update(updated_date_kst=stamp,current_stage='WORD100_ROLE_DTW_APPLIED_MIND_DIAGNOSIS_COMPLETE',next_single_action=result['next_single_action'])
state['word100_role_dtw_20261017']=dict(status=result['status'],updated_kst=stamp,unit=str(U),result=str(U/'RESULT.json'),report=str(U/'RESULT_KO.html'),summary='멀다/공항/별 required R DTW계산복구. 저장13회 OK10→13,양손10값보존. 마음에들다2x2진단에서C입력무관오답,정확원인미확정.',server_pid=38708,training=0,diagnostic_TCN_forwards=8,user_learning_tuning=False,bank_sha256=ready['bank_sha256'],next_single_action=result['next_single_action'])
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
entry8="""### 2026-10-17 필요한 손 중심 DTW 웹 적용·마음에 들다 진단 완료
- 공개 TRAIN 역할 기반 공통 비교 적용. 멀다·공항·별 required R, 다른9연습단어 양손유지(정지 받침손 포함). 필요손누락은보류,쉬는손은None/비교제외. 100단어자동역할·왼손주도변형·정오답기준미검증.
- TRAIN216쌍/18검사/DEV36·필수손제거63조건·쉬는손제거9조건PASS. 양손DEV27기존동일. 별도stdlib81집계/9분포검산(동일Codex,에이전트검토아님). 공개36영상각8장면관찰이며모든프레임/전문가인증아님. 은행동결후사용자평가.
- 저장13회 DTW계산OK10→13,멀다/공항/별복구. 양손10회거리·비율및TCN13순위보존. 새3회모든항목참고상한초과; 계산가능개선이지정답증명아님. 첫12회C11/12분류결과는그대로.
- 마음에들다2회×2입력×2모델 진단8forward: 첫M0모델두입력모두2위,C모델기존입력10/C입력8위(둘다수어오답);재시도4조합모두2위. 검출하나로설명불가,분류혼동잔존. 공개DEV동일3명두보기각3/3저장점수와실사용차이확인. 정확세부동작원인/수행정오답미확정.
- 웹·Node·HTTP·실브라우저검증완료. PID38708일반실행8765/카메라꺼짐. M0기본/C선택·기존DTW은행보존. 사용자학습·튜닝·후보선택0,새학습0. 근거 ops/word100_role_dtw_20261017/RESULT.json,RESULT_KO.html,WEB_DTW.png.

"""
entry9="""### 2026-10-17 DTW 보완 결과 확인 및 분류 후속 범위 결정
현재요청완료. 웹새로고침후기존촬영기록에서필요손DTW확인가능. 마음에들다TCN1위실패해결로표기금지. 후속분류보완은공개TRAIN목표/혼동단어감사·사전조건계획부터사용자와범위확인. 사용자촬영은진단·평가전용,학습/튜닝/후보선택/임계값조정금지. 새학습·기본교체자동실행없음. st-gcn자동화삭제유지. 현재서버38708/8765상태확인후중복실행금지.

"""
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text(encoding='utf-8-sig')
assert entry8.splitlines()[0] not in s
for old,new in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n',entry8),('## 9. 다음 단일 작업\n\n',entry9)]:
 assert old in s;s=s.replace(old,old+new,1)
ctx.write_text(s,encoding='utf-8')
handoff=f"""# WordsInSilence 최신 인수인계

갱신 {stamp}

## 현재 목표와 마지막 완료
사용자최신요청: 마음에들다 오인식추가진단+한손DTW보완, 사용자학습·튜닝금지. 완료·웹적용.
단위 research/front_keyphase_v1/ops/word100_role_dtw_20261017의 SCOPE/RESULT.json/RESULT_KO.html 기준.
공개TRAIN근거로 멀다·공항·별required R,다른9연습단어양손유지. 받침손은정지해도필수. 안보인다는이유로자동제외금지. 선택안한손None/비교제외,필수손누락보류.

## 검증과 결과
TRAIN216쌍/18검사/DEV36(필수손누락63조건/쉬는손특징삭제9조건)/양손DEV27기존일치. 별도stdlib81집계/9분포검산PASS,동일Codex담당이며별도에이전트아님. 공개36영상각8장면표본관찰,전문가인증아님.
공개규칙·은행동결후사용자13회평가: DTW계산OK10→13,3단어복구;양손10거리·비율/TCN13순위유지. 새3회9항목모두참고상한초과;정오답/유사도정확성검증아님.
마음에들다첫2x2진단: M0모델입력무관2위,C모델기존입력10/C입력8위,둘다수어1위。재시도4조합모두2위. 검출만으로설명불가,분류혼동잔존。정확동작원인미확정。진단8forward,새학습/튜닝0.
이전첫12회 M0 8/12(66.67%)→C11/12(91.67%,+25%p),Top5 10→11 그대로. 본DTW보완으로분류개선된것아님. 1명탐색평가이지맹검일반화성능아님.

## 웹과 보존
C http://127.0.0.1:8765/practice.html?model=c / M0 ?model=m0 / 기본M0유지.
서버 PID38708 일반실행8765,로그현재단위/server.stdout.log,server.stderr.log. 기존34640을camera/session null확인후교체. 카메라접근0. Node표시/기존선택·HTTP·실제저장기록웹확인완료,WEB_DTW.png.
C SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab;M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f.
새ROLE_BANK SHA{ready['bank_sha256']}, config/word100_role_dtw.json핀. 이전DTW은행보존,새계산.role_dtw_27da98d31469.json별도저장. 원본result/NPZ/model해시유지확인.
DTW=기존primary관측+공개TRAIN동작비교,C가중치를읽지않음. 고정구간·샘플링유지,손추정복원없음.

## 남은 문제와 다음 한 작업
사용자와결과확인. 마음에들다TCN실패,왼손주도변형,100단어역할,정오답기준,겹침센서누락미해결.
후속분류보완은공개자료감사·사전조건계획부터범위확인. 사용자촬영학습/튜닝/후보선택/임계값조정금지.
TRAIN899/DEV300/보류14~18유지. 새학습/기본교체/카메라/다운로드/외부전송/Git없음. st-gcn자동화삭제유지. 완료점수재계산하지말고저장근거재사용.
"""
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
print(json.dumps({'status':result['status'],'report':str(U/'RESULT_KO.html'),'docs_synced':True,'updated_kst':stamp},ensure_ascii=False))

