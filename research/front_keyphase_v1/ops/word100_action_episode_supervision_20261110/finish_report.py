from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),'utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(U/'RESULT.json');v=read(U/'VERIFICATION.json');assert v['passed'] and v['result_sha256']==sha(U/'RESULT.json')
causal=dict(review='Same Codex original RGB and saved geometry review, after predictions; no training/tuning from this audit.',independent_expert=False,source_strips=read(U/'CAUSAL_RGB_SOURCES.json'),new_warnings=[
 dict(clip_id='NIA_SL_WORD2523_REAL13_F',primary_seed=42,kind='reversed_core_auto_order',old_partial=[68,80],new_complete=[69,77],old_signed_change=.0314,new_signed_change=-.6656645,interpretation='Original RGB in reversed test order shows both hands approaching. New span ends before artificial splice at80; old partial includes that splice. New direction warning has visible motion evidence. This is a synthetic reverse-direction test, not a real learner error or solved pure order detection.'),
 dict(clip_id='NIA_SL_WORD2523_REAL12_F',seeds=[43,44],primary_seed_gain=False,old_partial=[67,78],new_complete=[67,75],old_signed_change=.0614,new_signed_change=-.544327,interpretation='Same direction evidence in source RGB. Secondary seeds only; do not select them from evaluation scores.')],remaining=[
 dict(word='잘하다',clip_id='NIA_SL_WORD1133_REAL13_F',kind='reversed_core_auto_order',result='False similar remains',evidence='Saved brush_arm binding contains only partial50..59 /49..60 spanning a trajectory jump and partial70..77 predicted hold; near-wrist arrival shares parent family but no clean reversed travel proposal. This does not establish true contact or shape correctness.',next='Separate splice/observation discontinuity from real movement; check complete motion support using FIT labels, without globally smoothing short changes.'),
 dict(word='조상',clip_id='NIA_SL_WORD1567_REAL14_F',kind='whole_reverse',result='New-only matching loses one repetition, old matches it',evidence='Reviewed73..85 vs new73..76 and77..93 fragments; old76..85 matches. Automatic event splitting/linking, not manual-window family/direction error.',next='Preserve reversal points and repeated events; one candidate cannot stand for two repetitions.'),
 dict(word='이집트',clip_id='NIA_SL_WORD2523_REAL14_F',kind='normal',result='Both models lack adequate overlap for actual outward interval',evidence='Reviewed onset48..51/end54..57; old44..50, new37..53. Preparation included and actual later motion shortened.',next='Add real start/end/hold distinctions on FIT footage; avoid fixed timing ratios.'),
 dict(word='순종',clip_id='NIA_SL_WORD0739_REAL12_F',kind='reversed_core_auto_order',result='Still uncertain, semantic role conflict',evidence='raise_forearm reversed candidate conflicts with old phase roles; close_chest has no reliable direction. No override based on expected answer.',next='Label same appearance in different action contexts and compare direct/automatic windows before retraining linkage.')])
write(U/'CAUSAL_AUDIT.json',causal)
plan=dict(status='REVIEWED_NOT_EXECUTED',goal='주요 행동의 실제 시작·진행·끝을 찾고, 준비·전환·마무리와 구분해 방향·순서·누락을 설명에 연결한다. 공통 기능을 사용하며 단어별 예외는 만들지 않는다.',basis=dict(manual_windows='historical7 + additional14 all correct family/direction for both old and new; small coarse-motion diagnosis only',automatic='additional old13/14 vs new12/14 primary42; candidate linking is measured bottleneck',limited_gain='historical9 reversed-core cases4→5 warnings primary; three initializations gain1/2/2, all gains in one expression',not_resolved='one false similar,6 omitted-final conditions uncertain, semantic conflicts and endpoint errors remain'),stages=[
 dict(step=1,name='실제 움직임과 잘못 연결된 구간을 구분',work=['기존 공개 FIT의 시작·방향 전환·끝·유지 구간을 앞뒤 원본 RGB로 확인한다.','이동 중 / 도착 후 유지 / 다음 행동으로 이동 / 관측 불가를 분리하고, 한 행동 ID로 시작·진행·끝을 잇는다.','기존4표현 외 경계 주석이 있는6표현(WORD0003/0500/0678/1139/1290/2551)에서 같은 문제 유무를 점검한다. 의미 판단을 강제로 붙이지 않고 확인 가능한 움직임만 추가한다.','좌표 급변은 실제 빠른 변화·검출 오류·시험 편집 경계를 나눠 기록한다. 마지막 모습이 비슷하다는 이유로 중간 동작을 완료 처리하지 않는다.'],outputs=['공개FIT 사건ID·역할·방향전환·관측근거 정답','구간 분리/연결 오류 목록'],gate='판단 불가 항목은 학습 손실에서 제외하되 평가 분모/보류 수에 남긴다. 반복 횟수와 짧은 변화는 별도 보존.'),
 dict(step=2,name='기존 움직임을 보존하며 후보 구간 연결 보완',work=['기존 좋은 후보와 이번 후보를 모두 보존하고, 같은 행동의 조각인지 다른 행동인지 구분하는 공통 연결부를 만든다.','합칠 근거는 몸 기준 상대 움직임·관측 품질·방향 전환·행동 역할이다. 정답 방향이나 손모양과 다르다는 이유로 후보를 제외하지 않는다.','손을 모았다 펴기처럼 구별되는 행동은 합치지 않는다. 반복의 왕복도 각각의 구간과 전체 반복을 함께 기록한다.','준비/마무리가 섞인 구간과 좌표의 불연속은 방향 판단 근거를 낮추며, 바로 오답/정답으로 바꾸지 않는다.'],outputs=['기존만/새것만/함께 사용 비교 가능한 공통 연결부'],gate='먼저 FIT 진단에서 실제 구간 보존이 가능한지 확인. 관측 자체가 부족하면 연결 모델의 추가 학습으로 숨기지 않는다.'),
 dict(step=3,name='필요한 부분만 지도학습',work=['직접 지정한 구간에서는 잘 맞으므로 기존 손모양·방향 모델은 우선 고정한다.','1단계의 정답으로 후보 구간의 시작·끝·조각 연결/분리 판정부를 학습한다. 학습 가능한 오류가 확인될 때만 작은 판정부와 필요한 인코더 층을 조정한다.','공통100단어899영상 학습을 유지하되 세부 정답이 없는 부분을 정답처럼 채우지 않는다.','여러 초기값과 조건을 실행 전에 동결하고, 추가 평가 점수로 seed/가중치/경계를 선택하지 않는다.'],outputs=['구간 연결 모델 후보','고정 모양/방향 대비 회귀 비교'],gate='FIT 내부 사람 분리 개발 비교에서 기존 사건 누락이 늘지 않고 연결이 나아져야 최종 공개 비교로 진행.'),
 dict(step=4,name='같은 조건으로 비교하고 적용 여부 결정',work=['기존78조건은 회귀용으로 유지하고 정답 구간 제공/자동 구간을 분리해 비교한다.','이번ID14는 이미 본 평가자료로 표시한다. 별도 공개 기록ID가 확보돼 있으면 원본 해시·과거 노출을 감사한 뒤 새 평가를 예약한다. 새 외부 다운로드는 시작하지 않는다.','정상·속도·교차·유지·짧은 변화·반대 방향·순서·중단/누락·관측 부족을 각각 집계한다. 인위적 시험과 실제 촬영을 혼동하지 않는다.','주요 구간 한 개가 두 번 재사용되는지, 잘못된 모양의 시도가 정렬 과정에서 사라지는지, 빠진 동작을 DTW가 늘려 채우는지를 검사한다.','통과한 경우에만 어느 순간·어느 손·어떤 방향을 고칠지 짧은 문장과 색상 표시로 연결한다. 웹/기존C·T3-F 기본 교체는 별도 결과 확인 후 한다.'],outputs=['이전/현재 조건별 결과','누락 포함 자동 구간 성능','학습 모델 계산시간과 실제 촬영 후 전체 대기시간을 분리한 측정'],gate='고정 기준에서 정상·속도·교차 경고 증가 및 기존 손모양/방향 저하 없음, 자동 구간/순서·누락의 실질 개선, 초기값별 안정성 확인. 실패하면 모델 전체 교체 없이 확인된 병목만 다음 대상으로 제한.' )],branch_rules=[
 '직접 구간도 틀리면 행동/손모양 정답과 판정부를 보강한다. 자동 구간만 틀리면 구간 경계·연결을 우선한다.',
 '관측이 끊겨 실제 행동을 확인할 수 없으면 보류한다. 동작을 하지 않았다는 근거가 있을 때만 누락을 지적한다.',
 '후보가 잡히지만 설명 역할만 틀리면 역할 연결 주석을 보강한다. 후보 자체가 없으면 역할 학습만 반복하지 않는다.',
 '추가 표현에서 효과가 없으면 범용성 미확인으로 보고하며, 단어명 예외로 점수를 올리지 않는다.' ],protections=['C·T3-F/운영웹/기존TCN·DTW 유지','사용자·멘토 자료 학습·튜닝·선택0','새 모델 다운로드0','동결 완료 산출물 재추론0'],review='Saved-output arithmetic, original RGB causal audit, manual-vs-automatic and seed comparison by same Codex. No external expert review or guaranteed improvement claim.')
write(U/'NEXT_PLAN.json',plan)
report='''이번 작업 결과 — 한 행동의 시작·진행·끝 연결

핵심
공통 기능으로 학습·비교를 마쳤다. 뒤집은 동작을 추가로 찾았지만, 새 구간 검출기를 전체 교체할 만큼 안정적이지는 않았다. 현재 웹과 기존 연구 기준은 유지한다.

1. 한 단어 전체를 하나의 행동으로 뭉친 것이 아니다.
예: ‘잘하다’의 팔을 따라 이동하기와 손목 부근에 도착하기는 한 행동의 진행과 끝으로 연결했다. ‘조상’의 양손 높낮이는 번갈아 움직이는 행동 안의 상태로 연결했다. 모으기와 펼치기처럼 구별되는 행동은 각각 남긴다.
단어명별 예외 코드는 없다. 같은 공통 기능을 다른 표현에도 쓰려는 구조지만, 그 표현의 설명·사건 정답 연결과 검증은 필요하다.

2. 무엇을 실제로 학습했나?
공개100단어899영상(기록ID01~09)의 공통 학습을 유지했다. 시간 합성곱+GRU 인코더와 행동 역할·시작/끝 판정부를 3개 초기값으로 각각12회 학습했다.
세부 행동 역할은4표현36영상, 실제 움직임 경계는10표현18영상29움직임을 이용했다. 그중21움직임을 설명의 행동과 연결했다. 100단어 모두에 정밀 피드백 정답을 붙였다는 뜻은 아니다.
기존 순간 손모양·방향 판정부는 고정했다. 이번 변경은 행동 구간을 찾고 설명에 연결하는 쪽이다.

3. 이전과 무엇이 달라졌나? (미리 정한 주 결과 모델)
• 주요 동작을 뒤집은9조건: 차이 경고4→5, 계속 비슷하다고 나온 사례1→1, 보류4→3.
• 정상12조건+속도 변경24조건: 기존 표시 유지. 정상12 중9는 비슷,3은 보류였고 이번에도 같다. 전부 정답 판정했다는 뜻이 아니다.
• 조상의 양손 교차3영상과 그 속도 변경6조건: 기존 표시 유지.
• 이미 확인하던 손모양/방향 변형9조건: 경고 유지.
• 마지막 동작을 지운6조건: 계속 보류. 누락을 정확히 설명하는 단계는 아직 해결하지 못했다.
• 손 전체 또는 마지막 구간이 안 보이는18조건: 비슷하다고 올려주지 않았다.
78조건을3개 초기값에서 비교했다. 서로 다른234개 원본 영상이 아니라 같은78조건의 반복 비교다. 추가 경고는 초기값별1/2/2개였으며 모두 이번 이집트 역방향 시험에서 나왔다.

새 경고의 이유
기존 구간에는 동작 뒤집기 시험의 연결 지점까지 섞였다. 새 후보는 그 앞의 실제 ‘두 손이 가까워지는’ 구간을 잡았고, 원본 영상과 몸 기준 거리 변화가 같은 방향임을 확인했다. 이것은 인위적으로 뒤집은 동작의 방향 차이를 추가로 찾은 결과다. 실제 학습자의 오류나 순서 오류 전체를 해결했다는 뜻은 아니다.

4. 왜 새 모델로 전부 바꾸지 않았나?
추가 공개4영상에서7개 움직임을 직접 확인했다. 원본과 전체 역재생으로14개 조건을 만들었다.
동작 구간을 직접 알려주면 기존과 새 모델 모두14/14에서 큰 행동 종류·방향이 맞았다. 스스로 구간을 찾게 하면 기존13/14, 새 모델12/14였다. 정상 원본7개에서는 둘 다6/7, 역재생7개에서 기존7/7→새6/7이었다.
즉 ‘그 구간의 행동을 구별하는 능력’과 ‘어디부터 어디까지가 그 행동인지 찾는 능력’은 다르다. 이번 작은 표본에서는 뒤쪽이 더 큰 병목이었다. 손가락의 모든 세부 모양을14/14 맞혔다는 의미도 아니다.
추가 자료는 이 피드백 학습에서 제외한 기록ID14지만 과거 TCN 평가에는 사용됐다. 프로젝트 전체에서 처음 본 자료라고 부르지 않는다. 원본4개에서 만든 짝 조건이므로14개 독립 영상도 아니다.

5. 다른 단어에서도 문제가 있나?
• 잘하다: 뒤집은 시험1개를 여전히 비슷하다고 판단한다. 중간 움직임보다 좌표 급변이 섞인 부분 구간을 근거로 삼는 문제가 남았다.
• 조상: 새 구간 검출기는 역재생의 반복1개를 조각내 놓쳤다. 기존 후보는 잡았다.
• 이집트: 추가 정상 영상에서 준비 동작을 포함하고 실제 펼침의 끝은 짧게 잡았다. 기존·신규 모두 구간 겹침 기준을 통과하지 못했다.
• 순종: 앞뒤 행동 역할에 대한 근거가 충돌해 보류하는 사례가 남았다. 정답에 맞춰 강제로 연결하지 않았다.
따라서 특정 한 단어만의 문제가 아니지만, 모든100단어를 같은 세부 기준으로 검사한 것도 아니다.

6. 속도·보존·검증
새 판정부와 연결 계산만은 중앙값 약0.041초,95백분위 약0.173초였다. 관절 추출·기존 특징 계산·TCN·DTW·웹 화면을 제외했으므로 사용자가 기다리는 총시간이 아니다.
추가 공개4영상의 RTMW 추출은483프레임, 각 약4.0~5.3초였다. 이는 원본을 순서대로 읽는 오프라인 추출 시험이다.
저장 결과의 해시·수치·한 움직임 중복 배정·기존 판정 보존을 별도 코드로 검산했다. 같은 Codex의 별도 검산이며 독립 전문가가 검증한 것은 아니다.
C·T3-F, 운영웹, 기존TCN·DTW, 사용자·멘토 자료 미사용을 유지했다. 새 후보는 연구 폴더에 보관하고 운영에 적용하지 않았다. 최초 준비 단계의 모듈명 충돌은 학습 전에 수정하고 실패 기록을 보존했다. 평가 전 중복 배정 검사를 수정한 이력도 보존했다.

다음 작업
새 모델을 크게 바꾸기보다, 실제 움직임의 경계와 조각 연결부터 보강한다. 상세4단계와 실패 시 분기는 NEXT_PLAN_KO.txt / NEXT_PLAN.json에 정리했다. 이 후속 계획은 검토 완료이며 아직 새 학습을 실행하지 않았다.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
lines=['다음4단계 계획 — 검토 완료, 아직 후속 실행 전','',plan['goal'],'','계획 근거: 직접 구간에서는 기존·신규 모두21/21의 큰 행동 종류·방향을 맞혔다(기존 정상7+추가 원본/역재생14). 자동 구간에서는 새 검출기의 추가 표본 결과가13/14→12/14였다. 따라서 당장 필요한 것은 더 큰 행동 분류기보다 구간을 정확히 찾고 이어 붙이는 보완이다. 이 작은 표본으로 전체 성능을 보장하지 않는다.','']
for s in plan['stages']:
 lines.append(f"{s['step']}. {s['name']}");lines.extend('• '+x for x in s['work']);lines.append('통과 조건: '+s['gate']);lines.append('')
lines.extend(['문제가 겹칠 때의 분기']+['• '+x for x in plan['branch_rules']]+['','유지 사항']+['• '+x for x in plan['protections']]+['','검토 방법: 실제 원본 화면 확인, 저장 수치의 별도 검산, 직접/자동 구간 비교, 초기값3개 비교. 독립 전문가 검증은 아니다.','목표는 네 단어의 예외 수정이 아니라 같은 움직임에도 재사용할 수 있는 연결 기능이다.4표현 밖의6표현은 기존 공개 경계 자료로 확장 검증하며,100단어 전체의 세부 피드백 검증 완료로 확대해 말하지 않는다.'])
(U/'NEXT_PLAN_KO.txt').write_text('\n'.join(lines)+'\n','utf-8')
summary='100단어899 공통학습 유지, 행동 시작·진행·끝 역할 연결3seed 학습·78조건234결과 비교 완료. 주요 역순9조건 경고4→5(주seed42), 정상/속도/교차 기존표시 유지. 직접구간 기존7+추가14의 행동종류·방향 모두맞음; 추가 자동구간 기존13/14→새12/14로 전체교체 미채택. 이집트 제한 방향근거 발견, 잘하다 거짓비슷1·누락6보류·순종역할충돌·조상반복구간 문제 잔존. 기존C·T3-F/웹/TCN·DTW 유지, 개인자료0. 같은Codex 별도검산2862통과, 독립전문가검증 아님.'
nexttext='NEXT_PLAN의4단계: 공개FIT 실제 경계/반복/불연속 정답 보강→기존후보를 보존하는 공통 조각 연결→필요한 경계·연결부만 학습(100단어 유지)→정상/속도/교차/짧은변화/순서·누락 비교. 기존4외 경계자료6표현도 점검. 현재학습·평가 재실행금지. 추가ID14는 이미 평가노출, 후속평가 분리감사. L 기준유지·U는격리후보.'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text('utf-8')
for sec,title,body in [('## 8. 현재 상태','### 2026-11-10 행동 시작·진행·끝 연결 검증 완료·제한 개선',summary),('## 9. 다음 단일 작업','### 2026-11-10 행동 경계·조각 연결 보강 계획',nexttext)]:
 start=text.index(sec);a=text.index('### ',start);b=text.find('\n### ',a+4);b=len(text) if b<0 else b
 text=text[:a]+title+'\n\n'+body+'\n근거 '+str(U/'RESULT_KO.txt')+' / '+str(U/'NEXT_PLAN_KO.txt')+'\n\n'+text[b:]
ctx.write_text(text,'utf-8')
statepath=O/'STATE.json';st=read(statepath);st['current_stage']=U.name;st['updated_at']=now;st['next_single_action']=nexttext
st[U.name]=dict(status=r['status'],summary=summary,next=nexttext,result=str(U/'RESULT.json'),report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFICATION.json'),causal_audit=str(U/'CAUSAL_AUDIT.json'),next_plan=str(U/'NEXT_PLAN.json'),execution=str(U/'CURRENT_EXECUTION.json'),implemented_by='Codex',verified_by='Same Codex separate saved-output arithmetic + original RGB review; not independent expert',default_adopted=False,reference_baseline=str(O/'word100_event_guide_linkage_20261110'),candidate_only=str(U/'runs/42/model.pt'))
write(statepath,st)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+nexttext+'\n\n현재 단위: '+str(U)+'\n결과: '+str(U/'RESULT_KO.txt')+'\n계획: '+str(U/'NEXT_PLAN_KO.txt')+'\n이번 학습·평가·검산 모두 종료. 진행 중인 이번 단위 프로세스 없음. 후속4단계 미실행. 다른 웹 서버 상태는 이번에 변경·확인하지 않음.\n','utf-8')
history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with history.open('a',encoding='utf-8') as f:f.write('\n\n2026-11-10 행동 단위 연결과 병목 분리\n사용자 요청: 한 행동의 시작·진행·끝을 연결하고 행동 판단/구간 연결을 분리해 검증. 특정단어 예외 대신 공통 기능인지 확인.\n진행 이유: 앞/뒤 자세가 유사하면 역할과 시간순서가 혼동됨. 이동과 도착을 하나의 사건으로 연결하되 다른 핵심행동은 유지.\n'+summary+'\n판단: 이번 결과는 인위적 역방향 일부 개선. 추가 자동구간 성능은 나빠져 전체교체 불가. 원점 반복 대신 직접구간/자동구간 대조로 경계 연결 병목을 좁힘.\n'+nexttext+'\n근거 '+str(U/'RESULT_KO.txt')+'\n')
write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE',pid=None,training_complete=True,evaluation_complete=True,verification_passed=True,default_adopted=False,next_plan_executed=False,updated_at=now))
with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind='complete',time=now,primary_changes=1,default_adopted=False,verification_count=v['count']),ensure_ascii=False)+'\n')
print('Reports, reviewed next plan, cause audit and project handoff synchronized.')
