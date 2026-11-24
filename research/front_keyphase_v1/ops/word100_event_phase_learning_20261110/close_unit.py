from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(U/'VERIFICATION.json')['passed']
assert read(U/'VERIFICATION.json')['result_sha256']==sha(U/'RESULT.json')
now=datetime.datetime.now().astimezone().isoformat()
causes=[
 dict(kind='observed',finding='Strict complete-phase requirement retains4/21 matched intervals vs14/21 active-run proposals on primary42; same probabilities, isolated decoder comparison.',implication='Do not require all3phase predictions before using valid partial direction evidence.'),
 dict(kind='known_data_limit',finding='Reviewed interval endpoints are not exact movement onset/offset labels;20/60/20 interior phase targets are weak estimates.',implication='4/21 is not densephase accuracy. Improving true boundaries needs actual observed boundary supervision.'),
 dict(kind='hypothesis_not_isolated',finding='Weakphase supervision and hard per-frame monotonicity likely both affect fragmentation; this experiment does not isolate a causal share for each.',implication='Do not claim that more labels alone guarantees success or tune decoder thresholds on historicalDEV.'),
 dict(kind='limited_observed_gain',finding='One compound synthetic error has new contrary-direction evidence on63..71 instead of confused full47..78; same partial evidence supports2cards.',implication='1case, not2; existingclipwarning countunchanged; posthoc, not freshvalidation.'),
 dict(kind='remaining',finding='Automatic reversed-order9 remains3warnings,1similar_without_warning,5uncertain. Fullnormal12 remains9similar3uncertain.',implication='No whole-stage or completion mastery; no web adoption.'),
 dict(kind='code_correction',finding='Partial followup used100classindex for unused4guide-stage output; latent path is independent of that input. Originalcode preserved;4word deterministic equivalence checks pass.',implication='Correction does not change saved results and does not justify repeated inference.')]
write(U/'CAUSE_AUDIT.json',dict(causes=causes,implemented_by='Codex',verified_by='Same Codex separate saved arithmetic, not independent human',no_new_expert_gold=True))
next_steps=[
 '완료된 3학습·234비교·78부분보완 결과를 재실행하지 않는다. 기존 Pcoupled42 모양/방향과 현재 운영 모델을 보존한다.',
 '공개 FIT의 실제 RGB와 몸 기준 관계를 함께 확인하여 이동 시작·방향 전환·유지 진입·종료의 관측 가능한 경계/범위를 주석으로 남긴다. 시간 비율 추정은 별도 약한 정답으로 유지하고 100단어 전체가 정밀 주석됐다고 하지 않는다.',
 '사람별 FIT 안에서 검토/개발을 나눠, 단계 연결이 불완전해도 관측된 방향은 남기는 보조 출력을 검증한다. 뒤 동작을 억지로 만들거나 짧은 반대 방향·미배정 구간을 삭제하지 않는다.',
 '비교 전에 조건/대표 실행/채택 기준을 동결한다. 정상 속도·교차·방향·순서·생략·관측 부족을 동일 입력으로 비교한다. 과거 DEV에 맞춘 문턱/seed 선택 금지. 새 공개 평가 가능성은 현1199 manifest 밖의 기록/출처부터 점검하고 개인 자료는 사용하지 않는다.',
 '범용 경계 성능과 실제 오류 피드백의 이득이 입증되기 전에는 현재 웹8814와 C·T3F/TCN을 바꾸지 않는다. 실제 새 사용자 영상의 총시간과 교정 정확도는 별도 검증 대상이다.'
]
write(U/'NEXT_PLAN.json',dict(steps=next_steps,primary_bottleneck='Reliable observed event-boundary supervision and non-destructive partial evidence, not model size.',current_unit_complete=True,product_event_segmentation_complete=False,training_not_running=True,adoption='RESEARCH_ONLY_LIMITED_CUE_NO_DEFAULT_CHANGE'))
summary='100단어899공개FIT를 매회 유지한 단계 보조 Conv+GRU 3seed×10회 학습·234자동비교·78부분근거 보완 검증 완료. 기존 모양/방향 가중치·출력 및 TCN/DTW 보존. 정상12·속도24·교차3·기존오류9·부족/삭제24 결과 유지. 복합변형1건에서 부분 이동63~71의 역방향 근거 추가(2안내 같은근거, 영상경고수 불변). 주요 이동21개 중 대표14구간 검출, 시작·진행·끝 완전연결4개뿐; 순서역전9는3경고/1비슷함/5보류로 해결안됨. 경계끝점과20/60/20약한주석은 실제정밀경계정답이 아님. 3546저장검산 PASS(같은Codex). 후속코드 단어번호 혼동수정·사용한latent결과무영향4검사확인. 역사DEV사후보완·독립성능아님. 웹8814/C/T3F/현재TCN 보존·개인자료0·HTML생략.'
nxt='완료된 3학습·234비교·78부분보완 재실행금지. 다음은 공개FIT 실제 RGB에서 이동시작/방향전환/유지/종료의 관측경계·범위를 보강하고, 약한 비율주석과 분리해 단계연결을 검증하는 것. 불완전연결도 관측방향은 보존하되 전체완료로 승격하지 않는다. 기존100공통·모양/방향 보존, 특정단어 예외/역사DEV문턱튜닝 금지. 새평가자료 가능성은 기록출처 감사부터. 현재 웹미교체, 개인자료금지. NEXT_PLAN.json 참조.'
tag='### 2026-11-10 주요 동작별 시작·진행·끝 학습 진행'
newtag='### 2026-11-10 동작 단계 보조 학습·부분 방향 근거 검증 완료'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig')
for section,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
 start=s.index(tag,s.index(section));end=s.index('\n### ',start+len(tag));s=s[:start]+newtag+'\n\n'+body+'\n근거 '+str(U/'RESULT_KO.txt')+'\n\n'+s[end:]
p.write_text(s,'utf-8')
d=read(O/'STATE.json');d.update(current_stage=U.name,next_single_action=nxt,last_updated=now);d[U.name]=dict(status='COMPLETE_RESEARCH_LIMITED_CUE_NOT_ADOPTED',summary=summary,next=nxt,execution=str(U/'CURRENT_EXECUTION.json'),report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFICATION.json'),default_web_changed=False,private_access=False);write(O/'STATE.json',d)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업\n'+nxt+'\n\n현재단위 '+str(U)+'\n학습/평가 실행 종료. 사용자 재요청에 따라 다음 단위부터 이어간다. 실행중인 현재단위 작업 없음.\n','utf-8')
hp=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt';hs=hp.read_text('utf-8-sig');title='2026-11-10 — 동작별 시작·진행·끝과 부분 관측의 분리'
assert title not in hs
hs+='\n\n'+title+'\n사용자 요청: 기존 모양·방향을 유지하면서 주요 동작의 시작·진행·끝 구분.100단어 공통, 정상 속도/교차와 오류 검증.\n'+summary+'\n왜: 넓은 창에 준비·복귀가 섞이면 좋은 방향 모델도 잘못 판단할 수 있어, 단계 탐색을 별도 보조로 분리했다.\n발견: 완전한3단계연결을 요구하면 실제 관측한 이동까지 버렸다. 전체완료와 일부방향근거를 분리하니 복합변형1건의 설명근거를 보강했다. 약한시간분할주석으로 정확한행동경계까지 배웠다고 할 수 없어 웹미채택.\n다음 '+nxt+'\n근거 '+str(U/'RESULT_KO.txt')+'\n'
hp.write_text(hs,'utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_RESEARCH_LIMITED_CUE_NOT_ADOPTED',pid=None,finished=now,verification_passed=True,training_runs=3,automatic_comparisons=234,partial_followup=78,product_event_segmentation_complete=False,default_web_changed=False,private_access=False))
print('Result, cause audit, next plan, history and current project documents synchronized.')
