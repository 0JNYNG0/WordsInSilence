import hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
m=read(U/'VIEW_MANIFEST.json');qa=read(U/'BROWSER_QA.json')
assert qa['fragment_sha256']==sha(m['fragment'])
assert len(qa['results'])==2 and all(v['status']=='PASS' and all(c['ok'] for c in v['checks']) for v in qa['results'])
for p,h in m['pins'].items():assert sha(p)==h,p
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
report='''남은 문제 — 실제 장면으로 이해하기

이번은 완료한 실험을 설명하는 자료다. 새 학습·관절검출·DTW 계산을 하지 않았다. 공개 원본에서 이미 검토한 7프레임과 저장 검출 후보를 사용했다. 화면의 확대는 보기 위한 표시이며 분석 영역을 바꾼 것이 아니다.

1. 손이 보이지만 좌표를 확정하지 못한 4회
잘하다(P048) 2.467초와 조상(P063) 2.400초: 전체 화면에서는 기존 오른손만 찾았고, 손 주변 확대에서 왼손 후보가 나왔다. 현재 규칙은 둘에서 같은 손을 확인해야 추가하므로 왼손은 보류했다. 화면에 손이 없거나 손동작이 틀렸다는 뜻이 아니다. 한쪽 검출만으로도 쓸 수 있게 하려면 실제 손 정체와 관절 품질을 검증할 별도 근거가 필요하다.
조상(P063) 1.900초: 손이 겹친 장면에서 두 배율이 그린 관절 위치가 다르다. 어느 쪽을 믿을지 확정하지 못했다.
조상(P066) 2.367초: 새 손 후보는 있지만 어느 팔의 손인지 유일하게 연결하지 못했다. 빈 오른손 자리에 그냥 넣으면 같은 손을 두 번 넣거나 좌우를 바꿀 수 있어 보류했다.
4회는 4단어 또는 4영상이 아니라, 한 프레임의 한쪽 손을 세는 단위다.

2. 서로 다른 동작 순간을 비교한 문제
조상의 공개 영상 A(P066) 2.267초는 오른손이 위이고, 참고 B(P063) 1.933초는 왼손이 위다. 기존 DTW가 두 순간을 연결했다. 시간 숫자가 다른 것이 문제인 것은 아니다. 사람마다 속도가 다르므로 시간은 달라도 된다. 중요한 것은 실제로 같은 동작 단계인지다.
이 상태로 손 위치 차이만 보고 사용자에게 틀렸다고 말하면, 비교 시점을 잘못 잡아 생긴 차이를 사용자 탓으로 돌릴 수 있다. 두 영상 모두 공개 참고이며 이 장면으로 어느 촬영자가 잘못했다고 판정하지 않는다. 올바른 대응 시점을 새로 확정한 것도 아니다.

3. 손을 찾은 것과 손가락 정밀 위치를 아는 것은 다름
이집트(P072) 2.333초는 두 손의 후보 42점 중 36점을 근거가 있는 추정점으로 추가했고, 엄지 쪽 6점은 보류했다. 왼손 1~4번, 오른손 3~4번 관절 후보다. 채택한 36점도 사람이 찍은 실제 관절 정답과 비교해 정확성을 확정한 것은 아니다. 따라서 이 자료만으로 엄지를 얼마나 펴거나 접어야 한다는 정밀 교정을 확정하지 않는다.

개선된 부분과의 관계
이전에 없던 손 관측 18회를 더 확보하고 기존 비교에도 연결한 것은 실제 구현 성과다. 그러나 같은 단계끼리 대응됐는지와 손가락 하나하나가 정확한지는 따로 검증해야 한다. 원래 보이던 정보와 비교할 시간 범위는 보존했다.
다음 핵심은 앞뒤 양손 관계·이동 방향·반복 순서로 같은 주요 단계끼리 대응하는지 확인하는 것이다. 근거 부족은 보류로 남기고 정상 손내림/속도/교차를 오류로 만들지 않는다.

기존 C·T3-F/운영 웹·DTW 보존. 사용자·멘토 촬영 접근·학습·튜닝 없음. 시각 자료는 채팅 안에 표시하며, 8800의 이전 보고서를 교체하지 않았다.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
statep=ROOT/'research/front_keyphase_v1/ops/STATE.json';state=read(statep)
state.update(updated_at=now,updated_date_kst=now)
state[U.name]={'status':'COMPLETE_VISUAL_EXPLANATION_ONLY','at':now,'fragment':m['fragment'],
 'report':str(U/'RESULT_KO.txt'),'public_source_frames':7,'held_hand_cases':4,
 'browser_checks':sum(len(x['checks']) for x in qa['results']),'new_inference':0,'new_training':0,'new_DTW':0,
 'user_mentor_used':False,'production_changed':False,'research_next_action_unchanged':True}
write(statep,state)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig')
h=re.search(r'^## 8\.[^\n]*\n',s,re.M);assert h
entry='''
### 2026-11-01 남은 문제 시각 설명 완료 — 새 실험 없음
사용자 요청으로 `word100_remaining_issues_view_20261101`에서 기존 공개7프레임과 저장후보를 재사용한 채팅 시각자료 제작. 4보류손(확대만후보2/배율불일치1/팔정체모호1),P06668↔P06358 반대손높이 순간대응,P07270 엄지6점보류를 구분. 관측보류≠사용자수어오류,채택추정점≠관절정답임을 설명. 원이미지해시/시점/좌표표시연결확인·브라우저2폭46검사PASS. 초기숨김스타일충돌수정;검사기시작타이밍실패기록보존후수정. 연구결과·원본·모델·운영불변/추론·학습·DTW0/개인·멘토자료사용0. 8800이전보고서그대로;채팅내`remaining-sign-evidence.html`표시. 연구 다음단계는 이전과 동일.

'''
s=s[:h.end()]+entry+s[h.end():].lstrip('\n\r')
h9=re.search(r'(^## 9\.[^\n]*\n\s*### [^\n]*\n)',s,re.M);assert h9
s=s[:h9.end()]+'남은문제시각설명은완료(remaining_issues_view/RESULT_KO.txt);추가정렬·학습실행없고아래연구계획유지.\n'+s[h9.end():]
p.write_text(s,encoding='utf-8')
handoff=ROOT/'NEW_CHAT_HANDOFF.md';t=handoff.read_text('utf-8-sig');first=t.index('\n')
t=t[:first+1]+'\n2026-11-01 최신요청은남은문제시각설명. word100_remaining_issues_view_20261101 완료:공개7프레임/4보류손·단계대응·엄지6점보류설명,채팅내 remaining-sign-evidence.html. 새추론/학습/DTW0. 원연구완료와다음단계는아래그대로.\n'+t[first+1:]
handoff.write_text(t,encoding='utf-8')
paths=[U/'VIEW_MANIFEST.json',U/'BROWSER_QA.json',U/'RESULT_KO.txt',Path(m['fragment']),p,handoff,statep]
write(U/'COMPLETION.json',{'status':'COMPLETE_VISUAL_EXPLANATION_ONLY','at':now,'pins':{str(x):sha(x) for x in paths},
 'source_pins_preserved':True,'new_experiment':False,'old_results_preserved':True,
 'visual_check':'Root viewed original 4 residual overlays, both opposite-height frames, and rendered desktop/mobile views. UI46 checks PASS.',
 'runtime_fallback':'Native browser/Node tools kernel-assets failure; separate-profile network-blocked local Chrome renderer used.',
 'active_execution':None})
print('Visual explanation complete; source pins verified and authoritative records synchronized')
