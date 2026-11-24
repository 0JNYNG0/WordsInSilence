"""Record validated negative result; does not run detections or classifiers."""
import json,hashlib,html,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p):return json.loads(p.read_text('utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
s=read(U/'SUMMARY.json');v=read(U/'INDEPENDENT.json');e=read(U/'EVIDENCE_INDEX.json');r=read(U/'RESULTS.json');a=read(U/'ANNOTATED.json');c=read(U/'CONTEXT.json')
assert v['status']=='PASS' and s['changed_events']==0 and not (U/'RUNNING.json').exists()
assert len([x for x in a if x['remaining9']])==9

visual={
 'at':now,'reviewer':'Codex, same implementer; no separate human reviewer',
 'scope':'All9 remaining saved actual sensor RGB frames plus all11 frames/14 hand candidates with positive pixel evidence and5 established normal crossing counterexamples. Prior anatomical labels reused; not new blinded labels or full785frame human visual review.',
 'remaining9':[
  {'id':x['id'],'word':x['word'],'finding':'Color support joins visible forearms through torso/clothing; cannot isolate one anatomical forearm. Border partial hand still retained.',
   'new_correction':False,'observed_points':x['observed_points'],'common_component_fraction':x['shared_component_frame_fraction']} for x in e['remaining9']],
 'exception_detail':{'id':309,'finding':'Two candidates around one visible right hand; support also floods large background area (76.9%). No anatomical choice can be justified from this mask.'},
 'positive14':'The14 accepted paths in11frames visually follow visible forearms and agree with existing sides. They provide no correction. Most examples are separated resting forearms, not the hard crossing cases.',
 'normal_crossing5':'IDs37/38/39/90/291 remain unchanged; masks merge both arms, and37/38/39 nearly fill the image. Abstention preserves correct sides but is not evidence the RGB method can solve crossings.',
 'conclusion':'NO_NEW_CORRECTION; color connectivity is not arm-part connectivity. Input RGB can contain human-visible arm contours even when this method fails.',
 'viewed_files':[str(p) for p in sorted((U/'evidence').glob('remaining_sheet_*.jpg'))]+[str(p) for p in sorted((U/'evidence').glob('positive_sheet_*.jpg'))]+[str(U/'evidence'/f'normal_{i}.png') for i in [37,38,39,90,291]],
 'limitations':['Single reviewer','Known correlated development samples','No video semantic ground truth relabeling','No arbitrary outfit/skin/lighting invariance established']}
write(U/'VISUAL_REVIEW.json',visual)
next_plan={
 'status':'AWAIT_USER_RESULT_REVIEW_NO_TRAINING',
 'reject':'Do not add current chroma connectivity to learning or production. Do not sweep color thresholds to fit remaining9.',
 'preserve':['C and live web/TCN/DTW','Previous U observed points, provisional hints, identity-unknown state','Normal crossing and confirmed assignment'],
 'proposed_next':'If continuing image evidence, first test a separate forearm-boundary/direction method that distinguishes arm from torso and compares both possible anatomical paths. Freeze protocol on publicTRAIN before results; separate image path from weak Pose wrist estimates. Fail closed for overlap or competing paths.',
 'validation':['Partial-hand entry/exit, single-hand, real crossing, duplicate controls together','Require remaining cases corrected AND no normal crossing/observed-point regression','Require different attire/lighting/skin-condition controls before claiming appearance robustness','No TCN training or C replacement until user reviews results and training-input contract is fixed'],
 'not_guaranteed':'Contour/part evidence is a research direction, not a demonstrated performance gain. Some occluded identities may remain unknown.',
 'forbidden':['User/mentor material for training/tuning/selection','Word-specific exceptions','Hidden off-sensor pixels','Forced nearest elbow side','Changing classifier or inference defaults']}
write(U/'NEXT_PLAN.json',next_plan)
result={**s,'status':'VALIDATED_NEGATIVE_RESULT_NOT_ADOPTED_AWAITING_REVIEW','at':now,
 'remaining9_recovered':0,'remaining9_identity_unknown':9,'remaining9_points_preserved':72,
 'normal_regressions':0,'crossing_regressions':0,'source_frames':785,'source_clips':367,
 'pixel_both_arms_connected_candidates':1210,'pixel_single_arm_support_candidates':14,
 'new_confirmed_identity_promotions':0,'method_adopted':False,'independent':'PASS','contract_tests':15,
 'protected_pins':len(c['pins']),'full899_rgb_replayed':False,'production_changed':False,
 'training_authorized_now':False,'visual_review':'SameCodex;9failure+11positiveframes+5normalcounterexamples',
 'conclusion':'色連結は腕の解剖学的連結ではない'.replace('色連結は腕の解剖学的連結ではない','색 연결이 몸통까지 합쳐져 좌우 근거를 구분하지 못함. 이번 방법은 추가 개선 0으로 미채택.'),
 'result':str(U/'RESULT_KO.html'),'next_plan':next_plan['proposed_next']}
write(U/'RESULT.json',result)
text=f'''화면 안 팔–손 영상 연결 검증 결과
검증일: {now}

결론: 남은9건 추가 해결0건. 이번 색 연결 방법은 미채택한다.
공개TRAIN367개 영상에서 미리 정한785프레임·1224손후보를 처리했다. 전체899개 영상 RGB 전수검사는 아니다.
정상 배정441/441, 정상 손 교차315/315를 유지했다. 실제 배정 변경0건, 관측점18552개 보존, 남은9건72점 보존.
정상 유지 수치는 검토표본의 배정 보존 여부이며 TCN 인식 정확도가 아니다. 변경을 보류했기 때문에 유지된 것이므로 새 RGB 방법의 효과로 과장하지 않는다.

어떻게 확인했나
검출 좌표만 보던 이전 진단에 해당 프레임의 RGB 픽셀을 추가했다. 실제 보이는 범위로 먼저 자른 뒤 팔꿈치·손목 근처의 색을 기준으로 연결 영역을 찾았다. 한쪽 팔과 하나의 손만 연결될 때만 보조 근거로 허용했다. 현재프레임만 사용하고 미래프레임이나 화면밖픽셀은 쓰지 않았다.
색범위20·최대폭640 등 계수와 통과규칙은 공개 결과를 보기 전에 고정했고 결과에 맞춰 바꾸지 않았다. 특정단어/촬영자 예외는 없다.

왜9건이 해결되지 않았나
9건 모두 양팔이 몸통/옷을 거쳐 같은 색 영역에 묶였다. 사람 눈에는 서로 다른 팔인데 알고리즘은 하나의 큰 덩어리로 본 것이다. 특히 공항 장면은 배경까지 포함해 화면76.9%가 연결됐다. 이는 현재 알고리즘의 한계이며 영상에 팔 연결 정보가 전혀 없다는 뜻이 아니다.
전체1224후보 중1210개는 두 팔 연결로 모호했고,14개(11프레임)는 한쪽 팔 경로를 얻었지만 모두 원래 배정과 같았다. 새 정정은 없었다. 초기 임시 좌우가 틀린9건은 확정좌우=-1(미확정)을 유지하며 실제 손 관측은 지우지 않았다.
특정 피부색을 고정하지 않아도 색을 쓰는 이상 옷·조명·배경의 영향은 남는다. 외형과 촬영조건에 독립적이라고 주장할 수 없다.

검증 근거
합성 계약15개 통과.785프레임 각각 화면밖픽셀 교란/검출후보 순서 반전/흑백 대조 검사 통과. 흑백에서는 색 근거를 보류했으며 이것은 색 의존성 확인이지 색 독립성 증명이 아니다.
별도 검산코드로 색 영역, 연결 성분, 최단 경로, 최종 배정, 기존 좌표와 해시23507개를 확인했다.785프레임·1224후보·18552관측점 모두 일치. 독립된 사람이나 별도 에이전트의 정답 평가가 아니라 동일Codex의 별도 계산 구현이다.
시각검토는 남은9장면, 색 연결이 통과한11프레임/14후보, 정상교차 반례5장면이다.785프레임 모두를 새로 사람 정답화한 것은 아니다.

다음 방향
색이 이어지는지만 보는 규칙을 더 느슨하게 조정하지 않는다. 필요하면 몸통과 구별되는 팔 윤곽·방향을 따라 손으로 이어지는지, 양쪽 팔 경로가 동시에 가능한지 확인하는 별도 후보를 먼저 검증한다. 실제 교차·부분손·중복을 함께 확인해야 한다. 이는 아직 효과가 입증되지 않은 후속안이다.
이전 관측 보존/미확정 분리 후보와 기존C를 유지한다. 학습 입력 연결·학습·C비교는 사용자와 이번 결과를 확인한 다음 결정한다.

보존: C/TCN/DTW/웹 변경0, 새학습0,분류추론0, 사용자·멘토·DEV·보류자료사용0. 원본과 기존동결결과 읽기전용.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
rows=''.join(f'<tr><td>{html.escape(x["word"])}</td><td>{x["observed_points"]}점</td><td>양팔이 한 영역으로 합쳐짐</td><td>미확정 유지</td></tr>' for x in e['remaining9'])
figs=''.join(f'<details><summary>#{x["id"]} {html.escape(x["word"])} · 관측 {x["observed_points"]}점</summary><img loading="lazy" src="evidence/remaining_{x["id"]}.png" alt="실제 입력과 색 연결 영역 비교"></details>' for x in e['remaining9'])
positive=''.join(f'<img loading="lazy" src="evidence/positive_sheet_{i}.jpg" alt="기존 배정과 일치한 색 경로">' for i in range(1,5))
controls=''.join(f'<img loading="lazy" src="evidence/normal_{i}.png" alt="정상 손 교차 유지">' for i in [37,38,39,90,291])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>팔–손 영상 연결 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1422;color:#e6edf7;font:17px/1.75 system-ui,"Malgun Gothic",sans-serif}main{max-width:1120px;margin:0 auto;padding:46px 24px 80px}h1{font-size:37px;line-height:1.3;margin:16px 0}h2{font-size:26px;margin:0 0 18px}h3{font-size:20px}p{margin:12px 0}.tag{color:#8bbcff;font-weight:700;letter-spacing:.06em}.lead{font-size:21px;color:#c5d5e9}.result{padding:24px;background:#30291d;border-left:5px solid #efbc68;border-radius:8px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.stat{background:#192a40;padding:22px;border-radius:12px}.stat strong{display:block;font-size:35px;color:#a2d3ff}.stat span{display:block;color:#c9d7e8}.section{background:#111f32;border:1px solid #283b56;border-radius:16px;padding:28px;margin:26px 0}.small{font-size:15px;color:#a8bad0}img{width:100%;display:block;margin:16px 0;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:16px}th,td{padding:12px 10px;border-bottom:1px solid #31435c;text-align:left}th{color:#a2d3ff}details{border:1px solid #344760;border-radius:8px;padding:12px 16px;margin:10px 0}summary{cursor:pointer;font-weight:700}li{margin:8px 0}a{color:#9ac9ff}.flow{font-size:20px;background:#1a3047;border-radius:10px;padding:18px;text-align:center}.muted{color:#aac0d9}@media(max-width:700px){main{padding:24px 14px}h1{font-size:29px}.grid{grid-template-columns:1fr}.section{padding:18px}table{font-size:13px}th,td{padding:8px 4px}}
</style><main><div class="tag">WORDS IN SILENCE · 공개 자료 검증 · 2026.10.18</div><h1>팔과 손의 영상 연결을 확인했지만,<br>남은 9건은 해결되지 않았습니다.</h1><p class="lead">이번에는 실제 영상의 색 연결을 근거로 추가했습니다. 정상 배정은 보존했지만, 새로 좌우를 바로잡은 사례가 없어 이 방식은 채택하지 않습니다.</p><div class="grid"><div class="stat"><span>남은 9건 추가 해결</span><strong>0 / 9</strong><span>손 관측은 유지 · 좌우는 미확정</span></div><div class="stat"><span>기존 정상 배정 유지</span><strong>441 / 441</strong><span>새 오배정 0건</span></div><div class="stat"><span>정상 손 교차 유지</span><strong>315 / 315</strong><span>검토한 교차 장면 기준</span></div></div><div class="result"><b>이 수치는 TCN의 단어 인식률이 아닙니다.</b><br>공개 학습 자료 367개 영상의 선택된 785프레임에서 좌우 배정과 손 관측을 확인한 결과입니다. 전체 899개 영상의 RGB 전수검사는 아닙니다. 배정을 바꾸지 않았기 때문에 정상이 유지된 것이며, 새로운 인식 성능 향상이 확인된 것은 아닙니다.</div>
<section class="section"><h2>무엇을 시험했나</h2><p>팔꿈치와 손목 주변의 색을 기준으로 <b>영상 픽셀이 실제로 이어지는지</b> 확인했습니다. 하나의 손이 한쪽 팔에만 이어질 때 보조 근거로 사용하고, 양팔이나 여러 손 후보가 함께 연결되면 보류하도록 정했습니다.</p><div class="flow">화면 안 RGB → 색 연결 영역 → 한쪽 팔·한 손인지 확인 → 근거가 모호하면 보류</div><p class="small">현재 프레임만 사용했습니다. 실제 검출 화면으로 먼저 자른 뒤 분석해, 화면 밖 정보는 들어오지 않습니다. 계수와 규칙은 결과 확인 전에 고정했습니다. 기존 확정 배정·보류 결정·관측 좌표는 보호합니다.</p></section>
<section class="section"><h2>왜 잘 보이는 팔도 구분하지 못했나</h2><p>사람은 팔의 윤곽과 방향을 보고 알 수 있습니다. 하지만 이번 방식은 색이 이어지는지만 봐서, <b>팔과 검은 옷·몸통을 하나의 큰 영역으로 묶었습니다.</b> 그러면 왼팔도 오른팔도 같은 영역에 들어가 어느 쪽인지 결정할 수 없습니다.</p><img src="evidence/remaining_23.png" alt="가래떡: 왼팔·오른팔·몸통이 같은 초록 영역으로 합쳐짐"><p>오른쪽 초록색은 <b>팔을 정확히 분리한 결과가 아니라 색이 연결된 영역</b>입니다. 남은 9건 모두 이 문제가 발생했습니다. ‘공항’에서는 배경까지 묶여 화면의 76.9%가 한 영역이 되었습니다.</p><p><b>영상에 정보가 전혀 없다는 뜻은 아닙니다.</b> 이번 색 연결 알고리즘이 그 정보를 팔별로 분리하지 못한 것입니다. 고정 피부색을 쓰지 않아도 옷·조명·배경의 영향은 남습니다.</p></section>
<section class="section"><h2>남은 9건</h2><table><thead><tr><th>단어</th><th>남긴 관측</th><th>이번 실패 원인</th><th>결과</th></tr></thead><tbody>__ROWS__</tbody></table><p>기존 임시 좌우가 틀린 상태는 해결되지 않았습니다. 다만 그것을 확정 정답으로 올리지 않고 <b>좌우 미확정 상태와 실제 관측 72점</b>을 유지했습니다. 아래에서 각각의 입력과 계산 영역을 볼 수 있습니다.</p>__FIGS__</section>
<section class="section"><h2>정상 손과 손 교차도 함께 확인했습니다</h2><p>총 1,224개 손 후보 중 1,210개는 양팔이 연결되어 모호했습니다. 14개 후보(11프레임)에서 한쪽 팔 경로를 찾았지만, 모두 기존 배정과 같았습니다. <b>새 배정 변경은 0건</b>입니다.</p><p>정상 441건과 교차 315장면은 기존 결과를 유지했고, 선택된 입력의 손 관측 18,552점도 보존했습니다. 이들 표본은 겹칠 수 있으므로 더해서 독립 시험 횟수로 계산하지 않습니다.</p><details><summary>한쪽 팔 경로가 나온 11프레임 보기</summary><p class="small">분홍 선은 계산된 경로입니다. 대부분 팔이 따로 보이는 장면이며, 어려운 손 교차 해결을 입증하지 않습니다.</p>__POSITIVE__</details><details><summary>정상 손 교차 반례 5장면 보기</summary><p class="small">이전 최근접 팔꿈치 규칙으로는 잘못 바뀌던 정상 사례들입니다. 이번 방식은 모호함을 이유로 보류해 기존 배정을 유지했습니다.</p>__CONTROLS__</details></section>
<section class="section"><h2>계산 오류인지도 따로 확인했습니다</h2><ul><li>별도 코드로 785프레임의 색 영역·연결 성분·경로·최종 배정을 다시 계산해 일치를 확인했습니다.</li><li>합성 계약 15개, 화면 밖 픽셀 변경·후보 순서 변경·흑백 대조를 확인했습니다.</li><li>흑백에서는 색 근거를 보류했습니다. 이는 색에 의존함을 확인한 검사이며 색에 독립적이라는 증거가 아닙니다.</li><li>기존 파일과 동결 코드의 해시 23,507개를 확인했습니다.</li></ul><p class="small">구현과 시각 검토는 동일 Codex, 수치 검산은 별도 구현입니다. 독립된 사람의 정답 평가가 아닙니다. 계산 검증 통과와 방법의 효과는 별개이며, 이번 방법의 추가 개선은 0건입니다.</p></section>
<section class="section"><h2>다음에는 무엇이 달라져야 하나</h2><p>색 범위를 느슨하게 바꾸는 것만으로는 해결 근거가 부족합니다. 다음 후보는 <b>몸통과 구별되는 팔 윤곽·방향</b>을 보고, 손까지 이어지는 두 팔의 가능한 경로를 함께 비교해야 합니다. 겹치거나 근거가 끊긴 부분은 계속 미확정으로 남겨야 합니다.</p><p>이 방향도 아직 성능이 검증된 것은 아닙니다. 부분 손·정상 교차·중복 검출을 함께 검증하고, 옷과 촬영 조건이 달라진 경우도 확인한 뒤 판단해야 합니다.</p><p><b>기존 C와 이전 관측 보존 후보는 그대로입니다.</b> 사용자 촬영, 학습, 튜닝, TCN·DTW·웹 운영 코드는 건드리지 않았습니다. 결과를 함께 확인한 뒤 다음 검증이나 학습 입력 연결 조건을 정합니다.</p></section>
<p class="small"><a href="RESULT_KO.txt">전체 텍스트</a> · <a href="RESULT.json">집계 결과</a> · <a href="INDEPENDENT.json">별도 검산</a> · <a href="VISUAL_REVIEW.json">시각 검토 범위</a> · <a href="NEXT_PLAN.json">후속 조건</a></p></main></html>'''
for k,val in [('ROWS',rows),('FIGS',figs),('POSITIVE',positive),('CONTROLS',controls)]:page=page.replace('__'+k+'__',val)
(U/'RESULT_KO.html').write_text(page,encoding='utf8')

# Synchronize the latest state only; prior frozen reports stay untouched.
status='WORD100_IMAGE_ARM_CONNECTIVITY_VALIDATED_NEGATIVE_AWAITING_REVIEW'
summary='ops/word100_image_arm_connectivity_20261018. 화면안RGB 팔–손 색연결 별도후보 검증완료. 공개TRAIN367영상·선택785프레임/1224후보(전체899RGB전수아님). 남은9해결0/미확정9/관측72점보존. 정상441·교차315·관측18552점유지,배정변경0.1210후보는양팔이몸통/옷을거쳐한영역으로합쳐짐,14후보/11프레임단일팔색경로는기존배정과동일.계산PASS와효과를구분;색연결은해부학적팔분리근거로부족하여미채택.계약15·785화면밖/순서/흑백검사·별도scipy연결/BFS검산PASS·보호23507핀유지.9실패/11양성/5교차반례시각검토 동일Codex,독립사람검토아님. C/웹/TCN/DTW유지·사용자/멘토/DEV/보류사용0·학습/분류0.'
nexttext='이번 색연결 후보의 남은9추가해결0/정상보존 결과를 사용자와 확인한다. 기존C와 U관측보존/미확정상태 유지·학습대기. 새RGB방식이필요하면몸통과구별되는팔윤곽·방향및양쪽가능경로를검증하는별도후보를먼저설계한다. 옷/조명/배경에독립적이라고주장하지않으며알려진9건에맞춰색문턱·시간조건완화금지. 정상교차/부분손/중복통제와새조건검증을함께동결.학습입력연결은사용자결과확인후조건부터정한다. 완료785RGB검증/별도검산재실행불필요.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf8')
for header,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','### 2026-10-18 화면 안 팔–손 RGB 연결 검증 완료 — 추가 개선 없음',summary),('## 9. 다음 단일 작업','### 2026-10-18 영상 연결 검증 결과 검토·학습 대기',nexttext)]:
 assert t.count(header)==1
 t=t.replace(header+'\n',header+'\n\n'+title+'\n'+body+'\n',1)
p.write_text(t,encoding='utf8')
p=ROOT/'research/front_keyphase_v1/ops/STATE.json';state=read(p);state['updated_at']=now;state['updated_date_kst']=now;state['current_stage']=status;state[U.name]={**result,'next_action':nexttext};write(p,state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}

### 현재 목표·마지막 완료
{summary}

### 다음 단일 작업
{nexttext}

### 구현·검증·근거
- candidate.py: 화면 안 픽셀만 사용한 프레임별 색 연결 후보. 운영 연결 없음. 결과 전 계수 동결.
- CONTEXT.json: 367영상/391영상조건/785프레임 선택, 기존 근거와 코드23507핀. run.py 실행 exec72591 종료0·RUNNING 잠금없음.
- CONTRACTS.json 15PASS. INDEPENDENT.json: exec29667 종료0,785프레임/1224후보/18552점 별도코드 검산PASS. 동일Codex의 별도구현이며 다른사람/에이전트의 독립판정아님.
- 남은9건 모두양팔색영역합쳐짐. 정상441/교차315보존은배정변경0에따른보존이며새인식성능향상아님. RGB양성14후보도기존과동일. 새좌우확정승격0.
- VISUAL_REVIEW.json: 남은9실제입력/양성11프레임14후보/정상교차5반례검토. 나머지785전부새로운시각정답화아님. evidence/ 비교그림, RESULT_KO.html/txt·RESULT/NEXT_PLAN·FINAL_AUDIT.
- 현재RGB후보미채택. 이전 word100_uncertain_hand_preservation_20261018(U)와 C 유지. 남은9관측72점과확정좌우=-1 유지. 현재 학습용 최종입력에연결되지않음.
- 원본/W:/sources/과거동결결과읽기전용. 사용자·멘토·DEV·보류읽기0. 학습/분류추론0, C/TCN/DTW/웹변경0. 카메라/다운로드/외부전송/Git없음. st-gcn자동화삭제유지.
''',encoding='utf8')

# Final audit checks artifacts, all pre-existing pins, and the three authoritative docs.
for p,h in c['pins'].items():assert sha(p)==h,p
for p,h in read(U/'OUTPUT_PINS.json').items():assert sha(p)==h,p
links=re.findall(r'(?:src|href)="([^"]+)"',page)
for link in links:assert (U/link).is_file(),link
for p in U.glob('*.json'):read(p)
assert sum(x['observed_points'] for x in e['remaining9'])==72
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md']:assert U.name in p.read_text('utf8')
write(U/'FINAL_AUDIT.json',{'status':'PASS','at':now,'protected_pins':len(c['pins']),'output_pins':len(read(U/'OUTPUT_PINS.json')),'all_report_links_exist':len(links),'processes_completed':[72591,29667],'running_lock':False,'independent':'PASS','method':'NOT_ADOPTED_NO_IMPROVEMENT','documentation_synchronized':True,'artifact_hashes':{str(p):sha(p) for p in U.glob('*') if p.is_file() and p.name!='FINAL_AUDIT.json'},'prohibited_activity':{'training':0,'classifier_inference':0,'user_reads':0,'C_changed':False}})
print('FINAL PASS',U/'RESULT_KO.html')
