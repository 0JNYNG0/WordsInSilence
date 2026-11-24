import json,hashlib,html,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
c=read(U/'CONTEXT.json');s=read(U/'SUMMARY.json');v=read(U/'INDEPENDENT.json');a=read(U/'ANNOTATED.json');r=read(U/'RESULTS.json');changes=read(U/'CHANGES.json');idx=read(U/'EVIDENCE_INDEX.json');blocked=read(U/'BLOCKED_CONFLICTS.json')
assert v['status']=='PASS' and s['remaining9_correct']==8 and s['normal_regressions']==0 and s['crossing_changed_frames']==0
assert len(changes)==13 and not (U/'RUNNING.json').exists()
rem=[x for x in a if x['remaining9']];normal=[x for x in a if x['verdict']=='normal_identity'];side={0:'왼손',1:'오른손',-1:'미확정'}
visual=[]
for x in changes:
    z=r[x['record_index']]
    visual.append({**x,'word':z['word'],'judged_actual_side':x['after'],'corrected':True,'source_support':'Saved actual sensor input shows the candidate wrist continuing into that anatomical forearm. Both forearm contours visible; opposite path crosses torso without paired forearm edges. No offsensor pixels consulted.'})
write(U/'VISUAL_REVIEW.json',{'at':now,'reviewer':'Codex implementer; separate numeric verifier, no independent human/agent',
 'scope':'All13 actual changes, unresolved9thcase,5known normal crossing counterexamples,12representative clip/view blocked conflicts. Not all785frames newly visually labeled.',
 'all_changes':visual,'remaining9_corrected_ids':[x['id'] for x in rem if x['after']==x['actual_side']],
 'unresolved':{'id':309,'word':'공항','problem':'Two raw candidates around one visible right hand; distorted wrist placement and oblique/foreshortened palm-forearm region. Intended right path only6/12 paired sections with discontinuity and edgejump; other candidate2/12. No reliable unique path. Keep original provisional hints and unknown identity, no forced merge/discard.'},
 'normal_guard_counterexample':{'id':38,'word':'후원','frame':75,'raw':1,'actual_side':1,'standalone_contour_side':0,'guard':'Both candidates propose same arm, so no side change. Evidence alone would be wrong.'},
 'blocked_conflicts':{'count':29,'keep_confirmed':28,'duplicate_destination':1,'representative_views_seen':12,'interpretation':'28 disagreements with prior evidence-backed assignments are not28 newly labeled errors. They occur in overlap/crossing and were not applied. Single-frame contour alone is unsafe as an override.'},
 'viewed_files':[str(U/'evidence'/f'changes_sheet_{i}.jpg') for i in range(1,6)]+[str(U/'evidence'/f'case_{i}.png') for i in [309,37,38,39,90,291]]+[str(U/'evidence'/f'conflict_sheet_{i}.jpg') for i in range(1,5)],
 'limitations':['Known development sample','Mostly border entry/rest forearms among the corrected9case subset','No finger geometry reconstruction','No arbitrary skin/outfit/lighting invariance proof','No recognition accuracy measurement']})
normal_evidence=Counter('unknown' if x['evidence_side']<0 else 'correct' if x['evidence_side']==x['actual_side'] else 'wrong' for x in normal)
nextplan={
 'status':'AWAIT_USER_REVIEW_NO_TRAINING',
 'retain':'Research-only guarded contour candidate; originalC and U results unchanged.',
 'next_single_action':'Review8/9 corrections and remaining duplicate/oblique forearm case with user. Before training, define how contour-supported provisional assignments differ from anatomical confirmation and how to integrate observed coordinates and uncertainty without erasing normal hands.',
 'if_continuing_validation':'Investigate wrist endpoint quality and multiple raw candidates on one physical hand jointly, using same public controls and preserved original points. Do not relax contour thresholds for 공항. Validate outside the known785frame sample before claiming general robustness.',
 'before_training':['Freeze a publicTRAIN-only broader validation and input contract after user review','Preserve confirmed assignments and duplicate-destination guards','Keep ambiguous contour as unknown; never fabricate missing finger coordinates','No user/mentor/DEV/heldout tuning','Compare with C only under approved frozen conditions'],
 'not_claimed':['General hand identity solved','Increased TCN accuracy','Full899videoRGB validation','Ready to replace web or C']}
write(U/'NEXT_PLAN.json',nextplan)
result={**s,'status':'VALIDATED_BOUNDED_PARTIAL_IMPROVEMENT_AWAITING_REVIEW','at':now,
 'remaining9_corrected':8,'remaining9_unresolved':1,'remaining9_points_preserved':72,
 'changes_source_verified_correct':13,'additional_neighbor_corrections':5,'changed_clips':len({x['clip_id'] for x in changes}),
 'normal_standalone_contour':dict(normal_evidence),'blocked_conflicts':{'previous_confirmed':28,'duplicate_destination':1},
 'all_observed_points_preserved':18552,'image_frames':785,'selected_clips':367,'full899_rgb_replayed':False,
 'research_candidate_retained':True,'production_applied':False,'prior_identity_state_promotion':False,
 'independent':'PASS','contract_tests':16,'protected_pins':len(c['pins']),
 'verification_recovery':'First scalar-verifier outside-image interpolation mismatch preserved. Reference coefficients repaired; frozen candidate and saved results unchanged. Valid samples already agreed; full finalcheckPASS.',
 'training_authorized_now':False,'source_visual_reviewer':'SameCodex','result':str(U/'RESULT_KO.html'),'next_action':nextplan['next_single_action']}
write(U/'RESULT.json',result)
text=f'''팔 윤곽·방향 활용 검증 결과
{now}

결론: 보호 규칙을 함께 사용한 별도 후보에서 남은9건 중8건의 임시 좌우를 올바르게 정정했다. 정상441건·교차315장면은 유지했다. 기존C/웹에는 적용하지 않았다.
범위: 공개TRAIN367영상의 고정785프레임·1224손후보. 전체899영상RGB전수 또는 새사람 시험이 아니다.

방법
팔꿈치에서 각각의 손목 후보로 향하는 양쪽 경로를 같은 기준으로 조사했다. 경로를 가로지르는12개 단면에서 팔 양쪽 윤곽, 팔에 맞는 폭, 방향, 연속성을 검사했다. 몸통을 가로지르는 넓거나 끊긴 경로는 통과하지 못하도록 했다. 현재프레임의 실제화면안 흑백명암만 사용했다. 가까운 팔꿈치 선택이나 화면좌우반분 규칙이 아니다. 색이나 촬영자/단어ID는 특징으로 사용하지 않았다.
하나의 팔 경로만 통과하고 다른 손과 충돌하지 않을 때만 미확정 임시 좌우를 정정한다. 기존 확정 배정·보류 결과는 바꾸지 않는다. 모호한 좌표를 없애거나 손가락을 만들어 넣지 않는다.

결과
- 남은9건:8정정/공항1미해결.8건은가래떡·독재자·추출·유성구주민지원센터·가평군주민지원센터·강진군주민지원센터·진도군수어통역센터·칠천.
-8건의 올바른 팔은12/12단면을 충족했고 반대팔은0/12였다. 기존의 색연결후보는0/9였다.
- 인접5프레임도정정되어 전체13손/프레임이 바뀌었다.13전체 실제 입력 대조로 올바른 해부학적 측과 일치함을 확인했다.8영상의 상관된 장면이며13독립사용자시험이 아니다.
- 정상441/441·교차315/315유지.새실제오배정0. 전체관측18552점/남은9의72점보존.최종학습텐서/신뢰상태승격은아직미연결.

남은문제와필수보호
공항에는같은오른손주변에두검출후보가있다.손목위치가서로다르고팔이카메라쪽으로향해윤곽이짧고손바닥윤곽과섞인다.문제후보의오른팔경로는6/12단면만충족하고연속성과폭변화조건도실패했다.다른중복후보도2/12에그쳤다.안전하게정정/합치기/삭제할근거가없어미확정을유지했다.
정상후원#38에서는윤곽만보면잘못된왼팔을제안한다.두손후보가같은팔로몰리는것을막는규칙덕분에기존오른손이유지됐다.전체선정범위에서기존확정배정과충돌한윤곽제안28건도차단했다.28건전부를새정답오류로라벨링한것은아니다. 따라서윤곽방법을기존추적대신단독사용하면안된다.
정상441의단독윤곽결과는맞는측388/보류52/틀린측1이다.보호규칙포함결과와단독윤곽성능을구분한다.

검산
합성계약16PASS.785프레임각각후보순서변경·무정보영상·명암반전검사통과.색정보미사용은피부/옷/조명변화에강하다는보장이아니다.
별도scipy필터와scalar샘플링/윤곽쌍/결정코드로785프레임·2448경로가설·29316단면·1224손후보를검산했다.25,089보호핀보존.
최초검산실패는검산코드의화면밖무효샘플보간계수차이였다.첫오류와코드보존후검산식만수정했다.유효영역은이미일치했고,동결후보·계수·저장결과변경없이전체PASS했다.
구현·시각검토동일Codex/수치검산별도구현.독립사람/새사용자평가아님.

한계및다음
이번8/9는알려진좌우문제장면의정정수이며TCN정확도88.9%라는뜻이아니다.대부분화면아래부분손/준비구간의문제였다.수어인식률향상과동작중겹침해결은아직별도검증이필요하다.
먼저이결과와공항의중복/손목근거문제를사용자와검토한다.이후영상근거를기존좌우확정·관측보존정보와어떻게연결할지정하고,알려진표본외공개자료에서도검증해야한다.학습과C비교는결과확인후동결조건으로진행한다.
C/TCN/DTW/웹변경0.새학습/분류추론0.사용자·멘토·DEV·보류자료사용0.카메라·다운로드·외부전송·Git없음.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
rows=''.join(f'<tr><td>{html.escape(x["word"])}</td><td>{side[x["before"]]}</td><td>{side[x["after"]] if x["after"]==x["actual_side"] else "미해결 유지"}</td><td>{"윤곽 12/12 · 정정" if x["after"]==x["actual_side"] else "윤곽 불완전 · 중복 후보"}</td></tr>' for x in rem)
gallery=''.join(f'<details><summary>변경 {x["display_id"]} · {html.escape(x["word"])} · 프레임 {x["index"]}</summary><img loading="lazy" src="evidence/change_{x["display_id"]:02}.png" alt="실제 입력과 팔 윤곽 대조"></details>' for x in idx['changes'])
blocked_gallery=''.join(f'<img loading="lazy" src="evidence/conflict_sheet_{i}.jpg" alt="기존 배정과 충돌해 차단한 윤곽 근거">' for i in range(1,5))
css=re.search(r'<style>(.*?)</style>',(U.parent/'word100_image_arm_connectivity_20261018/RESULT_KO.html').read_text('utf8'),re.S).group(1)
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>팔 윤곽·방향 검증 결과</title><style>__CSS__</style><main><div class="tag">WORDS IN SILENCE · 공개 자료 검증 · 2026.10.18</div><h1>팔의 윤곽과 방향으로<br>남은 9건 중 8건을 정정했습니다.</h1><p class="lead">정상 손과 교차 장면을 유지한 별도 후보 결과입니다. 기존 C와 웹에는 아직 적용하지 않았습니다.</p><div class="grid"><div class="stat"><span>남은 문제 9건 중</span><strong>8건 정정</strong><span>이전 색 연결 방식: 0건</span></div><div class="stat"><span>정상 좌우 배정 유지</span><strong>441 / 441</strong><span>검토 범위 내 새 오배정 0건</span></div><div class="stat"><span>정상 손 교차 유지</span><strong>315 / 315</strong><span>기존 배정 보호 규칙 포함</span></div></div><div class="result"><b>이것은 TCN의 단어 인식률이 아닙니다.</b><br>공개 TRAIN 자료 367개 영상에서 고른 785프레임의 손 좌우 문제를 검증했습니다. 학습·새 사용자 시험·전체 899개 영상의 RGB 전수검사는 하지 않았습니다.</div>
<section class="section"><h2>이번에는 어떻게 달라졌나</h2><p>이전에는 비슷한 색이 이어지는지 보다가 팔과 몸통이 한 덩어리로 묶였습니다. 이번에는 팔꿈치에서 손목으로 향하는 길을 따라 <b>양쪽 가장자리가 팔처럼 좁고 연속적으로 이어지는지</b> 확인했습니다.</p><div class="flow">양쪽 팔 경로를 각각 조사 → 윤곽·폭·방향 확인 → 한쪽만 확실하고 충돌이 없으면 정정</div><p>가까운 팔꿈치만 고르거나 화면을 좌우로 반씩 나누지 않습니다. 손가락이 일부만 보여도 팔 윤곽이 충분히 보이면 어느 손인지 확인할 수 있습니다. <b>안 보이는 손가락 모양을 복원한 것은 아닙니다.</b></p><img src="evidence/case_23.png" alt="가래떡 왼손으로 이어지는 양쪽 팔 윤곽"><p class="small">초록 선은 조건을 통과한 팔 양쪽 윤곽입니다. 12/12는 조사한 12개 단면에서 윤곽이 잡혔다는 뜻이며, 동작 점수나 성공 확률이 아닙니다. 좌우는 촬영자 본인 기준입니다.</p></section>
<section class="section"><h2>남은 9건의 결과</h2><table><thead><tr><th>단어</th><th>기존 임시 좌우</th><th>별도 후보 결과</th><th>영상 근거</th></tr></thead><tbody>__ROWS__</tbody></table><p>정정된 8건은 맞는 팔의 윤곽이 12개 단면 모두에서 이어졌고, 반대 팔을 향한 경로는 0개였습니다. 관련 인접 5프레임도 바뀌어 <b>총 13개 손/프레임의 정정</b>을 확인했습니다. 모두 실제 입력과 대조했습니다.</p><p class="small">8개 영상의 서로 연관된 프레임입니다. 13명의 사용자나 13번의 독립 시험을 의미하지 않습니다. 기존 신뢰 상태를 자동 승격하거나 학습용 입력을 바꾸지 않았습니다.</p><details><summary>정정한 전체 13프레임 보기</summary>__GALLERY__</details></section>
<section class="section"><h2>‘공항’ 1건은 아직 남았습니다</h2><img src="evidence/case_309.png" alt="공항: 중복 손 후보와 불완전한 윤곽"><p>하나의 오른손 주변에 두 검출 후보가 잡혀 있고 손목 위치도 서로 다릅니다. 팔이 카메라 쪽으로 향해 보이는 길이가 짧고, 팔과 손바닥 윤곽이 섞여 있습니다.</p><p>문제 후보의 오른팔 방향은 12개 단면 중 6개만 잡혔고, 윤곽이 끊기거나 급하게 바뀌었습니다. 다른 중복 후보도 충분한 연결을 얻지 못했습니다. <b>좌우를 억지로 정하거나 손을 삭제하지 않고 미확정 상태를 유지했습니다.</b> 기존 임시 좌우의 오류도 이 장면에서는 아직 해결되지 않았습니다.</p></section>
<section class="section"><h2>윤곽만 믿으면 안 되는 이유도 확인했습니다</h2><p>정상 ‘후원’ 교차 장면에서는 윤곽만 보면 오른손을 왼팔에 연결하는 잘못된 제안이 나왔습니다. 하지만 <b>두 손 후보를 같은 팔에 중복 배정하지 않는 규칙</b>이 이를 막아 정상 배정을 유지했습니다.</p><img src="evidence/case_38.png" alt="정상 후원: 잘못된 윤곽 제안을 중복 배정 방지로 차단"><p>전체 선정 범위에서 기존 확정 배정과 다른 윤곽 제안 28건도 보호 규칙으로 차단했습니다. 이 28건을 모두 새롭게 오답으로 판정한 것은 아닙니다. 핵심은 <b>윤곽 근거가 기존 추적을 무조건 덮어쓰면 안 된다</b>는 것입니다.</p><table><thead><tr><th>정상 441건에서</th><th>결과</th></tr></thead><tbody><tr><td>윤곽 근거만 단독으로 볼 때</td><td>맞는 측 388 · 보류 52 · 틀린 측 1</td></tr><tr><td>보호 규칙을 포함한 최종 후보</td><td>정상 441건 모두 유지</td></tr></tbody></table><details><summary>차단한 충돌의 대표 장면 보기</summary>__BLOCKED__</details></section>
<section class="section"><h2>보존과 검산</h2><ul><li>선택된 손 관측 18,552점과 남은 9건의 72점을 그대로 보존했습니다.</li><li>합성 검사 16개, 785프레임의 후보 순서 변경·빈 영상·명암 반전 대조를 통과했습니다.</li><li>별도 코드로 2,448개 팔 경로 가설과 29,316개 단면을 다시 계산해 일치함을 확인했습니다.</li><li>기존 자료와 동결 코드 25,089개 해시를 확인했습니다.</li></ul><details><summary>검산 중 발견한 구현 문제와 처리</summary><p>첫 검산에서는 화면 밖 무효 샘플을 계산하는 검산 코드의 보간식 차이를 발견했습니다. 최초 오류를 남긴 뒤 검산식만 바로잡았습니다. 유효 영역은 이미 일치했고, 동결 후보·계수·저장 결과는 바꾸지 않은 채 최종 전체 검산을 통과했습니다.</p></details><p class="small">구현·영상 검토는 동일 Codex, 수치 검산은 별도 코드입니다. 독립된 사람의 평가가 아닙니다. 검산 통과와 새 사용자 성능 검증은 별개입니다.</p></section>
<section class="section"><h2>현재 판단과 다음 단계</h2><p><b>부분 손이 보일 때 잘못 붙은 좌우를 바로잡는 보조 근거로는 가능성을 확인했습니다.</b> 특히 이번에 정정한 문제들은 화면 아래쪽에서 손이 등장하는 준비 구간에 해당합니다. 동작 중 복잡한 겹침까지 해결했다고 볼 수는 없습니다.</p><p>색을 사용하지 않더라도 옷·조명·가림 때문에 윤곽이 약해질 수 있습니다. 알려진 표본 밖의 공개 자료에서도 확인해야 하며, TCN 인식률이 얼마나 오를지는 아직 알 수 없습니다.</p><p>이번 결과를 함께 확인한 뒤, 남은 중복 후보·손목 위치 근거를 보완할지와 학습 입력에 어떤 정보로 연결할지 정합니다. <b>기존 C·TCN·DTW·웹은 유지했고, 사용자 촬영은 읽거나 학습·튜닝에 사용하지 않았습니다. 학습은 아직 시작하지 않았습니다.</b></p></section><p class="small"><a href="RESULT_KO.txt">전체 텍스트</a> · <a href="RESULT.json">집계</a> · <a href="INDEPENDENT.json">별도 검산</a> · <a href="VISUAL_REVIEW.json">영상 검토 범위</a> · <a href="NEXT_PLAN.json">다음 조건</a></p></main></html>'''
for key,value in [('CSS',css),('ROWS',rows),('GALLERY',gallery),('BLOCKED',blocked_gallery)]:page=page.replace('__'+key+'__',value)
(U/'RESULT_KO.html').write_text(page,encoding='utf8')

# Current documents only; previous reports and original arrays stay untouched.
summary='ops/word100_forearm_contour_20261018. 팔윤곽·방향+기존배정보호 별도후보 검증완료. 공개TRAIN367영상/고정785프레임·1224손후보. 남은9중8임시좌우정정/공항1미해결,인접5포함13손프레임정정(8영상),전체13원본입력대조일치. 정상441·교차315유지·관측18552점/잔존9의72점보존. 단독윤곽은정상441중388맞음/52보류/1오답이며#38중복배정방지가정상보호;기존확정과충돌28도차단→윤곽단독교체금지. 공항은중복후보/왜곡손목·짧은윤곽6/12로미해결. 계약16·785순서/빈영상/명암반전·별도scipy2448경로/29316단면검산PASS·보호25089핀유지. 최초독립검산무효샘플보간불일치증거보존후검산식만수정,동결후보/결과불변. 전체899RGB전수/새사람/TCN정확도아님. 동일Codex시각검토+별도검산,운영/학습입력미적용·C/웹/사용자자료유지·학습/분류0.'
nexttext='팔윤곽후보의8/9정정·공항1미해결·교차보호규칙필수결과를사용자와확인한다. 기존C/U유지·학습대기. 남은중복후보와손목위치의신뢰를함께검증하되정상손삭제·공항맞춤윤곽문턱완화금지. 후보는보조근거이며확정배정덮어쓰기금지. 이후알려진785표본외공개TRAIN검증과최종입력연결(좌우잠정/확정/관측좌표분리)계약을먼저정한다. 사용자/멘토/DEV/보류로튜닝하지않는다. 학습·기존C비교는사용자결과확인후동결조건으로만. 완료785검증·별도검산재실행불필요.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf8')
for header,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','### 2026-10-18 팔 윤곽·방향 후보 검증 완료 — 남은9 중8 정정',summary),('## 9. 다음 단일 작업','### 2026-10-18 팔 윤곽 후보 검토·입력 연결 및 학습 대기',nexttext)]:
 assert t.count(header)==1
 t=t.replace(header+'\n',header+'\n\n'+title+'\n'+body+'\n',1)
p.write_text(t,encoding='utf8')
p=ROOT/'research/front_keyphase_v1/ops/STATE.json';state=read(p);state['updated_at']=now;state['updated_date_kst']=now;state['current_stage']='WORD100_FOREARM_CONTOUR_VALIDATED_PARTIAL_IMPROVEMENT_AWAITING_REVIEW';state[U.name]={**result,'next_action':nexttext};write(p,state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}

### 현재 목표·마지막 완료
{summary}

### 다음 단일 작업
{nexttext}

### 근거·실행·검증
- candidate.py: 흑백Sobel,양팔방향12단면의대칭윤곽·폭·연속성,유일팔/기존확정우선/중복목적지차단.현재프레임·화면안정보만.공개결과전PARAMS동결.임시side정정출력만,기존identity_side승격/좌표수정없음.
- run.py freeze후run exec77155종료0,785프레임실행49.6초.임시좌우13정정/정상441·교차315유지.그중알려진9는8정정.나머지5는인접프레임.누적13독립사용자시험이아님.
- verify.py 최초실패FIRST_VERIFY_ERROR/verification_recovery보존.화면밖무효샘플의검산보간계수만복구.동결후보/RESULTS해시불변.최종exec96391종료0/24.96초,785프레임2448가설29316단면검산PASS.다른사람/에이전트판정아님.
- VISUAL_REVIEW:13변경전부/공항1/정상교차5/차단충돌12대표조건시각검토.색연결단위의입력RGB재사용·원본검증핀유지.이전단위재추출/재검산없음.
- 단독윤곽정상388맞음/52불확실/1오답.후원#38두후보같은팔금지규칙이막음.별도28기존확정충돌도보호.새785전체시각정답/28전부오류라벨주장금지.
- 공항sensor40raw0:오른팔윤곽6/12·끊김/폭급변,동일물리손중복raw1은2/12.미해결로관측보존.8정정은모두초기/준비구간부분손.수어인식률향상/새환경일반화는미검증.
- RESULT_KO.html/txt,RESULT/NEXT_PLAN/FINAL_AUDIT,evidence/원본+윤곽그림.보호25089핀유지.
- 현재실행/잠금없음. 기존C/TCN/DTW/웹·사용자자료유지.학습/분류/사용자·멘토·DEV·보류사용0.원본W:/sources/과거결과읽기전용.카메라/다운로드/외부전송/Git없음.st-gcn자동화삭제유지.
''',encoding='utf8')
for pinset in [c['pins'],read(U/'OUTPUT_PINS.json')]:
 for path,h in pinset.items():assert sha(path)==h,path
recovery=read(U/'verification_recovery/RECOVERY.json');assert sha(U/'candidate.py')==recovery['candidate_unchanged'] and sha(U/'RESULTS.json')==recovery['results_unchanged']
for p in U.glob('*.json'):read(p)
links=re.findall(r'(?:src|href)="([^"]+)"',page)
for link in links:assert (U/link).is_file(),link
assert normal_evidence=={'correct':388,'unknown':52,'wrong':1} and len(blocked['blocked'])==29
assert sum(x['usable_points'] for x in rem)==72
write(U/'FINAL_AUDIT.json',{'status':'PASS','at':now,'protected_pins':len(c['pins']),'output_pins':len(read(U/'OUTPUT_PINS.json')),'report_links':len(links),'recovery_candidate_and_results_unchanged':True,'documentation_synchronized':True,'processes_completed':[77155,96391],'running_lock':False,'new_training':0,'user_reads':0,'production_changed':False,'files':{str(p):sha(p) for p in U.glob('*') if p.is_file() and p.name!='FINAL_AUDIT.json'}})
print('FINAL PASS',U/'RESULT_KO.html')
