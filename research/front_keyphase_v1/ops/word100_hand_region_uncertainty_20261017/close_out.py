import json,hashlib,re
from pathlib import Path
from datetime import datetime
U=Path(__file__).resolve().parent;R=U.parents[3];O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
    with (U/n).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def write(n,s):
    with (U/n).open('x',encoding='utf-8') as f:f.write(s)
c=read(U/'INDEPENDENT_CHECK.json');s=read(U/'SCOPE.json');assert c['status']=='PASS'
at=datetime.now().astimezone().isoformat(timespec='seconds')
assert c['states_checked']==15926 and c['regions_checked']==2005
assert len(s['records'])==60 and len({x['word_code'] for x in s['records']})==28
assert not c['changed_confident_core_states']
result={
 'status':'IMPLEMENTED_VALIDATED_NOT_ADOPTED_EXCESSIVE_ABSTENTION', 'at':at,
 'operator':'Codex local implementation and scalar independent verification; same operator, no agent or expert review',
 'unit':'word100_hand_region_uncertainty_20261017',
 'public_scope':{'videos':60,'word_types':28,'known_train_people':9,'previous_videos':48,'fresh_word_videos':12,
                 'frames':7963,'hand_states':15926,'core_frames':3158,'manual_missing_hand_frames':109,'manual_observed_controls':32},
 'visual_location_summary':c['visual_location_summary'],'overlap_specific':c['overlap_specific'],
 'improvements':['Prior R016 and fresh N001 boundary-to-inside errors became uncertainty, not recovered correct boundary labels.',
                 'Fresh N023 visually separated hands no longer receive overlap-likely; changed to uncertainty.',
                 'Existing usable coordinates, masks and assignments unchanged. History expires, missing duration/shape variation expands diagnostic envelope.'],
 'regressions':['Fresh38 missing samples: correct broad location35->25, error1->0, abstentions2->13.',
                'Previous71 missing samples: correct66->24, error1->0, abstentions4->47.',
                'Fresh10 visual-overlap samples: specific overlap detection7->1. Previous19:1->0.',
                'Mind06 gap14frames: previously14 generic interior-missing, now1 interior and13 uncertain; no overlap recovery.'],
 'cause':'Largest per-landmark hand-to-pose-wrist change between last two observed frames conflates shape change, tracking fluctuation and pose bias. Linear growth over gap makes envelope too broad; requiring intersection at all envelope corners removes many true overlaps. Independent calculation passes; not unit/timestamp arithmetic error.',
 'acceptance':{'calculation':'PASS','observation_preservation':'PASS','location_and_overlap_usefulness':'REGRESSION; reject as replacement','recognition_improvement':'NOT_MEASURED'},
 'post_output_safety':'No core frame acquired a new confident outside/overlap state; saved state pairs independently checked. Thus zero changed-confident runs to render under frozen protocol. Fresh54 blinded images and4 explanatory examples viewed.',
 'limits':['Selected TRAIN frames, correlated frames and one visual reader; not independent user performance.',
           'Region is a convex envelope from21 previous landmarks, not RGB silhouette or 3D hand.',
           'Uncertainty is an engineering bound, not calibrated probability; no proof true hand lies within it.',
           'Visible projected contact/overlap cannot prove why detector failed.',
           'Observed controls may be partly cut; observed means prior usable palm points, not validated whole-hand truth.',
           'Baseline fallback remains when reliable full-hand history absent; not every missing hand gets region evidence.'],
 'new_training':0,'classifier_inference':0,'new_detection':0,'user_recording_reads':0,'production_changes':0,
 'next_single_action':'사용자와 미채택 결과 확인. 손 위치 근거와 겹침·형태 불확실성을 별도 정보로 유지하는 방향 검토. 현재 후보 기준 완화나 자동 학습 금지; 학습은 함께 결과 확인 후.',
 'evidence':{n:str(U/n) for n in ['SCOPE.json','region.py','SYNTHETIC_TESTS.json','VISUAL_LABELS.json','PREDICTIONS.json','INDEPENDENT_CHECK.json','REGION_EXAMPLES.jpg','FIRST_TEST_ERROR.txt']}}
save('RESULT.json',result)
text='''손 전체 영역·움직임 불확실성 보완 결과

결론: 구현과 공개 검증은 완료했지만, 기존 구분 방식을 대체하는 후보로는 미채택한다.
손목 한 점 대신 마지막 실제 관측의 손21점 영역을 사용하고, 누락 시간과 손–팔 상대좌표 변화가 커질수록 추정 범위를 넓혔다. 일부 잘못된 단정을 줄였지만, 너무 많은 정상 구분까지 판단 보류로 바뀌었다. 특히 겹침 검출이 좋아지지 않았다. 모델 인식률을 높였다는 결과가 아니다.

무엇을 구현했나
1. 마지막 실제 손 관절21점의 외곽을 감싸는 영역을 저장한다. RGB 윤곽선이나 3D 복원이 아니다.
2. 현재 팔의 손목 위치를 이용해 그 영역이 있을 만한 위치를 추정한다. 추정 영역을 학습용 손 관절로 넣지는 않는다.
3. 직전 두 관측의 손–팔 상대좌표 변화와 안 보인 시간을 이용해 불확실성 범위를 넓힌다. 이는 확률로 보정된 신뢰구간이 아니다.
4. 경계에 걸릴 수 있거나 범위가 너무 커지면 판단을 보류한다. 겹침은 불확실성 범위 안에서 두 손 영역이 계속 교차하는지를 본다.
단어명·사람ID·미래 프레임·원본에서 가려진 영역은 입력으로 사용하지 않는다. 관측된 손 좌표·관측 마스크·좌우 배정은 그대로 둔다. 신뢰할 전체 손 이력이 없으면 이전 근거에 머무르므로 모든 미관측에 영역을 추정하는 것은 아니다.

자료와 검증
AI Hub 공개 TRAIN28표현·9명·60영상. 이전48영상과 새로 고른12표현12영상(기존과 다른 단어, 촬영자04/05)을 비교했다. 새 표현은 새 사람을 뜻하지 않는다.
전체7963프레임/양손15926개 상태를 저장 관측으로 계산했다. 새 MediaPipe 검출, TCN 채점, 학습은 없다.
새 표본은 규칙·대상을 고정하고 출력 전에 영상54장면을 판독했다. 미검출38장면과 정상 관측16장면이다. 이전87개 시각 판독은 회귀 확인용으로 재사용했다. 한 장면은 한 프레임의 한 손이며, 연속 장면이 포함된다.

새 공개 표본38개에서의 화면 안·밖 구분
                    기존 방식   이번 후보
시각 확인과 일치       35          25
잘못 구분              1           0
판단 보류              2          13
정상 관측 대조16개는 모두 그대로 유지했다.
보류가 늘어 오류가 줄었으므로 정확도가 좋아졌다고 해석하면 안 된다. 틀린1개를 정답으로 고친 것이 아니라 답을 보류한 것이다. 맞게 구분하던10개도 더 이상 답하지 못했다.

이전 표본71개에서의 회귀
기존:일치66/오류1/보류4 → 후보:일치24/오류0/보류47.
기존 정상 관측 대조16개는 그대로 유지했다.

겹침을 얼마나 찾았나
새 표본에서 눈으로 겹침을 확인한10장면:기존7개→후보1개만 겹침 의심으로 특정했다. 기존에 겹침이 아닌데 겹침이라고 했던1장면은 보류로 바뀌었다.
이전 겹침19장면:기존1개→후보0개. 마음에 들다의 누락14프레임도 기존에는 모두 화면 안 미관측이었지만 이번에는1개만 유지하고13개는 보류했다. 사라진 관절을 복구하지 못했다.

개선된 부분
R016유성구주민지원센터·N001화상:손목은 화면 안쪽이지만 손 일부가 경계에 걸친 사례를 화면 안이라고 단정하지 않게 됐다. 결과는 올바른 경계 판정이 아니라 불확실성 표시다.
N023공항시장:두 손이 떨어진 장면의 겹침 오판도 보류로 바뀌었다.
좌표 보존, 이력 만료, 판단 근거와 불확실성 수치 기록은 구현·검산됐다. 단어별 예외나 사용자 자료를 사용하지 않았다.

왜 실용성이 떨어졌나
손가락 모양이 실제로 바뀌는 것, 검출 좌표가 흔들리는 것, 몸 손목과 손 검출 손목의 차이를 직전 두 프레임의 최대 상대좌표 변화 하나로 합쳤다. 그 값을 누락 시간 동안 계속 커지는 오차처럼 반영해 불확실성 범위가 지나치게 넓어졌다.
마음에 들다 예:손 영역 크기는 약85px, 누락0.033초에서 불확실성39px,0.233초에서190px. 손보다 큰 범위를 보류하니 겹침을 놓친다. 또한 손 형태는 회전·변형하는데 이번 추정은 과거 영역을 평행 이동하므로 형태를 정확하게 예측하지 못한다.
손의 위치 판단과 손 모양/겹침 추정의 불확실성을 하나의 판정으로 묶은 점도 문제다. 팔 연결로 위치를 말할 수 있는 경우까지 과거 손 모양이 불확실하다는 이유로 보류했다.

검산 및 안전 확인
합성16조건 통과. 독립 scalar 좌표/외곽/다각형 교차 코드로15926상태 일치,2005영역 추정 검산. 최대 불확실성 차이0.000075px 미만,경계 차이0.000198px 미만. 단위·타임스탬프 계산 오류로 생긴 결과가 아니다. 자료240핀·모델/실행코드10보호파일 해시 유지.
기존에 없던 확신 있는 화면 밖/겹침 판정이 핵심구간에 새로 생긴 경우는0. 그래서 동결 계획의 ‘확신 판정이 새로 생긴 구간’ 추가 검토 대상도0이다. 새54장면과 설명4장면을 시각 확인했다. 모든 프레임에 전문가 정답이 있다는 의미는 아니다.
합성 테스트 첫 실패는0.4초에서 불확실성 상한과 정확히 같은 값인데 초과를 기대한 테스트 오류였다.0.45초로 수정하고 경계 장면의 합성 팔 길이를 바로잡았다. 규칙·계수는 바꾸지 않았고 공개 출력 전에 수정했다. 최초 증거 FIRST_TEST_ERROR.txt 보존.
구현·검산·판독은 같은 Codex 담당이며 다른 에이전트/전문가의 독립 판독은 아니다.

현재 판단과 다음 방향
이번 후보는 자동 구분 대체안으로 미채택한다. 실패 장면이 통과하도록 계수나 보류 기준을 낮추지 않는다. 기존 방식의 결과와 새 영역·불확실성 근거를 별도 필드로 남겼으므로, 향후 손 위치 정보와 겹침/형태 불확실성을 분리하는 설계에 참고할 수 있다.
새 손 영역이 있다는 이유만으로 겹침을 확정하거나 화면 밖 손을 학습·DTW에서 자동 제외하지 않는다. 다음에는 위치 판단을 유지할 근거와 겹침을 말할 근거를 분리하고, 더 긴 실제 관측 이력에서 검출 흔들림과 형태 변화를 따로 측정할 계획부터 검토하는 것이 타당하다. 그것이 성능을 높인다는 보장은 아직 없다. 사용자와 이 결과를 확인한 다음 결정한다.

사용자/멘토 촬영 읽기·학습·튜닝·후보선택0. 새 학습0,TCN 추론0,새 검출0. 기존C/M0·웹·DTW 유지. 원본과 과거 동결 결과 보존. 운영 반영이나 새 학습은 실행하지 않았다.
'''
write('RESULT_KO.txt',text)
gallery=''.join(f'<details><summary>출력 전 공개 표본 판독 {i}/7</summary><img src="review_{i:02d}.jpg" loading="lazy" alt="새 공개 시각 판독 {i}"></details>' for i in range(1,8))
html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 영역·불확실성 보완 결과</title>
<style>body{{background:#edf1f5;color:#1c2d40;font:17px/1.75 'Malgun Gothic',sans-serif;margin:0}}main{{max-width:1050px;margin:auto;padding:38px 22px 70px}}h1{{font-size:34px;line-height:1.4}}h2{{font-size:23px}}section{{background:white;border:1px solid #d5dfe8;border-radius:15px;padding:26px;margin:22px 0}}.notice{{background:#fff0d4;border-left:5px solid #bb6e20;padding:18px}}.lead{{font-size:21px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #d5dfe8}}img{{width:100%;height:auto}}summary{{cursor:pointer;font-weight:bold}}details{{padding:12px 0}}a{{color:#215f9b}}.small{{font-size:15px;color:#485d72}}.bad{{color:#a43c2b;font-weight:bold}}.eyebrow{{color:#265f99;font-weight:bold}}@media(max-width:600px){{main{{padding:22px 10px}}section{{padding:18px}}h1{{font-size:27px}}}}</style>
<main><div class="eyebrow">WordsInSilence · 공개 자료 비교 · 2026-10-17</div><h1>손 전체 영역과<br>움직임 불확실성을 함께 확인한 결과</h1>
<p class="lead">구현·검증 완료. <b>기존 방식을 대체하는 후보로는 미채택합니다.</b></p><div class="notice">일부 잘못된 단정은 줄었지만, 너무 많은 장면을 판단하지 못하게 됐습니다. 겹침 탐지와 단어 인식이 개선됐다고 볼 수 없습니다. 새 학습은 하지 않았습니다.</div>
<section><h2>무엇을 만들었나</h2><p>마지막으로 실제 관측된 손21점의 영역을 기억하고, 현재 팔 연결을 참고해 손이 있을 만한 범위를 추정합니다. 안 보이는 시간이 길거나 관절 변화가 크면 그 범위를 넓혀 판단을 보류합니다.</p><p>보이지 않는 손가락 좌표를 만들어 채우는 기능은 아닙니다. 확률로 보정된 신뢰구간도 아닙니다. 단어명·사람ID·미래 프레임을 사용하지 않았습니다.</p></section>
<section><h2>새로 고른 공개 장면에서의 비교</h2><p>공개 TRAIN <b>28표현·9명·60영상</b>을 계산했습니다. 아래 표는 이번에 추가한12표현에서 출력 전에 고른 <b>손 미검출38장면</b>의 화면 안·밖 구분입니다.</p><table><tr><th>결과</th><th>기존</th><th>이번 후보</th></tr><tr><td>시각 확인과 일치</td><td>35</td><td class="bad">25</td></tr><tr><td>잘못 구분</td><td>1</td><td>0</td></tr><tr><td>판단 보류</td><td>2</td><td class="bad">13</td></tr></table><p><b>오류1개를 정답으로 고친 것이 아니라 보류했습니다.</b> 동시에 맞게 구분하던10개도 더 이상 답하지 못했습니다.</p><p class="small">한 장면은 한 프레임의 한 손. 연속·선별 표본이며 새 사용자 정확도가 아닙니다. 정상 관측 대조16개는 기존대로 유지했습니다. 추가12표현은 새 사람12명이 아닙니다.</p></section>
<section><h2>겹침은 더 잘 찾았나?</h2><table><tr><th>시각 확인 표본</th><th>기존 겹침 의심</th><th>이번 후보</th></tr><tr><td>새 겹침 장면10개</td><td>7개</td><td class="bad">1개</td></tr><tr><td>이전 겹침 장면19개</td><td>1개</td><td>0개</td></tr></table><p>겹침으로 잘못 보던 공항시장1장면은 보류로 바뀌었습니다. 하지만 실제 겹침도 많이 놓쳤습니다. 마음에 들다의 누락14프레임은 이번에1개만 화면 안 미관측으로 유지하고13개는 보류했습니다.</p><p>이전 미검출71장면 전체에서도 일치66→24,오류1→0,보류4→47로 구분 범위가 줄었습니다.</p></section>
<section><h2>범위가 왜 너무 넓어졌나</h2><p>직전 두 관측의 상대좌표 변화에는 <b>실제 손 모양 변화와 검출 흔들림</b>이 섞여 있습니다. 가장 큰 변화량을 누락 시간 동안 계속 커지는 오차처럼 계산해 과도한 불확실성이 생겼습니다.</p><p>마음에 들다 예시:손 영역 크기는 약85px인데, 누락0.033초의 불확실성39px가0.233초에서190px로 커졌습니다. 손의 위치를 말할 수 있는 경우까지 형태 추정의 불확실성 때문에 보류한 점도 문제입니다.</p><img src="REGION_EXAMPLES.jpg" alt="누락 시간에 따라 커지는 불확실성 영역 비교"><p class="small">초록:마지막 실제 손 영역을 현재 팔 위치로 옮긴 추정. 주황:공학적 불확실성 범위. 노랑:실제 관절 관측 화면의 하단. 원본 전체 영상은 시각 확인용이며 선 아래 숨겨진 정보는 알고리즘에 제공하지 않았습니다.</p></section>
<section><h2>확인된 성과와 남은 문제</h2><p><b>성과:</b>손목만 안쪽에 있다는 이유로 손 전체를 안쪽이라고 단정하던2개 사례를 보류했습니다. 관측 좌표·마스크·좌우 배정을 보존하고 영역·불확실성 근거를 기록하는 기능을 구현했습니다.</p><p><b>문제:</b>판단 보류가 너무 많고 겹침 탐지가 줄었습니다. 영역은 과거 손 모양의 평행 이동이므로 현재 손의 회전·변형을 정확히 예측하지 못합니다. 전체 손 이력이 없는 경우에는 영역을 만들 수도 없습니다.</p><p><b>결정:</b>현재 자동 분류의 대체안으로 채택하지 않습니다. 기존 위치 근거와 새 형태·겹침 불확실성을 별도 정보로 유지하는 설계를 다음 검토 대상으로 기록했습니다. 이번 실패에 맞춰 기준을 낮추거나 특정 단어 예외를 넣지 않았습니다.</p></section>
<section><h2>검증과 보존</h2><p>합성16조건 통과. 별도 외곽·다각형 교차 코드로15,926상태와2,005영역 계산을 검산했습니다. 최대 경계 차이는0.000198px 미만으로 계산 오류가 원인은 아니었습니다. 자료240핀·보호파일10개 해시 유지.</p><p>사용자·멘토 촬영 읽기0 · 새 학습0 · TCN 추론0 · 새 MediaPipe 검출0 · 운영 모델/웹/DTW 변경0.</p><p class="small">같은 Codex 담당자의 별도 코드 검산이며 다른 에이전트나 수어 전문가의 독립 판독은 아닙니다. 핵심구간에서 새로 생긴 확신 있는 화면 밖/겹침 판정은0개여서, 동결 계획의 해당 추가 시각검토 대상도0개였습니다. 전체 물리적 정확성 보장은 아닙니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="RESULT.json">결과 원장</a> · <a href="INDEPENDENT_CHECK.json">독립 계산</a> · <a href="SCOPE.json">실행 전 고정 조건</a></p><details><summary>출력 전 시각 판독 근거</summary>{gallery}</details></section></main></html>'''
write('RESULT_KO.html',html)

# Preserve current root documents, then update only current status and next step.
before=U/'before';before.mkdir(exist_ok=False)
cp=R/'WordsInSilence_PROJECT_CONTEXT.md';hp=R/'NEW_CHAT_HANDOFF.md';sp=O/'STATE.json'
for q in (cp,hp,sp):(before/q.name).write_bytes(q.read_bytes())
stage='WORD100_HAND_REGION_UNCERTAINTY_COMPLETE_NOT_ADOPTED_REVIEW_BEFORE_TRAINING'
state=read(sp);new={}
for k,v in state.items():
    new[k]=v
    if k=='current_stage':new['word100_hand_region_uncertainty_20261017']={**result,'report':str(U/'RESULT_KO.html'),'text_report':str(U/'RESULT_KO.txt'),'final_check':str(U/'FINAL_CHECK.json'),'running_processes':[]}
new['current_stage']=stage;new['updated_date_kst']=at;new['next_single_action']=result['next_single_action']
new['resume_command']='Read latest PROJECT_CONTEXT3/8/9, NEW_CHAT_HANDOFF and word100_hand_region_uncertainty_20261017. Region candidate completed but not adopted: excessive abstention/overlap regression. Reuse frozen outputs; no coefficient sweep, training or production changes before user review.'
sp.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n','utf-8')
section8='''
### 2026-10-17 손 영역·움직임 불확실성 후보 검증 완료 — 과도한 보류·겹침 회귀로 미채택
사용자승인:손전체영역과움직임불확실성보완·공개검증보고,학습은결과확인후. ops/word100_hand_region_uncertainty_20261017 구현/검산완료·운영미채택. 마지막실제21점convex영역을현재pose손목으로평행이동,직전2관측손–pose상대변화+누락시간으로불확실성확대. 공통규칙·계수·표본을출력전동결,단어/사람ID·미래/native숨김정보/가상좌표입력없음. 기존위치근거baseline_state와새영역근거는별도저장.
공개TRAIN28표현·9명·60영상(이전48+서로다른새12표현04/05),7963프레임15926손상태. 새54시각표본출력전판독(미검출38+정상16),이전87재사용. 새38의큰위치구분기존일치35/오류1/보류2→후보25/0/13. 이전71은66/1/4→24/0/47. 오류를정답으로고친것아니라보류해감소,구분범위회귀. 정상대조총32관측유지. 선별연속TRAIN손프레임이며인식률/새사람성능아님.
겹침특정새10개7→1,이전19개1→0. 마음06L55~68=14중이번1화면안/13보류,좌표복구없음. R016유성구센터·N001화상의경계걸침오판과N023공항시장의겹침오판을보류로전환. 최대상대좌표변화가형태변화/검출흔들림/pose오차를합쳐누락동안과도하게확대;손위치와형태불확실성을묶어유용한위치판단도보류한한계. 마음예85px손에서불확실성39px@0.033s→190px@0.233s. 단위계산버그아님,결과후계수완화안함.
합성16PASS·별도scalar외곽/다각형교차코드15926상태일치/2005영역검산;최대u차이0.000075px미만/경계0.000198px미만. 자료240핀·모델/코드10보호핀유지. 새확신밖/겹침핵심판정0개여서해당추가시각검토대상0. 새54장면+설명4장면확인,전문가/타에이전트검토아님. 최초합성테스트기댓값오류FIRST_TEST_ERROR보존후공개출력전테스트fixture만수정,후보규칙불변.
새학습/TCN추론/검출/사용자자료읽기/운영변경0. C/M0·웹·DTW보존. 완료exec59943/86752종료0,별도검산종료0,진행중없음. SCOPE/PREDICTIONS/VISUAL_LABELS/INDEPENDENT_CHECK/RESULT_KO.html/RESULT.json/REGION_EXAMPLES.jpg/FINAL_CHECK 재사용. 자동분류대체미채택,추가학습안함.

'''
section9='''
### 2026-10-17 영역·불확실성 후보 미채택 결과 확인 — 자동 학습 금지
ops/word100_hand_region_uncertainty_20261017 완료결과를사용자와확인. 오판감소는보류증가의대가이며신규38큰위치일치35→25/겹침7→1로회귀했으므로기존분류대체/DTW자동제외/학습연결금지. 정상관측·운영모델보존. 현재동결계수느슨하게재탐색안함.
다음검토방향:손위치의현재팔근거와손형태·겹침추정의불확실성을별도출력으로분리,과거두프레임최대변화의무제한선형확대대신더긴실제관측에서측정흔들림/형태변화를먼저분석. 현재코드baseline_state/region근거는이미별도기록,이를정답처럼자동사용안함. 새후보조건은결과확인후정리;학습은추가승인전진행하지않음. 사용자/멘토자료학습·튜닝·후보선택금지,새가상손복원/단어특례금지.

'''
context=cp.read_text('utf-8')
for head,block in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',section8),('## 9. 다음 단일 작업',section9)]:
    assert context.count(head)==1;context=context.replace(head+'\n',head+'\n'+block,1)
cp.write_text(context,'utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {at}

## 현재 목표·마지막 완료
사용자승인:손목점대신손전체영역+움직임불확실성보완/공개검증,학습은결과확인후. ops/word100_hand_region_uncertainty_20261017 구현·검산완료,과도한보류/겹침회귀로자동분류대체미채택. RESULT_KO.html/RESULT_KO.txt/RESULT.json/FINAL_CHECK 재사용.

## 결과
공개TRAIN28표현·9명·60영상(이전48+새12표현),7963프레임15926양손상태. 새표현은새사람아님. 새54눈판독을출력전고정(미검출38/정상16),이전87회귀판독재사용.
새38큰위치구분기존35일치/1오류/2보류→후보25/0/13. 이전71은66/1/4→24/0/47. 오류를정답으로고친것아니라보류증가,구분범위회귀. 정상대조총32유지. 인식률아님.
겹침특정새10개7→1,이전19개1→0. 마음06L55~68=14프레임중후보1화면안/13보류,복구0.
R016유성구센터와N001화상경계오판, N023공항시장겹침오판이보류로변경. 최대손–pose상대변화를누락시간동안확대하여과한범위:마음85px손의불확실성39px@0.033s→190px@0.233s. 형태변화/검출흔들림/pose편차가합쳐지고위치판단까지보류되는한계.

## 구현·검증
region.py:마지막실제21점convex영역을현재pose손목으로옮기고누락시간/상대변화로불확실성계산. 새추정점을모델에넣지않음. 같은sensor시야의현재/과거만사용,단어ID·사람ID·미래/native숨김신호없음. 원래baseline_state/새region근거별도저장.
SCOPE에규칙·계수·표본동결,결과후수정없음. 합성16PASS,별도scalar hull/clip15926상태일치/2005영역검산PASS. 최대u차이0.000075px미만/경계0.000198px미만. 자료240핀/보호10파일유지. 같은Codex담당별도코드,타에이전트/전문가판독아님.
새확신밖/겹침핵심상태0이므로변경확신구간추가검토대상0. 새54장면과REGION_EXAMPLES4장면확인. 전프레임물리정확도보장아님. FIRST_TEST_ERROR는합성기댓값/팔길이fixture문제만공개출력전수정,후보계수불변.
prepare exec59943/replay86752종료0,검산/보고완료,진행중없음. 재생/검출/추론재실행불필요. 사용자촬영읽기0·새학습0·TCN채점0·MediaPipe실행0·운영변경0.

## 다음 한 작업
{result['next_single_action']}
현재계수완화재탐색/단어예외/가상손복원금지. 촬영간격보강보류. 다른후보를하려면현재결과먼저확인하고조건고정. 사용자/멘토자료학습·튜닝·후보선택금지.

## 보존과 웹
원본/W:/sources/과거동결자료읽기전용,Git/카메라/다운로드/외부전송없음. 새공개12영상은기존ZIP에서정확CRC/크기/SHA확인로컬해제.
C http://127.0.0.1:8765/practice.html?model=c / M0 ?model=m0 / 기본M0. 이번서버점검·재시작없음.
C SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab;M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f 유지. 필요손DTW완료상태유지. st-gcn자동화삭제유지.
'''
hp.write_text(handoff,'utf-8')
links=re.findall(r'(?:href|src)="([^"]+)"',html)
assert all((U/x).is_file() for x in links)
for n in ['RESULT.json','INDEPENDENT_CHECK.json','SCOPE.json','VISUAL_LABELS.json']:read(U/n)
for path,h in {**s['pins'],**c['protected_pins']}.items():assert sha(path)==h,path
assert sha(U/'region.py')==s['code_sha256']['region.py']
assert read(sp)['current_stage']==stage
save('FINAL_CHECK.json',{'status':'PASS','at':at,'candidate_status':result['status'],'root_documents_synchronized':True,'report_links_checked':len(links),'source_pins':240,'protected_pins':10,'new_training':0,'user_recording_reads':0,'running_processes':[], 'document_sha256':{str(q):sha(q) for q in [cp,hp,sp,U/'RESULT_KO.html',U/'RESULT.json']}})
print(json.dumps({'status':result['status'],'report':str(U/'RESULT_KO.html'),'root_docs':'PASS','links':len(links)},ensure_ascii=False))
