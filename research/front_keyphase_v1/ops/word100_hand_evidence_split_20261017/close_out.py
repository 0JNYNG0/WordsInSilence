import json,hashlib,re
from pathlib import Path
from datetime import datetime
U=Path(__file__).resolve().parent;O=U.parent;R=U.parents[3]
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    with (U/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def write(n,x):
    with (U/n).open('x',encoding='utf-8') as f:f.write(x)
c=read(U/'INDEPENDENT_CHECK.json');s=read(U/'SCOPE.json');assert c['status']=='PASS'
at=datetime.now().astimezone().isoformat(timespec='seconds')
next_action='사용자와 위치·모양·겹침 근거 분리 결과 확인. 기존 위치 단서는 보존됐지만 손 전체 경계2사례와 겹침 원인 미확정은 남음. 현재 모듈은 진단 기록 전용이며 자동 학습·DTW 제외·운영 연결하지 않음. 학습은 결과 함께 확인한 다음.'
result={'status':'SEPARATED_DIAGNOSTIC_CONTRACT_VERIFIED_NO_PRODUCTION_OR_TRAINING', 'at':at,
 'unit':'word100_hand_evidence_split_20261017',
 'implementation':'Codex local; same operator separate saved-contract verification, not other agent or expert',
 'changes':['Position from current observation/arm-chain retained independently of hand-region uncertainty.',
            'Usable observed shape points vs unobserved shape are separate, region estimate never treated as observation.',
            'Proximity hint separated from area-supported overlap suspicion; neither confirms cause.',
            'No aggregate state or automatic coordinate/mask/exclusion decision. No numeric coefficient changes.'],
 'public_scope':{'videos':60,'word_types':28,'train_people':9,'hand_states':15926,
                 'existing_visual_labels_reused':141,'missing_hand_samples':109,'observed_control_samples':32,
                 'new_blinded_public_labels':0,'stream_clips':2,'stream_hand_states':570},
 'validation':{'position_and_evidence_preserved':15926,'observed_states_preserved':6946,
               'correct_existing_missing_samples_preserved':101,'restored_vs_rejected_combined':52,
               'metamorphic_cases':210,'source_pins':240,'protected_pins':10},
 'compatibility_comparison':{'baseline':{'agree':101,'whole_hand_disagree':2,'undetermined':6},
                             'rejected_combined':{'agree':49,'whole_hand_disagree':0,'undetermined':60},
                             'separated_position':{'agree':101,'whole_hand_disagree':2,'undetermined':6},
                             'note':'Same109 known correlated hand-frame samples. Separated location denotes wrist/arm evidence; not certified whole-hand visibility. No new accuracy gain versus baseline.'},
 'remaining_issues':c['remaining_whole_hand_boundary_disagreements'],
 'overlap_counts':{'whole_sequence_proximity_hints':25,'region_supported_suspicions':1,
                   'confirmed_occlusion_causes':0,'note':'15926 hand-state outputs, not annotated accuracy denominators. Absence of support is not proof of absence of overlap.'},
 'acceptance':{'separation_and_nonregression_contract':'PASS','physical_cause_recognition':'UNRESOLVED','model_accuracy_gain':'NOT_MEASURED'},
 'new_training':0,'classifier_inference':0,'new_detection':0,'user_recording_reads':0,'production_changes':0,
 'next_single_action':next_action,
 'evidence':{n:str(U/n) for n in ['separate.py','SCOPE.json','OUTPUT.json','CONTRACT_TESTS.json','INDEPENDENT_CHECK.json']}}
save('RESULT.json',result)
text='''손 위치와 손 모양·겹침 근거 분리 — 구현·검증 결과

결론
손 모양이 불확실하다는 이유로 기존 손 위치 정보까지 판단 보류로 덮어쓰던 문제를 분리했다. 공개 자료에서 기존에 맞게 구분한101개 미검출 장면의 위치 단서를 모두 유지했다. 직전 미채택 후보에서 보류됐던52개를 기존 수준으로 복원했다. 기존 방식보다 물리적 원인 분류나 TCN 인식률이 높아졌다는 의미는 아니다.

지금은 무엇을 따로 기록하나
① 손 위치:현재 관측과 어깨–팔꿈치–손목 연결로 ‘손목이 화면 안쪽’,‘손목이 경계/밖 쪽’,‘판단 불가’ 등을 기록한다. 손 전체가 보인다는 확정 판정은 아니다.
② 손 모양 관측:실제로 쓸 수 있는 손 관절이 관측됐는지,부분 관측인지,사용 가능한 관측이 없는지 기록한다. 관측됨은 기존 손바닥5점 사용 기준이며 전체21점의 정확한 모양이 검증됐다는 뜻은 아니다. 과거 영역 추정을 실제 관측으로 바꾸지 않는다.
③ 겹침 근거:두 손이 가깝다는 기존 단서와 영역 교차를 뒷받침하는 근거를 구분한다. 근접만으로 겹침을 확정하지 않는다. 영역 불확실성은 이 항목에 남기고 위치를 덮어쓰지 않는다. 근거가 없다는 것은 겹치지 않았다는 증명이 아니다.

실제 사례로 이해하기
마음에 들다의 누락 구간:‘손목은 화면 안쪽으로 추정된다’와‘손가락 모양은 관측되지 않아 겹침 원인을 확정할 수 없다’를 동시에 기록한다. 직전 후보처럼 모양이 불확실해서 위치까지 모른다고 바꾸지 않는다. 사라진 손 관절을 복구한 것은 아니다.
유성구주민지원센터·화상의 경계 사례:‘손목은 안쪽’과‘손 전체가 들어와 있는지는 불확실’을 함께 기록한다. 손목 한 점을 손 전체로 해석하면 생기는 기존2개 불일치는 해결된 것으로 계산하지 않았다.
공항시장:두 손 근접 단서는 있어도 영역 근거가 불확실하므로 겹침 확정으로 올리지 않는다.

공개 자료 검증
AI Hub 공개 TRAIN28표현·9명·60영상의 저장 관측15926손상태를 사용했다. 이전 시각 판독141개(미검출109/정상대조32)를 회귀 확인에 재사용했으며 새 미공개 표본이나 새 사람 시험이 아니다. 한 표본은 한 프레임의 한 손,연속 프레임 포함이다.
동일109개 미검출 표본의 과거 ‘손 전체 안·밖’ 시각 판독과 비교:
                         기존 방식    직전 통합 후보    이번 위치 정보
시각 확인과 일치             101             49              101
손 전체 경계와 불일치          2              0                2
판단 보류                     6             60                6
이번 위치 정보는 손목·팔 근거로 표현 범위를 명확히 했다. 이 표는 기존 정보 보존을 확인하는 같은 기준의 비교이며 새로운 손 전체 정확도가 아니다. 직전 후보의 오류0은 많은 보류를 대가로 얻은 값이었다.
이전 단계에서 추가한38개 표본만 보면 기존35일치/1불일치/2보류,직전25/0/13,이번35/1/2로 기존 수준 유지다.
정상 관측 대조32개와 전체6946개 관측 상태도 보존했다.

독립 검산 및 결합 검사
별도 검산 코드로15926개 위치 상태·근거가 기존과 일치함을 확인했다. 불확실성 상태와 크기를 바꾸는210개 조건에서 위치가 영향을 받지 않았다. 부분/미관측 손이 관측된 손으로 승격되지 않고,공통 상태 하나가 위치를 덮어쓰지 않도록 검사했다.
공개 영상2개 전체를 실제 스트리밍 어댑터로 연결해570손상태가 저장 결과와 일치했다. 원래 관절 배열·마스크·좌우 배정을 수정하지 않았다. 파일240핀과 모델/실행 코드10보호핀 유지. 구현·검증은 같은 Codex 담당자의 별도 코드이며 다른 에이전트나 전문가의 독립 판독은 아니다.

남은 문제
손 전체의 화면 경계 여부는 손목 위치만으로 확정할 수 없다. 같은2개 경계 사례를 오류 해결로 보고하지 않았다.
겹침 원인을 더 잘 찾는 기능을 새로 만든 것이 아니다. 전체15926손상태 중 근접 단서25개,영역이 뒷받침하는 의심1개를 분리 기록했지만,이것은 정확도 수치가 아니다. 보이지 않는 손 모양은 여전히 모른다.
이전 영역 불확실성의 과도한 확대는 그대로 한계로 남는다. 이번에는 그 한계가 위치 정보를 지우지 않도록 분리했다. 수치를 낮춰 특정 장면이 통과하도록 조정하지 않았다.

현재 적용 범위와 다음 단계
이번 분리 모듈은 공개 자료 진단 부품으로 구현·검증됐다. 웹·TCN·DTW에는 연결하지 않았다. 이 결과로 손을 자동 제외하거나 관절을 만들어 채우거나 사용자 동작이 틀렸다고 판단하지 않는다.
사용자와 위치 보존 및 남은 경계·겹침 문제를 확인한 뒤 다음 단계를 정한다. 학습은 아직 실행하지 않았다. 기존 학습 모델의 성능 향상이나 새 사용자 일반화를 주장하지 않는다.
사용자/멘토 촬영 읽기·학습·튜닝·후보선택0. 새 학습0,TCN 추론0,새 MediaPipe 검출0,운영 변경0. 기존C/M0·웹·DTW·공개 원본·과거 동결 결과 보존.
'''
write('RESULT_KO.txt',text)
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 위치와 모양·겹침 근거 분리</title><style>
body{margin:0;background:#eff3f6;color:#213448;font:17px/1.75 'Malgun Gothic',sans-serif}main{max-width:1040px;margin:auto;padding:40px 22px 70px}h1{font-size:34px;line-height:1.35}h2{font-size:23px}section{background:white;border:1px solid #d5dfe7;border-radius:15px;padding:25px;margin:22px 0}.lead{font-size:21px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.card{border-radius:12px;background:#eef4fa;padding:18px;border-top:5px solid #366e9e}.card:nth-child(2){border-color:#738390}.card:nth-child(3){border-color:#bb8b30}.card b{display:block;font-size:22px}.notice{background:#fff2d9;padding:17px;border-left:5px solid #bc8a27}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border-bottom:1px solid #d6e0e9;padding:12px;text-align:left}.small{font-size:15px;color:#4e6274}a{color:#1e5d95}.kicker{font-weight:bold;color:#30638d}@media(max-width:700px){.cards{grid-template-columns:1fr}main{padding:20px 10px}section{padding:18px}h1{font-size:28px}}
</style><main><div class="kicker">WordsInSilence · 공개 자료 검증 · 2026-10-17</div><h1>손의 위치는 유지하고,<br>모양·겹침의 불확실성은 따로 기록</h1><p class="lead">기존에 맞게 구분하던 위치 단서 <b>101개 모두 유지.</b><br>직전 후보에서 보류됐던52개를 기존 수준으로 복원했습니다.</p><div class="notice">이번 개선은 정보가 서로 덮어쓰이지 않도록 한 구조 보완입니다. 기존 모델보다 인식률이 높아졌다는 결과는 아닙니다. 새 학습과 웹 적용은 하지 않았습니다.</div>
<section><h2>한 가지 판단으로 합치지 않습니다</h2><div class="cards"><div class="card"><b>① 위치</b><p>팔 연결을 보면 손목이 어디에 있는가?</p><p>화면 안쪽 / 경계·밖 쪽 / 판단 불가</p></div><div class="card"><b>② 모양 관측</b><p>사용할 수 있는 손 관절을 실제로 봤는가?</p><p>관측 있음 / 부분 관측 / 관측 없음</p></div><div class="card"><b>③ 겹침 근거</b><p>가까운 것인가,영역도 겹칠 가능성이 있는가?</p><p>근접 단서와 영역 근거를 따로 기록</p></div></div><p><b>“손 모양은 모르지만,팔을 보면 손목은 화면 안쪽이다”</b>라는 두 정보를 함께 남깁니다. 모양이 불확실하다고 위치 정보까지 지우지 않습니다.</p><p class="small">손목 위치가 손 전체의 가시성을 증명하지는 않습니다. 관측 있음도 기존 손바닥5점 사용 기준이며 전체 손 모양의 정확성을 인증하지 않습니다.</p></section>
<section><h2>같은 공개 표본으로 비교</h2><p>공개 TRAIN28표현·9명·60영상에서 기존 관측을 사용했습니다. 아래는 이전에 시각 확인한 미검출109장면의 비교입니다.</p><table><tr><th>과거 손 전체 안·밖 판독과 비교</th><th>기존</th><th>직전 통합 후보</th><th>이번 위치 정보</th></tr><tr><td>일치</td><td>101</td><td>49</td><td><b>101</b></td></tr><tr><td>경계 불일치</td><td>2</td><td>0</td><td>2</td></tr><tr><td>보류</td><td>6</td><td>60</td><td>6</td></tr></table><p>이전 단계에서 추가했던38장면만 보아도 일치35→25→35로 기존 수준을 유지했습니다. 정상 관측 대조32개도 그대로입니다.</p><p class="small">이번 위치 항목은 손목·팔 근거입니다. 손 전체가 보이는지의 판정을 새로 개선한 것으로 해석하면 안 됩니다. 이전 판독141개를 재사용한 회귀 확인이며 새 사람·새 독립 표본 시험이 아닙니다. 한 장면은 한 프레임의 한 손으로 연속 장면이 포함됩니다.</p></section>
<section><h2>실제로 어떻게 읽으면 되나</h2><table><tr><th>사례</th><th>보존하는 위치 정보</th><th>따로 남기는 한계</th></tr><tr><td>마음에 들다의 누락</td><td>손목은 화면 안쪽 추정</td><td>손 모양 미관측,겹침 원인 미확정</td></tr><tr><td>유성구센터·화상의 경계</td><td>손목은 화면 안쪽 추정</td><td>손끝까지 들어와 있는지는 불확실</td></tr><tr><td>공항시장</td><td>화면 안쪽 위치 단서 유지</td><td>두 손 근접만으로 겹침 확정 금지</td></tr></table><p>손목 안쪽을 손 전체 안쪽으로 읽으면 생기는 <b>기존2개 경계 불일치는 해결된 것으로 계산하지 않았습니다.</b></p></section>
<section><h2>검증한 개선과 남은 문제</h2><p><b>개선:</b>15,926개 위치 상태와 근거가 기존과 일치했습니다. 불확실성 종류·크기를 바꾸는210개 조건에서도 위치가 덮어써지지 않았습니다. 공개 영상2개 전체를 스트리밍으로 연결한570손상태도 저장 결과와 일치했습니다.</p><p><b>남은 문제:</b>겹침 원인을 더 잘 찾았다는 증거는 없습니다. 전체 출력에서 근접 단서25개와 영역이 뒷받침하는 의심1개를 구분했지만,겹침 원인이 확정됐다는 뜻은 아닙니다. 근거 부족은 겹침이 없다는 증명도 아닙니다.</p><p>이전 후보의 과도한 영역 불확실성은 그대로 한계로 기록합니다. 이번에는 그 한계가 유효한 위치 정보를 없애지 않도록 분리했습니다. 보이지 않는 관절을 만들어 채우거나 손을 자동 제외하지 않습니다.</p></section>
<section><h2>적용 범위</h2><p>분리 모듈은 공개 자료 진단 부품으로 구현·검증했습니다. <b>운영 웹·TCN·DTW는 유지했으며 새 학습은 하지 않았습니다.</b> 사용자와 이 결과를 확인한 후 다음 단계를 정합니다.</p><p>사용자·멘토 촬영 읽기0 · 학습/튜닝/후보 선택0 · TCN 추론0 · 새 검출0. 자료240핀과 보호파일10개 해시 유지.</p><p class="small">같은 Codex 담당자가 별도 코드로 검산했습니다. 다른 에이전트·사람·수어 전문가의 독립 판독을 의미하지 않습니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="RESULT.json">결과 원장</a> · <a href="INDEPENDENT_CHECK.json">별도 검산</a> · <a href="SCOPE.json">고정 검증 조건</a></p></section></main></html>'''
write('RESULT_KO.html',html)

before=U/'before';before.mkdir(exist_ok=False)
cp=R/'WordsInSilence_PROJECT_CONTEXT.md';hp=R/'NEW_CHAT_HANDOFF.md';sp=O/'STATE.json'
for path in (cp,hp,sp):(before/path.name).write_bytes(path.read_bytes())
stage='WORD100_HAND_EVIDENCE_SPLIT_VERIFIED_DIAGNOSTIC_ONLY_REVIEW_BEFORE_TRAINING'
old=read(sp);new={}
for k,v in old.items():
    new[k]=v
    if k=='current_stage':new['word100_hand_evidence_split_20261017']={**result,'report':str(U/'RESULT_KO.html'),'text_report':str(U/'RESULT_KO.txt'),'final_check':str(U/'FINAL_CHECK.json'),'running_processes':[]}
new['current_stage']=stage;new['updated_date_kst']=at;new['next_single_action']=next_action
new['resume_command']='Read latest PROJECT_CONTEXT3/8/9, NEW_CHAT_HANDOFF, word100_hand_evidence_split_20261017. Split diagnostic verified: location preserved, shape/overlap uncertainty separate. Reuse outputs. Whole-hand boundary2 discrepancies and overlap causes unresolved. No training or production connection before user review.'
sp.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n','utf-8')
sec8='''
### 2026-10-17 위치·모양·겹침 근거 분리 구현·검증 완료 — 기존 위치 보존, 원인 인식 향상 아님
최신사용자승인:손위치판단과모양/겹침불확실성분리·기존잘구분보존·공개검증,학습은결과확인후. ops/word100_hand_evidence_split_20261017 완료. separate.py는기존현재팔연결위치/실제손관측유무/근접단서와영역교차근거를분리. 모양불확실성으로위치덮어쓰기없음,단일종합state없음. 수치계수변경0·단어예외0·가상관절0·자동제외eligible=False·겹침원인확정False.
공개TRAIN28표현9명60영상15926손상태,이전141시각판독(미검출109/정상32)회귀재사용. 새독립표본아님. 기존맞던미검출위치101모두유지;직전미채택통합후보에서보류됐던52복원. 같은109의기존101일치/2손전체경계불일치/6보류,통합49/0/60,분리위치101/2/6. 이전추가38은35/1/2→25/0/13→35/1/2. 손목위치의정보보존이지새물리정확도/TCN인식률향상아님.
R016유성구센터L70/N001화상L136의손목안쪽·손끝경계걸침은미해결2사례로유지. 출력은손목위치와손전체범위불확실성을함께표기. 마음06L55~68의14손목안쪽단서복원,손모양미관측/겹침미확정. 전체15926출력의근접단서25/영역지지의심1분리,원인정답률아님. 영역근거없음은겹침없음의증명아님.
독립저장계약검산15926위치및근거일치·관측6946보존·미관측8980승격없음. 불확실성변형210조건위치불변,공개2영상전체어댑터570손상태동일. 자료240핀·모델/코드10보호핀PASS. 동일Codex담당별도코드이며타에이전트/전문가아님. 실행모두종료0·새학습/TCN/검출/사용자읽기/운영변경0.
근거 SCOPE/OUTPUT/CONTRACT_TESTS/INDEPENDENT_CHECK/RESULT_KO.html/RESULT.json/FINAL_CHECK. 사용자와결과확인후다음단계,진단기록전용·웹TCNDTW미연결. 이전영역불확실성후보한계를분리했을뿐추정품질개선은미검증.

'''
sec9='''
### 2026-10-17 위치·모양·겹침 분리 결과 확인 — 진단 기록만 완료
ops/word100_hand_evidence_split_20261017의결과를사용자와확인. 위치단서101기존정합표본유지/직전후보대비52복원은정보보존성과이지인식률향상이아님. 손전체경계2사례·겹침원인은미해결. 기존위치단서와형태/영역불확실성을별도기록하는현재구조유지,자동제외/복원/운영연결안함.
학습은결과확인후이며이번진단을새TCN학습승인으로해석금지. 다음단계제안시진단정보의사용처/통과기준을명확히하고단어별예외·현재실패맞춤임계완화·사용자자료튜닝금지. 공개회귀근거는재사용하고불필요한재생·검출·채점반복하지않음. 기존C/M0/DTW/웹보존.

'''
context=cp.read_text('utf-8')
for head,sec in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',sec8),('## 9. 다음 단일 작업',sec9)]:
    assert context.count(head)==1;context=context.replace(head+'\n',head+'\n'+sec,1)
cp.write_text(context,'utf-8')
hp.write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

## 목표·마지막 완료
사용자:위치판단과손모양/겹침불확실성분리,기존잘구분보존,공개검증보고. 사용자촬영학습/튜닝금지,학습은결과확인후.
ops/word100_hand_evidence_split_20261017 구현/검증완료,진단기록전용·운영미연결. RESULT_KO.html/RESULT_KO.txt/RESULT.json/FINAL_CHECK 재사용.

## 확인한 결과
공개TRAIN28표현9명60영상15926손상태. 이전141판독(미검출109/정상32)회귀재사용,새독립표본아님.
기존맞던위치101모두유지,직전통합후보에서보류됐던52복원. 동일109기존101일치/2손전체경계불일치/6보류,직전49/0/60,분리101/2/6. 이전추가38은35/1/2→25/0/13→35/1/2. 기존수준의위치근거보존이지새인식률향상아님.
위치=현재팔/손목근거,모양=실제관측유무,겹침=근접단서/영역근거별도. 종합state없음,모양모름이위치덮어쓰기못함. 위치는손목이므로손전체가시성확정아님. 관측기준은기존palm5이며전체21점정확성검증뜻아님.
R016유성구센터L70/N001화상L136은손목안쪽·손끝경계걸침미해결2사례. 마음06L55~68=14위치단서유지,모양미관측/겹침미확정. 전체출력근접단서25/영역지지의심1,확정원인0. 없는근거를겹침없음으로해석금지.

## 구현·검증·실행
separate.py공통순수변환+SeparatedHandDiagnostics스트리밍어댑터. 수치계수변경0,단어/사람ID·native숨김정보·미래정보미사용. 모양추정을실제관측으로승격안함,자동제외/좌표복원False.
별도저장계약검산15926위치및근거일치/관측6946보존/미관측8980승격없음. 불확실성변형210조건위치불변,공개2영상전체570손상태어댑터일치. 자료240핀/보호10파일PASS. 동일Codex별도코드,다른에이전트/전문가검토아님.
SCOPE/OUTPUT/CONTRACT_TESTS/INDEPENDENT_CHECK완료,전실행종료0,진행중없음. 새학습/TCN추론/MediaPipe검출/사용자촬영읽기/운영변경0. 완료근거재실행불필요.

## 다음 한 작업
{next_action}
이전영역후보의과도한불확실성자체는해결안됨. 현결과를기존TCN재학습승인으로해석금지. 사용자/멘토자료학습·튜닝·후보선택금지,단어특례·임계값재탐색·가상손복원없음.

## 기존 모델·웹 보존
C http://127.0.0.1:8765/practice.html?model=c / M0 ?model=m0 / 기본M0. 이번서버점검/재시작없음.
C SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab;M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f 유지. 필요손DTW완료상태유지. st-gcn자동화삭제유지.
원본/W:/sources/과거결과읽기전용. 다운로드/카메라/외부전송/Git변경0.
''','utf-8')
links=re.findall(r'(?:href|src)="([^"]+)"',html)
assert all((U/x).is_file() for x in links)
for p,h in {**s['dependencies'],**s['source_pins'],**s['protected_pins']}.items():assert sha(p)==h,p
assert read(sp)['current_stage']==stage
for n in ['RESULT.json','INDEPENDENT_CHECK.json','OUTPUT.json','CONTRACT_TESTS.json']:read(U/n)
save('FINAL_CHECK.json',{'status':'PASS','at':at,'candidate_status':result['status'],'root_documents_synchronized':True,'links_checked':len(links),'source_pins':240,'protected_pins':10,'new_training':0,'user_recording_reads':0,'running_processes':[],
                       'document_sha256':{str(p):sha(p) for p in [cp,hp,sp,U/'RESULT_KO.html',U/'RESULT.json']}})
print(json.dumps({'status':'PASS','result':result['status'],'report':str(U/'RESULT_KO.html')},ensure_ascii=False))
