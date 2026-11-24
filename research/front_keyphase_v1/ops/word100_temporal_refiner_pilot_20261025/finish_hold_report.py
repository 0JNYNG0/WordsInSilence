"""Record the failed visible-teacher readiness gate; never train on failure."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, hashlib, shutil, html, ast

U=Path(__file__).resolve().parent; ROOT=U.parents[3]; B=U.parent/'word100_temporal_public_bundle_20261025'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    with (U/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    protected=read(B/'PROTECTED_BEFORE.json')
    assert all(sha(p)==h for p,h in protected.items())
    check=read(U/'VISIBLE_BASELINE_CHECK.json'); labels=read(U/'VISIBLE_LABELS_FROZEN.json')
    for name in ['rgb_features.py','refiner_model.py','train_compare.py']:
        ast.parse((U/name).read_text(encoding='utf-8'))
    assert not (U/'runs').exists() and not (U/'RGB_INDEX.json').exists()
    save('AUTHORIZATION.json',{'at':now,'user_request':'보이는 손 실제 위치 작은 표본 검증; 문제없으면 후보학습과 네조건 비교; 중요한 문제 즉시 보고; C/T3-F 보존; 사용자 촬영 학습/튜닝 금지',
        'conditional_training_authorized':True,'condition_met':False,'reason':'Frozen image assessment exposed a suspect wrist pseudo-target in V05; equal-trust21-point supervision is not justified.',
        'user_data_access':False,'runtime_replacement_authorized':False})
    save('VISIBLE_READINESS.json',{'status':'HOLD_SUSPECT_TEACHER_COORDINATE','at':now,
        'training_allowed':False,'ready_for_feature_extraction':False,
        'reason':'Existing MediaPipe observations cannot be treated as uniformly correct21-point targets; V05 wrist materially outside image-only interval.',
        'candidate_training_completed':False,'four_condition_comparison_completed':False,
        'conditional_user_authorization_exists':True,'affected_sample':'V05',
        'frozen_labels_sha256':sha(U/'VISIBLE_LABELS_FROZEN.json'),
        'scope':'coarse2D image assessment by two AI reviewers on12overlap/24total; not human-expert landmark annotation or3D truth',
        'do_not':['loosen frozen rectangles','silently exclude specific word/sample','claim proxy error improvement as anatomical accuracy','promote estimates to observations'],
        'next':'Class-independent per-point supervision reliability; preserve coordinates but distinguish usable observation from trustworthy learning target, then recheck normal/crossing controls.'})
    save('DRAFT_COMPARISON.json',{'status':'DRAFT_NOT_FROZEN_NOT_EXECUTABLE','training_authorized_for_execution':False,
        'planned_conditions':['B0 full-frame existing observations','B1 same-image re-detection merged beforefixedROI','B2 coordinate+time infill','B3 RGB+coordinate+time infill'],
        'planned_splits':{'train':[1,2,3,4,5,6],'dev':[7],'test':[8,9]},
        'planned_seeds':[17,29,43],'planned_epochs':80,'window_frames':11,
        'model':'Matched small bidirectionalGRU, observedcoordinates preserved, predictions onlyestimates',
        'rgb':'Frozen locallyavailableMobileNetV3Small features on body-wrist crops; no hiddenhand target cropcenter',
        'current_blocker':'Equal-weight existing MP21-point pseudo-targets include suspect source errors.',
        'lane_separation':'Natural missing B0/B1/recovery distinct from synthetic holes; synthetic mask AFTER B1, no cachedtargetcopyback',
        'metrics':['wrist location','wrist-relative finger geometry','1/3/7frame holes','observed preservation','false/duplicate/swapped new hand','coverage and uncertainty'],
        'codes_unexecuted':[str(U/f) for f in ['rgb_features.py','refiner_model.py','train_compare.py']],
        'no_claim':'No pilottraining, learnedperformance, recognitiongain, independentnew-usergeneralization.'})
    save('GATE_REVIEW.json',{'decision':'HOLD_TRAINING','reviewers':['parent','temporal_refiner_review'],
        'support':'Both independently localized12overlapimages before viewing coordinates; all12 overlap/unscorabledecisions agreed. Frozen labels were not moved after comparison.',
        'V05':'Targetleftwrist interval[992,495,1035,535]; savedB0/B1[972.6311,464.2286]; minimumdistance outside36.36px. Image shows connection closer to forearm than savedpoint. Suspect target error, not certifiedanatomy measurement.',
        'V21':'11.42px outsidecoarsebox withblur; uncertainty, notconfirmederror.',
        'nonblockers':['subpixel/near-edge box discrepancies do notprove wrongjoints','outsideROI loweredhand is not restorationfailure','occludedwrist labels remainunscorable'],
        'rationale':'A learner can reproduce a badpseudo-target and improveitsproxy score. Currentpreservation protects thiscoordinate too. Actualhandrecovery goal requires a separate supervision-evidence gate.',
        'source_files':{f:sha(U/f) for f in ['VISIBLE_SAMPLE_SELECTION.json','VISIBLE_REVIEW_PARENT.json','VISIBLE_REVIEW_INDEPENDENT.json','VISIBLE_LABELS_FROZEN.json','VISIBLE_BASELINE_CHECK.json']}})
    text='''보이는 손 위치 검증 결과 — 학습 직전 중요 문제 발견
2026-10-25 | WordsInSilence | 기존 C·T3-F 유지

결론
작은 공개 표본의 영상 확인과 기존/재검출 비교를 마쳤다. 새로 찾아낸 손 위치는 도움이 되는 사례가 있었지만, 기존 좌표 자체가 손목 연결부와 맞지 않는 것으로 의심되는 사례도 확인했다. 현재 후보는 기존 좌표를 모두 같은 신뢰도의 정답으로 배우므로 '문제없으면 학습' 조건을 통과시키지 않았다. 학습·RGB특징추출·B2/B3후보실행은0이며 네 조건의 최종 성능 비교는 아직 하지 않았다.

어떻게 확인했나
공개 영상24장면을 후보 결과 확인 전에 고정했다. 정상 구간9, 과거 교차 후보6, 짧은 누락 뒤 재등장 후보9다. 원본 이미지에 관절 표시를 하지 않은 상태에서 손목/손바닥 밑부분의 대략적인 위치 범위를 기록했다. 별도 에이전트가24장면, 주담당이 겹치는12장면을 독립적으로 봤고,12장면은 위치 범위가 겹치거나 평가불가 판단이 일치했다. 범위는 예측값을 보기 전에 고정했다.
이는 AI의 영상 기반 대략적2D 위치 확인이다. 사람이 정밀 주석한21개 관절 정답이나3D 굽힘 정답은 아니다. 보이지 않는 관절 위치를 만들어 넣지 않았다.

24장면의 구성
- 범위 안에서 손목 위치를 비교할 수 있는 장면20개.
- 손목 자체가 다른 손에 가려진 장면2개: 평가불가로 유지.
- 내려간 손의 위치가 상반신 범위 경계에 걸친 장면1개, 범위 밖인 장면1개: 복원 실패로 세지 않음.

기존 관절과 현재 이미지 재확인 결과
20개 평가 가능한 장면에서 기존 관절B0의 손목값은11개에 있었고,9개는 없었다. 저장된 재검출B1을 연결하면20개 모두에 손목값이 있었다.
새로 추가된9개 중8개는 미리 표시한 위치 범위 안에 들어왔다. 나머지1개는 범위에서약11.4픽셀 벗어났지만 영상이 흐려 실제 오류라고 단정할 수 없다.
전체20개 중 범위 안에 들어온 값은B0 6개, B1 14개다. 이것은 수어 인식률이나 정확한 관절 정답률이 아니다. 대략적인 이미지 위치 범위와의 일치 건수이며, 범위 가장자리에서1픽셀 내외 차이가 난 경우도 포함한다.

중요한 문제: 틀릴 수 있는 답안을 그대로 배우게 되는 구조
V05의 왼손 손목은 사진에서 팔과 손이 이어지는 부분보다 손가락 쪽에 좌표가 찍혀 있다. 기존B0와 재검출B1 모두 같은 값이다. 미리 정한 위치 범위에서약36.4픽셀 벗어났다. 두 검토자가 원본을 다시 확인했으며 단순한1픽셀 경계 차이보다 중요한 문제 가능성이 있다고 판단했다. 정밀 해부학 오차를 확정한 것은 아니지만, 그 값을 확실한 정답으로 취급할 근거는 부족하다.
쉽게 말하면, 답안지의 일부가 틀렸을 수 있는데 AI에게 그 답을 그대로 외우게 할 위험이다. 그대로 학습하면 저장된 좌표를 닮는 점수는 좋아져도 실제 손 위치는 더 정확해지지 않을 수 있다.
현재 초안은 이미 관측된 좌표를 보존하고 빈 부분만 채운다. 따라서 잘못 관측된 값도 그대로 남는다는 한계가 있다. 정상 관절 보존과 관측된 모든 관절의 정확성은 다른 문제다.

정상 손과 교차 확인
V11·V12·V13의 교차 장면에서는 보존된 손목 좌표가 사전에 기록한 위치 범위와 맞았다. 관절이 교차했다는 이유만으로 버리는 방식은 사용하지 않았다. B1은 기존 유효 좌표를 그대로 보존한다. 다만 이 사실만으로 모든 교차나 새로 추정할 손의 안전성이 검증됐다고 할 수 없다.
원본24장면, 위치표시, 좌표 비교 결과를 모두 보존했다. 문제 장면을 몰래 제외하거나 표시 범위를 넓혀 통과시키지 않았다.

다음 보완 방향
1. '관측된 좌표'와 '학습 정답으로 신뢰할 좌표'를 분리한다. 좌표는 보존하되 불확실한 값까지 강하게 따라 배우게 하지 않는다.
2. 손목 위치, 손가락 각각의 위치/모양을 따로 판단한다. 팔·손목 연결, 앞뒤 움직임, 실제 이미지 근거가 엇갈리면 해당 부분의 정답 근거를 보류한다. 특정 단어명이나V05번호에 대한 예외는 만들지 않는다.
3. 신뢰도 규칙을 정할 때 정상 교차·손내림·빠른 변화를 잘못 버리지 않는지 같은 공개 표본과 더 넓은 정상 대조에서 확인한다. 평가 영상/점수를 보고 규칙을 맞추거나 사용자 촬영을 이용하지 않는다.
4. 그 검증이 통과하면 같은 사람 분리·시간창·예산으로 관절시간B2와RGB관절시간B3를 학습한다. 실제 누락과 인위적 누락을 나눠 비교하고, 손목/손가락/비교 가능 범위를 따로 보고한다.
전 단어의 의미와 모든 움직임을 사람이 일일이 정의하자는 뜻은 아니다. 학습에 쓸 좌표의 신뢰 근거를 공통 방식으로 확인하려는 것이다.

실제 작업 상태
- 완료: 공개24장면의 눈에 보이는 위치 확인, 기존B0/재검출B1의 제한된 비교, 입력/비교 누설 위험 검토, 별도 후보 코드 초안과 독립 정적 검토.
- 미실행: RGB특징추출, 새 후보 모델 실행, 학습6회, 네 조건의 최종 비교, C·T3-F 새 인식률 측정.
- 코드 정적검토에서 입력차원577/575 불일치를 실행 전에575로 바로잡았고, 손목·손가락·누락길이별 지표를 분리했다. 이것은 성능 개선 결과가 아니다.
- 기존 C·T3-F/운영DTW·웹 관련32파일은 고정 해시와 일치한다. 사용자·멘토 촬영 접근/학습/튜닝0, 원본·과거 결과 수정0.
- 작성: 주담당Codex/temporal_bundle_split. 영상독립확인:temporal_refiner_review. 코드독립검토:temporal_bundle_inventory. 현재 실행 중인 학습/특징추출 없음.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    cards=[('V05','학습 전 해결할 문제','주황 점: 기존·재검출 손목값(동일). 초록 상자: 관절 표시 없이 이미지에서 먼저 정한 손목 위치 범위. 점이 손가락 쪽으로 치우쳐 보인다.'),
           ('V11','정상 교차 보존 사례','교차 상태에서도 기존 좌표가 사전에 표시한 손목 범위 안에 있다.'),
           ('V20','재검출로 보완된 사례','기존에 없던 손목값이 생겼고, 독립적으로 표시한 위치 범위 안에 들어왔다.'),
           ('V21','단정하지 않는 사례','위치 범위에서약11.4픽셀 벗어났지만 모션 블러가 있어 명확한 오류로 확정하지 않는다.')]
    gallery=''.join(f'<figure><h3>{title}</h3><img src="evidence/{cid}_comparison.jpg" alt="{html.escape(title)}"><figcaption>{desc}</figcaption></figure>' for cid,title,desc in cards)
    paras=text.split('\n\n');body=''.join('<section>'+''.join('<p>'+html.escape(line)+'</p>' for line in x.splitlines())+'</section>' for x in paras[1:])
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보이는 손 검증 · 학습 전 문제 발견</title><style>body{font-family:system-ui,"Malgun Gothic",sans-serif;line-height:1.8;color:#192c3a;background:#f4f6f8;max-width:1080px;margin:auto;padding:30px 24px}h1{font-size:34px;line-height:1.35}.badge{color:#9a4311;font-weight:700}section,figure,.lead{background:#fff;border:1px solid #dfe5eb;padding:22px;border-radius:14px;margin:20px 0}section p:first-child{font-weight:700;font-size:20px}p{margin:9px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}img{width:100%;height:auto}figcaption{font-size:14px;color:#4e6070}a{color:#185aad}table{width:100%;border-collapse:collapse;background:#fff}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}@media(max-width:750px){.grid{grid-template-columns:1fr}h1{font-size:26px}}</style><body><div class="badge">표본 검증 완료 · 조건부 학습 통과 보류</div><h1>좌표가 있다고<br>정확한 학습 정답인 것은 아니었다</h1><div class="lead">일부 손을 다시 찾는 효과는 보였다. 하지만 기존 손목값의 오류 가능성이 발견되어, 그 값을 그대로 배우는 학습은 시작하지 않았다. 기존 C·T3-F는 그대로다.</div><table><tr><th>공개 표본</th><th>이번 확인</th></tr><tr><td>24장면</td><td>관절 표시 없이 위치 확인</td></tr><tr><td>20장면</td><td>범위 안에서 손목 비교 가능</td></tr><tr><td>재검출 손목9개</td><td>8개는 이미지 확인 범위 안,1개는 불확실</td></tr><tr><td>새 학습0회</td><td>기존 좌표를 모두 정답으로 취급할 근거 부족</td></tr></table>'''
    page+='<h2>실제 확인 장면</h2><div class="grid">'+gallery+'</div>'+body+'<section><a href="VISIBLE_BASELINE_CHECK.json">24장면 전체 수치</a> · <a href="VISIBLE_READINESS.json">학습 보류 근거</a> · <a href="GATE_REVIEW.json">교차 검토 판단</a> · <a href="code_REVIEW.json">코드 정적 검토</a></section></body></html>'
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    next_action='보이는 손 표본에서 발견한 학습 정답 신뢰 문제를 먼저 보완한다. 관측좌표는 보존하되 위치/손가락별 정답 근거를 따로 표시하고, 팔 연결·앞뒤 시간·이미지 근거가 엇갈리는 좌표를 동일 신뢰도로 학습시키지 않는 공통 규칙을 검토·검증한다. V05/특정단어 임의제외·정답범위 사후확대 금지. 정상 교차/손내림 보존을 재확인해 조건부 학습 자격이 회복된 뒤 B0~B3 비교로 재개한다. 현재 VISIBLE_READINESS HOLD이며 학습/RGB특징추출 금지. C·T3-F/운영웹 유지, 사용자·멘토 촬영 접근·학습·튜닝·선택 금지.'
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';handoff=ROOT/'NEW_CHAT_HANDOFF.md';state=U.parent/'STATE.json'
    before=U/'before';before.mkdir(exist_ok=False)
    for p in [context,handoff,state]:shutil.copy2(p,before/p.name)
    c=context.read_text(encoding='utf-8-sig')
    entry='''### 2026-10-25 보이는 손24장면 검증 — 기존 정답좌표 불일치 의심으로 후보 학습 보류
- 사용자 작은표본 확인→문제없으면 학습/4조건 비교를 조건부 승인. 공개24장면 image-only 범위고정, 별도에이전트24/부모중복12 확인.22위치표시,20ROI내/1경계/1밖/2가림평가불가. 범위를 예측에 맞춰 변경하지 않음.
- ROI내20장면 B0손목가용11→B1가용20. 추가9중8개 사전범위 내,1개V21은11.4px차이+블러로 불확실. 수어인식률·정밀관절정답률 아님.
- V05 기존B0/B1왼손목이 사전위치범위 밖36.4px; 실제손목보다손가락/반대손쪽 치우침 의심. 기존MP21좌표를 동일신뢰도로 정답학습하면 오류복제 점수만 좋아질 위험. VISIBLE_READINESS=HOLD,학습/특징추출 불허. 특정샘플제외/범위완화 없음.
- V11/12/13정상교차 위치범위와 일치. 기존값 보존과 모든값 정확성은 별개. C·T3-F/운영32보호해시 유지,사용자촬영 접근0. 학습/새ML추론/다운로드0.
- 격리RGB추출/GRU후보/6run비교 코드초안+독립정적검토만 완료(차원577→575 수정). 런타임preflight/실제학습 미실행. 근거 `research/front_keyphase_v1/ops/word100_temporal_refiner_pilot_20261025/RESULT_KO.html`, `VISIBLE_READINESS.json`, `GATE_REVIEW.json`.

'''
    c=c.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+entry,1)
    c=c.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-10-25 좌표를 관측한 것과 학습 정답으로 신뢰하는 것을 분리\n\n'+next_action+'\n\n',1)
    context.write_text(c,encoding='utf-8')
    s=read(state);s.update(current_stage='TEMPORAL_REFINER_VISIBLE_TEACHER_GATE_HOLD_NO_TRAINING',updated_date_kst=now,updated_at=now,next_single_action=next_action)
    assert U.name not in s
    s[U.name]={'status':'VISIBLE_CHECK_COMPLETE_TRAINING_HOLD_SUSPECT_PSEUDO_TARGET','at':now,'report':str(U/'RESULT_KO.html'),
        'readiness':str(U/'VISIBLE_READINESS.json'),'conditional_training_authorization':True,'training_condition_met':False,
        'visible_cases':24,'coarse_scorable':22,'roi_inside_scorable':20,'new_B1_wrist_available':9,'added_inside_image_box':8,
        'suspect_existing_teacher_case':'V05','uncertain_added_case':'V21','new_training':0,'new_model_inference':0,'new_rgb_extraction':0,
        'runtime_change':0,'protected_unchanged':32,'user_recording_access':0,'prototype_code_runtime_tested':False,'active_execution':None,'next':next_action}
    state.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    handoff.write_text(f'''# WordsInSilence 最新 인수인계

갱신 {now}

현재 목표: RGB·관절·시간 보완으로 실제 손의 관측 근거 개선. 사용자 작은 표본 통과 시 후보학습·4조건 비교를 승인했으나, 이번 visible gate는 중요 문제로 통과하지 않았다.

마지막 완료: 공개24장면 image-only 손목 위치 확인/기존B0·재검출B1 제한비교.22범위표시 중20ROI내/1경계/1밖,2가림평가불가. 새학습/추론0.

다음 한 작업: {next_action}

근거: {U/'RESULT_KO.html'}
현재gate: {U/'VISIBLE_READINESS.json'}
평가: VISIBLE_SAMPLE_SELECTION / VISIBLE_REVIEW_PARENT / VISIBLE_REVIEW_INDEPENDENT / VISIBLE_LABELS_FROZEN / VISIBLE_BASELINE_CHECK / GATE_REVIEW.json.

핵심 사실
- 원본이미지에서 관절예측 없이24장면 고정·관측.별도에이전트24/부모12겹침확인.12개는 위치범위중첩 또는평가불가판정일치. AI의대략2D영상판단이며 사람전문가의정밀21점/3D정답 아님.
- ROI내20손목 B0가용11→B1가용20. 새9개중8사전범위내, V21약11.4px밖+블러는불확실.전체범위내B0 6/B1 14는인식률아님. 작은경계차이도포함.
- V05 후원REAL05 f53 왼손목 기존B0/B1=(972.631,464.229), 고정이미지범위[992,495,1035,535] 밖36.36px. 손목보다손가락쪽 치우침의심. 기존좌표를정답으로그대로배우면오류복제가능. 학습조건미충족,범위완화/해당단어제외금지.
- V11/12/13교차는기록범위와일치. 정상기존좌표보존과관측값전체정확성은별개.

준비코드 상태
- rgb_features.py / refiner_model.py / train_compare.py는별도초안.정적검토 code_REVIEW.json통과이나 모델forward/런타임preflight/학습/특징추출은실행안함. 입력차원577→575실행전수정.
- 계획B2/B3같은GRU/11창/3seed[17,29,43]/80epoch/signer1~6train7dev8~9test. 이들은기존C가본사람이므로새사람전체시스템검증아님. DRAFT_COMPARISON이며동결실행조건아님.
- 합성누락은B1통합후동일하게숨김,캐시B1로정답copyback금지. 실제누락lane와합성proxy구분. 몸손목으로만RGBcrop,숨긴손좌표cropcenter금지. 관측vs추정분리.

보존
- 이전준비61영상/24표현/9명/7647프레임,394시험정의는읽기전용유지(앞작업word100_temporal_public_bundle_20261025).
- C·T3-F/운영DTW·웹32보호파일해시동일. 원본/과거결과/사용자·멘토촬영미변경. 사용자촬영접근·학습·튜닝·선택0. 새학습/ML추론/RGB특징추출/다운로드0.
- 현재실행중프로세스/학습잠금없음. 조건부사용자승인과실제학습자격미통과를구분할것.
'''.replace('最新','최신'),encoding='utf-8')
    assert all(sha(p)==h for p,h in protected.items())
    assert next_action in context.read_text(encoding='utf-8') and next_action in handoff.read_text(encoding='utf-8')
    save('FINAL_AUDIT.json',{'status':'VISIBLE_CHECK_COMPLETE_CONDITIONAL_TRAINING_NOT_STARTED','at':now,
        'conditional_work_incomplete_reason':'visible teacher quality gate held; not a toolapproval failure',
        'new_training':0,'new_inference':0,'new_rgb_features':0,'protected_unchanged':32,
        'artifacts':{p.name:sha(p) for p in U.glob('*') if p.is_file() and p.suffix in ['.py','.json','.txt','.html']},
        'root_sync':{str(p):sha(p) for p in [context,handoff,state]},'active_execution':None})
    print(json.dumps({'status':'HOLD_REPORT_COMPLETE','report':str(U/'RESULT_KO.html'),'training':0},ensure_ascii=False))

if __name__=='__main__':main()
