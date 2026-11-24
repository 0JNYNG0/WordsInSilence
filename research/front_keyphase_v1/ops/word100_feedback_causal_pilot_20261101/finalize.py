"""Close the bounded four-step cycle after separate checks; preserve history."""
from pathlib import Path
import json, hashlib, datetime, shutil
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert not (U/'COMPLETION.json').exists()
    checks={n:read(U/'review'/n) for n in ['VERIFY_POINT.json','VERIFY_RECOVERY.json','VERIFY_COMBINED.json']}
    assert all(x['status']=='PASS' and not x['failures'] for x in checks.values())
    point=read(U/'POINT_DIAGNOSIS.json');rec=read(U/'ONE_RECOVERY_RESULT.json');comb=read(U/'COMBINED_CHECK.json')
    assert all(x['usable']==26 and x['unknown']==4 and x['false_attribute_difference']==0 for x in point['summary'])
    assert rec['added_feature_cells']==11
    assert [x['shared'] for x in comb['rows']]==[38,38,49,49]
    for manifest,key in [('FROZEN_POINT_DIAGNOSIS.json','source_pins'),('FROZEN_ONE_RECOVERY.json','pins'),('FROZEN_COMBINED_CHECK.json','pins')]:
        for path,h in read(U/manifest)[key].items():assert sha(path)==h,path
    decision=dict(status='NO_ADDITIONAL_TRAINING_JUSTIFIED_BY_THIS_DIAGNOSTIC',
        identified_learned_errors_in_supported30=0,unknown_feature_opportunities=4,
        why_not_training='Visible missing hand was absent from accepted native landmarks, not a proven classifier error. Reuse fixed recovery first. No supported dense-stage/noncircular contrast emerged.',
        full_oracle_2x2='Closed after bounded review; no independent unique stage or contrasting endpoint. Not an unfinished forced experiment.',
        implemented='Word-independent attribute consumer plus unchanged current-image recovery at one new public endpoint; original posterior evidence unchanged.',
        operational_replacement=False,educational_correctness=None,
        next_scope='Keep the recovered position channel separate. Meaningful phase/order feedback still needs supported temporal-event evaluation; do not launch another cost sweep or train on current diagnostic labels.',
        limitations=['Repeated DEV,9 shared pair/reference-frame positions,not a fresh accuracy cohort',
            'One recovered public frame,not global recognition improvement',
            'Same-Codex visual review,not independent KSL expert truth',
            'Actual learner-error truth remains absent; synthetic counterexamples do not provide it'])
    write(U/'TRAINING_DECISION.json',decision)
    result='''이번 작업 결과 — 손 모양 판단, 시간 연결, 관측 부족을 나눠 확인했습니다.

무엇이 달라졌나요?
이전에는 조상의 한 참고 장면에서 왼손 정보가 비어 있었습니다.
이번에는 기존 재확인 방법을 그 장면에 적용해 왼손 위치 정보 일부를 확보했습니다.
같은 두 장면을 비교할 수 있는 수치가 69개 중 38개에서 49개로 늘었습니다.
원래 있던 오른손과 비교값은 그대로 유지했습니다. 단어 인식률이 오른 수치는 아닙니다.

1. 손 모양 판정부부터 확인했습니다.
공개 4표현의 기존 정답이 겹치는 30항목을 기존 학습 seed42/43/44 각각 확인했습니다.
30항목은 30영상이 아니라 9개 (영상쌍, 참고시점)에 모인 작은 표본입니다.
각 seed에서 26항목은 비교할 관측 근거가 있었고 확인 정답과 맞았습니다. 4항목은 보류였습니다.
예측끼리 같지만 둘 다 틀린 경우도 별도 확인했으며 이 표본에서는 없었습니다.
이 결과만으로 전체 손 모양 정확도가100%라고 말할 수는 없습니다.

2. 반복 중 다른 순간을 연결한 문제도 함께 확인했습니다.
필요한 공개 2구간·4영상·89프레임만 원본으로 다시 확인했습니다.
조상은 손 모양이 비슷한 동작이 다시 나타납니다. 현재 연결점과 다른 후보 모두 비슷한 손 모양입니다.
따라서 손 모양을 맞혔다는 사실만으로 반복의 어느 차례를 연결해야 하는지 확정할 수 없었습니다.
이집트의 참고영상은 펼친 손을 확인했지만, 비교영상의 측면 손에서는 네 손가락을 각각 확정하기 어려웠습니다.
이를 ‘주먹이다’ 또는 ‘잘못했다’로 바꾸지 않았습니다.
정확한 단계 정답을 넣는 시험은 근거가 끝내 부족해 이번 범위에서 종료했습니다.

3. 원인이 확인된 빈칸은 실제로 보완했습니다.
보류4항목은 모두 P065 48프레임의 왼손 근거 부족에 연결되어 있었습니다.
원검출에 후보2개가 있었지만 모두 오른손 주변 대체후보였습니다. 하나를 억지로 왼손으로 넘기지 않았습니다.
원본 전체화면1회와 왼팔 주변 확대1회로 재확인하고, 현재 팔 연결과 두 화면 크기의 일치 조건을 그대로 적용했습니다.
그 결과 왼손21점 중12점, 손목·손바닥 위치/방향과 양손 위치관계11수치가 추가되었습니다.
나머지9점은 채우지 않았으며, 손가락 각도14개의 유효 지원은 여전히0입니다.
따라서 위치 정보가 생겼다고 손가락 모양까지 확인됐다고 하지 않습니다.

4. 관측 보완과 시간 연결을 함께 시험했습니다.
기존/보완 관측 × 현재/다른 반복시점의 네 조건을 계산했습니다.
두 연결 후보 모두 비교 가능한69수치 중38→49, 미관측31→20이었습니다.
기존에 계산 가능했던 차이값76개(두 후보 각각38개)는 정확히 보존되었습니다.
새 위치정보는 같은 왼손-위/오른손-아래 관계를 보여줬지만, 올바른 반복 회차를 고르는 문제는 남았습니다.
다른 후보를 ‘정답 연결’이라고 부르거나 이 시험을 정렬 정확도 상승으로 계산하지 않았습니다.

추가 코드 보강
두 출력이 모두 ‘편 손이 아님’이어도 같은 손 모양이라는 뜻으로 쓰지 않도록 항목의 뜻을 명확히 했습니다.
같은 손이 동시에 ‘펴짐’과 ‘접힘’으로 나오는 모순은 별도로 표시하고, 원래 차이값과 관측 정보는 지우지 않습니다.
기존 비교부도 전체 수어의 정답을 선언하지 않았으므로 과거의 전체손모양 판정버그를 고친 것이라고 주장하지 않습니다.
정상 속도/손내림/교차/짧은 변화/미관측을 보존하는 기술검사46개와15,309항목 보존검사를 통과했습니다.
손 모양 오류와 연결 오류가 겹치면 우연히 비슷해 보이는 계산 반례도 확인했습니다. 실제 발생률을 측정한 것은 아닙니다.

학습·적용·속도
이번 추가 모델학습0, SignRep/TCN 판정부 새 추론0, 새DTW경로0입니다.
기존 판정부를 다시 학습해야 한다는 증거가 이 표본에서 나오지 않아, 관측 보완과 비교 출력 보강으로 마쳤습니다.
기존 학습 예측의 지원은26/30 그대로입니다. 새 좌표를 붙여 예전 예측까지 신뢰 가능한 것으로 바꾸지 않았습니다.
새 MediaPipe 관측만2회입니다. 이 한 프레임의 검출 합계 약0.054초, 초기화·해당 원본 디코드·특징 계산 포함 약0.55초였습니다.
영상 전체 피드백 대기시간이나 다른PC의 속도가 아닙니다.
연구 후보에만 반영했습니다. 웹/기존C·T3-F·기존DTW는 교체하지 않았습니다. 8800은 이전 보고서입니다.
사용자·멘토 촬영 접근/학습/튜닝/선택0, 원본/기존학습모델/과거실험결과 보존, 다운로드/외부전송/Git0입니다.

검증과 한계
별도 에이전트 검산: 저장 예측13,233건, 복원279건, 네조건1,103건 PASS.
독립 작성 검산이지만 동일Codex 계열이며 독립 사람의 한국수어 정답 검증이 아닙니다.
기존 전략 검토문서 한 파일의 후속 수정으로 과거핀 불일치가 생긴 이력은 PRODUCT_REVIEW_ADDENDUM.json에 기록했습니다.
모델·자료·실험결과는 그 수정의 영향을 받지 않았습니다. 원문을 추정 복원하거나 과거핀을 바꾸지 않았습니다.

현재 결론
계획1~4의 한정된 진단·보완·복합조건 검증은 마쳤습니다. 조건부 추가학습은 진행조건 미충족으로 실행하지 않았습니다.
실제로 얻은 것은 빈 왼손의 위치 근거 일부와, 손 모양 문제/시간 연결 문제를 섞지 않는 출력입니다.
수어 인식률이나 교육 피드백 정확도가 높아졌다는 증거는 아직 없습니다.
남은 핵심은 정밀 손 모양의 관측 부족, 반복 회차/동작 단계 대응, 의미상 중요한 차이를 평가할 근거입니다.
다음에는 확인된 위치 근거를 유지하면서 동작의 순서를 평가할 자료가 있는 범위만 진행합니다.
현재 진단용4개 새정답을 학습에 넣거나, 같은 자료에서 기준을 계속 바꾸는 작업은 하지 않습니다.

그림: visuals/RESULT_BEFORE_AFTER_KO.png
근거: POINT_DIAGNOSIS.json, RGB_REVIEW.json, ONE_RECOVERY_RESULT.json, COMBINED_CHECK.json, review/VERIFY_*.json
'''
    (U/'RESULT_KO.txt').write_text(result,encoding='utf-8')
    before=U/'before';before.mkdir(exist_ok=True)
    docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']
    for p in docs:
        assert not (before/p.name).exists();shutil.copy2(p,before/p.name)
    now=datetime.datetime.now().astimezone().isoformat()
    state=read(O/'STATE.json')
    state['current_stage']='FEEDBACK_CAUSAL_PILOT_COMPLETE_PARTIAL_POSITION_RECOVERY'
    state['next_single_action']='완료30지원/2창/4조건재실행X. 복원위치·기존모양근거분리유지. 반복회차/단계피드백은독립평가근거있는범위만;새거리식순회/진단라벨학습/운영교체X.'
    state[U.name]=dict(status='COMPLETE_BOUNDED_PLAN_PARTIAL_OBSERVATION_GAIN',at=now,unit=str(U),active_session=None,
        steps_completed=[1,2,3,4],conditional_training='not warranted by observed supported failures and missing temporal truth',
        point_opportunities=30,seeds=[42,43,44],usable_per_seed=26,unknown_per_seed=4,
        bounded_rgb_frames=89,paired_windows=2,new_detector_calls=2,new_classifier_inferences=0,new_training=0,new_dtw_paths=0,
        recovered_frames=1,recovered_points=12,added_geometry_features=11,added_finger_angle_support=0,
        geometry_shared_before=38,geometry_shared_after=49,geometry_denominator_per_endpoint=69,
        alternative_endpoint_is_corrected_stage=False,educational_accuracy=None,recognition_gain=None,
        verify_checks={n:v['checks'] for n,v in checks.items()},report=str(U/'RESULT_KO.txt'),production_changed=False,
        prior_review_pin_change=str(U/'review/PRODUCT_REVIEW_ADDENDUM.json'))
    (O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text(encoding='utf-8-sig')
    s8='''\n### 2026-11-01 4단계 인과 진단·보완 완료 — 왼손 위치 근거 추가, 단계·교육 정확도 미입증
`word100_feedback_causal_pilot_20261101` 완료. 기존 보호3+센터신규4 저장3seed/30기회 진단:각26지원/4unknown/지원오류0,9고유pair-ref시점에집중. 2공개paired창/89frame 확인:같은손모양반복으로단계oracle·차이대조미확보,강제2×2/학습분기종료. P065r48왼손없음은raw두개가오른손대체후보인계보확인. 기존범용full+crop재관측2회로왼손12점/11geometry추가,기존R/38지원정확보존,손가락14각지원0유지. old/newgeometry×q74/q98가설4조건각69기회38→49/unknown31→20,원차이76보존. 반복회차정답/교육정오/인식률개선아님. 새typed7소비부는음성일치/모순/미관측분리,기존whole-shape버그라고주장X. 기술46+15309보존검사/별도예측13233·복원279·결합1103PASS. 새학습/TCN·SignRep추론/DTW경로0,기존posterior지원26/30불변. C/T3F·운영·개인/멘토/원본보존. 결과txt+한글전후png,새HTML없음. 과거PRODUCT리뷰핀변경이력addendum공개보존.\n'''
    s9='''\n### 2026-11-01 다음 단일 작업 — 복원된 위치와 단계 평가의 남은 범위 분리
이번승인1~4는한정진단·필요관측복원·4조건민감도·별도검산까지완료. 완료30기회/89frame/재관측1frame/4조건재실행X. 기존모델/웹유지. 새위치정보를기존학습예측의마스크로소급승격하지않고,손가락미관측/회차모호함분리. 다음모델학습은학습필요오류와시간정답/별도평가근거가함께있는항목에만제안. 현재진단라벨4개학습/기준튜닝금지. 전단어주석/새모델·거리식순회로자동확대X. 전체목표는동작방법·순서와중요차이피드백이며아직완료아님.\n'''
    for title,entry in [('## 8.',s8),('## 9.',s9)]:
        i=txt.index(title);j=txt.index('\n',i)+1;txt=txt[:j]+entry+txt[j:]
    p.write_text(txt,encoding='utf-8')
    handoff=f'''# WordsInSilence 최신 인수인계

2026-11-01 사용자 승인 TEAM_PLAN1~4를 {U.name}에서 완료했다. 결과는 RESULT_KO.txt와 visuals/RESULT_BEFORE_AFTER_KO.png.
4단계 완료는 제한된 진단·필요보완·검증 완료이며 전체 수어 피드백 목표 달성은 아니다.

저장 보호원3 native + 새4 exactcenter, seed42/43/44 모두 각각30기회(9고유pair/ref시점) 중26지원/4unknown,지원오류0. 기존oldp/newp센터108일치. 새판정부학습 필요근거0,unknown을모델정답으로승격X.
공개2창/4영상89frame 추가시각확인: P066q98R/Lclosed1,P070r62R/Lopen1,진단전용4label학습선택금지. q74/q98같은모양반복,올바른회차oracle없음. P07295/1074fingerunknown유지. full confirmed2×2지원없어해당분기종료,좋은창재선택X.
P065r48의nativeL없음 원인: raw2후보[-1,1] 모두오른손대체(추정강제L금지). 미실행프레임인것확인후기존full/crop고정method새1+1회. L12points/11geometry추가,Langles0,기존오른손/원38지원/타프레임보존. approx0.054s검출/0.55s한프레임전체,영상전체대기시간아님.
기존/복원geometry×현재q74/다른회차q98 네조건각69기회,공통38→49·unknown31→20,기존76차이값보존. 두stage가설을검토했으나q98을정답으로부르지않음. oldposterior26지원/4unknown은그대로,재추론안함. 새action_evidence7항목소비부는same0/positive/차이/unknown/모순분리;기존버그주장X,실제이표본에서guard이득0. 기술46+15309보존검사.
별도검산 semantic_feedback: VERIFY_POINT13233,VERIFY_RECOVERY279,VERIFY_COMBINED1103 PASS. semantic_data_audit시각교차검토,semantic_model방법/복원경로검토. 동일Codex≠독립사람KSL정답. 과거전략PRODUCT리뷰지연수정으로핀불일치1건은review/PRODUCT_REVIEW_ADDENDUM에old/new와내용보존,추정복원/과거핀변경X;모델·자료·실험결과무영향.

새학습/TCN·SignRep추론/DTW경로0,기존C/T3F/운영웹DTW/사용자·멘토/원본보존. HTML생략,8800은과거페이지. 카메라/다운/외부전송/Git0. 실제실행모두종료,중복실행금지.
다음은위치관측을유지하며손가락관측과반복회차/단계평가근거를분리해야함. 현재좁은진단으로교육적허용범위/중요차이성능을단정할수없다. 완료실험다시실행/단어예외/단계답안강제채움/진단라벨학습금지. 필요학습은지원되는실제오류+정답+별도평가가함께있을때만;새거리식순회로돌아가지말것.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
    hist=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
    with hist.open('a',encoding='utf-8') as f:f.write('\n\n2026-11-01 수정계획1~4 실행: 지도학습량 증가 대신 관측·판정·시간연결 분리진단. 공통30지원 중26정상/4관측부족;2창에서 반복의동일모양 확인되어단계oracle강행안함. 기존MP재확인으로 실제빈왼손12점/11위치관계추가,38/69→49/69비교지원. 손가락모양·반복회차는미해결. 원본예측/전체인식률·교육정확도상승주장없음. 동작방법·순서피드백목표유지,불필요재학습·비용식순회대신확인된누락의좁은보완으로진행. 상세 '+str(U/'RESULT_KO.txt')+'\n')
    (U/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(status='COMPLETE',active_session=None,all_commands_exited=True,
       completed_at=now,new_training=0,new_classifier_inferences=0,new_detector_calls=2,new_dtw_paths=0),ensure_ascii=False,indent=2),encoding='utf-8')
    pins={str(p):sha(p) for p in [*docs,U/'RESULT_KO.txt',U/'TRAINING_DECISION.json',U/'CURRENT_EXECUTION.json',
         U/'POINT_DIAGNOSIS.json',U/'RGB_REVIEW.json',U/'ONE_RECOVERY_RESULT.json',U/'COMBINED_CHECK.json',
         *sorted((U/'review').glob('*.json')),U/'visuals/RESULT_BEFORE_AFTER_KO.png']}
    write(U/'COMPLETION.json',dict(status='COMPLETE_PARTIAL_OBSERVATION_GAIN_NOT_EDUCATIONAL_ACCURACY',at=now,
           user_plan_steps_completed=[1,2,3,4],conditional_training_not_run=True,normal_sign_classification_preserved=True,
           production_models_changed=False,private_recordings_accessed=False,report=str(U/'RESULT_KO.txt'),pins=pins))
    assert read(O/'STATE.json')[U.name]['new_training']==0
    assert all(sha(p)==h for p,h in read(U/'COMPLETION.json')['pins'].items())
    print('FINALIZATION PASS: 4-step bounded cycle complete, docs synchronized; training0/classifier0/DTW0/detector2')
if __name__=='__main__':main()
