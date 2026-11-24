"""Finalize the local preparation report and current handoff; no ML work."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, html, json, shutil

U = Path(__file__).resolve().parent
R = U.parents[3]

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(x, f, ensure_ascii=False, indent=2)

def main():
    assert (R / 'AGENTS.md').exists(), R
    prep = read(U/'PREPARE_RESULT.json'); verification = read(U/'VERIFY_RESULT.json')
    loader = read(U/'LOADER_INDEPENDENT.json'); scenario = read(U/'SCENARIO_CHECK.json')
    manifest = read(U/'MANIFEST.json'); protocol=read(U/'EVALUATION_PROTOCOL.json')
    assert verification['status']=='PASS' and loader['status']=='PASS'
    assert scenario['counts']['intact_controls']==61 and scenario['cases']==394
    assert read(U/'SCENARIO_INDEPENDENT.json')['status']=='PASS'
    protected=read(U/'PROTECTED_BEFORE.json')
    assert all(sha(p)==h for p,h in protected.items())
    now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    groups=defaultdict(list)
    for row in manifest: groups[(row['word_code'],row['display_label'])].append(row)
    clock_cases=[r['clip_id'] for r in manifest if r['max_average_fps_clock_difference_us']>1]
    write(U/'ISSUES_RESOLVED.json',{
        'at':now,'source_timestamps_unchanged':True,
        'average_fps_check':{'first_error_preserved':'FIRST_ERROR.txt','affected_clips':clock_cases,
            'cause':'Container average FPS differs from actual per-frame presentation times; frame-index/FPS was an invalid comparison clock.',
            'resolution':'Compared every decoded frame PTS to preserved observation timestamp_us; all61 pass within1us. No source or observation time changed.'},
        'model_adapter':{'issues':['Audit-only unusable coordinates must not enter model tensors.','Some valid hand coordinates lack per-joint confidence (NaN).'],
            'resolution':'Unified model adapter zero-masks unusable coordinates; retains valid coordinates, adds confidence-availability masks; excludes audit raw/native streams, class labels and targets.',
            'evidence':['LOADER_CHECK.json','LOADER_INDEPENDENT.json','SCENARIO_INDEPENDENT.json'],
            'first_scenario_artifacts':'before_scenario_adapter_fix/'},
        'scope':'Preparation defects corrected; not a change to existing C/T3-F or proof of recognition gain.'})
    text=f'''공개 입력·평가 묶음 준비 결과
2026-10-25 | WordsInSilence | 준비 완료, 학습 전 공동 검토 단계

1. 이번에 실제로 완료한 일
이미 확보한 AI Hub 원본 RGB 영상61개와 같은 시점의 기존 관절·관측 정보를 연결했다. 24개 표현,9명,7,647프레임이다. 새로운 영상 다운로드·MediaPipe 추출·모델 추론·학습은 하지 않았다.
기존 문제·정상 대조36개와 여러 사람 비교27개를 합쳐 중복2개를 제거했다. '수어·후원·멀다'는 각각9명이고 나머지 표현은1~5명이다. 전체100단어 중 첫 관측 보완 시험용 작은 묶음이며,24개 모두9명이 수행한 것은 아니다.
기존 원본 RGB 약280.6MB는 제자리에서 읽는 연결 방식이다. 새 좌표 묶음은 약15.5MB다. 전체 약1TB 자료를 복사·재추출하지 않았다.

2. 처리 순서와 입력 분리
원본 전체 화면의 기존 검출 → 현재 이미지에서 재확인한 손과 통합 → 동일한 상반신 범위의 관절만 사용한다. 원본 이미지와 관절의 프레임 순서·시간을 그대로 맞췄으며, 화면을 잘라 검출했던 과거 sensor 결과는 이번 전체 화면 비교의 기준값으로 섞지 않았다.
감사용 파일에는 원래 좌표도 보존한다. 실제 모델 입력 어댑터는 범위 밖/사용 불가 좌표를 가리고, 목표 단어·파일명·주석 구간·정답 참고·숨겨 둔 원본 좌표를 전달하지 않는다. 정상 좌표의 신뢰도 값만 없는 경우에는 좌표를 버리지 않고 '신뢰도 정보 없음'을 별도로 표시한다.
이번에는 학습용280차원 변환·손크기 재계산·새 모델 구조나 학습 단계 수를 실행하지 않았다. 다음 비교에서 손크기 기준은 보완 전 기준으로 고정하여 크기 변화와 RGB 효과가 섞이지 않게 한다.

3. Gloss 연결
기존100개 표현의 출처 단어 ID와 표시 이름을 연결했다. Gloss는 자료 식별·같은 표현 참고 연결·표현별 집계용이다. 동작을 복원할 때 정답 단어를 미리 알려 주지 않는다.
61개 영상 사이에 서로 다른 사람의 같은 표현 참고 연결268개를 만들었다. 쿼리 자신과 같은 사람은 제외한다. 이 작은 묶음 안에서 다른 사람의 같은 표현 참고가 없는15개 영상도 표시했다. 다른 한국어 뜻/말뭉치/형태변이를 자동으로 하나의 표현으로 합치지 않는다.

4. 준비한 시험
원래 입력 그대로의 대조61개와,21개 손 관절이 연속 관측되는 구간의 한 손 좌표를1·3·7프레임씩 숨기는333개를 합쳐394개 시험 정의를 준비했다. 영상 RGB·몸·반대손·시간은 그대로다. 이 변형을 적용할 수 있는 손은111개이며,연속 완전 관측 구간이 부족한11개 손은 미지원으로 명시했다. 같은 원본의 변형들은 동일 그룹에 둔다.
이 시험은 '원래 저장됐던 좌표를 되찾는가'를 확인한다. 원래 MediaPipe 좌표도 틀릴 수 있으므로 실제 손가락 정답이나 물리적 가림 복원의 증명으로 사용하지 않는다. 394개는 독립 촬영394개가 아니며,같은61개 촬영에서 만든 시험 조건 수다.
자연적인 관절 누락 후보583구간과 확인용305프레임도 색인했다. 이 숫자는 오류583개라는 뜻이 아니다. 정상 손내림·한손 동작·실제 가림·센서 검출 실패가 섞일 수 있어 아직 유형 정답을 확정하지 않았다.

5. 비교할 네 조건
B0 기존 전체 영상 관절: 출발점.
B1 현재 이미지 재확인: 같은 장면을 다시 보아 찾은 정보의 효과.
B2 관절·시간 보완: 앞뒤 좌표만 함께 보았을 때의 효과.
B3 RGB·관절·시간 보완: 앞뒤 영상도 함께 보았을 때 추가 효과.
B0/B1의 기존 입력은 연결했고 B2/B3는 비교 계약만 정리했다. 새 보완 모델을 실행한 결과가 아니다. 각 후보는 같은 영상·시간·범위·기준 크기를 사용한다. B2/B3 학습을 시작할 때에는 같은 자료 노출·학습 예산·선택 규칙을 먼저 고정한다.

6. 무엇이 좋아져야 통과인가
첫째,보이는 손 위치를 독립 확인한 지점에서 위치 오차가 줄어야 한다. 손 위치와 손가락 각도는 따로 평가한다.
둘째,정상 손·정상 교차·짧은 실제 움직임이 보존되어야 한다. 가짜 손,중복 손,좌우 뒤바뀜이 새로 생기면 채택을 보류한다.
셋째,관측/재검출/추정/미확정을 구분한다. 손이 사라졌다고 비교 시간이나 평가 분모가 줄어 좋은 점수를 받으면 안 된다.
넷째,C·T3-F를 고정한 인식 결과와 DTW 비교 범위/잘못된 경고를 별도로 확인한다. 좌표를 더 찾았다는 사실만으로 인식률이나 교정 정확도가 좋아졌다고 하지 않는다.
다섯째,실제 처리시간도 함께 측정한다. 첫 후보는 촬영 후 앞뒤 장면을 쓰는 분석이며,즉시 실시간으로 동작한다는 뜻은 아니다.

7. 입력 검증 결과와 발견한 문제
원본61개7,647프레임을 끝까지 디코딩하여 관절 시간표와 연결했다. 다른 에이전트가61개 배열의8,648,757개 값과483개 원본 파일 해시를 재검산했다. 과거 고정 해시459개도 일치했다. 기존 유효 좌표271,653개가 보존됐고,과거 재검출 결과의 추가 가용점8,590개를 연결했다. 이는 이번에 새로 개선한 성능 수치가 아니다.
기존 C·T3-F 체크포인트와 웹/DTW 관련 보호파일32개는 변경되지 않았다. 사용자·멘토 촬영 자료에는 접근하지 않았다.
검사 중3개 영상의 평균FPS 표기로 계산한 시간에 최대약0.0165초 차이가 있었다. 실제 각 프레임 시간으로 대조하니 관절 기록과1마이크로초 이내로 일치했다. 검사식을 바로잡았고 원본 시간은 수정하지 않았다.
입력 어댑터 검토에서는 사용 불가 좌표의 노출과 비어 있는 관절별 신뢰도 문제를 보완했다. 현재 두 조건122건과 양손 누락244건의 독립 검산에서 값 유한성·원본 우회 방지·정상 좌표 보존을 통과했다. 최초 검사 실패와 수정 근거는 별도 보존했다.
대표3영상의 화면 미리보기도 확인했다. 이는 전체 머리/상반신과 범위 선이 표시되는지에 대한 화면 점검이며,모든 관절의 해부학적 정확성 판정은 아니다.

8. 아직 확인되지 않은 것
이61개는 이미 학습·개발 검토에 사용된 공개 자료다. 새 사용자 일반화 시험으로 보고하지 않는다.17·18번 촬영자도10/19 평가에 이미 사용됐다.14~16번 예약 자료는 이번에 열지 않았으며,후속 최종 평가 전 노출 이력을 다시 확인한다.
독립적인 손가락 정답 좌표와 완전히 가려진 손의 실제3D 정답은 없다. 일부 보이는 장면의2D 위치를 별도 확인하는 일은 필요하지만,모든 단어의 의미·필수 동작을 사람이 일일이 정의해야 시작할 수 있는 구조는 아니다.
고정된 카메라 화면 밖의 정보는 실제 사용 중에는 얻을 수 없다. 전체 영상 검출 후 범위를 제한한 시험과,픽셀 자체가 화면 밖인 시험을 구분한다. 후자는 이번 묶음의 완료된 검증으로 주장하지 않는다.

9. 다음 한 단계
이번 준비 결과를 함께 확인한 다음,보이는 손의 위치를 확인할 작은 평가 표본과 남은 가림/교차 구간의 근거를 고정하고 B0~B3 비교를 진행한다. 학습이 필요한 후보는 사람·원본 그룹,학습/선택/평가 역할과 예산을 먼저 확정한다. 기존 C·T3-F는 그대로 둔다.
목표는 특정24개/100개 단어에 대한 예외가 아니라,어떤 표현에서도 쓰는 관측 보완이다. 다만 효과의 범위는 실제 시험한 자료에서부터 확인하고 확대한다. 현재 새 인식률 향상은 측정하지 않았으며,학습은 사용자 검토 전까지 시작하지 않는다.

구현: Codex 주담당. 자료·수치 독립 검산: temporal_bundle_inventory. 자료 노출/비교 계약/합성시험: temporal_bundle_split. 현재 실행 중인 준비 프로세스 없음.
주요 근거: MANIFEST.json, GLOSS_REFERENCE_INDEX.json, EVALUATION_PROTOCOL.json, SCENARIOS.json, VERIFY_RESULT.json, LOADER_INDEPENDENT.json, SCENARIO_INDEPENDENT.json.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    table=''.join(f'<tr><td>{html.escape(k[1])}</td><td>{len(v)}</td><td>{len(set(r["signer_id"] for r in v))}</td></tr>' for k,v in groups.items())
    gallery=''.join(f'<figure><img src="preview/{cid}.jpg" alt="공개 원본 전체 화면과 상반신 하단 경계"><figcaption>{label} · 노란 선은 사용 범위의 하단. 미리보기에만 표시.</figcaption></figure>' for cid,label in [('NIA_SL_WORD0072_REAL01_F','가래떡'),('NIA_SL_WORD0949_REAL01_F','멀다'),('NIA_SL_WORD1675_REAL04_F','성남시수어통역센터')])
    sections=[]
    for block in text.split('\n\n')[1:]:
        lines=block.splitlines()
        if lines and lines[0][:1].isdigit() and '. ' in lines[0]:
            sections.append('<section><h2>'+html.escape(lines[0])+'</h2>'+''.join('<p>'+html.escape(line)+'</p>' for line in lines[1:])+'</section>')
        else: sections.append('<p>'+html.escape(block).replace('\n','<br>')+'</p>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 RGB·관절·시간 평가 준비</title><style>
    :root{font-family:system-ui,"Malgun Gothic",sans-serif;color:#1d2939;background:#f3f5f8;line-height:1.8}body{max-width:1100px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.35}h2{font-size:23px;margin:0 0 16px}section,.box{background:white;border:1px solid #e4e8ef;border-radius:16px;padding:26px;margin:22px 0}p{margin:10px 0}.tag{color:#14644b;font-weight:700}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stats div{background:#112f49;color:white;padding:20px;border-radius:14px}.stats b{display:block;font-size:30px}figure{margin:26px 0}img{width:100%;height:auto;border:1px solid #ddd}figcaption{color:#52606d;font-size:14px}table{border-collapse:collapse;width:100%}th,td{padding:9px 15px;border-bottom:1px solid #e1e5ed;text-align:left}a{color:#1255a3}summary{cursor:pointer;font-weight:700}@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}body{padding:18px}h1{font-size:27px}}
    </style><body><div class="tag">준비 검증 완료 · 새 학습 없음</div><h1>RGB·관절·시간 보완을<br>같은 기준에서 비교할 준비</h1><p>기존 C·T3-F를 보존한 별도 연구 입력 묶음 · 2026.10.25</p><div class="stats"><div><b>61</b>공개 원본 영상</div><div><b>24 / 9</b>표현 / 수행자</div><div><b>7,647</b>연결한 프레임</div><div><b>394</b>시험 정의 · 실행 성능 아님</div></div>'''
    page+=''.join(sections[:3])+gallery+''.join(sections[3:])+f'<details class="box"><summary>표현별 영상·사람 수 보기</summary><table><tr><th>표현</th><th>영상</th><th>사람</th></tr>{table}</table></details><div class="box"><a href="EVALUATION_KO.txt">비교 기준 상세</a> · <a href="VERIFY_RESULT.json">독립 입력 검산</a> · <a href="LOADER_INDEPENDENT.json">입력 어댑터 검산</a> · <a href="SCENARIO_INDEPENDENT.json">시험 정의 검산</a></div></body></html>'
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    next_action='공개 입력·평가 묶음 준비 결과를 사용자와 확인한다. 확인 후 보이는 손의 독립2D 확인 표본/자연 가림·교차 근거와 B0기존·B1이미지재확인·B2관절시간·B3RGB관절시간 비교 조건을 고정한다. 학습이 필요한 후보의 사람/원본 그룹·학습/선택/평가 역할·예산은 실행 전 확정. 기존 C·T3-F/운영DTW·웹 보존; 사용자·멘토 촬영 학습/튜닝/선택 금지. 준비 완료를 새 성능 향상이나 새 사람 검증으로 보고하지 않음. 현재 요청상 사용자 확인 전 추가 학습 없음.'
    before=U/'before';before.mkdir(exist_ok=False)
    context=R/'WordsInSilence_PROJECT_CONTEXT.md';handoff=R/'NEW_CHAT_HANDOFF.md';state=U.parent/'STATE.json'
    for p in [context,handoff,state]:shutil.copy2(p,before/p.name)
    body=context.read_text(encoding='utf-8-sig')
    entry8='''### 2026-10-25 공개 RGB·관절·시간 입력/평가 묶음 준비 — 검산 통과, 학습 전 확인
- 기존 공개36+27−중복2=61영상/24표현/9명/7647프레임. 원본RGB280565724bytes 제자리 연결, 새NPZ15530473bytes. 전원 기존 TRAIN01~09 자료로 새 사람 최종평가 아님.
- 원본 전체화면 검출/재확인 통합 후 동일ROI. Gloss100개 출처ID 대응/다른사람 참고268연결(15쿼리 묶음내 같은표현 다른사람 없음). 정답단어/참고는 관측보완기 입력 제외.
- 원형61+1/3/7프레임 손좌표 합성누락333=394시험 정의. 독립 해부학 정답 아님. 자연누락583구간/305프레임은 검토용, 오류 정답으로 확정하지 않음.
- 전수 소스 대조 및 독립검산8648757값/483해시, 기존32보호파일 불변. 평균FPS와 실제PTS 차이3영상은 검사식 수정으로 해결(시간불변). 모델입력의 미사용좌표 노출과 미제공 신뢰도NaN은 마스크로 분리; 정상좌표 보존. 입력/합성어댑터 검산 통과.
- B0/B1기존자료 재사용, B2관절시간/B3RGB관절시간 비교 계약 준비. 실제 새 후보추론/학습/다운로드0, 사용자·멘토 촬영 접근0. C·T3-F/운영웹·DTW 유지. 근거: `research/front_keyphase_v1/ops/word100_temporal_public_bundle_20261025/RESULT_KO.html`, `VERIFY_RESULT.json`, `LOADER_INDEPENDENT.json`, `SCENARIO_INDEPENDENT.json`.

'''
    anchor8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
    anchor9='## 9. 다음 단일 작업\n'
    assert anchor8 in body and anchor9 in body
    body=body.replace(anchor8,anchor8+'\n'+entry8,1).replace(anchor9,anchor9+'\n### 2026-10-25 공개 입력·평가 준비 결과 공동 검토\n\n'+next_action+'\n\n',1)
    context.write_text(body,encoding='utf-8')
    d=read(state);d.update(current_stage='TEMPORAL_PUBLIC_INPUT_EVALUATION_BUNDLE_VERIFIED_AWAITING_REVIEW_NO_TRAINING',updated_date_kst=now,updated_at=now,next_single_action=next_action)
    key=U.name; assert key not in d
    d[key]={'status':'PREPARATION_VERIFIED_AWAITING_USER_REVIEW_NO_TRAINING','at':now,'report':str(U/'RESULT_KO.html'),'counts':prep['counts'],'words':24,'signers':list(range(1,10)),'scenario_cases':394,'synthetic_dropout_cases':333,'unsupported_synthetic_hands':11,'reference_links':268,'queries_without_reference':15,'independent_verification':str(U/'VERIFY_RESULT.json'),'loader_verification':str(U/'LOADER_INDEPENDENT.json'),'scenario_verification':str(U/'SCENARIO_INDEPENDENT.json'),'protected_unchanged':32,'new_inference':0,'new_training':0,'user_or_mentor_recording_access':0,'runtime_change':0,'active_execution':None,'next':next_action}
    state.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    handoff.write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}

현재 목표: 공개 RGB·관절·앞뒤 시간으로 순간 검출누락을 보완하되, 실제 행동을 정답 자세로 덮어쓰지 않는다.

마지막 완료: 공개 입력·평가 묶음61영상/24표현/9명/7647프레임 준비와 독립 검산. 입력 연결·시험 정의 완료이며 새 후보 성능 비교나 학습 완료가 아니다.

다음 한 작업: {next_action}

근거: {U/'RESULT_KO.html'}
실제 입력: {U/'MANIFEST.json'}
비교 계약: {U/'EVALUATION_PROTOCOL.json'}
검증: VERIFY_RESULT.json / LOADER_INDEPENDENT.json / SCENARIO_INDEPENDENT.json (위 작업 폴더).

준비 내용
- 기존 문제/정상 대조36개 + 다인27개 − 중복2개. 수어/후원/멀다는9명씩,다른표현1~5명. 원본RGB약280.6MB 제자리 읽기, 새NPZ약15.5MB. 새 다운로드/검출추론 없음.
- 실제 전체 원본 관측과 재확인 결과를 연결. 과거 cropped sensor 광류는 같은조건 baseline이 아님. 원본영상framePTS와 관절timestamp를 전수 확인.
- Gloss100개 출처 대응/같은표현 다른사람 참고268연결.15쿼리는 이묶음 내 같은표현 다른사람 없음. 목표Gloss/라벨/정답구간/참고동작을 보완기에 입력하지 않는다.
- intact61+합성손누락333=394시험.111손 지원/11손은 연속11프레임 부족. 기존MP좌표 재현 proxy이며 실제가림/3D정답 아님.583누락후보/305확인프레임은 미주석이며 자동오류판정 금지.
- B0기존/B1이미지재확인 입력 있음. B2관절시간/B3RGB관절시간은 비교계약만. 손크기 고정·시간·ROI 통일. 모델용 입력은 load_inputs.py/검증된 scenario adapter만 사용; 감사용NPZ 전체를 모델에 넘기지 않는다.

검증 및 수정
- 다른 에이전트가8648757저장값/483소스해시/459과거핀/32보호핀 검산. RGB별도3영상306프레임 확인; 주담당 전수61영상7647프레임 디코딩.
- 평균FPS 계산과 실제PTS 차이가3영상에 있었으나 원본시간은 정확; 검사식 수정으로 해결. FIRST_ERROR/first_attempt 보존.
- 미사용좌표 입력노출 및 신뢰도미제공NaN 문제 수정. 좌표는유지,conf_available로구분. 입력122조건/누락244조건 독립검산. scenario adapter초기결과는before_scenario_adapter_fix보존.
- 작성/실행: 주담당Codex, 자료독립검산:temporal_bundle_inventory, 계약/시험:temporal_bundle_split, 부모통합검산. ML성능 향상이나 수어 전문가 정확성검증으로 과장하지 않음.

보존/미확인
- 모든61은 이미 C·T3-F 학습/개발에서 본 자료. ID17/18 200개도10/19이미평가.14~16예약299+1식별미해결은 열지 않음;후속최종평가전 노출감사필요.
- 독립 손가락정답/실제가림3D정답은 없음. 제한된 보이는2D 표본검토와 전단어 의미수작업은 다른 일;후자는선행조건아님.
- 기존 C·T3-F/운영DTW·웹/원본·과거동결결과 유지. 사용자·멘토촬영 접근/학습/튜닝/선택0. 이번학습/새추론/다운로드0. 이번 준비 실행 종료,활성프로세스/잠금 없음.
''',encoding='utf-8')
    assert read(state)[key]['scenario_cases']==394
    assert next_action in context.read_text(encoding='utf-8') and next_action in handoff.read_text(encoding='utf-8')
    assert all(sha(p)==h for p,h in protected.items())
    artifacts=['prepare.py','load_inputs.py','prepare_scenarios.py','MANIFEST.json','GLOSS_REFERENCE_INDEX.json','EVALUATION_PROTOCOL.json','SCENARIOS.json','PREPARE_RESULT.json','VERIFY_RESULT.json','LOADER_CHECK.json','LOADER_INDEPENDENT.json','SCENARIO_CHECK.json','SCENARIO_INDEPENDENT.json','ISSUES_RESOLVED.json','RESULT_KO.txt','RESULT_KO.html']
    write(U/'FINAL_AUDIT.json',{'status':'PASS_PREPARATION_COMPLETE_NOT_MODEL_PERFORMANCE','at':now,'artifact_hashes':{f:sha(U/f) for f in artifacts},'root_sync':{str(p):sha(p) for p in [context,handoff,state]},'protected_unchanged':len(protected),'new_training':0,'new_inference':0,'active_preparation_execution':None})
    print(json.dumps({'status':'REPORT_AND_HANDOFF_COMPLETE','report':str(U/'RESULT_KO.html'),'training':0},ensure_ascii=False))

if __name__=='__main__': main()
