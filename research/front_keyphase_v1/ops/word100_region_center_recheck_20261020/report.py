import json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
from PIL import Image
U=Path(__file__).resolve().parent;V=U/'revised';P=U.parent/'word100_image_tracking_pair_20261020'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    frozen=read(V/'FROZEN.json');ver=read(V/'VERIFY.json');targets=read(U/'REVIEW_TARGETS.json')
    for p,h in {**frozen['pins'],**frozen['protected']}.items():assert sha(p)==h,p
    notes={
        0:'수영: 손내림 중 일부 손 영역에 보라색 추적점이 남음. 중심은 불확실하여 보류. 손 전체 윤곽/손가락 복원 아님.',
        1:'수어: 새 중심 추정은 이전 추정과 가까운 같은 손 영역 안. 정확한 해부학적 정답은 확인 불가.',
        2:'대지01 본동작: 주먹은 영상에 보이나 새 후보는 중심 보류. 이전 보조가 제공했던 위치 지원을 잃은 반례.',
        3:'대지06 손내림: 부분 가시 손은 남아 있지만 새 추적 위치는 보류.',
        4:'대지08 손내림: 주먹이 보이는데 새 위치는 보류.',
        5:'어린이날07 본동작: 보이는 손의 중심 누락을 이어주지 못함. 이전 위치 지원을 잃은 반례.',
        6:'어린이날07 손내림: 경계의 손 영역은 보이지만 새 추정 없음.',
        7:'가래떡08: 손을 돌려 내리는 장면. 보이는 손을 새 후보가 이어주지 못함.',
        8:'어린이날01: 손내림 중 손은 보이나 새 중심은 보류.',
        9:'상처07 왼손: 이전 추정은 손 영역 안이지만 임시 관절 중심과 벌어짐. 새 후보는 좌표 수정이 아니라 보류.',
        10:'상처07 왼손 동일 구간: 새 중심 보류, 복원 성공 아님.',
        11:'상처07 왼손 동일 구간: 새 중심 보류, 복원 성공 아님.',
        12:'상처07 왼손 동일 구간: 새 중심 보류, 복원 성공 아님.',
        13:'진도군수어통역센터08: 두 추정 모두 같은 손 영역에 있음. 새 추정이 MP 임시 중심에서 더 멀어진 수치 반례. 수작업 관절 정답은 없음.',
        14:'상처07 오른손: 새 추정은 같은 손 위. MP 임시 중심 대비 이전보다 약간 나빠진 프레임.',
        15:'상처07 오른손: 동일 회귀 구간의 앞 프레임. 다른 손으로 옮겨간 것은 보이지 않음.',
        16:'대지01: 새 추정은 주먹 영역 안이며 임시 관절 중심과 이전보다 가까움. 정밀 좌표 정답 검증 아님.',
        17:'대지01 오른손: 임시 중심 기준 개선, 두 점 모두 손 영역 안.',
        18:'대지01 왼손: 임시 중심 기준 개선, 두 점 모두 손 영역 안.',
        19:'수어04: 관측이 재개되면 분홍 현재 관측을 사용. 초록 예측과 벌어져도 현재값을 예측으로 끌어가지 않음.',
        20:'어린이날07: 손 방향이 바뀌는 시점에서 분홍 현재 관측 우선. 다음 누락 프레임은 보류되어 아직 공백 문제는 남음.',
        21:'성남시수어통역센터05: 손 방향 변화 중 현재 관측 우선. 예측점은 진단용으로만 남음.'}
    assert set(notes)=={t['id'] for t in targets}
    visual=dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        reviewer='Same Codex, visual inspection of all8 sheets; not independent human or sign expert',
        targets=[dict(t,review=notes[t['id']],precise_joint_truth=False) for t in targets],
        covered='All2 natural new support states, all9 previous natural supports, old high-error4 frames, paired3largest increases/3decreases,3current-observation recheck events.',
        conclusion='Natural new center1 is on visible hand; region-only1 remains on visible partial hand. Seven prior supports lost. No new wrong-hand drift seen in selected sheets; no exhaustive manual truth for all4641frames.')
    write(U/'VISUAL_REVIEW.json',visual)
    # Retain an explicit account of regression instead of selecting only supported outputs.
    failures=[]
    for ep in read(P/'EPISODES.json'):
        x=read(V/'records'/f"{ep['clip_id']}.json");e=x['frames'][ep['start']];s=ep['side'];pr=x['frames'][ep['start']-1]['natural']
        failures.append(dict(ep,new_state=e['natural']['source'][s],
            prior_seed_reason=pr['seed_reason'][s],
            retained_points=len(pr.get('retained_region_points',{}).get(str(s),[])),
            fit_reason=e['natural']['traces'].get(str(s),{}).get('center_fit',{}).get('reason')))
    write(U/'OLD9_RECHECK.json',failures)
    audit=dict(status='COMPLETE_STRUCTURAL_SEPARATION_VERIFIED_RECOVERY_NOT_IMPROVED_NOT_ADOPTED',
        final_candidate='revised/candidate.py',first_candidate_preserved=True,
        numeric_parameters_unchanged=read(U/'FROZEN.json')['params']==frozen['params'],
        result=ver,known_regression='Old natural support9->new center1+region_only1, no support7; annotated-core2->0.',
        continuity_repair='358current-hand states retain forward/backward-valid pixels inside current palm. Synthetic centers33->38; natural support unchanged.',
        primary_conclusion='Do not conflate missing-coordinate suppression with accurate coordinate recovery. On same38 synthetic supported targets average proxy error2.10525%->2.11137% shoulder, essentially unchanged/slightly worse.',
        no_user_data=True,no_training=True,no_model_or_web_change=True,
        next='Review result. Keep separation/current-priority contract; do not adopt sparse LK center filling. Prefer a separately frozen current-image re-detection comparison on known visible-missed hands, with public normal/cross controls and same-pixel/identity constraints. No extra training/download now.',
        limitations='Known TRAIN diagnostic cases; result-informed continuity repair is disclosed and not an independent test. MediaPipe proxy not exact hand geometry; geometry gates not sign correctness. Browser preview not attempted due known file URL policy, images inspected locally.')
    write(U/'FINAL_AUDIT.json',audit)
    texts=f"""손 영역·손바닥 중심 분리와 현재 관측 재확인 결과
2026-10-20 / Codex 로컬 구현·별도 저장 검산·시각 검토

결론
정보를 분리하고 현재 관측을 우선하는 구조는 구현·검증했다. 그러나 손 위치 복원 성능이 좋아졌다는 근거는 없다. 이번 후보는 학습 입력/운영에 채택하지 않는다.
이전 정상 관측과 좌우는 보존했지만, 자연 누락에 새로 보태는 중심은9개에서1개로 줄었다. 영역만1개, 나머지7개보류다. 본동작주석 안의 지원2개는0이 되었다.

변경 내용
1. 현재 관절에서 계산한 중심(observed), 영상 변환으로 추정한 중심(estimated), 영상 특징점 영역만 따라간 상태(region_only), 근거없음(missing)을 분리.
2. 이전 단순 이동에 더해2D평면 회전/균일크기변환을 반영하는 similarity transform. 이는 실제3D회전이나 관절 재검출이 아님.
3. 현재 관측이 돌아오면 그 좌표를 사용하고 팔·손 연결로 추적 근거를 재확인. 이전 예측값으로 현재값을 끌어가지 않음.
4. 관측중심 갱신과 영상점 폐기를 분리. 현재 손바닥 안에 남아 있는 유효 추적점만 유지. 좌우 이름은 기존C를 보존하며 강제정정하지 않음.

공개 범위와 조건
이전과 같은 TRAIN36영상·23단어·9촬영자·4641프레임. 저장C관절 및 동일sensor픽셀, 영상 전체의 현재/과거 프레임 사용.
기존LK 결과재사용. 새MediaPipe실행/분류재평가/모델학습/DEV·보류·사용자·멘토자료접근없음.
합성검출누락은 기존과같은70구간×6=420손프레임. 손검출입력만감추고RGB/팔정보유지. 숨긴손중심은후보에주지않고검산에만사용.
공통처리에는 단어명이나ID분기없음. 전체수어/새사람일반화 검증이라는 뜻은 아님.

보존·재확인
현재관측중심4086건 그대로사용. 알려진정상좌우42/42보존,교차표본54프레임배정변화0,기존오류12개유지.
전체좌우변경0이므로 오류정정성공으로해석하지않는다. 교차54도전부수기정답검증집합은아님.
손바닥중심 관측재개195시점에서현재값우선. 187은현재팔근거확인,8은정체성확정없이기존관측보존하고추적이름재시작보류.
195는부분관측이완전해진경우도포함,195영상이나실제손전체재등장을뜻하지않는다.
2D저장손방향변화15도이상367시점에서도현재관측우선. 3D물리각도나수어오류횟수아님.
손가락관측을새로만든수0. 기존관절검출오류자체를고쳤다는뜻도아님.

복원 비교(최종후보)
자연누락:이전중심추정9 -> 새중심1,영역만1,나머지7보류. 유지된1개도약0.033초한프레임.
검출숨김420:이전중심추정61 -> 새38,보류382. 그중같이추정한38개로공정비교:
평균오차:어깨너비2.10525% -> 2.11137%(사실상동일/미세악화).
중앙오차:1.46156% -> 1.41168%;95백분위5.09617% -> 4.75522%;최대7.80629% -> 7.19185%.
38중23개프레임오차감소,15개증가. 서로상관된프레임이며유의한일반화개선으로주장하지않음.
이전전체61의최대19.81%오차가새전체38에서7.19%가됐지만분모가다르다. 큰오차4개를보류한것이지정답좌표로고친게아니다.
상처07왼손4개큰오차는한구간. 0.10어깨너비선은분석요약기준이며수어허용오차/사용자잘못기준아님.

새 문제와 수정 이력
첫후보에서관측갱신때기존유효영상점까지폐기하여지원이줄어드는결합문제발견.
첫결과/FROZEN/records/VERIFY를보존하고revised에수정후보를따로고정. 숫자기준·표본은변경하지않음.
현재팔과손이확인되고유효추적점이현재손바닥영역안에있는경우만계속사용. 358시점에서기존점을유지했고숨김시험추정33->38.
그래도이전9중7자연지원프레임의직전관측에서재사용가능점은0~5개로최소6개를못채워추적이중단됨.
수영은점들이같은변환을충분히지지하지못해영역만표시. 수어1개중심은영상상손안에있지만정밀정답확정은불가.
손회전/가림에서현재관절을정확히찾는문제와영상점이따라가는문제가다르다는한계가남음.
revised준비도중중첩폴더STATE백업경로오류는실행전발생. 최초오류보존·경로만수정·기존백업해시확인후동결/실행. 조건변경없음.

검증·시각화
합성이동·회전·크기변화/특징점일렬/현재관측우선/재등장/시간역전/영역점유지 등자체계약11개PASS.
별도검산:저장flow2525건,중심변환2161건의산술·인과시간·관측보존·숨김자료미주입·유지점의현재손바닥포함확인.
22시점의이전/현재/다음원영상시각검토. 새자연중심/영역2개와이전9지원,큰오차4,개선/악화각3,재관측3포함(일부중복).
구현과검산/시각검토모두같은Codex. 별도코드검산이며다른사람/에이전트검증아님. MediaPipe중심은임시비교값,수작업정밀정답아님.
36영상/36이동그래프/8시각시트. 추가추적연산95백분위7.33ms,영상디코딩/MediaPipe제외이며실시간보장아님.
기존C/T3-F와32보호파일해시불변. 운영웹/DTW/학습변경없음. 첫시도·수정후시도모두재실행없이보존.

다음 계획
영역/관측/추정분리와현재관측우선은연구부품으로유지하되,추적중심을바로학습·DTW에넣지않는다.
추적문턱을낮추거나단어별예외를추가하기보다,화면에손이있는데관절만놓친공개장면에서현재이미지의손재검출을따로비교하는것을제안한다.
범위/해상도/가중치유무를확인해동결하고같은공개정상손/교차대조로실제누락감소와오배정을검증. 완전가림/화면밖은관측불가로유지.
첫잘못된좌우배정은별도현재팔·손근거문제이며손추적만으로해결했다고하지않는다.
새도구다운로드/추가학습/운영교체는이번에실행하지않았으며사용자와결과검토후다음단위를확정한다.

공식 방법 참고
https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html (estimateAffinePartial2D)
"""
    (U/'RESULT_KO.txt').write_text(texts,encoding='utf8')
    # Scientific evidence extracts, no new image content.
    for name,sheet,box in [('states',1,(0,0,1500,640)),('lost',1,(0,640,1500,960)),
                           ('reappear',7,(0,320,1500,640))]:
        with Image.open(U/'evidence'/f'review_{sheet}.png') as im:im.crop(box).save(U/'evidence'/f'{name}.png')
    data=[]
    for row in frozen['rows']:
        x=read(V/'records'/f"{row['clip_id']}.json")
        data.append(dict(id=row['clip_id'],word=row['label'],person=row['signer_id'],
            fps=1/(x['frames'][1]['time']-x['frames'][0]['time']),first=x['frames'][0]['time'],frames=len(x['frames']),
            events=[dict(time=t['time'],state=t['state']) for t in read(V/'NATURAL.json') if t['clip_id']==row['clip_id']]))
    (U/'RESULT_KO.html').write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False)),encoding='utf8')
    print('Report and visual review saved; candidate not adopted')

HTML=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>손 영역과 손바닥 중심은 다르다 · 공개 비교 결과</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:#193348;font:16px/1.75 "Malgun Gothic",sans-serif}main{max-width:1200px;margin:auto;padding:40px 24px 70px}h1{font-size:clamp(28px,4vw,44px);line-height:1.4;margin:15px 0}h2{font-size:25px;margin:0 0 20px}h3{font-size:19px;margin:15px 0 6px}section{background:white;border:1px solid #dce5ec;border-radius:16px;padding:28px;margin:26px 0}.kicker{color:#287565;font-weight:700}.lead{font-size:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{background:white;border:1px solid #dce5ec;border-radius:12px;padding:18px}.stat b{display:block;font-size:27px}.warn{background:#fff2de;padding:16px;border-left:4px solid #d29735}.good{background:#ebf6f0;padding:16px;border-left:4px solid #438b6d}a{color:#17678d}.small{font-size:14px;color:#536d7d}table{width:100%;border-collapse:collapse}th,td{padding:13px 10px;border-bottom:1px solid #dce5ec;text-align:left}th{background:#eef4f7}img,video{width:100%;height:auto;border-radius:8px}video{background:#163147}button,select{font:inherit;padding:8px 13px;border:1px solid #b9cad5;border-radius:7px;background:white;color:#193348}button{cursor:pointer}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.legend{display:flex;gap:15px;flex-wrap:wrap}.old{color:#b57408}.new{color:#267c39}.region{color:#9848a4}details{margin:16px 0}summary{font-weight:bold;cursor:pointer}.states{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.state{background:#f1f5f8;padding:17px;border-radius:10px}.state b{display:block}figure{margin:20px 0}figcaption{font-size:14px;color:#536d7d}
@media(max-width:800px){main{padding:24px 12px}.grid{grid-template-columns:1fr}.stats,.states{grid-template-columns:repeat(2,1fr)}section{padding:20px 16px}table{font-size:14px}td,th{padding:9px 6px}select{width:100%}}
</style><main>
<div class="kicker">WORDS IN SILENCE · 2026.10.20 · 공개 관측 비교</div>
<h1>손 영역을 따라가는 것과<br>손바닥 중심을 아는 것은 다릅니다.</h1>
<p class="lead">관측과 추정을 분리하고 재확인하는 구조는 검증했습니다.<br><b>하지만 누락 위치를 더 잘 복원한 결과는 아니므로, 학습에는 넣지 않습니다.</b></p>
<div class="stats"><div class="stat"><b>36개 영상</b>23단어 · 촬영자9명</div><div class="stat"><b>4,086건</b>현재 관측 중심 보존</div><div class="stat"><b>9 → 1</b>자연 누락의 중심 추정 지원</div><div class="stat"><b>학습 0회</b>C·T3-F 그대로 유지</div></div>
<section><h2>무엇을 구분했나요?</h2><div class="states">
<div class="state"><b>① 현재 관측</b>지금 검출된 관절로 중심을 계산. 이전 추정보다 우선합니다.</div>
<div class="state"><b>② 중심 추정</b>영상점의 이동·평면 회전·크기 변화를 이용한 추정. 실제 관측과 구별합니다.</div>
<div class="state"><b>③ 영역만 추적</b>손 안의 일부 영상점은 따라가지만, 정확한 중심은 보류합니다.</div>
<div class="state"><b>④ 근거 부족</b>위치·손 모양을 임의로 만들어 채우지 않습니다.</div></div>
<p>손이 다시 보이면 현재 좌표로 중심을 갱신합니다. 현재 팔·손 연결과 맞는지 확인하고, 아직 유효한 영상 추적점만 유지합니다. 손가락 관측을 새로 만들거나 좌우를 강제로 바꾸지 않았습니다.</p>
<p class="small">‘현재 관측’도 MediaPipe의 추정 결과이며 정밀한 실제 관절 정답을 뜻하지는 않습니다. 사용한 <a href="https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html">OpenCV 2D 변환</a>은 평면 회전·균일 크기 변화까지 다루며, 실제 손의 3D 회전과 굽힘을 완전히 복원하지는 않습니다.</p></section>
<section><h2>개선된 구조와 보존한 부분</h2><table><tr><th>항목</th><th>결과</th><th>뜻</th></tr>
<tr><td>현재 손바닥 중심</td><td><b>4,086건 보존</b></td><td>이전 예측 쪽으로 현재 관측을 끌어가지 않음</td></tr>
<tr><td>손바닥 관측 재개</td><td><b>195시점 현재값 우선</b></td><td>187시점 팔 근거 확인, 8시점은 추적 이름 재시작 보류</td></tr>
<tr><td>정상 좌우 배정</td><td><b>42 / 42 유지</b></td><td>새로 바뀐 배정 없음</td></tr>
<tr><td>손 교차 표본</td><td><b>54프레임 유지</b></td><td>현재 배정을 그대로 보존</td></tr>
<tr><td>알려진 오류 배정</td><td><b>12건 그대로 남음</b></td><td>이번 작업으로 좌우 오류를 고친 것은 아님</td></tr></table>
<p class="small">195는 부분 관측이 완전해진 경우도 포함한 손별 시점 수입니다. 195개 영상이나 실제 손 전체의 재등장 횟수가 아닙니다. 교차54는 기하학적 표본이며 전부 정답으로 수동 판정한 집합은 아닙니다.</p></section>
<section><h2>복원 성능은 왜 채택하지 않나요?</h2><table><tr><th>같은 조건의 비교</th><th>이전 추적</th><th>수정 후보</th></tr>
<tr><td>자연 누락에 중심 추정 추가</td><td>9개 순간</td><td><b>1개 순간</b> + 영역만1개</td></tr>
<tr><td>그중 주석상 본동작 구간</td><td>2개</td><td><b>0개</b></td></tr>
<tr><td>일부러 검출을 감춘420손-프레임</td><td>61개 중심 추정</td><td><b>38개 중심 추정</b></td></tr>
<tr><td>둘 다 추정한 같은38개의 평균 오차</td><td>어깨너비의 2.105%</td><td><b>2.111%</b> · 사실상 같음</td></tr>
<tr><td>같은38개의95백분위 오차</td><td>5.096%</td><td>4.755%</td></tr></table>
<p class="warn">큰 오차가 있던 4개 순간은 <b>정확한 좌표로 고친 것이 아니라 보류</b>했습니다. 나쁜 추정을 숨기면 평균이 좋아 보일 수 있으므로, 같은38개를 따로 비교했습니다. 23개는 가까워지고15개는 멀어졌습니다.</p>
<p>처음 구현에서는 중심 갱신 때 유효한 추적점까지 지우는 문제를 발견했습니다. 첫 결과를 보존한 뒤, 현재 손바닥 안에 남은 점을 유지하도록 고쳤습니다. 유지된 시점은358개, 검출 숨김 시험의 지원은33개에서38개로 회복됐지만, 이전 방식보다 누락을 더 잘 메우지는 못했습니다.</p>
<p class="small">오차는 숨겨 둔 MediaPipe 중심과의 거리이며 수작업 정밀 정답이 아닙니다. 수어 동작 정답률이나 사용자 허용오차도 아닙니다. 이 결과를 보고 문턱을 낮추거나 단어별 예외를 추가하지 않았습니다.</p></section>
<section id="watch"><h2>영상과 손 이동을 직접 비교</h2><label for="clip">공개 영상</label> <select id="clip"></select><p id="info" class="small"></p>
<div class="legend"><b>파랑·분홍: 현재 관측</b><b class="old">× 주황: 이전 추정</b><b class="new">○ 초록: 새 중심 추정</b><b class="region">보라: 추적점 영역</b></div>
<div class="grid"><div><video id="video" controls playsinline preload="metadata"></video><div class="controls"><button id="prev">이전 프레임</button><button id="next">다음 프레임</button><button id="speed">0.5배속</button><span id="time"></span></div><div id="events" class="controls"></div></div>
<div><h3>관측이 돌아오면 무엇을 쓰나요?</h3><p>파랑·분홍의 현재 관측을 사용합니다. 초록 예측이 다른 위치에 있어도 현재 관측을 그쪽으로 끌어가지 않습니다.</p><h3>보라색은 손 전체를 찾은 건가요?</h3><p>아닙니다. 손 안에서 따라간 작은 영상점의 범위입니다. 중심이나 손가락 모양까지 확인했다는 뜻이 아닙니다.</p><p class="small">아래 그래프의 가로축은 시간입니다. 연한 초록 배경은 제공 주석 구간이고, 회색은 비교할 중심이 없는 시간입니다. 빈칸을 임의의 선으로 잇지 않았습니다.</p></div></div>
<a id="graphLink" target="_blank"><img id="graph" alt="양손의 시간별 위치와 관측·추정·보류 상태"></a></section>
<section><h2>확인한 장면</h2><h3>영역만 남기기 / 중심 추정하기</h3><p>위 ‘수영’ 장면은 손내림 중 일부 영역만 추적했습니다. 아래 ‘수어’ 장면은 같은 손 위에 중심 추정이 남았습니다. 둘 다 정확한 손가락 모양을 복원한 것은 아닙니다.</p><img src="evidence/states.png" alt="수영 영역만 추적과 수어 중심 추정 대조">
<h3>남은 문제: 영상에는 손이 보이는데 새 후보는 보류</h3><p>‘대지’ 본동작에서 이전 주황 추정은 손 위에 있지만 새 후보는 비어 있습니다. 직전 관측에서 현재 손바닥 안에 남은 유효점이 최소6개보다 적어 추적이 끊겼습니다. 이런 사례 때문에 이번 후보를 복원 개선으로 채택하지 않았습니다.</p><img src="evidence/lost.png" alt="대지의 기존 위치 지원을 잃은 반례">
<h3>관측 재개: 현재 중심을 우선</h3><p>‘수어’에서 관측이 다시 잡히면 분홍 중심을 사용합니다. 초록 추정과의 차이는 진단에만 남겼습니다.</p><img src="evidence/reappear.png" alt="수어 관측 재개와 현재 중심 우선">
<details><summary>22개 검토 시점 전체 보기</summary><p>원영상·이전·현재·다음 장면을 함께 봅니다. 확대는 설명용이며 실제 검출 범위를 바꾼 것이 아닙니다. test 표시는 검출만 숨긴 부가 시험입니다.</p>
<p><a href="evidence/review_1.png">1: 영역·중심·지원손실</a> · <a href="evidence/review_2.png">2: 지원손실</a> · <a href="evidence/review_3.png">3: 손내림</a> · <a href="evidence/review_4.png">4: 이전 큰 오차</a> · <a href="evidence/review_5.png">5: 보류·회귀</a> · <a href="evidence/review_6.png">6: 회귀·개선</a> · <a href="evidence/review_7.png">7: 개선·재관측</a> · <a href="evidence/review_8.png">8: 방향 변화</a></p></details></section>
<section><h2>다음 단계의 판단</h2><p class="good"><b>유지할 것:</b> 관측·추정·영역만 보이는 상태의 구분, 현재 관측 우선, 불확실한 좌우를 억지로 확정하지 않는 원칙.</p>
<p><b>바로 학습에 넣지 않을 것:</b> 이 영상 추적기가 내는 중심 추정. 실제 본동작의 누락을 줄이지 못했고, 긴 가림과 기존 좌우 오류도 남아 있습니다.</p>
<p><b>다음 제안:</b> 추적 규칙을 계속 늘리기보다, 화면에는 손이 있는데 관절만 놓친 공개 장면에서 <b>현재 이미지의 손 재검출</b>을 별도로 비교합니다. 정상 손과 교차 장면의 오배정도 함께 검사하고, 완전가림·화면밖은 계속 확인 불가로 남깁니다.</p>
<p>추가 학습·새 도구 다운로드·운영 반영은 이번에 하지 않았습니다. 결과를 함께 확인한 뒤 다음 조건을 정합니다.</p></section>
<section><h2>검증 범위와 보존</h2><p>같은 공개 TRAIN36영상의 진단 결과입니다. 모든 수어·새 사용자 일반화 검증은 아닙니다. 사용자·멘토 촬영과 보류 자료는 사용하지 않았습니다.</p><p>자체 계약11개와 별도 코드의 저장추적2,525건·중심변환2,161건 검산을 완료했습니다. 원본 관측·숨어 있는 비교값의 미주입·현재 손바닥 안 추적점 유지 등을 확인했습니다. C·T3-F를 포함한32보호파일 해시가 유지됐습니다.</p><p>구현·실행·시각검토와 별도 계산 검산은 같은 Codex가 수행했습니다. 독립적인 다른 사람·에이전트의 검증은 아닙니다.</p>
<details><summary>첫 후보 보존 및 준비 오류 기록</summary><p>첫 후보의 감소 결과를 보존한 뒤, 숫자 기준을 바꾸지 않고 영상점과 중심의 갱신을 분리한 수정 후보를 따로 고정했습니다. 같은자료를다시확인한개발비교이며새독립시험이아닙니다.</p><p>수정 후보 준비 중 중첩 폴더의 STATE 백업 경로 오류가 발생했습니다. 영상 비교 실행 전에 경로만 고쳤고 최초 오류와 기존 백업을 보존했습니다.</p></details>
<p><a href="RESULT_KO.txt">상세 보고</a> · <a href="revised/VERIFY.json">최종 후보 검산</a> · <a href="VERIFY.json">첫 후보 결과</a> · <a href="VISUAL_REVIEW.json">시각 검토</a> · <a href="FINAL_AUDIT.json">최종 판정</a> · <a href="revised/FROZEN.json">동결 조건</a></p></section>
</main><script>
const DATA=__DATA__,sel=document.getElementById('clip'),video=document.getElementById('video'),graph=document.getElementById('graph'),graphLink=document.getElementById('graphLink');let row,pending=0;
for(const x of DATA){const o=document.createElement('option');o.value=x.id;o.textContent=`${x.word} · 촬영자${String(x.person).padStart(2,'0')}`;sel.append(o)}
function pick(id,t=0){row=DATA.find(x=>x.id===id)||DATA[0];sel.value=row.id;pending=t;video.src=`videos/${row.id}.mp4`;graph.src=`plots/${row.id}.png`;graphLink.href=graph.src;document.getElementById('info').textContent=`${row.frames}프레임 · 기존 C 관측 보존 · 별도 연구 후보`;const ev=document.getElementById('events');ev.replaceChildren();for(const e of row.events){const b=document.createElement('button');b.textContent=`${e.time.toFixed(3)}초 · ${e.state==='estimated'?'중심 추정':'영역만'}`;b.onclick=()=>{video.pause();video.currentTime=e.time-row.first+.006};ev.append(b)}}
video.addEventListener('loadedmetadata',()=>video.currentTime=Math.max(0,pending-row.first));
video.addEventListener('timeupdate',()=>document.getElementById('time').textContent=(video.currentTime+row.first).toFixed(3)+'초');
sel.onchange=()=>pick(sel.value);document.getElementById('prev').onclick=()=>{video.pause();video.currentTime=Math.max(0,video.currentTime-1/row.fps)};document.getElementById('next').onclick=()=>{video.pause();video.currentTime=Math.min(video.duration,video.currentTime+1/row.fps)};document.getElementById('speed').onclick=()=>{video.playbackRate=video.playbackRate===1?.5:1;document.getElementById('speed').textContent=video.playbackRate===1?'0.5배속':'1배속'};
pick('NIA_SL_WORD0003_REAL04_F',2.239);
</script></html>'''
if __name__=='__main__':main()
