"""Saved-result reporting only. Does not import or rerun the tracker."""
import json, hashlib, html
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image

U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    c=read(U/'FROZEN.json');v=read(U/'VERIFY.json');ep=read(U/'EPISODES.json')
    counts=Counter();reasons=Counter();examples=[];gaps=Counter()
    for r in c['rows']:
        x=read(U/'records'/f"{r['clip_id']}.json")
        with np.load(r['views']['sensor']['raw'],allow_pickle=False) as z:
            assert r['box'][:2]==[0,0]
            assert (z['width']==r['box'][2]).all() and (z['height']==r['box'][3]).all()
        for e in x['frames']:
            reasons.update(e['natural']['reasons'])
            for s in (0,1):
                if e['direct'][s] is None:
                    counts['missing_core' if e['core'] else 'missing_outside_annotation']+=1
                    if e['estimate'][s] is not None:
                        counts['supported_core' if e['core'] else 'supported_outside_annotation']+=1
                counts['seed_count']+=int(e['natural']['seed'][s])
        # Missing complete palm center is not equivalent to a wrongly executed sign.
        for s in (0,1):
            n=0
            for e in x['frames']+[{'direct':[[0,0],[0,0]],'core':False}]:
                if e['core'] and e['direct'][s] is None:n+=1
                elif n:gaps['single_frame' if n==1 else 'multiple_frames']+=1;n=0
        episodes=[e for e in ep if e['clip_id']==r['clip_id']]
        examples.append(dict(id=r['clip_id'],word=r['label'],person=r['signer_id'],
            frames=len(x['frames']),duration=x['frames'][-1]['time'],first_time=x['frames'][0]['time'],
            intervals=r['intervals'],reasons=r['selection_reasons'],episodes=episodes,
            width=r['box'][2],height=r['box'][3],fps=x['fps']))
    notes=[
        '수영: 하단 경계에 걸쳐 손을 내리는 장면. 추정점은 보이는 손 영역에 있음. 손 전체·손가락 복원 아님.',
        '대지01: 주먹이 영상에 보이지만 C 손바닥 위치가 없는 1프레임. 추정점은 주먹 위에 있음.',
        '대지06: 손을 내리며 일부가 하단 경계 밖으로 이동. 추정점은 남아 보이는 손 영역에 있음.',
        '대지08: 손을 내리는 장면. 추정점은 주먹 영역에 있음.',
        '어린이날07 2.667초: 움직이는 손이 화면 안에 보임. 추정점은 손 영역에 있음.',
        '어린이날07 2.867초: 하단으로 내리는 장면. 추정점은 손 영역에 있음.',
        '가래떡08: 손을 돌리며 내리는 장면. 추정점은 손 영역에 있으나 정확한 해부학적 중심은 확인 불가.',
        '어린이날01: 손을 내리는 장면. 추정점은 손 영역에 있음.',
        '수어04: 다른 손 쪽으로 이동 중인 손에 추정점이 있음. 손가락 관측이 회복된 것은 아님.']
    visual=dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        reviewer='Codex visual review of original sensor images and before/current/after magnifications; not independent human or sign-language expert',
        natural=[dict(e,review=notes[i],verdict='inside_visible_hand_region_position_support_only',
            anatomy_center_ground_truth=False,hand_shape_restored=False) for i,e in enumerate(ep)],
        synthetic_issue=dict(clip_id='NIA_SL_WORD1519_REAL07_F',frames=[44,45,46,47],
            review='All four predicted points remain on the visible hand, but differ from the hidden MediaPipe palm-center proxy. Hand rotation/shape change is consistent with texture tracking versus anatomical-center drift; exact cause is an inference, not isolated proof.',
            distinct_episodes=1,max_proxy_error_shoulder=.1980718327424657,
            wrong_hand_confirmed=False,ground_truth='saved MediaPipe proxy, not manual joint truth'),
        sheets=[f'evidence/review_{i}.png' for i in range(1,6)],
        limitations=['All new natural missing-position estimates reviewed; not all 4641 frames manually adjudicated.',
            'Known normal/identity labels reused, 54 crossing flags are geometric and are not all manually labeled correct.',
            '9 observed-region supports do not establish quantitative joint accuracy or long-occlusion recovery.'])
    write(U/'VISUAL_REVIEW.json',visual)
    for path,h in {**c['pins'],**c['protected']}.items():assert sha(path)==h,path
    audit=dict(status='COMPLETED_LIMITED_POSITION_SUPPORT_NOT_ADOPTED',counts=dict(counts),
        natural_reasons=dict(reasons),core_missing_runs=dict(gaps),
        unique_videos=len({e['clip_id'] for e in ep}),unique_words=len({e['word'] for e in ep}),
        all_new_natural_estimates_reviewed=9,all_natural_episodes_single_frame=True,
        static_and_dimensions='run.py crops image[:sensor_height,:sensor_width] before resize/grayscale. All36 metadata boxes origin0 and match raw widths/heights. Full original future/outside pixels not used by tracker.',
        protected_files_unchanged=len(c['protected']),frozen_pins_unchanged=len(c['pins']),
        observed_finger_landmarks_added=0,training=0,classifier_inference=0,new_detector_runs=0,user_reads=0,
        production_changes=0,independent_agent_review=False,
        verification='Separate saved arithmetic script without tracker import plus same-Codex visual review. Not an independent reimplementation of LK or human joint truth.')
    write(U/'FINAL_AUDIT.json',audit);write(U/'VIEW_DATA.json',examples)
    # Display extracts, not image editing or generated evidence: same contact-sheet pixels.
    for name,src,box in [
        ('visible_missed','review_1.png',(0,320,1600,640)),
        ('texture_drift','review_4.png',(0,0,1600,320)),
        ('rest_boundary','review_1.png',(0,0,1600,320))]:
        with Image.open(U/'evidence'/src) as im:im.crop(box).save(U/'evidence'/f'{name}.png')
    report="""공개 영상의 손 이동·영상 추적 보조 비교
2026-10-20 / 별도 연구 후보 / 기존 C·T3-F·DTW·웹 유지

결론
영상의 움직임을 보조로 쓰는 방향에는 제한된 근거가 생겼으나, 이번 LK 후보는 아직 학습 입력이나 DTW 점수에 채택하지 않는다.
자연 영상에서 손바닥 위치가 없던 9개 순간에 위치 추정이 더해졌다. 모두 1프레임(약0.033초)이고, 8영상·5단어에 분포한다.
원영상 대조에서 9개 추정점은 보이는 손 영역 안에 있었다. 정밀 관절 정답이나 손가락 복원이 아니다.
주석상 본동작 구간 2개, 주석 밖7개다. 후자는 주로 손내림이다. 주석 경계는 필수동작의 완벽한 의미 정답이 아니다.

무엇을 비교했나
공개 TRAIN 36영상,23단어,9촬영자,4641프레임. 이전 문제·교차·정상 대조에서 결과를 보기 전에 선정했다.
영상 처음부터 끝까지 같은 sensor 범위와 저장 C 검출·배정을 사용했다. 추적은 가시 범위를 먼저 자른 동일 픽셀만 사용하며 중간에 하단을 풀지 않았다.
OpenCV LK가 손의 작은 무늬/명암 특징점 이동을 추적한다. 실제 팔·손 연결로 시작점을 잡고 현재/과거 프레임만 사용한다.
최대0.20초, 전후방 추적 일치와 공간 일관성 등을 조건으로 위치 추정만 내며, 보이지 않는 손가락 좌표는 만들지 않는다.
좌우 정정 제안은 영상 궤적과 현재 팔 연결이 모두 맞을 때만 허용했다. 이번 실제 결과에서 제안 변경0.
공개 영상의 연구 진단용 표본이며 새 사용자/전체100단어/전체학습899영상/최종시험 성능으로 확대하지 않는다.

개선·보존·미해결
알려진 정상 배정42/42 유지, 새 오배정0. 기존 교차표본54프레임의 배정 변화0.
교차54는 기하학적 교차표본이며 전부 정답이라고 수동 판정한 집합은 아니다.
알려진 오류배정12/12 그대로 남았다. 그중 이전9사례도 포함한다. 처음 잘못 붙은 좌우 이름을 해결했다고 말할 수 없다.
원래의 손·팔 관측값은 보존됐고 손가락 관측을 늘린 수0. 위치만 이어진 경우는 별도 추정으로 표시한다.
5196개 손-프레임에 완전한 손바닥 위치가 없었지만 준비·손내림·한손수행·부분가시성을 포함하므로 5196개 오류라는 뜻이 아니다.
긴 가림·완전가림·화면밖 손은 여전히 해결되지 않았다. 추적할 시작 근거와 유지 가능한 특징점이 부족하면 보류한다.

일부러 관절 검출만 감춘 보조 시험
70구간에서6프레임씩, 총420손-프레임의 손 검출 후보를 숨겼다. RGB와 팔 관측은 남겼고 영상은 자르거나 반복 연결하지 않았다.
위치 추정61/420(14.52%), 보류359/420. 이 비율은 정답률이 아니라 추정값 제공률이다.
숨긴 MediaPipe 손바닥 중심을 임시 비교값으로 삼았을 때 오차 중앙값은 어깨너비1.79%,95백분위12.99%,최대19.81%.
어깨너비10%보다 큰4프레임은 상처07 한 구간에 몰렸다. 10%는 분석 요약선이지 수어 허용오차나 채택 기준이 아니다.
시각적으로 다른 손으로 옮겨가진 않았지만 손이 회전하면서 손가락 마디 쪽 특징점과 손바닥 중심이 어긋났다.
이것은 손 영역을 따라가는 것과 정확한 관절 위치를 복원하는 것이 다르다는 반례다. MediaPipe 자체 오차 가능성도 있어 정밀 정답률로 부르지 않는다.
프레임당 추적 추가연산95백분위7.39ms. 디코딩/리사이즈/MediaPipe 제외, 추적 미수행 프레임 포함이므로 웹 실시간 성능을 보장하지 않는다.

피드백에 적용할 원칙
손이 안 보임: 동작 오답이 아니라 확인 불가. '이 구간은 손이 겹쳐 손가락 모양을 확인하기 어려워요'와 같은 관측 안내.
손 위치만 추정: '손 이동은 일부 이어 볼 수 있지만 손가락 모양은 확인하지 못했어요.'
잘 보이는 동작에 차이: 같은 단계와 사용하는 손을 맞춰, 여러 참고와 비교한 차이를 위치/각도/궤적별로 설명.
공개 참고의 사람 간 차이 분포를 허용 범위의 출발점으로 삼을 수 있으나, 그 안=올바름/밖=틀림으로 즉시 확정하지 않는다.
참고에도 검출 실패가 있고 같은 의미의 자연스러운 변형이 존재한다. 현재 수치는 수어 오답·교정 방향을 검증한 기준이 아니다.
가림은 올바른 수어에서 자연스럽게 생길 수도 있다. 보여주려고 수어 동작 자체를 바꾸게 하지 않는다.

다음 한 작업 제안(아직 미실행)
손 영역을 따라가는 위치와 해부학적 손바닥 중심을 분리한 작은 보조 후보를 먼저 검토한다.
회전/크기변화가 있는 경우 여러 특징점의 기하 일관성과 손 재등장 관측을 대조하고, 불확실하면 중심 좌표·손모양을 채우지 않는다.
첫 등장/완전가림은 추적만으로 해결되지 않으므로 실제 이미지의 재검출 근거가 필요한지 별도로 확인한다.
같은 공개 고정 사례의 누락시간·위치오차·정상회귀를 먼저 비교하고, 미사용 공개 촬영자 검증은 후보를 고정한 뒤 수행한다.
이번 결과를 사용자와 검토하기 전 추가 학습/운영 반영/새 가중치 다운로드는 하지 않는다.

실행과 검증
작성·실행: Codex. 별도 verify.py가 추적 저장값2769건의 시간·좌표·판정 산술을 재구성하고, source/protected 해시를 확인했다.
같은 Codex의 별도 코드 및 시각검토이며 다른 에이전트/사람의 독립 검토는 아니다.
처음22영상 뒤 NaN 진단값 JSON 저장 오류. FIRST_ERROR 보존,22결과 해시 그대로 보존. 저장 시 비유한 진단값만null로 바꾸는 별도복구로14영상 완료.
동결된 추적 조건·원본·기존 C/T3-F32보호파일은 변경 없음. 학습0, 분류 재평가0, 새 MediaPipe실행0, 사용자촬영 접근0.
RESULT_KO.html에서36영상과 손 이동 그래프,9개 위치추정과4개 반례프레임을 확인할 수 있다.
"""
    (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
    data=json.dumps(examples,ensure_ascii=False).replace('</','<\\/')
    page=HTML.replace('__DATA__',data)
    (U/'RESULT_KO.html').write_text(page,encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

HTML=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>손이 보이지 않을 때, 영상은 얼마나 도와줄까? · WordsInSilence</title>
<style>
:root{--ink:#142a3a;--muted:#536675;--blue:#1887d8;--pink:#ce4087;--amber:#be7500;--line:#dce5e9}
*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:16px/1.7 "Malgun Gothic",sans-serif}
main{max-width:1240px;margin:auto;padding:40px 28px 80px}h1{font-size:clamp(28px,4vw,43px);line-height:1.35;margin:18px 0}h2{font-size:25px;margin:0 0 20px}h3{font-size:19px;margin:0 0 10px}p{margin:10px 0}a{color:#17668d}section{margin:30px 0;background:white;padding:30px;border:1px solid var(--line);border-radius:18px}.eyebrow{font-weight:700;color:#26746b;letter-spacing:1px}.lead{font-size:20px;max-width:950px}.muted{color:var(--muted);font-size:14px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}.stat{padding:20px;background:#fff;border:1px solid var(--line);border-radius:14px}.stat strong{display:block;font-size:29px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.warn{background:#fff4e1;border-left:4px solid #d39535;padding:15px 18px}.good{background:#edf7f4;border-left:4px solid #478b72;padding:15px 18px}table{border-collapse:collapse;width:100%}td,th{padding:13px 12px;text-align:left;border-bottom:1px solid var(--line)}th{background:#eff4f7}select,button{font:inherit;padding:9px 13px;border:1px solid #becdd6;border-radius:8px;background:white;color:var(--ink)}button{cursor:pointer}button:hover{background:#eaf4fa}select{max-width:100%;min-width:310px}
video{width:100%;background:#132a41;border-radius:10px;display:block}img{max-width:100%;height:auto}figure{margin:20px 0}figcaption{font-size:14px;color:var(--muted);margin:8px 0}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0}.left{color:var(--blue)}.right{color:var(--pink)}.extra{color:var(--amber)}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.bar{height:20px;background:#e5ebf0;border-radius:6px;overflow:hidden;margin:10px 0}.bar span{display:block;height:100%;width:14.52%;background:#d59525}
.badge{display:inline-block;font-size:13px;background:#e9eff4;border-radius:5px;padding:3px 9px}details{margin:15px 0;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;font-weight:bold}.casepic{border:1px solid var(--line);border-radius:8px}.steps{counter-reset:step}.step{padding:16px 0;border-bottom:1px solid var(--line)}.step b{display:block}.nav{display:flex;gap:18px;flex-wrap:wrap}.graph{border:1px solid var(--line);border-radius:8px;display:block;width:100%}
@media(max-width:800px){main{padding:22px 14px}.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}section{padding:22px 16px}table{font-size:14px}td,th{padding:9px 6px}select{min-width:0;width:100%}}@media print{button,select,video{display:none}section{break-inside:avoid}body{background:white}}
</style><main>
<div class="eyebrow">WORDS IN SILENCE · 공개 영상 비교 · 2026.10.20</div>
<h1>손이 보이지 않을 때,<br>영상은 얼마나 도와줄까?</h1>
<p class="lead">영상 추적은 <b>짧게 놓친 손 위치</b>에 조금 도움을 줬습니다.<br>하지만 긴 가림·손가락 모양·처음 잘못 붙은 좌우 이름은 아직 해결하지 못했습니다.</p>
<div class="nav"><a href="#result">비교 결과</a><a href="#watch">영상·이동 그래프</a><a href="#examples">실제 사례</a><a href="#feedback">피드백의 기준</a><a href="#next">다음 단계</a></div>
<div class="cards"><div class="stat"><strong>36개 영상</strong>23개 단어 · 촬영자 9명</div><div class="stat"><strong>9개 순간</strong>손 위치 추정 추가 · 모두 1프레임</div><div class="stat"><strong>42 / 42</strong>알려진 정상 좌우 배정 유지</div><div class="stat"><strong>학습 0회</strong>C·T3-F·사용자 촬영 보존</div></div>
<section><h2>이번에는 무엇을 바꿨나요?</h2><div class="grid"><div><h3>기존 방식</h3><p>MediaPipe가 찾은 관절과 팔·손의 연결 근거로 손 위치와 좌우를 정합니다. 관절을 놓친 순간에는 비교할 근거가 줄어듭니다.</p></div><div><h3>별도 영상 추적 보조</h3><p>잘 보이던 손의 작은 영상 특징점이 다음 장면에서 어디로 움직였는지 따라갑니다. 확실한 근거가 유지될 때만 <b>손 위치 추정</b>을 따로 표시합니다.</p></div></div><p class="good">원래 관측은 그대로 두고, 추정은 다른 색으로 표시했습니다. 추정된 위치를 실제 손가락 관절로 바꾸지 않았습니다.</p>
<p class="muted">같은 화면 범위·같은 저장 C 검출·같은 영상 전체 시간으로 비교했습니다. 현재와 과거 영상만 사용하고, 중간에 화면 아래 범위를 풀지 않았습니다. 방식은 <a href="https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html">OpenCV LK Optical Flow</a>이며, 이번 구현의 추적 지속 상한은 0.20초입니다.</p></section>
<section id="result"><h2>개선된 부분과 남은 부분</h2><table><thead><tr><th>확인한 내용</th><th>결과</th><th>해석</th></tr></thead><tbody>
<tr><td>자연스럽게 검출을 놓친 순간</td><td><b>9개 위치 추정 추가</b></td><td>모두 약 0.033초짜리 1프레임. 영상 대조상 손 영역 안에 있음.</td></tr>
<tr><td>그중 주석상 본동작 구간</td><td><b>2개</b></td><td>나머지 7개는 주로 손을 내리는 구간. 큰 복원 성과로 볼 수 없음.</td></tr>
<tr><td>알려진 정상 좌우 배정</td><td><b>42 → 42 유지</b></td><td>새 오배정 0. 이번 후보는 좌우를 바꾼 경우 자체가 없음.</td></tr>
<tr><td>기존 손 교차 표본</td><td><b>54프레임 변화 없음</b></td><td>기하학적 교차 표본. 모두 정답으로 수동 판정된 집합은 아님.</td></tr>
<tr><td>알려진 오류 좌우 배정</td><td><b>12 → 12 남음</b></td><td>처음의 좌우 혼동을 이번 추적만으로 고치지 못함.</td></tr>
<tr><td>보이지 않는 손가락 모양</td><td><b>복원하지 않음</b></td><td>위치가 이어져도 손가락이 어떻게 굽었는지는 모를 수 있음.</td></tr>
</tbody></table><p class="warn">현재 후보는 <b>진단용으로 보존하고 학습·DTW 점수·운영 웹에는 적용하지 않습니다.</b> 단어 인식률을 다시 측정한 실험도 아닙니다.</p>
<p class="muted">이전에 확인한 공개 TRAIN 문제·정상 대조 영상입니다. 전체 수어와 새 사용자 성능을 검증한 결과가 아닙니다. 4,641프레임 전체를 사람이 수동 판정한 것도 아닙니다.</p></section>
<section id="watch"><h2>같은 장면에서 기존 관측과 추정 비교</h2>
<label for="clip">영상 선택</label> <select id="clip" aria-label="공개 영상 선택"></select>
<p id="clipInfo" class="muted"></p><div class="legend"><b class="left">● 파랑: 기존 왼손 관측</b><b class="right">● 분홍: 기존 오른손 관측</b><b class="extra">○ 주황: 추가 위치 추정</b></div>
<div class="grid"><div><video id="video" controls preload="metadata" playsinline></video><div class="controls"><button id="prev">이전 프레임</button><button id="nextFrame">다음 프레임</button><button id="slow">0.5배속</button><span id="time" aria-live="off"></span></div><div class="controls" id="events"></div></div>
<div><h3>그래프는 이렇게 읽으세요</h3><p>아래 그래프의 가로축은 시간입니다. 첫째 줄은 손의 좌우 이동, 둘째 줄은 위아래 이동입니다.</p><p>파랑·분홍 선이 끊긴 곳은 <b>손바닥 위치를 충분히 관측하지 못한 구간</b>입니다. 일부 손가락만 보인 경우도 포함하며, 이것만으로 사용자의 실수라고 보지 않습니다.</p><p>주황 마름모는 영상이 보태 준 위치입니다. 없는 구간을 임의의 선으로 연결하지 않았습니다.</p><p>연한 초록 배경은 제공 주석의 동작 시간대입니다. 주석 밖 동작을 모두 불필요하다고 확정한 표시는 아닙니다.</p></div></div>
<a id="graphLink" target="_blank"><img id="graph" class="graph" alt="시간에 따른 양손 위치와 관측 구간"></a>
<p class="muted">좌우는 촬영자의 신체 기준입니다. 이동량은 한 영상에서 고정한 어깨너비 단위입니다. 손내림·한손 수행·관측 실패를 모두 ‘오류’로 표시하지 않습니다. 그래프를 누르면 원래 크기로 볼 수 있습니다.</p></section>
<section id="examples"><h2>실제로 도움이 된 순간과 한계</h2>
<h3>① 영상에는 주먹이 있는데 관절 위치가 잠깐 빠진 경우</h3><p>‘대지’ 촬영자 01의 1.633초입니다. 기존 손바닥 위치는 비었지만, 영상 추적점은 주먹 위에 남았습니다. <b>손의 대략적 위치를 1프레임 더 알 수 있었습니다.</b></p>
<figure><a href="evidence/visible_missed.png" target="_blank"><img class="casepic" src="evidence/visible_missed.png" alt="대지의 원장면과 앞·현재·다음 확대 비교"></a><figcaption>전체 가시 화면 + 앞·현재·다음 장면 확대. 확대 그림은 설명용이며, 추적 입력을 이 작은 사각형으로 바꾼 것이 아닙니다.</figcaption></figure>
<h3>② 손을 따라가도 손바닥 중심을 정확히 따라가는 것은 아님</h3><p>‘상처’ 촬영자 07에서 손 검출만 일부러 감춘 부가 시험입니다. 주황점은 같은 손 위에 있지만, 손이 돌아가며 파란 손바닥 중심과 벌어졌습니다. <b>손의 영상 무늬를 따라가는 것과 관절을 찾는 것은 다릅니다.</b></p>
<figure><a href="evidence/texture_drift.png" target="_blank"><img class="casepic" src="evidence/texture_drift.png" alt="상처의 영상 추적점과 숨겨 둔 관절 중심 비교"></a><figcaption>파랑: 숨겨 둔 MediaPipe 중심(임시 비교값). 주황: 추적점. 이 비교값도 수작업 정밀 정답은 아닙니다.</figcaption></figure>
<div class="grid"><div><h3>일부러 검출을 감춘 시험</h3><p>70구간 × 6프레임 = <b>420개 손-프레임</b><br>위치 추정 61개 · 보류 359개</p><div class="bar"><span></span></div><p>14.52%는 <b>추정값을 제공한 비율</b>입니다. 인식 정확도나 손동작 정답률이 아닙니다.</p></div><div><h3>추정값이 어긋난 반례</h3><p>61개 추정의 임시 중심 대비 오차 중앙값은 어깨너비의 1.79%, 최대 19.81%였습니다. 10%를 넘은 4프레임은 위 ‘상처’ 한 구간에 모였습니다.</p><p class="muted">10%는 결과 설명용 선이며, 수어의 허용오차로 정한 값이 아닙니다. 손 회전과 중심 정의의 차이가 원인일 가능성이 있지만 인과를 분리한 증명은 아닙니다.</p></div></div>
<details><summary>자연 발생 9개 추정 및 부가 반례 전체 보기</summary><p>모든 자연 추정점은 손 영역 안에 있었지만, 정밀 중심 좌표의 정답을 뜻하지는 않습니다. 손내림 장면도 성공 횟수로 과장하지 않았습니다.</p>
<a href="evidence/review_1.png" target="_blank">수영·대지 3사례</a> / <a href="evidence/review_2.png" target="_blank">대지·어린이날 3사례</a> / <a href="evidence/review_3.png" target="_blank">가래떡·어린이날·수어 3사례</a> / <a href="evidence/review_4.png" target="_blank">부가 반례 3프레임</a> / <a href="evidence/review_5.png" target="_blank">부가 반례 1프레임</a>
</details></section>
<section id="feedback"><h2>개인 맞춤 피드백으로 연결하려면</h2><table><thead><tr><th>시스템이 아는 것</th><th>가능한 안내</th><th>바로 단정하면 안 되는 것</th></tr></thead><tbody>
<tr><td>손이 가려져 모양을 못 봄</td><td>“이 구간은 손 모양을 확인하기 어려워요.”</td><td>“손 모양이 틀렸어요.”</td></tr>
<tr><td>위치는 추정했지만 손가락은 안 보임</td><td>“손 이동은 일부 이어 볼 수 있어요.”</td><td>“손가락 모양도 맞아요.”</td></tr>
<tr><td>동작이 잘 보이고 같은 단계끼리 비교 가능</td><td>“참고보다 손 위치가 높게 관측됐어요.”</td><td>검증 없이 “반드시 더 내려야 정답이에요.”</td></tr>
</tbody></table><p>여러 참고 수행자의 차이는 허용 범위를 만드는 <b>출발 자료</b>로 사용할 수 있습니다. 하지만 참고에도 검출 실패가 있고 자연스러운 표현 차이가 있으므로, ‘범위 밖 = 잘못된 수어’로 즉시 연결하지 않습니다.</p><p class="good">피드백의 순서: <b>충분히 보였는가 → 같은 단계·같은 손인가 → 어떤 차이가 있는가 → 교정이 필요한 차이인가</b>.</p><p>올바른 수어에서도 손이 겹칠 수 있습니다. 손을 보여주려고 수어 동작 자체를 바꾸게 해서는 안 됩니다.</p></section>
<section id="next"><h2>다음에는 무엇을 확인할까요?</h2><div class="steps">
<div class="step"><b>1. 손 영역 추적과 손바닥 중심 복원을 분리</b>손이 돌아가거나 크기가 달라져도 같은 관절을 나타내는지 확인합니다. 위치 추정만 된 경우는 계속 별도 표시합니다.</div>
<div class="step"><b>2. 다시 나타난 손은 현재 영상으로 재확인</b>처음부터 잘못된 좌우 이름과 완전가림은 추적만으로 해결하기 어렵습니다. 현재 팔·손 근거와 재검출을 대조하고 불확실하면 보류합니다.</div>
<div class="step"><b>3. 공개 영상에서 실제 누락 감소와 새 오류를 함께 비교</b>손내림·한손 동작·정상 교차를 보존하면서, 본동작의 관측 공백과 위치 오차가 줄어드는지 먼저 확인합니다.</div>
<div class="step"><b>4. 결과 확인 후에만 학습 입력 반영 검토</b>이번 후보는 채택하지 않습니다. C·T3-F는 유지하고 사용자 촬영은 학습·튜닝·모델 선택에 사용하지 않습니다.</div></div></section>
<section><h2>검증 기록</h2><p>저장 추적 2,769건의 시간·좌표·판정 계산을 별도 코드로 검산했습니다. 원본·동결 코드 및 C·T3-F를 포함한 보호 파일 32개는 해시가 유지됐습니다.</p><p>구현·실행·시각 검토는 Codex, 검산은 같은 Codex가 별도 코드로 수행했습니다. 다른 사람·에이전트의 독립 검증이나 수어 전문가 판정은 아닙니다.</p><details><summary>실행 중 오류와 복구 / 처리 시간</summary><p>22개 영상 처리 후 비유한 진단값의 JSON 저장 오류가 발생했습니다. 최초 오류와 완료 결과를 보존하고 저장 형식만 복구해 나머지 14개를 완료했습니다. 조건·좌표 계산은 변경하지 않았습니다.</p><p>추적 추가연산 95백분위 7.39ms/프레임. 디코딩·리사이즈·MediaPipe 제외, 추적 없는 프레임 포함이므로 웹 실시간 성능을 보장하는 수치가 아닙니다.</p></details>
<p><a href="RESULT_KO.txt">상세 보고서</a> · <a href="VERIFY.json">별도 계산 검산</a> · <a href="VISUAL_REVIEW.json">시각 검토</a> · <a href="FINAL_AUDIT.json">최종 감사</a> · <a href="FROZEN.json">동결 조건</a> · <a href="FIRST_ERROR.json">최초 오류</a></p></section>
</main><script>
const DATA=__DATA__;
const sel=document.getElementById('clip'),video=document.getElementById('video'),graph=document.getElementById('graph'),graphLink=document.getElementById('graphLink'),events=document.getElementById('events');
for(const r of DATA){const o=document.createElement('option');o.value=r.id;o.textContent=`${r.word} · 촬영자 ${String(r.person).padStart(2,'0')}${r.episodes.length?' · 추가 위치 '+r.episodes.length+'개':''}`;sel.append(o)}
let current=null,pendingTime=0;
function pick(id,time=0){current=DATA.find(r=>r.id===id)||DATA[0];sel.value=current.id;pendingTime=time;video.src=`videos/${current.id}.mp4`;graph.src=`plots/${current.id}.png`;graphLink.href=graph.src;
document.getElementById('clipInfo').textContent=`${current.frames}프레임 · ${current.width} × ${current.height} 가시 범위 · 기존 배정 변화 없음 · 추가 위치 ${current.episodes.length}개`;
events.replaceChildren();for(const e of current.episodes){const b=document.createElement('button');b.textContent=`추정 순간 ${e.time_start.toFixed(3)}초 (${e.side===0?'왼손':'오른손'})`;b.onclick=()=>{video.pause();video.currentTime=e.time_start-current.first_time+.006};events.append(b)}
if(!current.episodes.length)events.textContent='이 영상에서는 관측 공백에 새 위치 추정이 더해지지 않았습니다.';
}
video.addEventListener('loadedmetadata',()=>{video.currentTime=Math.max(0,pendingTime-current.first_time)});
video.addEventListener('timeupdate',()=>{document.getElementById('time').textContent=`${(video.currentTime+(current?.first_time||0)).toFixed(3)}초`});
sel.onchange=()=>pick(sel.value);
document.getElementById('prev').onclick=()=>{video.pause();video.currentTime=Math.max(0,video.currentTime-1/current.fps)};
document.getElementById('nextFrame').onclick=()=>{video.pause();video.currentTime=Math.min(video.duration,video.currentTime+1/current.fps)};
document.getElementById('slow').onclick=()=>{video.playbackRate=video.playbackRate===1?.5:1;document.getElementById('slow').textContent=video.playbackRate===1?'0.5배속':'1배속'};
pick('NIA_SL_WORD0401_REAL01_F',1.633+.006);
</script></html>'''
if __name__=='__main__':main()
