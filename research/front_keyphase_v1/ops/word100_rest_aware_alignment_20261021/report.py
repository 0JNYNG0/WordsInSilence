"""Publish saved verification and visual review, no experiment rerun."""
import html,json,re
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import study as d
U=d.U
r=d.read(U/'RESULT.json');v=d.read(U/'VERIFY.json');s=d.read(U/'STRESS.json');a=d.read(U/'DIAGNOSIS.json');m=d.read(U/'VISUAL_MANIFEST.json')
assert v['status']=='PASS'
review=dict(at=d.now(),reviewer='Codex raw-RGB visual review, not independent sign-language expert or anatomical ground truth',
 sheets=[Path(x).name for x in m['pages']],triplets=26,raw_panels=78,
 findings=[
  'Reviewed all 9 raw sheets. Selected all newly-added conflict-run endpoints/middles, largest shift of all8 changed pairs, and unchanged normal controls.',
  'REAL04→REAL09 후원: query f58 extended-finger pose matches new reference f40 more closely than old f48 bent pose. Query f61/f64 bent pose instead maps to new f42 extended pose, while old f48 better matches bend phase. Mixed local gain and regression, not an overall alignment improvement.',
  'REAL06→REAL09 query f28 is bent near face but new ref f42 extended, old f45 already bent: visible regression at selected transition. Most other changed cases shift1-4frames within similar poses; no strong overall improvement established.',
  'Automatic active witness means any observed point inside ROI, not semantic active motion. Many flagged left hands visibly hang at side near boundary; new25 flagged pair-handframes are NOT25 confirmed errors. Preserve geometry metrics but do not equate them to motion-error counts.',
  '수어 two-hand core/crossing selected raw panel and all72 paths unchanged. 멀다 one-hand selected panel and all72 paths unchanged. Prior80normal/cross/error controls are reused input-preservation evidence, not new DTW tests on80clips.',
  'No coordinate, side identity, angle-state, video or ROI rewrite. Stable rest not newly marked as wrong. Semantic phase correctness for every frame is not established.'
 ],decision='Do not adopt rest_alignment as demonstrated timing improvement. Retain research-only support calculation result and fixed reference denominator; review alignment safeguards before learning.')
d.write(U/'VISUAL_REVIEW.json',review)
d.write(U/'VERIFIER_CORRECTION.json',dict(first_error='VERIFY_FIRST_ERROR.txt',cause='Independent verifier used float32 mean; runtime sums float32 squares then divides integer counts in float64. First pair maximum difference1.5447332746276743e-08, evidence exactly equal.',
 change='Verifier only: sum squares in float32, cast sum to float64 then divide width. No tolerance relaxed; no candidate/inputs/thresholds/results changed.',outcome='All3215596cost cells and648pair results passed original precision tolerances. No experiment rerun.'))
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(9,3.4));vals=[373,378,378]
ax.barh(['기존 방식','계산만 변경','정렬까지 변경'],vals,color=['#647b94','#1b8e9b','#dd8a33']);ax.invert_yaxis();ax.set_xlim(0,675)
for i,n in enumerate(vals):ax.text(n+9,i,f'{n} / 675',va='center',fontsize=12)
ax.set_xlabel('여러 참고로 일부 비교 수치를 표시할 수 있는 각도 항목 수 (정답률 아님)');fig.tight_layout();fig.savefig(U/'visuals'/'availability.png',dpi=150);plt.close(fig)
# Already inspected raw correspondence rows, no pixel alteration beyond layout extraction.
Image.open(U/'visuals'/'raw_02.jpg').crop((0,0,1440,700)).save(U/'visuals'/'timing_counterexample.jpg',quality=95)

text='''손별 손내림을 시간 정렬에 반영한 세 조건 검증 결과

결론: 비교 근거 계산의 소폭 개선은 확인했지만, 실제 동작 시점을 더 정확히 맞췄다는 결론은 내릴 수 없다. 일부 전환 장면에서 오히려 앞 단계에 머무르는 반례가 있어 새 정렬은 운영에 적용하지 않는다. 기존 C·T3-F, 웹, 학습 데이터는 그대로 유지했다. 추가 학습·사용자 자료 접근은 0이다.

무엇을 비교했나
공개 TRAIN 수어·후원·멀다 3개 표현, 9명, 27영상. 각 영상을 같은 사람을 제외한 8명과 비교한216쌍. 세 조건648쌍 결과를 검산했다. 현재 대상은 작은 다중 수행자 묶음이며100단어 전체의 새 정렬 실험이 아니다.
1. 기존 방식: 이전 완료 결과의 경로와 계산 그대로.
2. 계산만 변경: 경로 그대로, 공개참고에서 확인된 손내림 시간에는 쉬는 손의 모양 정보까지 관측하도록 요구하지 않음.
3. 정렬까지 변경: 같은 손내림 정보를 장면 연결에도 반영. 팔 위치는 유지. 불확실 구간은 원래 규칙 유지.

결과
기존373/675 → 계산만378/675 → 정렬까지378/675. 정답률이나 동작 일치율이 아니다. 영상별·손별·15개 각도 중5명 이상 참고의 부분 비교 근거가 있어 표시 가능한 항목 수다. 두 손을 쓰는 두 표현540항목, 한 손 표현135항목으로675이고, 사용하지 않는 반대손135항목은 별도다. 기준5명/50%시간/8프레임/정렬근거0.35는 변경하지 않았다.
수어238/270 유지, 후원27→32/270, 멀다108/135 유지. 기존 표시 가능373항목이 부족으로 떨어진 경우0, 여전히297항목은 근거 부족. 추가5항목은 모두 계산 변경에서 발생했고 새 정렬의 추가 이득은0이다. 정렬 변경 후 기존 표시 항목10개의 각도 중앙값이 달라졌지만 최대1.31도 차이는 정확성 향상의 증거가 아니다.

실제 장면 연결
216쌍 중8쌍(모두 후원)의 연결 변경,208쌍 동일. REAL04→REAL09에서 비교 f58의 편 손가락은 새 참고 f40이 기존 f48보다 비슷했다. 그러나 비교 f61/f64처럼 손가락을 이미 굽힌 순간에도 새 참고 f42의 펴진 상태에 연결됐다. 기존 f48은 굽힘 단계와 더 가까웠다. 단계 전환을 잘 보존한다고 볼 수 없다. 최대 시간 이동0.567초는 대기 구간에서 생겨, 그 수치를 동작 정렬 오류량으로 쓰지 않았다.
자동 경고 active는 화면 범위 안에 점이 있다는 뜻이지 손이 수어 동작 중이라는 뜻이 아니다. 실제 원본에서 일부 경고 손은 옆에 내려놓은 상태였다. 따라서 새25쌍-손프레임 경고를25실제오류로 세지 않았다. 새 정렬은 개선/회귀가 섞여 있으며 전체 정확한 시간대 정답 주석은 없다.

손이 사라졌을 때
정상자료648결과 모두 참고의 각도별 비교 시간범위와 분모 유지. 한 손을 통째로 지운54조건(각8참고,432새 경로)에서도 참고범위/필요정보 가중치 분모 불변, 없는 손에서 비교가능 판정0. 손이 안 보인다는 이유로 확인해야 할 범위를 삭제하지 않았다.
중간3개 저장프레임만 지운27조건(216새 경로)에서는91경로가 바뀌었다. 삭제 시점은 비교 근거로 사용되지 않았지만, 남은 장면이 다른 참고 시점에 재연결되면서3240쌍-각도행 중39행의 관측비율이 올라갔고14행은 일부비교 기준을 새로 넘었다(108행은 반대로 기준 미달). 분모를 줄인 것은 아니지만 짧은 누락을 주변 장면으로 대신 맞출 위험이 남아 있다. 필수 동작의 실제 생략이 아니라 저장 특징을3프레임 지운 모의 시험이며, 이14건을 전체 동작 정답 판정이라고 해석해서는 안 된다.

정상 손내림과 교차
입력 좌표·좌우 배정·각도 상태는 이번 후보에서 수정하지 않았다. 수어72쌍과 멀다72쌍의 경로/결과 유지 및 원본 대조 장면 확인. 이전 정상·교차·알려진 오류80대조 근거는 입력 보존 확인으로 재사용했으며 이번에80개 새 DTW 비교를 한 것은 아니다. 정상 손내림을 동작 오류로 판정하는 기능은 추가하지 않았다. 다만 위 후원의 정상 단계 전환 회귀 때문에 정상 동작 전체가 더 잘 비교된다고 선언하지 않는다.

검증 및 한계
별도 수식으로3,215,596비용 셀,최소누적비용경로,648쌍의 시간분모·관측비율·27영상 요약 재검산 PASS. 원본26장면묶음(78패널)/9시트 직접 확인. 작성·실행·별도검산 모두Codex; 독립 수어 전문가 검토나 모든 손가락의3D정답 검증은 아니다. 최초 검산 실패는 float32 평균과 float64 나눗셈의 차이였고 검산 코드만 정정; 최초오류 보존. 임계값/결과/후보는 바꾸지 않았다.

다음 제안 — 아직 실행하지 않음
새 정렬을 학습이나 운영에 적용하지 않고, '범위 안에 손이 있음'과 '실제 동작 중/안정 손내림/알 수 없음'을 구분하는 진단부터 정리한다. 실제 이동·손 모양 전환과 팔 위치 등 현재 양성 근거를 함께 보며, 보이지 않으면 계속 모름으로 유지한다. 그다음 한 장면을 길게 늘려 짧은 전환·누락 구간을 대신하는 경우를 표시하는 공통 보호 후보를 별도로 비교한다. 속도 차이를 일률적으로 제한하거나 단어별 예외를 만들지 않는다. 이전 반복/압축 실험의 정상 속도 회귀도 대조군으로 재사용하고, 이번27영상과 바뀐8쌍의 단계 전환·전체손누락·짧은누락을 함께 확인한다. 참조범위고정은 유지한다. 효과 확인 전 문턱 완화·추가학습·모델교체는 하지 않는다.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
gallery=''.join(f'<details><summary>원본 장면 비교 {i+1} (왼쪽: 비교 영상 / 가운데: 기존 / 오른쪽: 새 연결)</summary><img loading="lazy" src="visuals/{Path(p).name}" alt="원본 장면 비교 {i+1}"></details>' for i,p in enumerate(m['pages']))
paths=''.join(f'<details><summary>후원 {p["query"].split("_")[-2]} → {p["reference"].split("_")[-2]} 연결 그래프</summary><img loading="lazy" src="visuals/{p["query"]}__{p["reference"]}_path.png" alt="시간 정렬 경로"></details>' for p in a['changed_pairs'])
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손별 손내림 · 시간 정렬 검증</title>
<style>body{margin:0;background:#edf2f6;color:#172a3b;font-family:"Malgun Gothic",sans-serif;line-height:1.75}main{max-width:1120px;margin:auto;padding:38px 24px}h1{font-size:32px;line-height:1.3}h2{font-size:23px;margin-top:0}section{background:white;border-radius:16px;padding:28px;margin:20px 0;border:1px solid #dce5ed}.lead{border-left:7px solid #c67b28;background:#fff8eb}.muted{color:#536879}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #dae2e9}th{background:#edf5f9}img{max-width:100%;height:auto;border-radius:8px}.kpis{display:flex;gap:16px;flex-wrap:wrap}.kpis div{background:#eff5f9;padding:15px 22px;border-radius:12px;flex:1;min-width:170px}.kpis strong{display:block;font-size:29px}details{padding:12px 0}summary{cursor:pointer;color:#235779}a{color:#166583}.note{padding:15px;background:#fff5e6;border-radius:9px}p{margin:12px 0}li{margin:8px 0}</style>
<main><p class="muted">2026-10-21 · 공개자료 연구 후보 · 추가 학습 없음</p><h1>손을 내려놓은 시간까지 고려하면<br>동작 시점을 더 잘 맞출까?</h1>
<section class="lead"><h2>결론: 비교 가능한 항목은 조금 늘었지만,<br>새 정렬이 더 정확하다는 근거는 부족합니다.</h2><p>일부 장면은 더 비슷하게 연결됐지만, 손가락을 굽힌 뒤에도 앞의 펴진 장면에 머무른 반례가 있습니다. <b>새 정렬은 운영에 적용하지 않습니다.</b> 기존 C·T3-F·웹은 유지하고 사용자 촬영은 사용하지 않았습니다.</p></section>
<section><h2>같은 영상으로 세 조건을 비교</h2><p>공개 3표현 × 9명 = 27영상. 자신을 제외한 8명을 참고하여 216쌍을 비교했습니다.</p><table><tr><th>조건</th><th>바뀐 부분</th><th>일부 비교 가능</th></tr><tr><td>기존</td><td>이전 장면 연결·계산 유지</td><td>373 / 675</td></tr><tr><td>계산만 변경</td><td>쉬는 손의 정보까지 요구하던 근거 계산 조정</td><td>378 / 675</td></tr><tr><td>정렬까지 변경</td><td>손내림 시간을 장면 연결에도 반영</td><td>378 / 675</td></tr></table><img src="visuals/availability.png" alt="일부 비교 가능 각도 항목 373,378,378"><p class="note"><b>인식률이나 동작 일치율이 아닙니다.</b> 영상별·손별 손가락 각도 중 여러 참고로 비교 수치를 보여줄 근거가 있는 항목 수입니다. 추가 5항목은 계산 변경에서 생겼으며, 새 정렬의 추가 이득은 없었습니다. 아직 297항목은 근거 부족입니다.</p><p>수어 238/270 유지 · 후원 27→32/270 · 멀다 108/135 유지. 사용하지 않는 반대손 135항목은 별도입니다.</p></section>
<section><h2>원본에서 확인한 장면 연결 문제</h2><p>216쌍 중 8쌍의 연결이 바뀌었고, 모두 ‘후원’이었습니다. 펴진 손가락을 더 비슷한 장면에 맞춘 사례가 있었지만, 이후 굽힌 순간까지 펴진 장면에 연결하는 회귀도 확인했습니다.</p><img src="visuals/timing_counterexample.jpg" alt="비교 영상은 손가락을 굽혔는데 새 참고 장면은 아직 펴진 상태"><p>왼쪽: 비교 장면 / 가운데: 기존 참고 장면 / 오른쪽: 새 참고 장면. REAL04의 f61·f64는 이미 굽힌 단계인데 새 정렬은 REAL09의 f42에 머뭅니다. 기존 f48은 굽힌 단계와 더 가깝습니다. 수어 표현의 언어적 정오 판정이 아닌, 보이는 단계의 대응 비교입니다.</p><p class="note">자동 경고의 ‘활동’은 사실 <b>관절점이 설정 범위 안에 있다</b>는 뜻이었습니다. 원본에는 옆에 내려놓은 손도 포함됐습니다. 경고 25개를 실제 동작 오류 25개로 세지 않았습니다.</p></section>
<section><h2>손이 사라져도 비교해야 할 범위는 유지</h2><div class="kpis"><div><strong>54조건</strong>한 손 전체 누락 시험</div><div><strong>범위 유지</strong>432개 재정렬 모두 확인</div><div><strong>0건</strong>없는 손을 비교 가능 처리</div></div><p>참고 영상이 정한 시간 범위·분모는 그대로이며, 삭제된 손 관측을 비교 근거로 사용하지 않았습니다.</p><h3>짧은 누락에는 남은 문제가 있습니다</h3><p>중간 3개 저장프레임을 지운 27조건에서, 남은 장면을 다시 연결하면서 39개 쌍-각도 항목의 관측비율이 오히려 올랐습니다. 또한 14개 항목은 일부 비교 기준을 새로 넘었습니다. 분모가 줄어든 것은 아니지만, <b>짧은 누락을 주변 장면으로 대신 맞출 위험</b>이 남습니다.</p><p class="muted">27조건 × 8참고 × 15각도 = 3,240쌍-각도 항목 중 집계. 전체 동작 정답 판정이 아닙니다. 필수 동작 생략의 정답 주석을 가진 시험도 아닙니다.</p></section>
<section><h2>정상 손내림·교차는 어느 범위까지 확인했나?</h2><ul><li>원래 관절 좌표, 좌우 배정, 각도 상태를 바꾸거나 손을 새로 삭제하지 않았습니다.</li><li>‘수어’ 72쌍과 ‘멀다’ 72쌍의 연결은 모두 그대로이며 정상·교차 대조 장면도 확인했습니다.</li><li>기존 정상·교차 등 80개 대조 근거는 입력 보존 확인으로 재사용했습니다. 이번에 80개 새 시간 정렬 시험을 했다는 뜻은 아닙니다.</li><li>정상 손내림을 동작 오류로 선언하지 않았습니다. 다만 ‘후원’의 단계 전환 반례 때문에 정상 동작 전체가 더 잘 비교된다고 결론 내리지 않습니다.</li></ul></section>
<section><h2>다음에는 무엇을 보완할까?</h2><ol><li><b>보이는 위치와 실제 활동 상태를 구분:</b> 범위 안의 점 하나만으로 활동 중이라고 부르지 않고, 이동·손 모양 전환·팔 위치와 안정 손내림 근거를 함께 확인합니다. 미검출은 계속 ‘모름’으로 둡니다.</li><li><b>한 장면이 여러 동작 단계를 대신하지 않는지 확인:</b> 짧은 굽힘 전환과 누락 구간을 남은 한두 장면으로 늘려 맞춘 경우를 표시하는 후보를 비교합니다.</li><li><b>정상 속도 차이도 함께 대조:</b> 속도가 다르다는 이유만으로 오류로 만들지 않도록 이전 반복·압축 시험과 이번 반례를 함께 확인합니다.</li></ol><p>위 내용은 다음 계획입니다. 이번 결과를 함께 확인한 다음 진행하며, 추가 학습은 실행하지 않았습니다.</p></section>
<section><h2>원본 비교와 연결 그래프</h2>'''+gallery+paths+'''</section>
<section><h2>검증 근거와 한계</h2><p>별도 수식으로 3,215,596비용 셀, 최적 경로 비용, 648쌍의 시간범위·관측비율·영상별 요약을 검산했습니다. 원본 26묶음(78패널), 9시트를 확인했습니다. 구현·검산·시각검토 담당은 Codex이며 독립 수어 전문가 검증은 아닙니다.</p><p>최초 검산에서 계산 자료형(float32 평균 / float64 나눗셈) 차이를 발견해 검산 코드만 정정했습니다. 최초 오류를 보존했고 후보·기준·저장 결과는 바꾸지 않았습니다.</p><p>실험은 공개 TRAIN 27영상에 한정됩니다. 실제 사용자 정확도, 전체 수어 일반화, 손가락의 3차원 정답을 검증한 결과가 아닙니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="VERIFY.json">별도 수식 검산</a> · <a href="STRESS.json">누락 시험</a> · <a href="DIAGNOSIS.json">진단 수치</a> · <a href="VISUAL_REVIEW.json">시각 검토 기록</a> · <a href="FROZEN.json">동결 조건</a></p></section></main></html>'''
(U/'RESULT_KO.html').write_text(doc,encoding='utf8')
links=re.findall(r'(?:src|href)="([^"]+)"',doc)
for p in links:assert (U/p).is_file(),p
d.write(U/'REPORT_CHECK.json',dict(status='PASS_STATIC_LINKS',links=len(links),browser_render_check=False,plot_visual_review_pending=True))
print('REPORT',len(links))
