import re,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study as d
U=d.U;a=d.read(U/'ANALYSIS.json');v=d.read(U/'VERIFY.json');assert v['status']=='PASS'
g=a['groups'];lost=a['lost_warnings'][0]
# Only presentation layout repair: keep legend away from bar labels.
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained');x=np.arange(5);labels=['원래 속도','빠르게','느리게','중간 속도 변화','잠깐 정지']
axs[0].bar(x-.18,[g[k]['old'] for k in d.KINDS],.36,color='#c99550',label='이전 시간 압축 경고');axs[0].bar(x+.18,[g[k]['new'] for k in d.KINDS],.36,color='#b84b52',label='주변 각도 범위 경고')
axs[0].set(xticks=x,xticklabels=labels,ylim=(0,280),ylabel='경고가 하나라도 있는 비교 / 216',title='다른 사람 비교에서는 경고가 거의 모두 발생');axs[0].legend(fontsize=9,loc='upper center',ncol=2)
for i,k in enumerate(d.KINDS):
    for dx,key in [(-.18,'old'),(.18,'new')]:axs[0].text(i+dx,g[k][key]+3,str(g[k][key]),ha='center',fontsize=9)
cv=[g['identity']['support']]+[v['support'] for v in g['identity']['geometry']]
axs[1].barh(range(4),np.array(cv)*100,color=['#6b65a8','#32a398','#3b72a1','#3b72a1']);axs[1].set(yticks=range(4),yticklabels=['손가락 각도','손바닥 위치','팔꿈치 위치','팔 손목 위치'],xlim=(0,110),xlabel='참고가 요구하는 칸 중 비교 가능 비율 (%)',title='항목별 관측 근거는 다르다 — 원래 속도')
for i,v in enumerate(cv):axs[1].text(v*100+1,i,f'{v*100:.1f}%',va='center')
fig.savefig(U/'visuals'/'summary.png',dpi=145);plt.close(fig)

text='''다른 사람의 공개 참고와 손 위치·팔 궤적 비교 결과

결론: 이번 작업은 TCN 단어 분류가 아니라 DTW 세부 비교의 신뢰도 점검이다. 같은 원본에서 보였던 경고 감소가 다른 사람 참고에서는 유지되지 않았다. 새 각도 기준을 사용자 오류/교정 지시로 적용하지 않는다. 손가락·손바닥·팔을 따로 보고 관측 부족을 표시하는 연구용 기능은 구현·검산했다.

자료와 시험
공개 TRAIN 수어·후원·멀다 3단어, 9명, 27영상. 각 영상을 같은 단어의 다른 8명과 비교한 216조합. 원래 속도, 빠르게, 느리게, 중간 속도 변화, 중간 잠깐 정지의 5조건으로 총1080조합이다. 원본이1080개인 것이 아니다. 기존 관절/재확인 정보를 재표본한 시험이며 RGB를 다시 검출한 것이 아니다. 100개 전체 검증·새 사용자 인식률은 아니다.

경고 결과
원래 속도216조합: 이전210, 새216. 빠르게213→215, 느리게214→216, 중간 속도변화216→216, 잠깐 정지209→216. 시간변형864조합 합계852→863. 이전 경고는 여러 장면이 한 장면에 연결되는 변화를 감지하고, 새 경고는 주변 각도 범위를 벗어나는지를 감지하므로 같은 종류의 오류 정답률은 아니다. 다른 사람끼리의 정확한 동작 대응시점·허용차 정답도 없다. 경고가 줄었다고 주장할 수 없고 이 상태로 모든 경고를 잘못된 수어라고 말할 수 없다.
원래부터216개 모두 새 경고가 있어 추가 경고0개는 좋은 결과가 아니라 집계가 포화된 것이다. 여러 참고를 쓰더라도 각 영상의8명 중5명 이상과 경고가 있는 경우는 모든조건27/27이다. 동일한 시점의 의미상 오류에5명이 합의했다는 뜻은 아니다.

왜 이런 한계가 생기는가
현재 각도 범위는 같은 장면에서 두 검출 결과가 얼마만큼 일치하는지를 표현한다. 다른 사람이 정상적으로 수행할 수 있는 다양성의 범위를 학습하거나 보정한 값이 아니다. 사람마다 다른 수행, 몸/카메라 방향, 투영 각도, 검출 오차와 시간 정렬 차이가 모두 들어 있다. 둘의 값이 다르다는 사실과 수어가 틀렸다는 결론은 구분해야 한다. 두 검출기는 독립된 정답 센서도 아니다.

새로 분리한 정보
손가락15각도는 각각 따로 비교하고 관측 부족을 남긴다. 손바닥은 재확인된 위치와 두 추정의 차이를 기록한다. 팔꿈치와 팔 손목은 기존 몸 기준 좌표의 궤적 및 연결 시점의 위치 차이를 기록한다. 어깨 중앙을 기준으로 어깨너비로 나눠 비교하며 좌우나 원본 좌표는 바꾸지 않았다. 팔 정보는 원래 DTW 시간 정렬에도 들어 있었고, 이번에는 별도 차이·관측 비율·그래프로 드러낸 것이다.
위치 차이0.1은 영상에서 어깨너비의10분의1만큼 차이난다는 뜻이다. 틀린 비율10%가 아니다. 팔과 손가락 차이를 임의로 한 점수에 합치지 않았다. 손바닥 위치는 실제 관측된 가까운 점을 쓰고 가상의 중간 점으로 맞추지 않는다. 주변점 선택은 단조로운 시간 궤적을 보장하지 않아 속도/동작순서 판정에는 쓰지 않는다. 그림의 이동 경로는 원래 관측의 시간순 경로이고 누락 구간은 연결하지 않는다.

관측 비율(원래 속도216조합)
손가락 각도63.0%, 손바닥 위치74.2%, 팔꿈치·팔 손목각99.4%. 참고가 요구한 시간×손×항목의 칸 중 비교 가능한 칸의 비율이며 정확도가 아니다. 모든 연결 장면이 관측된 엄격한 비율은 각도62.3%, 손바닥73.6%, 팔각99.4%. 일부만 보인 긴 연결 구간을 완전히 관측한 것으로 부르지 않는다. 팔 좌표가 많다고 손가락까지 맞다고 인정하지 않는다. 팔 좌표 자체가 실제로 정확한지는 별도 문제다.

경고가 사라졌지만 좋아졌다고 볼 수 없는 실제 예
후원REAL01을REAL05와 비교하여 빠르게 재표본했을 때 유일하게 경고가 사라졌다. 하지만 각도 비교 가능 비율은36.9%→21.8%로 줄었다. 전에 지속경고였던40칸 중28칸은 근거 부족,11칸은 차이가 남지만3연속 조건미달,1칸만 주변범위 안이었다. 따라서 잘 맞음으로 바꾸지 않는다. 전체적으로 정상 속도변형도 유효한 양끝 표본을 모두 요구하기 때문에 관측 비율이 떨어질 수 있다. 이것은 이 합성 시험의 재표본 특성이고 사용자가 빠르게 움직일 때의 실제 MediaPipe 성능측정이 아니다.

보존과 검산
손 전체 누락45·짧은3프레임 누락45·팔 누락45, 총135개 고정경로 통제시험에서 참고 비교범위를 줄이지 않았다. 관측이 전부 없으면 해당 항목은 전부 근거 부족, 다른 항목은 보존됐다. 손내림은 기존 참고측 양성근거가 있는 시간만 제외하고, 안 보인다는 이유로 손내림이라 바꾸지 않는다. 기존 한손역할·정상교차·좌우좌표·원본핀 유지. 새로운 검출기의 교차 성능을 다시 측정한 것은 아니다.
1080저장조건의 각도·위치 산식, 마스크와 부분관측, DTW최적비용,135누락시험 별도코드검산PASS. 동일Codex의 별도 수식검산이며 수어 전문가의 의미정답 검토가 아니다. 원본두명6장면과 그래프3개 시각검토. HTML은 정적링크검사, 브라우저렌더 미실행.
실행 중 진행상태 파일의 중복저장 오류1회가 발생했다. 최초오류와80개 부분결과를 보존하고 동일조건 계산의 배열일치를 확인해 이어서 완료했다. 조건·문턱 변경 및 제외0. 실행 전 문법오류 정정도 보존했다.

다음 계획
1. 자동교정용 각도 경고는 보류하고, 차이 수치·관측비율·미확인시간을 함께 보여주는 연구용 표현을 유지한다.
2. 공개 여러 수행자의 정상적인 차이를 사람별로 분리해 추정한다. 한 사람을 뺀 참고로 허용범위를 만들고 빠진 사람에서 과도한 경고와 누락을 평가한다. 사람별 결과와 모든 관측 부족을 함께 공개하고 넓은 평균으로 짧은 구별동작을 덮지 않는다. 현재3단어만으로 모든수어 허용범위를 정하지 않는다.
3. 모양 근거가 적은 구간에서도 관측된 팔/손 위치를 정렬근거로 사용할 수 있는지 별도후보로 비교한다. 현재도 팔을 쓰지만 새 가중치/경로는 이번에 적용하지 않았다. 손모양 미확인은 그대로 둔다. 정상속도와 알려진 누락/순서교란 통제시험을 함께 사용한다.
4. 결과를 함께 확인한 뒤 다음 비교를 확정한다. TCN추가학습은 아직 하지 않는다.

기존 C·T3-F·웹 보존. 사용자·멘토촬영 접근/학습/튜닝/후보선택 없음. 추가학습과 분류추론0회.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
table=''.join(f'<tr><td>{label}</td><td>{g[k]["old"]} / 216</td><td>{g[k]["new"]} / 216</td><td>{100*g[k]["support"]:.1f}%</td></tr>' for k,label in zip(d.KINDS,labels))
style='''*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#203346;font:18px/1.7 "Malgun Gothic",sans-serif}main{max-width:1180px;margin:auto;padding:42px 28px 80px}h1{font-size:42px;line-height:1.3;margin:14px 0 24px}h2{font-size:27px;line-height:1.4}section{background:white;border-radius:16px;padding:28px 32px;margin:22px 0;border:1px solid #d9e1e8}img{width:100%;height:auto;border-radius:8px}.tag{font-size:14px;color:#47677c}p{margin:14px 0}.notice{padding:18px 22px;border-left:5px solid #c45c55;background:#fff1ed}.good{padding:18px 22px;border-left:5px solid #369e94;background:#edf9f6}table{width:100%;border-collapse:collapse;font-size:17px}th,td{text-align:left;padding:13px 10px;border-bottom:1px solid #d9e1e8}.muted{font-size:15px;color:#607282}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.box{background:#f3f6f8;padding:20px;border-radius:10px}a{color:#206a98}details{margin:16px 0}summary{cursor:pointer;font-weight:700}@media(max-width:750px){main{padding:20px 12px}section{padding:20px}h1{font-size:30px}.grid{grid-template-columns:1fr}table{font-size:14px}}'''
doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>다른 사람 참고 · DTW 관측 근거 검증</title><style>{style}</style><main>
<div class="tag">2026-10-22 · 공개 3단어 / 9명 / 27영상 · 연구용 검증</div>
<h1>동작이 다른 것과<br>잘 보이지 않는 것을 나눠 비교하기</h1>
<div class="notice"><b>다른 사람의 참고에서는 경고 감소가 확인되지 않았다.</b><br>현재 각도 기준을 바로 사용자 오류로 판단하는 데 쓰면 안 된다. 손가락·손 위치·팔 궤적과 관측 부족을 따로 보여주는 기능은 구현하고 검산했다.</div>
<section><h2>지금은 TCN이 아니라 DTW의 세부 분석 단계다</h2><div class="grid"><div class="box"><b>TCN</b><p>“어떤 수어 단어인가?”<br>C·T3-F 유지. 추가 학습 없음.</p></div><div class="box"><b>DTW</b><p>“다른 속도로 한 장면끼리 어느 시점을 연결할까?”<br>기존 정렬 경로 계산을 유지.</p></div><div class="box"><b>이번 보완</b><p>“연결한 손 모양·위치·팔 움직임은 얼마나 다르고, 어디는 확인하지 못했을까?”</p></div></div><p>현재 결과는 TCN 단어 인식률 향상 결과가 아니다. DTW 분석에서도 <b>차이가 난다는 사실과 잘못된 수행이라는 판정을 구분</b>하고 있다.</p></section>
<section><h2>다른 8명의 같은 단어를 참고로 비교했다</h2><p>‘수어·후원·멀다’ 3단어를 수행한 9명의 공개 TRAIN 영상 27개를 사용했다. 각 영상을 같은 단어의 다른 8명과 비교하여 <b>216조합</b>, 시간 조건 5가지를 적용해 <b>1,080조합</b>을 확인했다.</p><p>기존 관절 입력의 속도를 바꾼 시험이다. 영상을 새로 촬영하거나 MediaPipe로 다시 추출하지 않았다. 100개 단어 전체나 일반 사용자 전체를 검증한 결과는 아니다.</p></section>
<section><h2>같은 사람에서는 줄던 경고가 다른 사람에서는 줄지 않았다</h2><img src="visuals/summary.png" alt="조건별 경고 수와 항목별 관측 가능 비율"><table><tr><th>시간 조건</th><th>이전 경고</th><th>새 각도 범위 경고</th><th>각도 비교 가능</th></tr>{table}</table><p>새 기준은 속도를 바꾸기 전부터 <b>216 / 216조합</b>에 경고했다. 따라서 “속도를 바꿔도 추가 경고가 없다”는 결과 역시 좋은 증거가 아니다. 이미 모든 조합에 경고가 있기 때문이다.</p><p>기존 경고와 새 경고는 보는 현상도 다르다. 이 수치는 경고 발생 건수이며 수어 오답률이나 TCN 인식률이 아니다. 다른 사람끼리의 정확한 대응 시점과 허용 차이의 정답은 아직 없다.</p></section>
<section><h2>왜 이런 결과가 나왔을까?</h2><p>현재 손가락 모양의 범위는 <b>같은 장면을 두 번 검출한 결과의 차이</b>를 바탕으로 만든 것이다. “다른 사람이 이 정도 다르게 해도 정상이다”라는 범위가 아니다.</p><p>같은 단어여도 실제 동작과 촬영 방향이 조금씩 다르고, 관절 검출과 시간 정렬에도 오차가 있다. 지금의 범위 밖 차이에는 이것들이 섞여 있다. <b>범위 밖이라는 이유만으로 손을 잘못 움직였다고 말할 수 없다.</b></p><div class="notice">전체 영상을 한 번에 맞음/틀림으로 바꾸는 경고 기준은 보류한다. 두 검출 결과가 비슷하다고 실제 손 모양의 정확성이 보장되는 것도 아니다.</div></section>
<section><h2>손 모양·손 위치·팔 궤적은 따로 볼 수 있게 했다</h2><img src="visuals/position_and_evidence.png" alt="후원 두 사람의 손바닥 및 팔 궤적, 위치 차이와 각도 근거 부족 시간"><p>위 그래프는 두 사람의 손·팔이 지나간 경로이고, 아래 왼쪽은 DTW로 연결한 시점의 위치 차이다. 아래 오른쪽은 손가락 각도별로 근거가 있는 곳과 부족한 곳이다. 빈 선은 관측이 부족한 구간이며 그 사이를 임의로 연결하지 않았다.</p><p><b>팔은 원래 DTW 정렬에도 사용하고 있었다.</b> 이번에는 팔꿈치·팔 손목의 차이와 관측 상태를 별도로 드러냈다. 손바닥 위치도 손가락 모양의 신뢰도와 분리했다.</p><p>위치 차이 0.1은 영상에서 <b>어깨너비의 10분의 1</b>만큼 차이 난다는 뜻이다. “10% 틀렸다”는 뜻은 아니다. 위치·각도를 임의로 하나의 정답 점수에 합치지 않았다.</p><details><summary>이 예시의 두 공개 원본 장면 보기</summary><img src="visuals/public_pair_raw.jpg" alt="후원 공개 REAL01과 REAL05의 대응 후보 장면"><p class="muted">원본 두 사람, 각 3장면이다. 큰 흐름은 비슷해 보이지만 각 손가락 모양·위치·정확한 대응 시점의 의미 정답까지 판정하지 않았다.</p></details><p class="muted">팔 궤적은 기존 관측을 시간순으로 그렸다. 가까운 주변 점을 고르는 위치 차이 계산은 단조로운 동작 순서나 속도를 보장하지 않으므로, 순서 오류 판정에 직접 사용하지 않는다.</p></section>
<section><h2>팔이 보여도 손가락까지 확인한 것은 아니다</h2><p>원래 속도 조합에서 비교 가능한 비율은 <b>손가락 각도 63.0% · 손바닥 74.2% · 팔꿈치/팔 손목 99.4%</b>였다. 이것은 정확도가 아니라, 참고가 요구한 시간과 항목 중 관측 근거가 남아 있는 비율이다.</p><p>모든 연결 장면이 관측된 엄격한 비율도 따로 기록했다. 각도 62.3%, 손바닥 73.6%, 팔 99.4%다. 일부 장면만 보인 연결을 완전히 관측한 것으로 처리하지 않는다.</p><div class="good">손가락이 안 보이면 “손가락은 비교 근거 부족”. 팔이 보이면 “팔 위치는 비교 가능”. 두 결과를 그대로 함께 유지한다.</div></section>
<section><h2>경고가 사라진 1건도 실제 개선은 아니었다</h2><p>‘후원’ REAL01을 REAL05와 비교해 빠르게 재표본하자 지속 경고가 사라졌다. 그러나 손가락 비교 가능 비율은 <b>36.9% → 21.8%</b>로 줄었다.</p><img src="visuals/warning_disappears.png" alt="빠르게 재표본하자 관측 근거 부족이 늘면서 지속 경고가 사라진 사례"><p>전에 경고였던 40칸 중 <b>28칸은 근거 부족</b>, 11칸은 차이가 남았지만 연속 길이 기준 미달, 1칸만 주변 범위 안이었다. 회색이 늘고 빨간 경고가 없어졌다고 잘 맞았다고 바꾸지 않았다.</p><p class="muted">표본 사이의 양끝이 모두 유효할 때만 비교하도록 했기 때문에 시간 재표본만으로도 관측 근거가 줄 수 있다. 실제 웹캠을 빠르게 움직였을 때의 검출률 실험은 아니다.</p></section>
<section><h2>정상 손내림·한 손 동작·기존 교차는 보존했다</h2><p>손 전체 누락 45건, 짧은 누락 45건, 팔 누락 45건의 <b>135개 통제시험</b>을 별도로 확인했다. 손이 사라져도 비교할 참고 시간 범위는 줄지 않았고, 없는 근거를 0점 차이로 바꾸지 않았다.</p><p>손내림은 기존 참고측의 확인된 시간만 제외했다. 안 보인다는 이유로 손내림을 새로 추정하지 않았다. 기존 한 손 동작의 역할, 정상 손 교차의 좌우 좌표와 원본 입력도 그대로 보존했다.</p><p class="muted">통제시험은 기존 정렬 경로를 고정하고 관측 근거를 제거한 시험이다. 누락 이후 정렬 경로까지 새로 계산하는 성능이나 새로운 손 검출기의 교차 정확도까지 검증한 것은 아니다.</p></section>
<section><h2>다음은 ‘다른 사람도 정상일 수 있는 범위’를 검증해야 한다</h2><ol><li><b>현재 경고를 자동 교정에 사용하지 않는다.</b> 차이 수치·관측 비율·미확인 시간을 함께 유지한다.</li><li><b>사람별 정상 차이의 범위를 공개 자료로 평가한다.</b> 한 사람을 뺀 참고들로 범위를 만들고, 빠진 사람에서 과도한 경고가 줄어드는지 확인한다. 단어마다 예외를 넣거나 사용자 촬영으로 맞추지 않는다.</li><li><b>관측된 팔·손 위치를 정렬의 근거로 더 활용할지 별도 비교한다.</b> 손가락 모양을 확인 못한 부분은 계속 보류한다. 위치가 맞는다고 모양까지 맞다고 인정하지 않는다.</li><li><b>짧은 차이를 평균으로 없애지 않는지도 함께 확인한다.</b> 알려진 누락·정상 속도 변형·관측 누락 시험을 함께 두고, 범위를 넓히기만 해서 경고를 없애는 후보는 채택하지 않는다.</li></ol><p>이번 결과를 함께 확인한 후 다음 비교를 확정한다. <b>추가 학습은 아직 진행하지 않았다.</b></p></section>
<section><h2>검증과 보존 기록</h2><p>1,080조합과 135통제시험의 산식·마스크·범위·부분관측·DTW최적비용을 별도 코드로 검산했다. 공개 원본 6장면과 그래프 3개를 시각적으로 확인했다. 같은 Codex의 별도 수식 검산이며 수어 전문가의 의미정답 검토는 아니다.</p><p>중간 진행상태 파일 저장 충돌은 최초 오류와 부분결과 80개를 보존한 뒤 동일조건·배열일치 확인 후 복구했다. 결과 제외나 기준 변경은 없다.</p><p><b>C·T3-F·운영 웹 유지. 사용자·멘토 촬영 접근·학습·튜닝·선택 없음. 추가학습·분류추론 0회.</b></p><p class="muted">HTML 정적 링크 검사와 그림 시각 검토 완료. 브라우저 렌더는 실행하지 않았다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="ANALYSIS.json">집계와 사례</a> · <a href="VERIFY.json">독립 산식 검산</a> · <a href="FROZEN.json">고정한 조건</a> · <a href="RECOVERY_AUTHORIZATION.json">복구 기록</a></p></section></main></html>'''
(U/'RESULT_KO.html').write_text(doc,encoding='utf8')
links=re.findall(r'(?:src|href)="([^"]+)"',doc)
for link in links:assert (U/link).is_file(),link
d.write(U/'REPORT_CHECK.json',dict(status='PASS',at=d.now(),links=len(links),charts=3,raw_frames=6,browser_render=False,layout_repair='summary legend moved above bars; experimental outcomes unchanged'))
print('REPORT',U/'RESULT_KO.html')
