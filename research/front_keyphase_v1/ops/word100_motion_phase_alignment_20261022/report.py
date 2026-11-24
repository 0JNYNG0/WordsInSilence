"""Research report from verified frozen outputs; no model or score execution."""
import re
from html import escape,unescape
import study as d
U=d.U
a=d.read(U/'ANALYSIS.json');v=d.read(U/'VERIFY.json');assert v['status']=='PASS'
normal=[('fast075','영상 길이 0.75배'),('slow15','영상 길이 1.5배'),('pace','구간마다 속도 변화'),('pause','중간에 정지 추가')]
labels={'angle':'손가락 각도','palm':'손바닥 위치','elbow':'팔꿈치 위치','wrist':'팔 손목 위치'}
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def rate(n,d):return f'{100*n/d:.1f}%'
time_table=table(['정상 시간 변화','기존 평균 오차','보조 적용 평균 오차','개선 / 악화 / 동일'],[(label,f'{a["known"][k]["timing"]["moving"]["old"]["mae"]:.3f}프레임',f'{a["known"][k]["timing"]["moving"]["new"]["mae"]:.3f}프레임',f'{a["known"][k]["improved"]} / {a["known"][k]["worsened"]} / {a["known"][k]["unchanged"]}') for k,label in normal])
cg=a['cross']['identity']['groups']
cross_table=table(['항목','기존 지속 차이 경고','보조 적용 경고','비교 가능 비율 기존 → 보조'],[(labels[g],f'{x["old"]["warning"]}/216',f'{x["new"]["warning"]}/216',f'{100*x["old"]["coverage"]:.2f}% → {100*x["new"]["coverage"]:.2f}%') for g,x in cg.items()])
omit_table=table(['시간 구간 제거 시험','기존 새 차이 발생','보조 적용 새 차이 발생','새 검출 / 검출 소멸'],[(f'{label} · {size_label}',f'{s["old"]["variants_any_raw"]}/81',f'{s["new"]["variants_any_raw"]}/81',f'{s["newly_detected_variants"]} / {s["lost_detection_variants"]}') for folder,label in [('known','같은 원본'),('cross','다른 사람 참고')] for size,size_label in [('skip3','3프레임'),('skip20pct','동작 구간 약 20%')] for s in [a['omissions'][folder][size]]])
new_miss=a['omissions']['cross']['skip3']['new']['variants_no_new_raw'];old_miss=a['omissions']['cross']['skip3']['old']['variants_no_new_raw']
countbetter=sum(a['known'][k]['improved'] for k,_ in normal);countworse=sum(a['known'][k]['worsened'] for k,_ in normal);countequal=sum(a['known'][k]['unchanged'] for k,_ in normal)
sections=[]
def add(title,body):sections.append('<section><h2>'+title+'</h2>'+body+'</section>')
add('결론: 이번 추가 가중 방식은 운영에 적용하지 않았어',f'''
<p><b>손 위치와 움직임 방향을 추가로 강조했지만, 실제 대응 시점을 더 잘 맞춘다는 개선은 확인되지 않았어.</b> 정상 시간 변화 108개 중 평균 시점 오차가 줄어든 경우는 {countbetter}개, 늘어난 경우는 {countworse}개, 같은 경우는 {countequal}개였어. 전체 평균의 변화도 작아서 이를 큰 성능 하락으로 과장하지는 않아.</p>
<p>다른 사람 참고의 짧은 3프레임 제거 시험에서는 새 차이가 나타나지 않은 변형이 기존 {old_miss}개에서 {new_miss}개로 바뀌었어. 자세한 검출 변화와 한계는 아래에 모두 남겼어. <b>낮은 차이만 보고 정상 수행으로 판정하는 기능은 추가하지 않았어.</b></p>
<div class="verdict">C·T3-F와 운영 웹은 유지했어. 이번은 DTW 연구 비교이고, 신경망 학습·분류 추론은 0회야. 사용자·멘토 촬영 자료는 열거나 학습·튜닝·모델 선택에 사용하지 않았어.</div>
''')
add('1. 무엇을 바꿔 비교했나', '''
<p>목적은 위치가 잠깐 비슷해 보이는 서로 다른 단계 대신, <b>같은 움직임을 하고 있는 순간끼리</b> 연결하는 것이었어. 기존 DTW에도 팔 정보가 있었고, 이번에는 손바닥·팔꿈치·손목의 위치와 움직임 방향을 별도로 강조했어.</p>
<div class="flow"><div><b>현재 위치</b><br>손바닥·팔꿈치·손목이 몸을 기준으로 어디 있는가</div><div><b>들어오고 나가는 방향</b><br>바로 앞·뒤 프레임에서 어느 방향으로 움직였는가</div><div><b>시간 연결에 추가</b><br>기존 연결 비용에 위치·방향 근거를 더함</div></div>
<p>여기서 팔 궤적을 활용했다는 것은 <b>시간에 따른 팔꿈치·손목의 위치와 바로 앞뒤 이동 방향을 연결 기준에 더했다</b>는 뜻이야. 긴 동작 전체를 하나의 단계로 이해하거나 앞에서 수행한 단계를 별도로 기억하는 기능을 새로 만든 것은 아니야. 그 차이를 이번 결과 해석에서도 구분해야 해.</p>
<p>속도 크기를 똑같이 맞추도록 요구하지는 않았어. 멈춰 있는 손에는 방향을 지어내지 않았고, 관측이 끊어진 구간을 건너뛰어 궤적을 만들지도 않았어. 손을 내렸다는 기존의 명확한 근거가 있으면 그 손바닥은 새 항목에서 제외하되 팔 위치는 남겼어.</p>
<p><b>여러 사람의 참고 범위는 이전 실험 값 그대로 고정했어.</b> 범위를 넓히면서 시간 연결까지 바꾸면 무엇 때문에 결과가 달라졌는지 알 수 없기 때문이야. 좌표·좌우 손 배정·정규화·기존 DTW 가중치와 연결 제한도 유지했어.</p>
<details><summary>실행 전에 고정한 수식과 한계</summary><p>새 비용 = 기존 비용 + 0.25 × 위치 차이 + 0.25 × 방향 차이. 위치 차이는 어깨너비 기준 거리에서 손바닥 추정 반경을 뺀 뒤 d/(1+d), 방향 차이는 (1−cos)/2야. 서로 관측된 항목만 사용했어. 방향은 연속 프레임 모두 관측되고 이동량이 두 반경의 합+0.01 어깨너비보다 클 때만 계산했어. 이 수치는 결과를 보기 전 고정한 연구용 값이며 검증된 수어 허용 문턱이 아니야. 방향 근거가 없으면 추가 비용도 없어지므로, 근거 없는 구간으로 연결이 몰리는지 별도로 검사했어.</p></details>
''')
add('2. 어떤 공개 자료로 시험했나', '''
<p><b>수어·후원·멀다 3개 단어, 공개 수행자 9명, 원본 27개 영상</b>을 사용했어. 기존 재확인 관절 자료를 재사용했고 새 영상 다운로드나 MediaPipe 재검출은 하지 않았어.</p>
<ul><li><b>같은 원본 297개 비교</b>: 원래 영상·정상 속도 변화·짧거나 긴 구간 제거. 원본과의 실제 대응 시점을 생성 과정에서 알고 있어서 시간 오차를 계산할 수 있어.</li><li><b>다른 사람 2,376개 비교</b>: 각 영상을 나머지 8명의 참고 시간축에 연결. 시험 대상 수행자는 다중 참고 범위 구성에서 제외돼 있어.</li><li><b>관측 누락 후 재정렬 135개</b>: 손 전체·짧은 손 관측·팔 관측을 없앤 뒤 기존 방식과 후보 방식 모두 시간 연결을 다시 계산했어.</li></ul>
<p class="note">변형 시험 수가 많아도 서로 다른 원본은 27개야. 100개 단어 전체나 완전히 새로운 사람의 분류 정확도 검증은 아니야. 다른 사람끼리는 정확한 동작 단계별 정답 시점이 없기 때문에, 경고 감소만으로 정렬 정확도가 올랐다고 말하지 않아.</p>
''')
add('3. 정상 속도 차이: 실제 시간 연결이 더 정확해졌나',time_table+'''
<p>표는 원본에서 변화가 관측된 구간의 평균 대응 시점 오차야. 정지된 자세에는 같은 모양인 시간이 여러 개 있으므로, 전체 영상 오차만으로 판단하지 않았어. 원래 속도의 자기 자신 비교는 두 방식 모두 0프레임이었어.</p>
<img src="visuals/timing.png" alt="정상 시간변형의 실제 대응 시점 오차 비교">
<p><b>기존 방식이 이미 평균 0.3프레임 미만으로 연결하던 쉬운 시험에서는 추가 보조의 이득이 나타나지 않았어.</b> 같은 원본이므로 체형·촬영 각도·검출 오차가 공유되는 한계가 있어. 이 결과를 실제 사용자 전체의 정렬 성능으로 확대할 수는 없어.</p>
<details><summary>가장 좋아진 사례와 가장 나빠진 사례</summary><img src="visuals/improvement.png" alt="대응 시점 오차가 가장 감소한 사례"><img src="visuals/regression.png" alt="대응 시점 오차가 가장 증가한 사례"><p>예시는 설명을 위해 전체 결과 중 선택한 것이고, 후보 조건을 바꾸거나 특정 단어의 예외를 만드는 데 사용하지 않았어.</p></details>
''')
angle=cg['angle'];comp=angle['comparison'];cp=a['cross']['identity']
add('4. 다른 사람끼리: 차이와 근거 손실을 분리했어',cross_table+f'''
<img src="visuals/cross.png" alt="다른 사람의 경고 수와 공통 관측 및 근거 손실 비교">
<p>손가락 각도의 공통 관측 칸에서는 기존 차이가 사라진 칸 {comp['resolved']:,}개와 새 차이가 생긴 칸 {comp['added']:,}개가 함께 있었어. 기존에 비교 가능했던 칸 중 {comp['old_comparable_now_unknown']:,}개는 새 연결에서 근거 부족이 됐고, 그중 기존 차이가 있던 칸은 {comp['old_difference_now_unknown']:,}개였어. 이런 소멸을 개선으로 세지 않았어.</p>
<p>참고의 동작 구간이 시험 영상의 주석 동작 구간 밖에 연결된 비율은 {100*cp['old_proxy']['query_outside_core_fraction']:.2f}% → {100*cp['new_proxy']['query_outside_core_fraction']:.2f}%였어. 이는 준비·마무리 혼입을 살펴보는 대리 지표야. 주석 경계가 사람마다 다를 수 있어서 이 비율 자체를 정렬 오답률로 부르지 않아.</p>
<p>반대로 <b>시험 영상에서 손내림이 확인된 손을 참고의 활동 시간에 연결한 횟수는 {cp['old_proxy']['query_rest_to_active_handsteps']} → {cp['new_proxy']['query_rest_to_active_handsteps']}로 늘었어.</b> 같은 프레임이 여러 번 연결된 횟수를 포함하는 진단 수치이며 동작 오류 개수는 아니지만, 정상 손내림의 시간 연결까지 더 좋아졌다고 말할 수 없는 이유야.</p>
<p>현재 경로에서 방향을 비교할 근거가 있는 비율은 {100*cp['support']['direction']['old']:.2f}% → {100*cp['support']['direction']['new']:.2f}%였어. 정지·작은 움직임·위치 불확실성·미관측이 모두 영향을 주므로 손 검출률과는 다른 값이야. 방향 근거만으로 긴 동작 순서를 충분히 판단할 수 있다고 보지는 않아.</p>
<p class="note">이 비교에서는 참고 범위가 바뀌지 않았어. 따라서 변화는 새 시간 연결에 의한 것이지만, 실제로 더 올바른 단계끼리 연결됐는지는 위 숫자만으로 확정할 수 없어. 정상 손과 손 교차의 좌표·좌우 정보는 그대로 유지했으며, 새 검출기의 교차 정확도를 평가한 시험은 아니야.</p>
''')
om=a['omissions']['cross']['skip3']
add('5. 짧은 동작 누락: 덜 놓치게 됐나',omit_table+f'''
<img src="visuals/omissions.png" alt="같은 원본 및 다른 사람의 짧은 시간 구간 제거 감도 비교">
<p>다른 사람 참고의 ‘새 차이 발생’은 8개 중 하나라도 제거 부근에서 새 차이를 보인 경우야. 5개 이상 참고가 함께 새 차이를 보인 짧은 제거 변형은 {om['old']['variants_five_raw']}개 → {om['new']['variants_five_raw']}개였어. 한 참고만 반응한 경우와 여러 참고가 반응한 경우를 구분해야 해.</p>
<p>제거한 구간의 위치는 처음 정한 기존 원본 경로로 고정해서 평가했어. 새 경로를 본 뒤 유리한 위치로 평가 구간을 옮기지 않았어. 제거 부분 밖에 새 차이가 생긴 경우도 별도 기록했어.</p>
<p><b>짧게 빠진 움직임이 여전히 넓은 관측 범위 안에 들어가거나 DTW 연결에 흡수될 수 있어.</b> 위치·방향을 강조한다고 이 문제가 자동으로 해결되지는 않았어. 또 3프레임을 지웠다는 사실과 수어의 필수 단계를 지웠다는 것은 달라. 이번은 시간 구간 제거에 대한 반응 검사이며, 필수 수어 동작 검출 정확도가 아니야.</p>
'''+('''<details><summary>새 차이를 놓친 짧은 제거 사례</summary><img src="visuals/missed_omission.png" alt="짧은 구간 제거에도 새 차이가 나타나지 않는 사례"><p>위 그래프의 원본 프레임 연결은 사람 간 언어적 단계의 정답 시점이 아니야. 제거 전후의 연결 변화와 차이·근거 부족을 확인하는 용도야.</p></details>''' if (U/'visuals/missed_omission.png').exists() else ''))
missing_table=table(['관측 제거','시험 수','참고 비교 범위 유지','원래 촬영의 미관측 보존'],[(label,x['n'],'확인' if x['all_reference_domain_unchanged'] else '실패','확인' if x['all_forced_native_unknown_preserved'] else '실패') for k,label in [('wholehand','손 전체'),('shortgap','손의 짧은 3프레임'),('wholearm','팔 전체')] for x in [a['missing'][k]]])
hiddenold=sum(x['counts']['old_forced_native_rows_only_mapped_to_excluded_reference'] for x in a['missing'].values());hiddennew=sum(x['counts']['new_forced_native_rows_only_mapped_to_excluded_reference'] for x in a['missing'].values())
add('6. 관측 부족은 정상으로 넘기지 않았나',missing_table+f'''
<p>이번에는 손바닥·각도 정보만 없앤 것이 아니라, 기존 DTW가 보던 해당 손의 특징과 관측 표시도 함께 없애고 시간 연결을 다시 계산했어. 숨긴 손 정보가 옛 입력에 남아서 몰래 도움을 주지 않도록 했어.</p>
<p>시간을 다시 맞추면 관측이 없던 시험 영상의 일부 시간이 참고의 손내림·비교 제외 시간으로 연결될 수 있었어. 그런 시험 시간행은 기존 {hiddenold:,}개, 후보 {hiddennew:,}개였어(서로 다른 통제 시험을 합친 행 수). 그래서 <b>참고에 맞춘 결과뿐 아니라 원래 촬영 시간표의 미관측도 따로 보존</b>했어. 모든 관측 제거 시험에서 이 기록은 유지됐어.</p>
<p>정상 원본의 확인된 손내림은 그대로 사용했어. 다만 관측을 일부러 지운 통제 시험에서는 손내림을 확인하던 근거도 함께 없애므로 ‘모르겠다’가 늘 수 있어. 이를 잘못된 수어 동작이라는 경고로 바꾸지는 않았어.</p>
<p><b>범위 안 / 범위 밖 / 비교 근거 부족</b>은 계속 구분되고, 전체 동작이 옳다는 판정값은 설정하지 않았어. 따라서 이번 검산 통과는 계산·근거 보존의 통과이지, 올바른 수어 판정 기능의 완성을 뜻하지 않아.</p>
''')
add('7. 왜 개선이 확실하지 않았고, 다음에는 무엇을 해야 하나', '''
<p><b>확인된 사실</b>은 기존 방식도 같은 원본 시간변형을 이미 가깝게 연결했고, 새 위치·방향 가중만으로는 결과가 더 좋아지지 않았다는 점이야. 또 참고 범위는 고정돼 있어서, 이전의 넓은 범위가 짧은 차이를 흡수하는 한계는 그대로 남아 있어.</p>
<p><b>아직 확정하지 못한 원인</b>은 위치 강조와 방향 강조 중 어느 쪽이 영향을 줬는지야. 이번에는 둘을 함께 추가했으므로 원인을 어느 하나로 단정하지 않아. 짧은 프레임 간 방향은 속도 재표본화·관절 흔들림의 영향을 받을 수 있고, 현재 위치가 비슷해도 동작 단계는 다를 수 있어.</p>
<ol><li>이번 후보는 연구 결과로 보존하고 운영에는 적용하지 않는 것이 맞겠어.</li><li>다음에는 가중치 조정이나 추가 학습부터 하지 말고, 남은 짧은 누락 사례에서 <b>구별 정보가 실제로 관측됐는지 → 시간 연결에서 압축됐는지 → 비교 범위에 흡수됐는지</b>를 분리하자.</li><li>그 결과를 바탕으로 시점별 최솟값·최댓값만 보는 대신, 실제 한 사람이 수행한 짧은 연속 동작과 그 앞뒤 순서를 함께 비교하는 후보가 필요한지 결정하자. 정상 속도·손내림·반복을 함께 검증하고, 특정 단어 예외는 만들지 않아.</li></ol>
<p>이 방향도 아직 성능이 좋아진다고 보장하는 제안은 아니야. 지금 확인한 결과를 함께 검토한 다음 다음 작업과 추가 학습 여부를 정하면 돼.</p>
''')
add('검증과 파일', '''
<p>Codex가 후보 구현·실행을 맡았고, 별도 Codex 에이전트가 비용·경로·관측·시간 오차 검산 코드를 작성했어. 다른 에이전트가 평가 집계를 작성했으며, 최종 수치와 그림은 부모 에이전트가 확인했어. 수어 전문가의 의미 판정은 아니야.</p>
<p>2,808개 결과의 새 비용·경로·각도/위치 비교·누락 처리와 보호 파일 해시를 확인했어. 실행 전 합성 시험 10개 통과 기록은 해시 확인 후 재사용했어. 원본 관측·기존 동결 결과·운영 모델을 바꾸지 않았어.</p>
<p>첫 검산에서는 관측 제거 시험의 계산에 약 0.00000001 차이가 생겼어. 별도 검산 코드가 평균을 계산할 때 사용한 숫자 정밀도 차이임을 같은 사례로 재현했어. 실험 결과와 통과 허용치는 바꾸지 않았고, 최초 오류를 보존한 채 검산 계산만 바로잡아 남은 135개를 확인했어. 먼저 통과한 2,673개의 전체 검산은 반복하지 않았어.</p>
<p><a href="FROZEN.json">고정 실험 조건</a> · <a href="VERIFY.json">독립 계산 검산</a> · <a href="ANALYSIS.json">전체 결과</a> · <a href="DESIGN_REVIEW.json">설계 검토</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p>
''')
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손·팔 위치와 방향을 활용한 시간 연결 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f4f7;color:#223244;font:17px/1.8 'Malgun Gothic',sans-serif}main{max-width:1120px;margin:auto;padding:36px 24px}header{background:#173e4d;color:white;border-radius:18px;padding:36px}header p{color:#d1e6e9}h1{font-size:34px;line-height:1.4}h2{font-size:24px;line-height:1.5}section{background:white;border:1px solid #dce5ea;border-radius:14px;padding:28px;margin:22px 0}img{width:100%;height:auto;margin:20px 0;border:1px solid #e1e7ec;border-radius:8px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce5ea}th{background:#edf5f5}p{margin:14px 0}.note{font-size:15px;color:#506174;border-left:4px solid #a9c1d0;padding-left:15px}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.flow div{padding:17px;border-radius:10px;background:#edf5f4}.verdict{padding:18px;border-left:5px solid #b8893e;background:#fff3dd}li{margin:10px 0}summary{cursor:pointer;font-weight:bold;color:#236b7e}a{color:#236b7e}.tag{font-size:14px;color:#b2d9df}@media(max-width:700px){main{padding:15px}section,header{padding:20px}h1{font-size:27px}.flow{grid-template-columns:1fr}table{min-width:580px}}
</style><main><header><div class="tag">WORDS IN SILENCE · DTW 연구 비교 · 2026.10.22</div><h1>손·팔의 위치와 움직임 방향으로<br>같은 동작 순간을 더 잘 찾을까?</h1><p>시간 연결만 변경한 고정 비교 · 정상 속도와 짧은 누락·관측 부족을 함께 검증</p></header>'''+''.join(sections)+'</main></html>'
plain=html.split('<main>',1)[1]
plain=re.sub(r'</(?:p|h1|h2|li|tr|section|header)>','\n',plain)
plain=re.sub(r'<[^>]+>',' ',plain);plain=re.sub(r'[ \t]+',' ',plain)
for name,content in [('RESULT_KO.html',html),('RESULT_KO.txt',unescape(plain))]:
    with (U/name).open('x',encoding='utf8') as out:out.write(content)
links=re.findall(r'(?:src|href)="([^"]+)"',html)
for link in links:assert (U/link).is_file(),link
d.write(U/'REPORT_CHECK.json',dict(status='PASS',at=d.now(),links=links,browser_render=False,static_link_check=True))
print('REPORT WRITTEN',flush=True)
