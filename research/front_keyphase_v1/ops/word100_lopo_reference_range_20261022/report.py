"""Render the completed frozen research; never rerun extraction or scoring."""
import re
from html import escape
import study as d

U=d.U
a=d.read(U/'ANALYSIS.json'); v=d.read(U/'VERIFY.json')
assert v['status']=='PASS'
g=a['groups']['identity']['groups']
labels={'angle':'손가락 각도','palm':'손바닥 위치','elbow':'팔꿈치 위치','wrist':'팔 손목 위치'}
def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
metrics=table(['비교 항목','단일 참고','다중 사람 범위','비교 가능한 칸 비율'],[(labels[k],f'{g[k]["base_warning"]}/216',f'{g[k]["warning"]}/216',f'{g[k]["coverage"]*100:.1f}%') for k in labels])
people=table(['참고에서 뺀 수행자','단일 참고 경고','다중 범위 경고','각도 비교 가능 비율'],[(f'공개 수행자 {p}',f'{r["groups"]["angle"]["base_warning"]}/24',f'{r["groups"]["angle"]["warning"]}/24',f'{r["groups"]["angle"]["coverage"]*100:.1f}%') for p,r in a['people'].items()])
timing=table(['시간 조건','단일 참고 각도 경고','다중 범위 각도 경고','각도 비교 가능 비율'],[(name,f'{a["groups"][k]["groups"]["angle"]["base_warning"]}/216',f'{a["groups"][k]["groups"]["angle"]["warning"]}/216',f'{a["groups"][k]["groups"]["angle"]["coverage"]*100:.1f}%') for k,name in [('identity','원래 속도'),('fast075','길이를 0.75배로 줄임'),('slow15','길이를 1.5배로 늘림'),('pace','구간별 속도 변화'),('pause','중간에 15프레임 정지 추가')]])
omissions=table(['제거 조건','시험 변형 수','1개 이상 참고에서 새 차이','5개 이상 참고에서 새 차이','어느 참고에서도 새 차이 없음'],[(name,s['n_source_variants'],s['variants_any_raw'],s['variants_five_raw'],s['variants_no_added_difference']) for k,name in [('skip3','짧은 3프레임'),('skip20pct','주석 동작 구간의 약 20%')] for s in [a['omission'][k]]])
sections=[]
def section(title, content): sections.append('<section><h2>'+title+'</h2>'+content+'</section>')
section('1. 이번 작업은 DTW 비교 기준을 만드는 연구야', '''
<p>TCN은 <b>“어떤 수어 단어인가?”</b>를 고르고, 지금 다루는 DTW 분석은 <b>“참고 동작과 어느 부분이 다른가?”</b>를 확인하는 쪽이야. 이번에는 여러 사람이 같은 단어를 수행한 자료에서 차이의 범위를 만들었어.</p>
<div class="steps"><div><b>9명 중 1명을 시험 대상으로</b><br>그 사람 자료는 범위를 만드는 데 전혀 넣지 않음</div><div><b>나머지 8명으로 참고 범위 구성</b><br>같은 시점으로 연결된 각도와 위치를 모음</div><div><b>빼 둔 사람에게 적용</b><br>9명 모두 돌아가며 같은 절차 수행</div></div>
<p>예를 들어 다른 사람들이 같은 순간의 손가락을 조금씩 다른 각도로 굽혔다면, 한 사람의 각도와 똑같아야 한다고 요구하지 않고 그 관측 범위를 함께 본다는 뜻이야. 다만 이 범위는 <b>공개 영상에서 관측한 범위</b>이지, 수어가 올바르다고 검증된 허용 범위는 아직 아니야.</p>
<p class="note">공개 TRAIN 자료의 수어·후원·멀다 3개 단어 × 9명 = 27개 영상을 사용했어. 100개 단어 전체 시험은 아니야. 이 사람들은 과거 모델 학습·개발에 사용된 사람들이므로, 이번 시험을 ‘모델이 전혀 본 적 없는 사람의 인식 정확도’라고 표현하면 안 돼. 이번에 제외한 것은 현재 참고 범위를 만드는 과정이야.</p>
''')
section('2. 비교할 수 있는 부분과 모르는 부분을 먼저 나눴어', '''
<p>손마다 손가락 각도 15개, 손바닥 위치, 팔꿈치 위치, 팔 손목 위치를 따로 계산했어. 각도는 도(°), 위치는 어깨너비를 기준으로 정리한 좌표를 사용해. 서로 다른 단위를 한 점수로 섞지 않았어.</p>
<ul><li>같은 시간·손·항목에서 최소 5명의 완전한 관측이 있어야 참고 범위를 만들었어. 8명의 관측 최솟값~최댓값을 사용했으며, 결과를 본 뒤 기준을 바꾸지 않았어.</li><li>시험 영상에서 연결된 관측이 일부라도 빠졌다면 그 부분은 <b>비교 근거 부족</b>으로 남겨. 팔이 보인다고 손가락 모양까지 확인했다고 하지 않아.</li><li>참고에서 확인된 손내림 시간과 기존 한 손/양손 구분은 유지했어. 시험 영상의 손이 사라졌다는 이유로 비교해야 할 시간대를 줄이지 않았어.</li><li>한두 프레임의 차이도 기록해. 3프레임 이상 이어지는 차이는 따로 집계하되, 짧다는 이유로 원래 차이 기록을 지우지 않았어.</li></ul>
<p class="note">기존 관절 좌표·좌우 배정·손 교차·시간 정렬 방식은 그대로 사용했어. 새 손 검출기를 평가한 것이 아니라, 같은 관측 위에서 참고 범위를 바꾼 비교야. 손내림과 필수 손 구분의 기존 추정이 수어적으로 모두 맞다는 새 보증은 아니야.</p>
''')
section('3. 한 사람에게만 맞추던 경고는 줄었어', metrics+'''
<p>표의 수는 원래 속도의 216개 비교 중, 해당 항목에서 <b>3프레임 이상 지속되는 차이가 하나라도 있었던 비교 수</b>야. 한 사람당 3단어 × 8개 참고 시간축 = 24개 비교이고, 9명 합계가 216개야. 영상 216개나 서로 독립적인 참가자 216명이라는 뜻은 아니야.</p>
<img src="visuals/summary.png" alt="항목별 지속 차이 경고와 감소 원인 분리">
<p>각도 경고는 216개에서 163개로 줄었어. 아홉 사람 모두 감소했지만, 여전히 상당수 비교에 차이가 남아 있어. <b>정답률이 올랐다는 결과가 아니라, 같은 단어의 사람 간 차이를 덜 엄격하게 취급하게 된 결과</b>야. 어느 경고가 실제 오류인지에 대한 수어 정답 주석은 없어.</p>
<p>기존에 범위 밖이던 각도 비교 칸 75,604개를 따로 확인했어. <b>62,608개는 관측을 유지하면서 다중 사람 범위 안</b>으로 들어왔고, <b>5,820개는 근거 부족으로 바뀌었어</b>. 나머지 7,176개는 함께 관측 가능한 칸에서도 차이가 남았어. 근거 부족으로 바뀐 것을 개선으로 세지 않았어.</p>
<p>전체 필요한 각도 비교 칸 중 실제 비교할 수 있던 비율은 <b>63.8%</b>야. 나머지 36.2%는 참고 자료 또는 시험 영상의 근거가 충분하지 않았어. 이 비율은 인식률이나 동작의 정확도가 아니야.</p>
<details><summary>한 사람씩 제외한 결과 보기</summary>'''+people+'''</details>
<img src="visuals/normal_range.png" alt="다른 여덟 명 범위와 제외된 수행자 관측의 비교">
<p class="caption">초록색은 다른 8명의 관측 범위, 파란색은 시험 영상의 연결된 관측이야. 아래 회색은 모르는 부분으로 남겨 두었어. 초록색 영역에 들어온다고 전체 동작이 정답이라는 뜻은 아니야.</p>
''')
section('4. 속도 차이에 대한 경고도 줄었지만 완전히 안정적이지는 않아', timing+'''
<p>같은 영상의 시간 간격을 바꾼 조건에서도 단일 참고보다 경고가 적었어. 다만 중간에 잠깐 멈추는 조건에서는 다중 범위의 각도 경고가 163개에서 176개로 늘었어. 따라서 <b>정상적인 속도 차이를 완전히 무시할 수 있는 단계는 아니야</b>.</p>
<p class="note">이 시험은 저장된 관측의 시간을 바꾼 것이야. 실제로 사람이 빨리 움직일 때 카메라 흔들림이나 MediaPipe 미검출이 늘어나는 현상을 재현한 시험은 아니야. 시간 재표본화 시 필요한 이웃 관측을 모두 요구하므로 비교 가능 비율도 달라질 수 있어.</p>
''')
section('5. 짧은 동작을 놓칠 위험은 남아 있어', omissions+'''
<p>각 영상의 주석 동작 구간 25%·50%·75% 부근을 정해 짧은 부분 또는 더 긴 부분을 제거했어. 영상마다 3가지라서 각 조건에 81개 변형이 있고, 각각 다른 8개 참고 시간축에 비교했어. 제거 위치는 결과를 보고 고르지 않았어.</p>
<img src="visuals/omissions.png" alt="짧거나 긴 구간 제거 후 참고별 새 차이 집계">
<p><b>짧은 3프레임을 지운 81개 중 12개에서는 어느 참고에서도 해당 부분에 새 범위 밖 차이가 나타나지 않았어.</b> 넓은 범위 안에 들어가 버리거나 시간 연결이 차이를 흡수할 수 있다는 뜻이야. 1개 참고만 차이를 발견한 결과와, 여러 참고가 함께 차이를 발견한 결과도 구분했어.</p>
<p>더 긴 부분을 지웠을 때는 81개 모두 적어도 한 참고에서 새 차이가 나왔지만, 이걸 필수 동작 검출률 100%라고 부를 수는 없어. 주석의 동작 구간을 지운 것이지, 그 모든 부분이 언어적으로 반드시 필요한 동작이라는 확인을 받은 것은 아니야. 사람 간 정확한 대응 시점도 없어서, 원본의 기존 시간 연결을 기준으로 근처를 확인했어.</p>
<img src="visuals/missed_all_references.png" alt="짧은 구간을 제거해도 여러 사람 범위 안에 남는 사례">
<p class="caption">‘수어’ 공개 수행자 1의 3프레임 제거 사례. 8개 참고 모두에서 제거 부분에 새 차이가 나타나지 않았고, 새 미관측도 없었어. 따라서 모든 누락을 단순히 ‘손이 안 보여서’라고 설명할 수는 없어. 이 장면을 필수 수어 단계의 누락이라고 확정한 것은 아니야.</p>
<details><summary>이 사례의 공개 원본 장면 보기</summary><img src="visuals/short_omission_raw.jpg" alt="제거 전후 공개 원본 세 장면"><p>원본 38·40·42프레임. 가운데는 제거 구간에 포함된 장면이야. 큰 움직임은 보이지만, 이 짧은 구간의 언어적 필수 여부는 이 이미지 세 장만으로 판정하지 않아.</p></details>
''')
section('6. 관측 부족을 정상으로 넘기지 않는 검사는 통과했어', '''
<ul><li>손 전체 관측 제거 45개, 짧은 손 관측 제거 45개, 팔 관측 제거 45개로 <b>135개 통제 시험</b>을 했어.</li><li>손이 없어져도 참고가 요구하는 비교 시간대와 항목 수는 그대로 유지됐어. 관측이 없으면 낮은 차이 또는 정상 대신 <b>근거 부족</b>으로 남았어.</li><li>관측이 없는 부분의 차이는 0으로 채우지 않았어. 보이는 다른 손·팔 항목은 유지했어.</li><li>별도 작은 합성 시험에서는 한 프레임 차이 보존, 3프레임 차이 검출, 4명만 관측된 참고의 보류, 부분 누락을 정상으로 보지 않음을 검산했어.</li></ul>
<p class="note">135개 시험은 시간 연결 경로를 고정한 근거 처리 검사야. 손이 사라진 뒤 시간 정렬 자체가 달라지는 모든 경우까지 검증한 것은 아니야. 이 제한은 다음 검사에서 따로 다뤄야 해.</p>
''')
section('7. 결론과 다음 한 작업', '''
<div class="verdict"><b>다중 사람 범위는 개인차를 덜 엄격하게 비교하는 데 도움이 됐어. 하지만 아직 자동으로 “잘 따라 했다”를 결정할 기준으로 쓰면 안 돼.</b></div>
<p>지금 범위는 각 시점·각도·좌표별 최솟값과 최댓값이야. 그래서 서로 다른 사람들이 한 동작들을 넓게 합치면서, 실제로는 아무도 하지 않은 손 모양 조합이나 짧게 빠진 움직임도 범위 안에 들어갈 수 있어.</p>
<p>추가 점검에서는 다른 사람의 준비·마무리 구간이 활동 구간에 연결되어 각도 범위의 끝값을 결정한 칸도 발견했어(참고가 충분한 각도 칸 175,603개 중 16,392개). 주석 경계 차이도 있으므로 전부 잘못된 정렬이라고 단정하지 않지만, <b>같은 동작 단계끼리 비교하고 있는지</b> 더 확인해야 하는 근거야. 이 결과를 보고 이미 실행한 기준을 바꾸지는 않았어.</p>
<ol><li><b>같은 단계끼리 연결되는지부터 보완</b>: 손 위치·팔 궤적·움직임 방향의 변화를 함께 보고 준비/마무리와 실제 동작이 섞이는 연결을 확인해. 기존 시간 연결과 같은 공개 자료에서 비교해.</li><li><b>한 점의 범위와 움직임 순서를 분리</b>: “어느 범위에 있었나”에 더해 “그 위치를 거쳐 어느 방향으로 움직였나”를 확인하는 연구 후보를 만들자. 관측이 없으면 추측으로 통과시키지 않아.</li><li><b>정상 속도·손내림·교차와 짧은 제거를 함께 시험</b>: 경고 수만 낮추지 않고 정상 동작 보존과 누락 보존을 같이 확인해. 관측 누락 후 재정렬되는 경우도 포함해야 해.</li></ol>
<p>전 단어에 수동 규칙을 붙이거나 특정 단어에 예외를 만드는 방향은 아니야. 위 연구 비교가 개선된 후 범위를 넓히고, 추가 학습 여부는 결과를 함께 확인한 뒤 결정하면 돼.</p>
<p><b>C·T3-F와 운영 웹은 그대로야. 이번 신경망 학습·분류 추론은 0회이고, 사용자·멘토 촬영은 열거나 학습·튜닝·선택에 사용하지 않았어.</b></p>
''')
section('검증 근거', '''
<p>216개 참고 범위의 수행자 제외와 범위 계산, 2,376개 비교의 관측·차이 계산, 재사용 경로 1,080개, 새 제거 시험 경로 1,296개의 저장 비용 최적값, 통제 시험 135개와 보호 파일 32개를 검산했어. 구현과 별도 산식 검산은 모두 Codex가 수행했으며, 수어 전문가의 의미 판정이나 다른 기관의 독립 검증은 아니야.</p>
<p>시각 자료는 로컬 이미지로 확인했고 HTML 파일 연결을 검사했어. 브라우저 화면 렌더 검사는 별도로 하지 않았어.</p>
<p><a href="FROZEN.json">실행 전 고정 조건</a> · <a href="VERIFY.json">별도 산식 검산</a> · <a href="ANALYSIS.json">전체 집계</a> · <a href="RANGE_AUDIT.json">범위 원천 점검</a> · <a href="SUPPLEMENT.json">짧은 제거 한계</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p>
''')
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>여러 사람 참고 범위 · 한 사람 제외 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f4f7;color:#202d3c;font:17px/1.8 'Malgun Gothic',sans-serif}main{max-width:1120px;margin:auto;padding:42px 24px}header{background:#173b4b;color:white;padding:38px;border-radius:18px}header p{color:#d7e9eb}.tag{font-size:14px;letter-spacing:.08em;color:#a8d8d7}h1{font-size:34px;line-height:1.4;margin:12px 0}h2{font-size:25px;line-height:1.5}section{background:white;padding:30px;margin:22px 0;border:1px solid #dde5e9;border-radius:15px}p{margin:14px 0}img{display:block;width:100%;height:auto;margin:22px 0;border:1px solid #e1e7ec;border-radius:8px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:16px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e0e7ed}th{background:#edf5f4}.note,.caption{font-size:15px;color:#4b5d6e}.note{border-left:4px solid #b4c9d4;padding-left:15px}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.steps div{background:#edf5f4;border-radius:10px;padding:18px}li{margin:10px 0}.verdict{background:#fff4df;border-left:5px solid #c69237;padding:18px}summary{cursor:pointer;color:#1c6c7d;font-weight:bold}a{color:#1c6c7d}footer{font-size:14px;color:#506170}@media(max-width:700px){main{padding:16px}section,header{padding:21px}.steps{grid-template-columns:1fr}h1{font-size:27px}table{min-width:560px}}
</style><main><header><div class="tag">WORDS IN SILENCE · DTW 연구 결과 · 2026.10.22</div><h1>여러 사람의 차이를 반영하면<br>비교 기준이 더 나아질까?</h1><p>한 사람씩 참고에서 제외한 공개 자료 검증<br>개인차 경고 감소 확인 · 짧은 구간 누락 판단은 아직 불완전</p></header>'''+''.join(sections)+'''<footer>연구 후보 결과이며 운영 모델·웹에는 적용하지 않았습니다.</footer></main></html>'''
txt=re.sub('<[^>]+>',' ',html.split('<main>',1)[-1])
txt=re.sub(r' +',' ',txt)
from html import unescape
for name,content in [('RESULT_KO.html',html),('RESULT_KO.txt',unescape(txt))]:
    with (U/name).open('x',encoding='utf8') as out:out.write(content)
links=re.findall(r'(?:src|href)="([^"]+)"',html)
for link in links: assert (U/link).is_file(),link
d.write(U/'REPORT_CHECK.json',dict(status='PASS',at=d.now(),links=links,browser_render=False,static_link_check=True))
print('REPORT WRITTEN',flush=True)
