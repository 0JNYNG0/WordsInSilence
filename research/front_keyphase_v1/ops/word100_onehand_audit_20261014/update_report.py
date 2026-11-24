"""Add actual user trial results to the previously requested local presentation."""
import json,re
from pathlib import Path
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
D=ROOT/'Docs/interim_report_20261014/web'
a=json.loads((U/'AUDIT.json').read_text('utf-8'));check=json.loads((U/'INDEPENDENT_CHECK.json').read_text('utf-8'))
assert check['status']=='PASS'
before=U/'presentation_before';before.mkdir(exist_ok=True)
for filename in ['index.html','report.css']:
    assert not (before/filename).exists()
    (before/filename).write_bytes((D/filename).read_bytes())
html=(D/'index.html').read_text('utf-8')
html=html.replace('<a href="#s9">다음 단계</a>','<a href="#s9">실사용 12회</a><a href="#s10">다음 단계</a>')
html=html.replace(' / 09</span>',' / 10</span>')
html=html.replace('<article id="s9"','<article id="s10"').replace('09 · 후속 계획','10 · 후속 계획').replace('09 / 10</span>','10 / 10</span>')
rows=''
for i in range(6):
    cells=''
    for r in (a['rows'][i],a['rows'][i+6]):
        cls='hit' if r['rank']==1 else 'miss'
        cells+=f'<td>{r["label"]}</td><td class="{cls}">{r["rank"]}위</td>'
    rows+=f'<tr>{cells}</tr>'
slide=f'''<article id="s9" class="page"><div class="slide user-audit"><div class="slide-top"><span>09 · 실제 웹캠 시험</span><span>학습 미포함 개발 참여자 1인 · 2026.10.14</span></div><h2 class="editable">12표현을 직접 수행하니, 관측 조건의 차이가 드러남</h2><div class="user-kpis"><div><b>50.0%</b><span>정답 1위 · 6 / 12</span></div><div><b>66.7%</b><span>정답 5위 이내 · 8 / 12</span></div><div><b>12회</b><span>12표현 × 각 1회 · 후보 100개</span></div></div><div class="audit-cols"><div><h3>전체 촬영의 목표 단어 순위</h3><table class="audit-table"><thead><tr><th>표현</th><th>순위</th><th>표현</th><th>순위</th></tr></thead><tbody>{rows}</tbody></table></div><div class="audit-cause"><h3>쓰지 않는 손이 안 보이면?</h3><p>학습 영상: 쉬는 왼손도 계속 보임<br>웹캠: 멀다·공항·별의 왼손 관측 3~8%</p><table class="audit-table"><thead><tr><th>공개 영상</th><th>원본 순위</th><th>왼손만 제거</th></tr></thead><tbody><tr><td>멀다</td><td>모두 1위</td><td>56~62위</td></tr><tr><td>공항</td><td>모두 1위</td><td>13~16위</td></tr><tr><td>별</td><td>모두 1위</td><td>5~10위</td></tr></tbody></table><p class="audit-takeaway">공개 3표현 × 3명에서도 현상 재현<br><b>현재 모델의 손 누락 민감성 확인</b></p></div></div><p class="foot editable">1인·표현별 1회 사례이며 일반 사용자 정확도는 미확정. 사용자 자료는 학습·튜닝·후보 선택에 미사용.</p><span class="page-no">09 / 10</span></div><section class="script"><div class="script-head"><b>발표 대본</b><button class="copy">대본 복사</button></div><p class="editable">공개 자료 검증과 별도로 제가 웹에서 제공하는 12개 표현을 한 번씩 촬영했습니다. 모델은 매번 100개 후보 전체 중에서 분류했고, 목표 단어가 1위인 것은 6회, 5위 안에 든 것은 8회였습니다. 저는 학습 자료에 포함되지 않았지만 개발 과정에 참여한 한 사람이므로, 일반적인 새 사용자 성능이 증명됐다고 말할 수는 없습니다. 특히 멀다, 공항, 별은 한 손으로 수행했을 때 반대손이 거의 보이지 않았습니다. AI Hub 학습 영상은 쉬는 손도 계속 화면에 남아 있었습니다. 원인을 확인하려고 공개 검증 영상에서 다른 입력은 유지하고 왼손 정보만 제거했습니다. 이 세 표현의 9개 영상은 원래 모두 정답이 1위였는데, 제거 후에는 모두 1위를 놓쳤습니다. 따라서 사람을 나누는 검증뿐 아니라, 실제 촬영의 손 관측 조건 변화에도 견디는지를 확인해야 한다는 문제를 찾았습니다. 양손이 보였던 잘하다, 믿다, 감사의 오답은 이 설명만으로 확정할 수 없어 별도로 남겼습니다.</p><details><summary>결과의 해석과 DTW 관측 상태</summary><p>순위는 저장된 원래 100개 출력으로 집계했다. 성공한 촬영만 골라 쓰거나 사용자 영상을 재추론하지 않았다. 공개 영상 개입은 이미 고정된 모델에 대한 원인 진단이며, 새 모델의 개선 결과가 아니다. 멀다·공항·별의 DTW는 왼손 근거 부족으로 종합 판단 보류, 오른손 비교는 남아 있다. 다른 표현에서도 참고 범위를 넘은 차이가 있어, 그 초과를 곧바로 실제 동작 오류로 간주하지 않는다.</p></details></section></article>
'''
html=html.replace('<article id="s10"',slide+'\n<article id="s10"',1)
html=html.replace('지금 화면은 기존 감사 촬영의 표시 예시이며, 새로운 실사용 시험의 통계는 실제 시험 후 별도로 집계합니다.','지금 화면은 기존 감사 촬영의 표시 예시입니다. 새 실사용 12회 결과는 9장에서 별도로 보여드립니다. 관측되지 않은 손은 일치한 것으로 처리하지 않고 판단을 보류합니다.')
html=html.replace('<h3>AI Hub 단어 범위 확대</h3><p>남은 약 2,900코드의 라벨·표현·품질 확인 → 사용 어휘 단계적 확대</p>','<h3>실제 촬영의 손 누락 조건부터 보완</h3><p>공개 학습 자료에서 쉬는 손·필수 손 구분 → 누락에 강한 후보 검증<br>기존 인식 성능과 양손 표현의 회귀를 함께 확인</p>')
html=html.replace('<h3>모두의 말뭉치 활용 준비</h3><p>2022–24 주석 JSON 329,401개 조사 완료 → 같은 연도 원영상 연결<br>2025 신청분은 내부 구성 확인 후 활용 여부 판단</p>','<h3>AI Hub·말뭉치 자료의 단계적 활용</h3><p>남은 AI Hub 코드의 라벨·표현 검토 후 범위 확대<br>말뭉치는 주석과 원영상을 연결하고 별도로 효과 비교</p>')
html=html.replace('다음에는 이미 확보한 AI Hub 자료에서 적합한 단어를 더 선정해 범위를 넓힐 계획입니다.','우선 이번에 확인된 손 누락 취약성을 공개 학습 자료로 보완하는 실험을 준비하겠습니다. 한 손 수어에서 쉬는 손이 안 보이는 경우를 다루되, 양손 수어의 필요한 손까지 지워버리지는 않도록 해야 합니다. 사용자의 12회 촬영 결과는 모델 선택이나 튜닝에 사용하지 않습니다. 그 다음 이미 확보한 AI Hub 자료에서 적합한 단어를 더 선정해 범위를 넓힐 계획입니다.')
html=html.replace('실제 저장 화면 두 개의 촬영을 구분해 사용했다. 새 촬영·학습·추론은 수행하지 않았다.','5~7장은 기존 저장 촬영의 기능 설명이며, 9장은 2026-10-14의 실제 12회 촬영 집계다. 원인 진단에서는 공개 자료 36개 × 3조건만 고정 모델로 추론했다. 사용자 촬영 재추론·학습·모델 변경은 없다.')
html=html.replace('<ul><li>전체 자료:', '<ul><li>실사용 12회 및 공개 손 누락 진단: <code>ops/word100_onehand_audit_20261014/INDEPENDENT_CHECK.json</code>, <code>RESULT_KO.txt</code></li><li>전체 자료:')
assert len(re.findall('<article id="s[0-9]+"',html))==10
(D/'index.html').write_text(html,'utf-8')
with (D/'report.css').open('a',encoding='utf-8') as f:f.write('''
/* Actual recorded user round, separated from the AI Hub development result. */
.user-audit h2{font-size:3.05cqw;margin-bottom:2%}.user-kpis{display:grid;grid-template-columns:1fr 1fr 1.2fr;gap:3%;border-bottom:1px solid var(--line);padding-bottom:1.4%;margin-bottom:2%}.user-kpis b{display:block;color:var(--teal);font-size:3.5cqw;line-height:1.2}.user-kpis span{font-size:1.25cqw}.audit-cols{display:grid;grid-template-columns:1fr 1fr;gap:4%}.audit-cols h3{font-size:1.7cqw;margin:0 0 .5em}.audit-table{width:100%;border-collapse:collapse;font-size:1.28cqw;line-height:1.45}.audit-table th{color:var(--muted);font-weight:500;text-align:left;border-bottom:1px solid var(--line);padding:.45em .3em}.audit-table td{border-bottom:1px solid var(--line);padding:.44em .3em}.audit-table .hit{color:var(--teal);font-weight:700}.audit-table .miss{color:var(--rust);font-weight:700}.audit-cause p{font-size:1.28cqw;margin:.4em 0 .7em}.audit-cause .audit-takeaway{background:var(--soft);padding:.5em .7em;margin-top:.65em}.user-audit .foot{font-size:1.18cqw}
''')
print('Updated',D/'index.html')
