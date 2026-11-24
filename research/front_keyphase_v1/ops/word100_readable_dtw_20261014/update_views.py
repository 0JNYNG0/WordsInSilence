from pathlib import Path
import json, re

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
web = U.parents[1] / 'web'
p = web / 'practice.html'
text = p.read_text('utf-8')
text = text.replace('20261014multiref1', '20261014difference1')
start = text.index('<p class="metric-legend">')
end = text.index('</details></section>', start) + len('</details>')
text = text[:start] + '''<p class="metric-legend">0에 가까울수록 학습 동작과 비슷해요 · 세 항목 모두 같은 방향으로 읽어요.</p><div id="dtw-metrics" class="dtw-metrics"></div><p class="analysis-hint">정답률이 아닌 차이 점수예요. 손이 충분히 관측되지 않으면 점수를 표시하지 않아요.</p><details class="criteria-details"><summary>비교 대상과 점수의 의미</summary><p>선택한 단어를 학습할 때 사용한 촬영자 9명의 동작과 비교합니다. TCN의 1위 후보가 달라도, DTW는 선택한 목표 단어를 기준으로 계산해요.</p><p><b>손가락 굽힘</b>: 화면에서 측정한 손가락 마디각의 차이<br><b>손 위치</b>: 몸을 기준으로 한 손목 위치의 차이<br><b>이동 궤적</b>: 첫 관측 위치부터 손목이 이동한 경로의 차이</p><p>0은 측정된 차이가 없음, 50은 학습자들 사이의 대표적 차이입니다. 100에 가까울수록 그 기준보다 큰 차이예요. 50은 합격선이 아니며, 80점이 80% 틀렸다는 뜻도 아닙니다.</p><p>각도·거리의 단위 차이를 학습 자료로 맞춘 뒤, 차이 ÷ (차이 + 학습자 간 대표 차이) × 100으로 표시합니다. 눈금의 같은 증가가 원래 차이의 같은 증가를 뜻하지는 않아요.</p><p>좌우 점수는 각 손의 기준으로, 전체 점수는 양손을 종합한 기준으로 계산하므로 좌우 표시 점수의 단순 평균은 아닙니다. 손바닥 방향·접촉·깊이와 올바른 수행의 허용 범위는 아직 충분히 검증되지 않았어요. 사용자 촬영으로 기준을 바꾸지 않습니다.</p></details>''' + text[end:]
text = text.replace('<script src="practice.js?', '<script src="dtw-display.js?v=20261014difference1"></script>\n  <script src="practice.js?')
assert text.count('id="dtw-metrics"') == 1 and text.count('dtw-display.js') == 1
p.write_text(text, 'utf-8')

p = ROOT / 'Docs/interim_report_20261014/web/index.html'
text = p.read_text('utf-8')
rows = json.loads((U.parent / 'word100_multiref_20261014/SAVED_ANALYSES.json').read_text('utf-8'))
row = next(r for r in rows if r['trial_id'] == 'tri_cc75cf631a1f')
metrics = row['result']['analysis']['training_comparison']['metrics']
bars = ''
for m in metrics:
    r = m['ratio']
    v = 100 * r / (1 + r)
    bars += f'<div style="margin:1.2em 0"><div style="display:flex;justify-content:space-between;font-weight:700"><span>{m["label"]} 차이</span><span>{v:.1f}점</span></div><div style="position:relative;background:#e9ecef;border-radius:9px;height:1.6cqw;margin:.8cqw 0"><div style="width:{v:.6f}%;height:100%;background:#8d4535;border-radius:9px"></div><i style="position:absolute;left:50%;top:-.3cqw;height:2.2cqw;border-left:2px dashed #344459"></i></div></div>'
section = '''<article id="s7" class="page"><div class="slide"><div class="slide-top"><span>07 · DTW 결과 상세</span><span>차이 점수 0–100 · 낮을수록 비슷함</span></div><h2 class="editable">같은 단어의 학습 동작과 얼마나 다른가?</h2><div class="two-col dtw-layout"><div style="font-size:1.8cqw;padding:1.2em;border:1px solid #e3e5e7;border-radius:18px"><p style="font-size:.7em;color:#596374">기존 감사 촬영의 표시 예시 · 새 실사용 시험 결과 아님</p>''' + bars + '''<div style="display:flex;justify-content:space-between;font-size:.7em"><span>0 · 차이 없음</span><span>50</span><span>100</span></div><p style="font-size:.7em">점선 50: 학습자 간 대표 차이 · 합격선 아님</p></div><div class="metric-explain"><div><b>학습 촬영자 9명과 비교 <em>TCN이 배운 같은 단어의 자료</em></b><p>내 수행과 학습 동작의 시간 흐름을 맞춤<br>손가락 굽힘·손 위치·이동 경로를 따로 비교</p></div><div><b>짧은 막대 = 더 가까운 동작 <em>전체 + 왼손·오른손</em></b><p>학습자 간 차이를 기준으로 0–100에 표시<br>관측 근거가 부족하면 점수 보류</p></div><div><b>인식과 비교는 서로 다른 결과 <em>단어 정답 여부는 TCN으로 확인</em></b><p>DTW는 선택한 목표 단어와의 차이를 계산<br>사용자 촬영은 학습·기준 조정에 미사용</p></div></div></div><p class="foot strong editable">차이 점수는 오답 비율이 아니다. 80점 ≠ 80% 틀림. 촬영 조건·개인차를 포함하며 교정의 근거로 추가 검증한다.</p><span class="page-no">07 / 09</span></div><section class="script"><div class="script-head"><b>발표 대본</b><button class="copy">대본 복사</button></div><p class="editable">TCN은 학습한 100개 단어 중 어떤 표현인지 순위를 매깁니다. DTW는 별도로, 사용자가 선택한 목표 단어를 학습할 때 쓴 9명의 동작과 비교합니다. 수행 속도를 맞춘 뒤 손가락 굽힘, 몸을 기준으로 한 손 위치, 이동 궤적의 차이를 계산합니다. 각 항목은 학습자들 사이의 자연스러운 차이를 기준으로 같은 0부터 100까지의 눈금에 표시했습니다. 세 항목 모두 낮을수록 유사하며, 50은 학습자 간 대표 차이이고 합격선이 아닙니다. 이 수치는 정확도나 틀린 비율이 아니라 동작 차이를 읽기 쉽게 바꾼 지표입니다. 지금 화면은 기존 감사 촬영의 표시 예시이며, 새로운 실사용 시험의 통계는 실제 시험 후 별도로 집계합니다.</p></section></article>'''
text, n = re.subn(r'<article id="s7".*?</article>', lambda _: section, text, flags=re.S)
assert n == 1
p.write_text(text, 'utf-8')
print('Updated practice and presentation; model, DTW arithmetic and original records unchanged.')
