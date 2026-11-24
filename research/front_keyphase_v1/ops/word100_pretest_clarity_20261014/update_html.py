from pathlib import Path

web = Path(__file__).resolve().parents[2] / 'web'
p = web / 'practice.html'
s = p.read_text(encoding='utf-8')
edits = {
    '20261014range1': '20261014clarity1',
    '<p class="countdown-hint">촬영 시작 → 3초 준비 → 동작 수행</p>': '<p class="countdown-hint">촬영 시작 → 3초 준비 → 동작 수행</p><p class="recording-hint">결과는 매번 자동 기록돼요. 실수한 시도도 ‘확인 기록’으로 표시해 주세요.</p>',
    '<h3 id="score-heading">상위 5개 후보 점수</h3>': '<h3 id="score-heading">어떤 단어로 인식했나요? <span class="method-tag">TCN</span></h3>',
    '<h3 id="dtw-heading">학습 동작들과 비교 <span class="method-tag">DTW</span></h3>': '<h3 id="dtw-heading">목표 동작과 얼마나 다른가요? <span class="method-tag">DTW</span></h3><p id="dtw-target" class="dtw-target"></p>',
    '<p class="metric-legend">항목별로 참고 범위 이내인지 먼저 확인해요. 같은 조건에서는 차이가 작을수록 더 비슷해요.</p>': '<p id="dtw-prediction-note" class="dtw-prediction-note"></p><p class="metric-legend">왼손·오른손을 각각 확인해요. 차이가 작을수록 학습 동작과 비슷해요.</p>',
    '좌우 점수는 각 손의 기준으로, 전체 점수는 양손을 종합한 기준으로 계산하므로 좌우 표시 점수의 단순 평균은 아닙니다.': '양손 종합은 좌우의 원래 차이를 평균한 뒤 점수로 바꿉니다. 한쪽 손의 큰 차이가 평균에 가려질 수 있어, 각 손의 범위 초과·판단 보류를 별도로 표시해요. 좌우 점수와 종합 점수는 기준이 달라 표시 점수의 단순 평균은 아닙니다.',
    '영상을 나란히 보며 비교해 보세요. 영상의 유사성만으로 모델이 판단한 원인을 확정할 수는 없어요.': 'TCN이 고른 후보의 영상입니다. 위 DTW의 비교 대상은 선택한 목표 단어로 유지됩니다.'
}
for before, after in edits.items():
    assert before in s, before
    s = s.replace(before, after)
p.write_text(s, encoding='utf-8')
print('HTML clarity updated')
