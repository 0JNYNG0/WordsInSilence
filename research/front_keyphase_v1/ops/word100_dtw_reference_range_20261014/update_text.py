from pathlib import Path
import json,re
U=Path(__file__).resolve().parent;LAB=U.parents[1];ROOT=U.parents[3]
p=LAB/'web/practice.html';s=p.read_text('utf-8')
s=s.replace('20261014difference1','20261014range1')
s=s.replace('0에 가까울수록 학습 동작과 비슷해요 · 세 항목 모두 같은 방향으로 읽어요.','항목별로 참고 범위 이내인지 먼저 확인해요. 같은 조건에서는 차이가 작을수록 더 비슷해요.')
s=s.replace('정답률이 아닌 차이 점수예요. 손이 충분히 관측되지 않으면 점수를 표시하지 않아요.','참고 범위 이내: 학습 자료에서 관측된 차이 상한 이하예요. 범위 초과: 동작과 촬영 상태를 확인해 주세요. 수어 전체의 정답·오답 판정은 아니에요.')
s=s.replace('<summary>비교 대상과 점수의 의미</summary>','<summary>참고 범위와 점수의 의미</summary><p><b>참고 범위 이내</b>: 이 항목의 차이가 학습 촬영자 9명에게서 관측한 상한 이내입니다.<br><b>범위 초과 · 확인 필요</b>: 학습 자료에서 본 것보다 큰 차이입니다. 실제 동작 차이인지, 관절 검출·촬영 조건의 영향인지 함께 확인해야 해요.<br><b>판단 보류</b>: 손 관측 또는 비교 근거가 부족합니다.</p><p>상한은 같은 단어를 수행한 학습자 9명을 서로 비교해 고정했어요. 사용자 점수에 맞춰 넓히거나 좁히지 않습니다. 학습에 쓰지 않은 3명의 같은 단어 영상 36개 중 범위 안에 들어온 수는 굽힘 30개, 위치 35개, 궤적 33개였습니다. 이 자료도 완벽한 수행인지 전문가가 별도로 판정한 것은 아니므로, 이를 수어의 정답 판정 기준으로 확대하지 않아요.</p>')
p.write_text(s,'utf-8')
p=ROOT/'Docs/interim_report_20261014/web/index.html';s=p.read_text('utf-8')
start=s.index('<article id="s7"');end=s.index('</article>',start)+len('</article>');section=s[start:end]
b=json.loads((U.parent/'word100_multiref_20261014/BASELINE.json').read_text('utf-8'))['words']['WORD1290']['baseline']
limits=[100*b[k]['max']/(b[k]['max']+b[k]['median']) for k in ['handshape','location','movement']]
for limit in limits: section=section.replace('left:50%;top:',f'left:{limit:.6f}%;top:',1)
section=section.replace('점선 50: 학습자 간 대표 차이 · 합격선 아님','점선: 각 항목의 학습 참고 상한 · 합격선 아님')
section=section.replace('짧은 막대 = 더 가까운 동작','참고 범위와 내 차이를 함께 표시')
section=section.replace('학습자 간 차이를 기준으로 0–100에 표시<br>관측 근거가 부족하면 점수 보류','참고 범위 이내 / 범위 초과·확인 필요<br>관측 근거가 부족하면 판단 보류')
section=section.replace('세 항목 모두 낮을수록 유사하며, 50은 학습자 간 대표 차이이고 합격선이 아닙니다.','같은 조건에서는 낮을수록 유사하며, 각 항목에 학습자 간 관측 차이의 상한을 점선으로 표시합니다. 범위 안이면 학습 예시와 비슷한 차이 수준이고, 초과하면 동작이나 촬영 상태를 더 확인하도록 안내합니다. 같은 단어를 한 개발 촬영자의 영상도 범위를 벗어날 수 있어 수어 정답·오답의 합격선은 아닙니다.')
s=s[:start]+section+s[end:];p.write_text(s,'utf-8')
print('Updated reference membership labels and presentation explanation; no user-tuned limit.')
