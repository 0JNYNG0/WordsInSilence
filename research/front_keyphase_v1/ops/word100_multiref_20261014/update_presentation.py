from pathlib import Path
import json,re
U=Path(__file__).resolve().parent;P=U.parents[3]/'Docs/interim_report_20261014/web/index.html'
text=P.read_text('utf-8')
rows=json.loads((U/'SAVED_ANALYSES.json').read_text('utf-8'))
row=next(r for r in rows if r['trial_id']=='tri_cc75cf631a1f')
metrics=row['result']['analysis']['training_comparison']['metrics']
bars=''
for m in metrics:
    v=m['ratio']
    bars+=f'<div style="margin:1.2em 0"><div style="display:flex;justify-content:space-between;font-weight:700"><span>{m["label"]}</span><span>{v:.2f}배</span></div><div style="position:relative;background:#e9ecef;border-radius:9px;height:1.6cqw;margin:.8cqw 0"><div style="width:{v/6*100:.4f}%;height:100%;background:#8d4535;border-radius:9px"></div><i style="position:absolute;left:16.6667%;top:-.3cqw;height:2.2cqw;border-left:2px dashed #344459"></i></div></div>'
section='''<article id="s7" class="page"><div class="slide"><div class="slide-top"><span>07 · DTW 결과 상세</span><span>학습 동작 대비 차이 · 낮을수록 가까움</span></div><h2 class="editable">세 항목을 같은 비교 기준으로 표시</h2><div class="two-col dtw-layout"><div style="font-size:1.8cqw;padding:1.2em;border:1px solid #e3e5e7;border-radius:18px"><p style="font-size:.7em;color:#596374">기존 감사 촬영에 새 비교 기준 적용 · 새 실사용 시험 아님</p>'''+bars+'''<div style="display:flex;justify-content:space-between;font-size:.7em"><span>0배</span><span>3배</span><span>6배</span></div><p style="font-size:.7em">점선: 학습자 간 대표 차이 1배</p></div><div class="metric-explain"><div><b>비교 대상 <em>같은 단어 · 학습 촬영자 9명</em></b><p>내 수행을 각 학습 동작과 시간 정렬 후 비교<br>손가락 굽힘·손 위치·이동 궤적을 따로 계산</p></div><div><b>공통 기준 <em>내 차이 ÷ 학습자 간 대표 차이</em></b><p>1배: 학습자들 사이에서 관측된 대표 차이<br>2배: 그 차이의 두 배 · 작을수록 가까움</p></div><div><b>표시 범위 <em>전체 + 왼손·오른손</em></b><p>관측이 부족하면 수치를 보류<br>본인 기록은 기준 계산·학습에 사용하지 않음</p></div></div></div><p class="foot strong editable">정답률·합격선이 아닌 상대적인 차이. 촬영 조건·개인차도 포함하므로 잘못된 수어라고 단정하지 않는다.</p><span class="page-no">07 / 09</span></div><section class="script"><div class="script-head"><b>발표 대본</b><button class="copy">대본 복사</button></div><p class="editable">TCN은 어떤 단어인지를 추정하고, DTW는 사용자가 선택한 단어의 학습 동작들과 얼마나 다른지를 별도로 계산합니다. 각도와 거리는 단위가 달라, 이제는 각 항목의 차이를 학습자 사이의 대표적인 차이로 나눠 같은 방식으로 표시합니다. 1배는 학습자 사이의 대표적인 차이 수준이고, 숫자와 막대가 작을수록 가까운 것입니다. 왼손과 오른손도 따로 볼 수 있습니다. 이 화면은 기존 감사 촬영을 새 기준으로 다시 분석한 구현 사례입니다. 정답률이나 수형의 합격 판정은 아니며 촬영 조건의 영향도 포함합니다.</p></section></article>'''
text,n=re.subn(r'<article id="s7".*?</article>',lambda _:section,text,flags=re.S)
assert n==1 and 'id="test-plan"' not in text and '실사용36회안내' not in text
P.write_text(text,'utf-8')
print('Presentation slide7 updated; prior self-test plan removed; no new user trial results invented')
