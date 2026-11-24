"""Render the verified paired comparison; no model execution."""
from common import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 r=read(U/'COMPARISON_INDEPENDENT.json');assert r['status']=='PASS_SAVED_SCORES_AND_ACTUAL_TRACES'
 note=read(U/'CONCLUSION.json')
 matched=read(U/'LATE_MATCHED_WINDOW_AUDIT.json')
 plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
 colors={'C':'#66788a','BAL31':'#147d73'}
 fig,axes=plt.subplots(1,3,figsize=(14,3.8),sharey=True,layout='constrained')
 allvals=[n for rows in r['late_per_seed'].values() for row in rows for n in row['sensor']['counts']]
 for i,ax in enumerate(axes):
  for arm in ['C','BAL31']:
   counts=r['late_per_seed'][arm][i]['sensor']['counts']
   ax.plot(range(61,61+len(counts)),np.array(counts)/3,color=colors[arm],lw=2,label='기존 C' if arm=='C' else '균등 31개')
  ax.set(title=f'동일 초기값 비교 {i+1}',xlabel='학습 회차',ylim=(min(allvals)/3-.5,100.5),xticks=[61,65,70,75,80]);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('공개 개발 검증 정확도 (%)');axes[-1].legend(loc='lower right')
 fig.savefig(U/'late_accuracy.png',dpi=180);fig.savefig(U/'late_accuracy.svg');plt.close(fig)
 def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
 selected=[];late=[]
 for i,seed in enumerate([20261013,20261014,20261015]):
  a=r['metrics']['C']['sensor'][i];b=r['metrics']['BAL31']['sensor'][i]
  selected.append([f'{i+1}회',f"{a['correct']}/300 · {a['correct']/3:.2f}%",f"{b['correct']}/300 · {b['correct']/3:.2f}%",f"{(b['correct']-a['correct'])/3:+.2f}%p",f"{r['selected_epochs']['C'][i]} → {r['selected_epochs']['BAL31'][i]}"])
  a=matched['paired'][i]['C']['sensor'];b=matched['paired'][i]['BAL31']['sensor']
  late.append([f"{i+1}회 · 61–{a['last']}회차",f"{a['adjacent_mean_pp']:.2f} → {b['adjacent_mean_pp']:.2f}%p",f"{a['adjacent_max_pp']:.2f} → {b['adjacent_max_pp']:.2f}%p",f"{a['sd_pp']:.2f} → {b['sd_pp']:.2f}%p",f"{a['mean_ce']:.4f} → {b['mean_ce']:.4f}"])
 ca=r['summary']['C']['sensor'];na=r['summary']['BAL31']['sensor']
 selected.append(['3회 평균',f"{ca['accuracy_mean']:.2f}%",f"{na['accuracy_mean']:.2f}%",f"{r['contrasts']['sensor']['mean_pp']:+.2f}%p",'같은 선택 규칙'])
 views=[]
 for v,label in [('sensor','실제로 화면 밖 손을 보지 못하는 조건'),('masked','전체 검출 후 영역 밖 좌표를 가린 조건')]:
  a=r['summary']['C'][v];b=r['summary']['BAL31'][v]
  views.append([label,f"{a['accuracy_mean']:.2f}% → {b['accuracy_mean']:.2f}%",f"{sum(a['top5_counts'])/9:.2f}% → {sum(b['top5_counts'])/9:.2f}%",f"{a['ce_mean']:.4f} → {b['ce_mean']:.4f}"])
 people=[]
 for s in ['11','12','13']:
  a=sum(x['signers'][s] for x in r['metrics']['C']['sensor'])/3;b=sum(x['signers'][s] for x in r['metrics']['BAL31']['sensor'])/3
  people.append([f'공개 개발 촬영자 ID{s}',f'{a:.2f}%',f'{b:.2f}%',f'{b-a:+.2f}%p'])
 details=[]
 for v in VIEWS:
  for g in r['metrics']['C'][v][0]['groups']:
   n=r['metrics']['C'][v][0]['groups'][g]['total'];a=sum(x['groups'][g]['correct'] for x in r['metrics']['C'][v]);b=sum(x['groups'][g]['correct'] for x in r['metrics']['BAL31'][v]);details.append([v,g,n,f'{100*a/(3*n):.2f}%',f'{100*b/(3*n):.2f}%',f'{100*(b-a)/(3*n):+.2f}%p'])
 def paras(items):return ''.join('<p>'+html.escape(x)+'</p>' for x in items)
 content=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>균등 배치 3회 비교 — WordsInSilence</title>
 <style>body{{margin:0;background:#edf2f5;color:#1a2b38;font:17px/1.65 "Malgun Gothic",sans-serif}}main{{max-width:1120px;margin:auto;padding:42px 24px}}h1{{font-size:36px;line-height:1.3;margin:10px 0}}h2{{font-size:25px;margin-top:0}}section{{background:white;padding:30px;border-radius:18px;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:13px 10px;text-align:left;border-bottom:1px solid #dce4e9}}th{{background:#edf4f3}}img{{width:100%}}.lead{{font-size:22px;font-weight:700;color:#146e65}}.label{{letter-spacing:2px;color:#376475;font-size:13px}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.card{{padding:20px;background:#e9f3f1;border-radius:12px}}.big{{font-size:32px;font-weight:700}}a{{color:#146e65}}details{{margin-top:18px}}@media(max-width:700px){{section{{padding:18px}}table{{font-size:12px}}.cards{{grid-template-columns:1fr}}}}</style>
 <main><div class="label">WORDS IN SILENCE · 2026.10.19 · 공개 개발 자료 비교</div><h1>마지막 3개 묶음을 없애면<br>정확도와 흔들림이 달라질까?</h1><p class="lead">{html.escape(note['headline'])}</p>
 <div class="cards"><div class="card">기존 C · 3회 평균<div class="big">{ca['accuracy_mean']:.2f}%</div>범위 {ca['accuracy_range'][0]:.2f}–{ca['accuracy_range'][1]:.2f}%</div><div class="card">균등 31개 · 3회 평균<div class="big">{na['accuracy_mean']:.2f}%</div>차이 {r['contrasts']['sensor']['mean_pp']:+.2f}%p · 범위 {na['accuracy_range'][0]:.2f}–{na['accuracy_range'][1]:.2f}%</div></div>
 <section><h2>1. 바꾼 것은 학습 묶음 경계 하나</h2>{table(['항목','기존 C','새 조건'],[['한 번 전체 학습','32개 × 28묶음 + 3개 × 1묶음','31개 × 29묶음'],['매 회차 자료/갱신','899개 · 29번','899개 · 29번'],['입력·순서·초기값','기존 C','동일'],['모델·학습률·선택 규칙','기존 C','동일'],['평가 묶음','32개','32개']])}<p>남는 3개를 버리거나 복제하지 않았습니다. 학습 자료 899개, 개발 검증 300개를 그대로 사용했습니다. 3회는 서로 다른 초기값으로 같은 실험을 반복한 횟수이며, 사용자 3명을 시험했다는 뜻은 아닙니다.</p></section>
 <section><h2>2. 같은 규칙으로 선택한 모델의 정확도</h2>{table(['실행','기존 C','균등 31개','차이','선택 회차 C → 새 조건'],selected)}<p>학습에 사용하지 않은 공개 촬영자 3명의 300개 영상 결과입니다. 다만 이 개발 자료는 모델 비교·선택에 반복 사용했으므로 최종 독립 시험이나 실제 웹캠 정확도로 해석하지 않습니다.</p>{table(['평가 조건','상위 1위 정확도','상위 5위 포함률','평균 손실'],views)}</section>
 <section><h2>3. 학습 후반의 흔들림</h2><p>결과를 보기 전에 정한 61–80회 학습 구간입니다. 새 조건의 세 번째 실행은 동일한 조기 종료 규칙으로 77회에서 끝났습니다. 그래프는 실제 기록만 표시하며, 아래 표는 공정한 보조 비교를 위해 세 번째 기존 C도 61–77회로 맞췄습니다. 첫째·둘째는 61–80회입니다.</p><img src="late_accuracy.png" alt="각 초기값에서 기존 C와 균등 배치의 후반 정확도 곡선. 세 번째 새 조건은77회에서 종료">{table(['실행·비교 구간','인접 회차 평균 변화','가장 큰 한 번의 변화','후반 정확도 표준편차','후반 평균 손실'],late)}<p>변화·표준편차는 낮을수록 덜 흔들립니다. 손실은 정답에 준 점수까지 반영하며 낮을수록 좋습니다. 각 셀은 기존 C → 새 조건입니다. 곡선의 오르내림과 최종 선택 모델 정확도는 서로 다른 지표입니다.</p>{paras(note['stability'])}</section>
 <section><h2>4. 좋아진 경우와 남은 문제</h2>{paras(note['accuracy'])}{table(['촬영자 · 각100영상','기존 C 평균','새 조건 평균','차이'],people)}<details><summary>관측 상태별 결과 보기</summary>{table(['조건','관측 그룹','영상 수','기존 C','새 조건','차이'],details)}</details>{paras(note['limits'])}</section>
 <section><h2>5. 판단과 다음 한 단계</h2>{paras(note['decision'])}<p>기존 C·웹·DTW는 그대로 보존했습니다. 사용자·멘토 촬영은 학습·튜닝·후보 선택에 사용하지 않았습니다.</p></section>
 <section><h2>검증 근거</h2><p>기존 입력 NPZ 2,398개 해시 일치, 텐서 4,196개 검증, 3초기값×80회 일정 215,760건 검산. 실제 학습 배치의 원본 ID·관측 선택 로그와 저장 모델·선택 회차를 확인했습니다. 저장 예측은 별도 산술로 재검산했으며 모델을 다시 추론하지 않았습니다.</p><p>구현·실행 및 별도 산술 검증 담당은 Codex입니다. 독립된 사람이나 외부 에이전트 검증을 받았다는 뜻은 아닙니다.</p><p><a href="COMPARISON_INDEPENDENT.json">전체 비교 검산</a> · <a href="READY.json">입력·분할 검증</a> · <a href="FROZEN_PROTOCOL.json">사전 고정 조건</a> · <a href="RESULT_KO.txt">텍스트 요약</a></p></section></main></html>'''
 (U/'RESULT_KO.html').write_text(content,encoding='utf-8')
 text='균등 배치 3회 학습 및 기존 C 비교\n\n'+note['headline']+'\n\n'
 for section in ['accuracy','stability','limits','decision']:text+='\n'.join(note[section])+'\n\n'
 text+='실행별 센서 정답수: C '+str(ca['top1_counts'])+' / BAL31 '+str(na['top1_counts'])+' (각300)\n'
 text+='입력2398파일 동일 해시,4196텐서 검증,240회 일정215760건 검산. 실제 배치ID·선택epoch·모델과 저장logit 별도산술 검산PASS. 사용자촬영미사용·운영C/웹/DTW유지.\n'
 (U/'RESULT_KO.txt').write_text(text,encoding='utf-8');print('HTML, text, PNG, SVG written')
if __name__=='__main__':main()
