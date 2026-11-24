"""All-class comparison and report from saved scores only, with no model inference."""
from common import *
import html,math
from collections import Counter
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEEDS=[20261013,20261014,20261015]

def main():
 assert not (U/'TRAIN_RUNNING.json').exists() and read(U/'EXIT.json')['train_exit']==0
 r=read(U/'COMPARISON_INDEPENDENT.json');matched=read(U/'LATE_MATCHED_WINDOW_AUDIT.json');ctx=read(U/'TRAIN_CONTEXT.json')
 assert r['status']=='PASS_SAVED_SCORES_AND_ACTUAL_TRACES';check(r['pins']);check(matched['pins'])
 labels=ctx['labels'];saved={};runtime={};head=[];pins={}
 for arm in ['C','T3']:
  runtime[arm]=[]
  for seed in SEEDS:
   p=BASE/f'training_C_{seed}_RESULT.json' if arm=='C' else U/f'{arm}_{seed}_RESULT.json'
   q=read(p);saved[arm,seed]=q;kp=BASE/f'training_C_{seed}_best.pt' if arm=='C' else U/f'{arm}_{seed}_best.pt'
   ck=torch.load(kp,map_location='cpu',weights_only=True);assert sha(kp)==q['checkpoint_sha256']
   assert all(torch.isfinite(t).all() for t in ck['state_dict'].values())
   num=sum(t.numel() for k,t in ck['state_dict'].items() if 'running_' not in k and 'num_batches_tracked' not in k)
   assert num==(37636 if arm=='C' else 47236)
   if arm=='T3':
    w=ck['state_dict']['phase_weight'];assert tuple(w.shape)==(100,96) and torch.count_nonzero(w)>0
    head.append({'seed':seed,'weight_norm':float(w.norm()),'zero_parameters':int((w==0).sum())})
   # Original C seconds are named elapsed_seconds; comparison is descriptive.
   seconds=q.get('seconds',q.get('elapsed_seconds'))
   if seconds is None:
    lp=BASE/f'training_C_{seed}_last.pt';last=torch.load(lp,map_location='cpu',weights_only=True);seconds=last['elapsed_seconds']
   epochs=q.get('completed_epochs',q.get('epochs_completed'))
   runtime[arm].append({'seed':seed,'epochs':epochs,'seconds':seconds,'seconds_per_epoch':seconds/epochs,'checkpoint_bytes':kp.stat().st_size,'parameters':num})
   pins[str(p)]=sha(p);pins[str(kp)]=sha(kp)
 words=[];events={v:[] for v in VIEWS};confusions={}
 for v in VIEWS:
  confusions[v]={}
  for arm in ['C','T3']:
   pairs=Counter()
   for seed in SEEDS:
    for row in saved[arm,seed]['predictions'][v]:
     if not row['correct']:pairs[(row['label_index'],row['prediction'])]+=1
   confusions[v][arm]=[{'true':labels[a],'predicted':labels[b] if b is not None else None,'count':n} for (a,b),n in pairs.most_common()]
  for i,seed in enumerate(SEEDS):
   before={p['clip_id']:p for p in saved['C',seed]['predictions'][v]};after={p['clip_id']:p for p in saved['T3',seed]['predictions'][v]}
   for cid,b in before.items():
    a=after[cid]
    if b['correct']!=a['correct']:events[v].append({'seed':seed,'clip_id':cid,'signer':b['signer_id'],'word':labels[b['label_index']],'C_prediction':labels[b['prediction']],'T3_prediction':labels[a['prediction']],'change':'gain' if a['correct'] else 'regression'})
 for i,label in enumerate(labels):
  row={'class_index':i,'word':label,'independent_DEV_videos':3,'predictions_per_condition':9}
  for v in VIEWS:
   counts={a:sum(p['correct'] for s in SEEDS for p in saved[a,s]['predictions'][v] if p['label_index']==i) for a in ['C','T3']}
   row[v]={**counts,'delta_correct':counts['T3']-counts['C']}
  words.append(row)
 analysis={'at':now(),'all_classes':words,'paired_events':events,'confusions':confusions,'runtime':runtime,'trained_extra_head':head,'pins':pins,'new_inference':0,'user_mentor_data_used':False,'production_changed':False,'limits':['Repeated seeds are not independent people or additional videos.','Runtime includes training and scheduled evaluation/checkpoint I/O, measured on different runs; not a controlled inference speed benchmark.','Head weight norm is not phase explanation or causal proof.']}
 write(U/'ALL_CLASS_ANALYSIS.json',analysis,new=True)
 ca=r['summary']['C']['sensor'];na=r['summary']['T3']['sensor'];delta=r['contrasts']['sensor']['mean_pp'];passed=r['review_gate']['pass']
 if passed:headline='T3는 정해 둔 개발 비교 기준을 통과했습니다. 운영 적용은 결과 검토 후 결정합니다.'
 elif delta>0:headline='평균 정확도는 높아졌지만, 모든 비교 조건을 통과하지 못해 기존 C를 유지합니다.'
 elif delta<0:headline='T3의 평균 정확도가 낮아져, 이번 후보는 채택하지 않고 기존 C를 유지합니다.'
 else:headline='평균 정확도는 기존 C와 같아, 개선이 확인되지 않았습니다. 기존 C를 유지합니다.'
 verdict={'at':now(),'headline':headline,'review_gate':r['review_gate'],'primary_delta_pp':delta,'decision':'REVIEW_CANDIDATE_NO_DEPLOYMENT' if passed else 'NOT_ADOPTED_KEEP_C','source':str(U/'COMPARISON_INDEPENDENT.json'),'no_automatic_next_training':True,'user_data_used':False,'production_changed':False}
 write(U/'CONCLUSION.json',verdict,new=True)
 plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
 colors={'C':'#758699','T3':'#087e8b'}
 fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained');xx=np.arange(3)
 for a,offset in [('C',-.18),('T3',.18)]:
  yy=np.array(r['summary'][a]['sensor']['top1_counts'])/3
  bars=ax.bar(xx+offset,yy,.33,label='기존 C' if a=='C' else 'T3',color=colors[a]);ax.bar_label(bars,labels=[f'{v:.2f}%' for v in yy],padding=3)
 ax.set(ylim=(0,106),xticks=xx,xticklabels=['1회','2회','3회'],ylabel='공개 개발 검증 정확도 (%)');ax.legend(loc='lower right');ax.spines[['top','right']].set_visible(False)
 fig.savefig(U/'selected_accuracy.png',dpi=170);fig.savefig(U/'selected_accuracy.svg');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,3.8),sharey=True,layout='constrained')
 vals=[z for runs in r['late_per_seed'].values() for run in runs for z in run['sensor']['counts']]
 for i,ax in enumerate(axes):
  for arm in ['C','T3']:
   yy=r['late_per_seed'][arm][i]['sensor']['counts'];ax.plot(range(61,61+len(yy)),np.array(yy)/3,color=colors[arm],lw=2,label='기존 C' if arm=='C' else 'T3')
  ax.set(title=f'실행 {i+1}',xlabel='학습 회차',ylim=((min(vals)/3-.5) if vals else 0,100.5),xticks=[61,65,70,75,80]);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('공개 개발 검증 정확도 (%)');axes[-1].legend(loc='lower right');fig.savefig(U/'late_accuracy.png',dpi=170);fig.savefig(U/'late_accuracy.svg');plt.close(fig)
 def esc(x):return html.escape(str(x))
 def table(h,rr):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rr)+'</tbody></table></div>'
 def fmt(x):return '자료 부족' if x is None else f'{x:.2f}'
 selected=[]
 for i,seed in enumerate(SEEDS):
  a=ca['top1_counts'][i];b=na['top1_counts'][i];selected.append([i+1,f'{a}/300 · {a/3:.2f}%',f'{b}/300 · {b/3:.2f}%',f'{(b-a)/3:+.2f}%p',f"{r['selected_epochs']['C'][i]} → {r['selected_epochs']['T3'][i]}"])
 selected.append(['평균',f"{ca['accuracy_mean']:.2f}%",f"{na['accuracy_mean']:.2f}%",f'{delta:+.2f}%p','같은 선택 규칙'])
 views=[];groups=[];people=[]
 for v,name in [('sensor','화면 밖 손을 센서가 보지 못하는 조건'),('masked','전체 검출 후 범위 밖 좌표를 가린 조건')]:
  a=r['summary']['C'][v];b=r['summary']['T3'][v]
  views.append([name,f"{a['accuracy_mean']:.2f}% → {b['accuracy_mean']:.2f}%",f"{sum(a['top5_counts'])/9:.2f}% → {sum(b['top5_counts'])/9:.2f}%",f"{a['ce_mean']:.4f} → {b['ce_mean']:.4f}"])
  for signer in ['11','12','13']:
   a=sum(x['signers'][signer] for x in r['metrics']['C'][v])/3;b=sum(x['signers'][signer] for x in r['metrics']['T3'][v])/3
   people.append([v,signer,f'{a:.2f}%',f'{b:.2f}%',f'{b-a:+.2f}%p'])
  for group in r['metrics']['C'][v][0]['groups']:
   n=r['metrics']['C'][v][0]['groups'][group]['total'];a=sum(x['groups'][group]['correct'] for x in r['metrics']['C'][v]);b=sum(x['groups'][group]['correct'] for x in r['metrics']['T3'][v])
   groups.append([v,group,n,f'{100*a/(3*n):.2f}%',f'{100*b/(3*n):.2f}%',f'{100*(b-a)/(3*n):+.2f}%p'])
 changes=[[x['word'],f"{x['sensor']['C']} → {x['sensor']['T3']}",f"{x['sensor']['delta_correct']:+d}",f"{x['masked']['C']} → {x['masked']['T3']}"] for x in words if x['sensor']['delta_correct'] or x['masked']['delta_correct']]
 focus=['동대문','속다','백육십이','상처','흡연실']
 focus_rows=[[x['word'],f"{x['sensor']['C']} → {x['sensor']['T3']}",f"{x['masked']['C']} → {x['masked']['T3']}"] for x in words if x['word'] in focus]
 late_rows=[]
 for i,pair in enumerate(matched['paired']):
  a=pair['C']['sensor'];b=pair['T3']['sensor']
  late_rows.append([i+1,str(pair['window']),f"{fmt(a['adjacent_mean_pp'])} → {fmt(b['adjacent_mean_pp'])}",f"{fmt(a['adjacent_max_pp'])} → {fmt(b['adjacent_max_pp'])}",f"{fmt(a['sd_pp'])} → {fmt(b['sd_pp'])}"])
 failures=[k for k,v in r['review_gate']['checks'].items() if not v]
 lateC=matched['three_seed_mean']['C']['sensor']['adjacent_mean_pp'];lateT=matched['three_seed_mean']['T3']['sensor']['adjacent_mean_pp']
 late_sentence=(f'동일 회차 범위에서 인접 회차 정확도 변화의 3회 평균은 {lateC:.2f} → {lateT:.2f}%p입니다. 낮을수록 덜 흔들립니다. 최종 선택 정확도와는 별개 지표입니다.' if lateC is not None and lateT is not None else '일부 실행이 61회 이전에 종료되어 후반 흔들림의 3회 평균을 비교할 수 없습니다. 기록이 있는 실행만 표에 표시하며, 조기 종료를 안정성 개선으로 간주하지 않습니다.')
 duration_rows=[[arm,i+1,q['epochs'],f"{q['seconds']:.1f}초",f"{q['seconds_per_epoch']:.2f}초",f"{q['checkpoint_bytes']/1024:.1f}KiB"] for arm,rr in runtime.items() for i,q in enumerate(rr)]
 total_batches=sum(x['batches'] for x in r['traces'].values());total_exposures=sum(x['actual_source_exposures'] for x in r['traces'].values())
 txt=f'''T3 시간 단계 요약 후보 — 3회 학습 및 기존 C 비교
{headline}

입력: 공개 TRAIN899(촬영자1~9) / DEV300(촬영자11~13), 기존 C와 동일.
변경: 기존 전구간 요약 + 앞/중간/뒤 상대시간 가중요약. 새 가중치0시작, 초기 출력C와 동일.
파라미터: 37,636 → 47,236(+25.5%). 원래32×28+3배치/학습률/최대80/조기종료12/CE선택 유지.
sensor 정답수: C {ca['top1_counts']} / T3 {na['top1_counts']} (각300).
sensor 평균: {ca['accuracy_mean']:.2f}% → {na['accuracy_mean']:.2f}% ({delta:+.2f}%p).
masked 평균: {r['summary']['C']['masked']['accuracy_mean']:.2f}% → {r['summary']['T3']['masked']['accuracy_mean']:.2f}%.
선택 회차: C {r['selected_epochs']['C']} / T3 {r['selected_epochs']['T3']}.
완료 회차: {[q['epochs'] for q in runtime['T3']]}.
{late_sentence}
사전 고정 판정: {'통과' if passed else '미통과'}; 미통과 항목: {failures}
검증: 동일입력2398파일/4196텐서, 초기출력오차0,9가지검증통과. 저장3600예측 산술/실제{total_batches}배치·{total_exposures}노출/선택epoch/checkpoint 검산PASS.
공개DEV 반복사용 결과이며 새사람·실제웹캠 정확도 아님. 3회는 동일300영상에 대한 초기값 반복.
상대시간요약이 의미단계 이해/손미검출복원이라는 뜻 아님. 추가파라미터효과와 순서효과 분리 불가.
기존C·웹·DTW 유지. 사용자/멘토 자료 미접근, 학습·튜닝·선택 미사용. 추가학습/운영교체 없음.
구현·실행·별도산술검증은 같은Codex. 독립사람/외부에이전트검증 아님.
'''
 assert not (U/'RESULT_KO.txt').exists();(U/'RESULT_KO.txt').write_text(txt,encoding='utf8')
 content=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T3 시간 단계 요약 비교 — WordsInSilence</title>
<style>body{{margin:0;background:#eef3f6;color:#1a2a38;font:17px/1.7 'Malgun Gothic',sans-serif}}main{{max-width:1120px;margin:auto;padding:42px 24px}}h1{{font-size:36px;line-height:1.35}}h2{{font-size:25px;margin-top:0}}section{{background:white;border-radius:18px;padding:30px;margin:22px 0}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{padding:12px 9px;text-align:left;border-bottom:1px solid #dde5eb}}th{{background:#edf6f5}}.scroll{{overflow-x:auto}}img{{width:100%}}.lead{{color:#086d79;font-size:22px;font-weight:700}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.card{{background:#dfefed;padding:24px;border-radius:14px}}.big{{font-size:34px;font-weight:700}}a{{color:#087885}}details{{margin:16px 0}}@media(max-width:700px){{section{{padding:18px}}.cards{{grid-template-columns:1fr}}}}</style>
<main><p>WORDS IN SILENCE · 2026.10.19 · 공개 개발 자료 비교</p><h1>동작의 앞·중간·뒤를 나눠 기억하면<br>수어 분류가 좋아질까?</h1><p class="lead">{esc(headline)}</p>
<div class="cards"><div class="card">기존 C · 3회 평균<div class="big">{ca['accuracy_mean']:.2f}%</div>범위 {ca['accuracy_range'][0]:.2f}–{ca['accuracy_range'][1]:.2f}%</div><div class="card">T3 · 3회 평균<div class="big">{na['accuracy_mean']:.2f}%</div>{delta:+.2f}%p · 범위 {na['accuracy_range'][0]:.2f}–{na['accuracy_range'][1]:.2f}%</div></div>
<section><h2>1. 무엇을 바꿨나</h2><p>기존 C는 시간에 따른 움직임을 분석한 뒤, 전체를 하나로 요약합니다. T3는 그 요약을 유지하면서 <b>앞·중간·뒤에 어떤 특징이 나타났는지</b>를 덧붙입니다. 구간 경계는 딱 자르지 않고, 가까운 구간에 부드럽게 나누어 반영합니다.</p><p>추가 부분은 0에서 시작해, 학습 전에는 C와 출력이 정확히 같습니다. 실제 동작의 의미를 찾아 나눈 것은 아닙니다. 손이 안 보이는 프레임도 원래 시간 위치와 미관측 표시를 유지합니다.</p>{table(['항목','조건'],[['입력·자료분리','기존 C 그대로: 학습899 / 공개 개발 검증300'],['변경한 부분','마지막 시간 요약 + 추가 분류 가중치9,600개'],['전체 파라미터','37,636 → 47,236 (+25.5%)'],['학습 배치','기존32개×28 + 마지막3개; 균등배치 미적용'],['학습·선택','기존과 같은 초기값3개 / 학습률 / 최대80회 / 조기종료12회 / 평균CE최소'],['운영 모델·DTW','기존 C 유지; 교체하지 않음']])}</section>
<section><h2>2. 같은 기준으로 선택한 결과</h2><img src="selected_accuracy.png" alt="기존C와 T3의 실행별 정확도">{table(['실행','기존 C','T3','차이','선택 학습회차 C→T3'],selected)}<p>300개는 학습에 쓰지 않은 공개 촬영자 3명의 영상입니다. 다만 모델 선택과 비교에 반복 사용한 개발 자료이므로, 최종 독립 시험이나 사용자 웹캠 정확도는 아닙니다. 3회는 같은 자료를 서로 다른 초기값으로 학습한 반복입니다.</p>{table(['관측 조건','1위 정확도','5위 내 정답 포함','평균 손실CE'],views)}<p>sensor는 잘린 화면만 검출한 입력, masked는 전체화면 검출 뒤 범위 밖 좌표를 가린 입력입니다. 실제 사용자 촬영 결과가 아닙니다.</p></section>
<section><h2>3. 반복 혼동과 새 오답</h2><p>아래 정답수는 각 단어의 공개 개발 영상3개 × 학습3회 = 9개 예측입니다. 독립된9명 시험이 아닙니다. 알려진 혼동 사례만 골라 성능을 판단하지 않고 100개 단어 전체를 비교했습니다.</p>{table(['이전 반복 혼동 단어','sensor C→T3 (9예측)','masked C→T3 (9예측)'],focus_rows)}<details open><summary>정답수가 달라진 모든 단어</summary>{table(['단어','sensor C→T3','차이','masked C→T3'],changes)}</details><details><summary>촬영자별·관측 상태별 전체 비교</summary>{table(['관측','촬영자ID','기존C','T3','차이'],people)}{table(['관측','그룹','고유영상','기존C','T3','차이'],groups)}</details><p>공개 자료의 정답률 변화만으로 손모양이나 순서를 이해하게 되었다고 단정할 수 없습니다. 정확도 개선이 있더라도 시간요약과 추가 가중치의 효과가 함께 포함됩니다.</p></section>
<section><h2>4. 학습 후반의 흔들림</h2><p>사전에 정한 61–80회 기록만 표시합니다. 조기 종료되었다면 그 이후를 채우지 않습니다. 표는 각 실행에서 C와 T3에 공통으로 남아 있는 회차만 비교합니다.</p><img src="late_accuracy.png" alt="실행별 학습후반 정확도 곡선">{table(['실행','비교회차','인접변화 평균 (%p)','인접변화 최대 (%p)','정확도 표준편차 (%p)'],late_rows)}<p>{esc(late_sentence)}</p></section>
<section><h2>5. 판정과 남은 한계</h2><p><b>사전에 정한 비교 기준: {'통과' if passed else '미통과'}</b>. 평균 sensor 정확도 향상, 3회 중2회 이상 향상, masked 평균·sensor 상위5위 유지, 촬영자·관측그룹 회귀 없음, 입력 실패 증가 없음을 함께 확인했습니다.</p><p>미통과 항목: {esc(', '.join(failures) if failures else '없음')}.</p><p>시작·끝의 여유 시간이나 중간 멈춤이 달라지면 같은 동작의 상대시간 위치도 달라질 수 있습니다. T3는 손의 미검출·좌우 오배정·틀린 손가락 좌표를 복구하지 않습니다. 이번 결과를 보고 추가 조건을 탐색하거나 특정 단어용 규칙을 넣지 않았습니다.</p><p>기존 C·웹·DTW는 그대로 두었습니다. 사용자·멘토 자료는 읽지 않았으며 학습·튜닝·후보 선택에도 사용하지 않았습니다. 결과를 함께 검토한 뒤 후속 작업을 정합니다.</p></section>
<section><h2>6. 검증·실행 근거</h2><p>입력2,398파일 해시와4,196텐서 확인, 초기 출력 차이0, 시간가중·패딩·미검출·기울기9가지 검사 통과. 실제{total_batches:,}배치/{total_exposures:,}자료노출, 선택 회차·저장 모델 및 총3,600개 저장 예측을 별도 산술로 검산했습니다. 완료 점수를 재추론하지 않았습니다.</p><p>구현·실행·별도 산술 검증은 같은 Codex가 수행했습니다. 독립된 사람이나 외부 에이전트의 검증을 뜻하지 않습니다.</p><details><summary>학습 시간과 저장 크기</summary>{table(['모델','실행','완료회차','총 시간','회차당','체크포인트'],duration_rows)}<p>시간은 학습·정기검증·파일저장을 포함한 관측값이며, 서로 다른 실행 시점의 장치 부하가 포함됩니다. 추론 지연 비교가 아닙니다.</p></details><p><a href="COMPARISON_INDEPENDENT.json">점수·학습추적 검산</a> · <a href="ALL_CLASS_ANALYSIS.json">전체 단어 비교</a> · <a href="READY.json">구현·입력 검증</a> · <a href="FROZEN_PROTOCOL.json">고정 계획</a> · <a href="RESULT_KO.txt">텍스트 요약</a></p></section></main></html>'''
 assert not (U/'RESULT_KO.html').exists();(U/'RESULT_KO.html').write_text(content,encoding='utf8')
 print(txt)

if __name__=='__main__':main()
