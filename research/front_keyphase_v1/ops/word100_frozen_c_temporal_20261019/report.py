"""Report verified C/T3/T3-F comparison without new inference."""
from common import *
import html,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def main():
 r=read(U/'COMPARISON_INDEPENDENT.json');assert r['status']=='PASS_SAVED_SCORES_ACTUAL_TRACE_AND_FROZEN_C';check(r['pins'])
 ready=read(U/'READY.json');assert ready['status']=='PASS'
 passed=r['review_gate']['pass'];s=r['summary'];new='T3F';labels=read(U/'TRAIN_CONTEXT.json')['labels']
 title='보조 시간요약만 학습한 결과';outcome='사전 비교 기준 통과 · 운영 적용은 사용자 검토 후' if passed else '사전 비교 기준 미통과 · 기존 C 유지'
 colors={'C':'#637a91','T3':'#d68c46','T3F':'#248c86'}
 plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
 fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for ax,v,caption in zip(axs,VIEWS,['전체 검출 후 범위 제한 (masked)','화면 제한 후 검출 (sensor)']):
  for j,a in enumerate(['C','T3','T3F']):
   values=[x/3 for x in s[a][v]['top1_counts']];ax.scatter(values,[j]*3,s=45,color=colors[a],alpha=.8);mean=s[a][v]['accuracy_mean'];ax.scatter([mean],[j],marker='D',s=90,color=colors[a]);ax.text(100.18,j,f'{mean:.2f}%',va='center',fontsize=11)
  ax.set(yticks=[0,1,2],yticklabels=['기존 C','T3','T3-F'],ylim=(2.6,-.6),xlim=(96,101.2),xticks=[96,97,98,99,100],xlabel='개발 검증 정확도 (%) · 96~100% 구간 확대',title=caption);ax.grid(axis='x',alpha=.2)
 fig.suptitle('점 3개: 각 학습 결과 · 마름모: 평균',fontsize=15);fig.savefig(U/'comparison.png',dpi=160);plt.close(fig)
 totals=sum(x['batches'] for x in r['traces'].values());exposures=sum(x['actual_source_exposures'] for x in r['traces'].values());epochs=r['completed_epochs']['T3F'];selected=r['selected_epochs']['T3F']
 results=[read(U/f'T3F_{seed}_RESULT.json') for seed in [20261013,20261014,20261015]];seconds=[x['seconds'] for x in results]
 main_rows=[]
 for a in ['C','T3','T3F']:
  main_rows.append([a.replace('T3F','T3-F'),' / '.join(map(str,s[a]['sensor']['top1_counts'])),f"{s[a]['sensor']['accuracy_mean']:.2f}%",' / '.join(map(str,s[a]['masked']['top1_counts'])),f"{s[a]['masked']['accuracy_mean']:.2f}%"])
 comparisons=[]
 for baseline in ['C','T3']:
  for v in VIEWS:
   event=r['paired'][baseline][v];gain=sum(len(x['gains']) for x in event);loss=sum(len(x['losses']) for x in event)
   comparisons.append([baseline,v,f"{r['contrasts'][baseline][v]['mean_pp']:+.2f}%p",gain,loss,gain-loss])
 changes=[]
 for row in r['all_classes']:
  if any(row[v]['T3F']!=row[v]['C'] for v in VIEWS):changes.append(row)
 changes.sort(key=lambda row:(-(abs(row['sensor']['T3F']-row['sensor']['C'])+abs(row['masked']['T3F']-row['masked']['C'])),row['index']))
 change_rows=[[x['label'],f"{x['sensor']['C']} → {x['sensor']['T3F']}",f"{x['masked']['C']} → {x['masked']['T3F']}"] for x in changes]
 focus_names=['상처','성남시수어통역센터','수영','백삼십육','감히','마음에들다']
 focus=[x for name in focus_names for x in r['all_classes'] if x['label']==name]
 focus_rows=[[x['label']]+[f"{x[v]['C']} / {x[v]['T3']} / {x[v]['T3F']}" for v in ['sensor','masked']] for x in focus]
 losses=[]
 for v in ['sensor','masked']:
  for i,event in enumerate(r['paired']['C'][v]):
   scores={x['clip_id']:x for x in results[i]['predictions'][v]}
   for cid in event['losses']:
    x=scores[cid];losses.append([v,f'{i+1}회차',labels[x['label_index']],f"REAL{x['signer_id']}",labels[x['prediction']]])
 late_rows=[]
 for a in ['C','T3F']:
  for v in ['sensor','masked']:
   values=[x[v]['adjacent_abs_mean_pp'] for x in r['late_per_seed'][a]]
   assert all(x['full_window'] for x in r['late_per_seed'][a])
   late_rows.append([a.replace('T3F','T3-F'),v,' / '.join(f'{x:.3f}' for x in values),f'{sum(values)/3:.3f}%p'])
 groups=[]
 names={'internal_reentry':'중간에 손 관측 재등장','one_hand_never_usable':'한쪽 손 끝까지 미관측','other':'나머지'}
 for v in ['sensor','masked']:
  for g,info in r['metrics']['C'][v][0]['groups'].items():
   nn=info['total']*3;counts=[sum(x['groups'][g]['correct'] for x in r['metrics'][a][v]) for a in ['C','T3','T3F']]
   groups.append([v,names[g]]+[f'{x}/{nn} ({100*x/nn:.2f}%)' for x in counts])
 signer=[]
 for v in ['sensor','masked']:
  for sid in ['11','12','13']:signer.append([v,sid]+[f"{sum(x['signers'][sid] for x in r['metrics'][a][v])}/300" for a in ['C','T3','T3F']])
 fail=[k for k,x in r['review_gate']['checks'].items() if not x]
 gain_text='전체와 그룹 기준을 통과했습니다. 다만 반복 사용한 개발 검증 결과이며, 실사용 모델 교체나 새 사용자 정확도 보장을 뜻하지 않습니다.' if passed else 'C의 학습 내용은 끝까지 보존됐지만, 보조 점수를 합친 최종 인식은 사전 기준을 통과하지 못했습니다. 기존 C를 유지하고 이 시간요약 계열의 추가 재탐색은 보류합니다.'
 next_text='결과를 사용자와 검토한 뒤, 필요하면 모델을 고정한 별도 평가 계획을 정합니다. 사용자 촬영은 평가 전용으로만 취급하며 학습·튜닝에 사용하지 않습니다. 현재 웹 모델을 자동 교체하지 않습니다.' if passed else '앞/중간/뒤 요약의 구간 수·가중치·학습 횟수를 이번 DEV 정답에 맞춰 다시 찾지 않습니다. 다음은 연속해서 실제 관측된 같은 관절의 방향·동작 전환을 활용할 수 있는지 공개 TRAIN 자료로 입력 수준부터 검토하는 계획을 제안합니다. 미검출0→좌표를 이동으로 계산하지 않고, 좌우 불확실·재등장 경계는 분리합니다. 새로운 후보 구현·학습은 결과 검토 후에 진행합니다.'
 conclusion={'at':now(),'status':'COMPLETE_KEEP_PRODUCTION_C','engineering_gate_pass':passed,'failed_checks':fail,'new_runs':3,'completed_epochs':epochs,'selected_epochs':selected,'auxiliary_seconds':seconds,'actual_batches':totals,'actual_source_exposures':exposures,'verified_predictions':5400,'frozen_parameter_count':37636,'trained_parameter_count':9600,'all_frozen_hashes_preserved':True,'base_probe_max_error':0,'sensor_delta_vs_C_pp':r['contrasts']['C']['sensor']['mean_pp'],'masked_delta_vs_C_pp':r['contrasts']['C']['masked']['mean_pp'],'sensor_delta_vs_T3_pp':r['contrasts']['T3']['sensor']['mean_pp'],'masked_delta_vs_T3_pp':r['contrasts']['T3']['masked']['mean_pp'],'interpretation':gain_text,'next':next_text,'user_mentor_data_used':False,'production_changed':False,'operator':'same Codex, independent arithmetic not an independent person/agent','limitations':['C checkpoints already chosen using repeated DEV; not a fresh-training matched one-variable causal control','time head does not identify semantic phases or restore missing hands','3seeds are3models, not3newpeople','head training time excludes original C pretraining'],'preparation_recovery':'First bootstrap syntax error and exact-byte copy line-ending mismatch preserved, fixed before training; no experimental conditions relaxed.'}
 write(U/'CONCLUSION.json',conclusion,new=True)
 body=f'''<header><p class="eyebrow">WORDS IN SILENCE · T3-F 공개 자료 비교 · 2026.10.19</p><h1>{title}</h1><p class="lead">{outcome}</p><div class="tags"><span>C 동결 검증 통과</span><span>3회 학습 완료</span><span>사용자 촬영 미사용</span></div></header>
<section><h2>결과부터</h2><p class="callout">{gain_text}</p>{table(['모델','sensor 정답 수 / 각300','sensor 평균','masked 정답 수 / 각300','masked 평균'],main_rows)}<p>세 정답 수는 같은 공개 DEV300개를 각 seed 모델로 평가한 결과입니다. 모델 선택에도 사용한 개발 검증이며, 일반 사용자의 웹캠 정확도가 아닙니다.</p><img src="comparison.png" alt="C,T3,T3-F 개발검증 비교"><p>sensor는 화면을 제한한 뒤 검출하여 실제 웹캠의 화면 밖 미관측을 모사합니다. masked는 전체 화면에서 검출한 뒤 지정 범위의 좌표만 남깁니다. 두 조건을 모두 확인했습니다.</p></section>
<section><h2>무엇을 고정하고, 무엇만 학습했나</h2><div class="cards"><article><b>기존 C · 37,636개 가중치</b><p>기존에 학습된 세 C 체크포인트를 각 회차에 대응시켰습니다. 관절 입력을 해석하는 부분과 기본 단어 점수, 정규화 통계를 모두 고정했습니다.</p></article><article><b>보조 부분 · 9,600개 가중치</b><p>같은 T3 방식으로 영상 앞·중간·뒤의 특징 차이를 요약하고, 기본 단어 점수에 더할 보조 점수만 학습했습니다. 총 모델 크기는47,236개입니다.</p></article></div><p><b>C 자체의 판단을 보존하는 것과, 보조 점수를 더한 최종 판단이 항상 좋아지는 것은 별개입니다.</b> 추가 점수가 원래 정답을 뒤집을 수 있어 전체와 그룹별 회귀를 함께 확인했습니다.</p><p>이전 T3는 처음부터 기반 특징과 보조 부분을 같이 학습했습니다. T3-F는 이미 학습된 C에서 출발합니다. 따라서 두 결과의 차이를 ‘동결 한 가지의 순수 인과 효과’로 단정하지 않습니다.</p></section>
<section><h2>동결이 끝까지 유지됐는지 확인</h2>{table(['검사','확인 결과'],[['학습 전11항목','초기 출력 C와 완전 동일 · 최대 차이0'],['시험 갱신','세 C 각각3번 보조 업데이트 후 C 가중치·통계·출력 불변'],['실제 학습 중',f'{totals:,}개 모든 배치에서 C 가중치·통계 해시와 eval 상태 확인'],['매 학습 회차',f'총{sum(epochs)}회 동안 고정 공개TRAIN 입력의 기본 출력 차이0'],['저장 모델','선택·마지막 체크포인트의 C 텐서를 원래 C와 직접 비교: 전부 동일'],['최적화 대상','보조 가중치1개 텐서(9,600개 값)만 optimizer와 기울기 대상'],['결과 검산','C/T3/T3-F 저장 예측5,400건 · 순위·CE·배치 노출·선택epoch 검산']])}<p>기존 입력2,398파일·4,196텐서는 원래 C/T3와 같습니다. 공개899TRAIN(촬영자1~9),300DEV(11~13),동일배치·뷰·구간일정과 학습률/조기종료/선택규칙을 유지했습니다. 사용자·멘토 자료와 보류 촬영자는 사용하지 않았습니다.</p><p>준비 중 실행 전 문법 오류와 복사 파일의 줄바꿈에 따른 해시 불일치가 발생했습니다. 최초 증거를 보존하고 문법·원본 바이트 복사만 복구한 뒤 검증을 통과했습니다. 학습 조건이나 검증 기준은 완화하지 않았습니다.</p></section>
<section><h2>개선과 회귀를 함께 계산</h2>{table(['비교 기준','입력','평균 차이','오답→정답','정답→오답','정답 순증'],comparisons)}<p>횟수는300영상×3모델=900개 예측 기준입니다. 같은 영상의 반복 모델 평가가 포함되므로 사람900명이나 독립 촬영900개가 아닙니다.</p><h3>C 대비 달라진 모든 단어</h3>{table(['단어','sensor C→T3-F /9','masked C→T3-F /9'],change_rows) if change_rows else '<p>단어별 정답 합계 변화 없음.</p>'}<p>단어당 공개 DEV3영상×3모델=9회입니다. 숫자9는 촬영자9명을 뜻하지 않습니다.</p></section>
<section><h2>이전에 문제였던 동작과 관측 그룹</h2>{table(['단어','sensor C / T3 / T3-F, 각9','masked C / T3 / T3-F, 각9'],focus_rows)}<p><b>상처와 성남시수어통역센터는 T3의 하락에서 C 수준으로 회복했습니다.</b> 상처는 여전히1회 오답이 남습니다. 수영은 masked에서 T3보다 개선됐지만 C보다1회 적게 맞혔습니다. 공개 자료의 마음에들다는 세 모델 모두9/9이며, 이번에 사용자의 해당 동작 문제까지 해결됐다는 근거는 아닙니다.</p>{table(['입력','관측 패턴','C','T3','T3-F'],groups)}<p>‘재등장’은 저장 관측점이 사라졌다가 다시 나타나는 자동 분류입니다. 필수 동작 손의 누락 외에 내려놓은 반대손이나 종료 후 경계점도 포함됩니다. 손 모양의 정확도나 누락 복원 성공률을 뜻하지 않습니다.</p><h3>C가 맞혔는데 새 후보가 틀린 사례 전부</h3>{table(['입력','모델 회차','정답 단어','공개 촬영자','새 오답'],losses)}<p>sensor1건, masked3건입니다. 백육십이는 다른 회차에서 개선된 경우와 상쇄되어 단어별 합계 표만 보면 드러나지 않습니다. <b>그룹 평균의 유지가 모든 영상·모든 회차에서 오류가 없다는 뜻은 아닙니다.</b></p><details><summary>촬영자별 결과</summary>{table(['입력','공개 촬영자ID','C','T3','T3-F'],signer)}</details></section>
<section><h2>학습 실행과 남은 한계</h2>{table(['학습','완료 epoch','선택 epoch','보조 학습 시간(초)'],[[i+1,epochs[i],selected[i],f'{seconds[i]:.1f}'] for i in range(3)])}<p>최대80회, 개발CE가12회 개선되지 않으면 종료하는 기존 규칙입니다. epoch1~80에서 같은CE기준으로 선택했고, 좋은 seed만 골라 보고하지 않았습니다. 위 시간에는 기존 C를 학습했던 시간이 포함되지 않습니다.</p><h3>후반61~80회에서 인식률의 흔들림</h3>{table(['모델','입력','3회차별 평균 변동(%p)','3회 평균'],late_rows)}<p>바로 이전 epoch와 정확도가 얼마나 달라졌는지 절댓값의 평균입니다. 낮을수록 이 구간의 변화가 작았습니다. 이미 학습된 C를 고정하고 작은 보조 부분만 학습한 결과이므로, 모든 학습 상황의 안정성이 증명됐다는 뜻은 아닙니다. T3는 조기 종료로 동일한20회 구간이 없어 이 표에서 비교하지 않았습니다.</p><p>손 관측 누락을 복원하거나 실제 수어의 의미 단계를 검출하는 모델은 아닙니다. 학습 예시와 반복·정지·마무리 시점이 다르면 상대시간 요약이 여전히 달라질 수 있습니다. 반복 사용한 DEV에서 나온 수치만으로 새로운 사람이나 모든 수어에서의 성능을 보장할 수 없습니다.</p><p><b>사전 비교 기준:</b> sensor평균상승·2seed이상상승,masked평균·sensorTop5유지,촬영자별·관측그룹별평균회귀없음,입력실패증가없음. 이는 프로젝트 내 공학적 검토 기준이며 통계적 유의성 검정이나 자동 배포 허가는 아닙니다.</p><p>미통과 항목: {html.escape(', '.join(fail)) if fail else '없음'}</p></section>
<section><h2>다음 한 작업</h2><p>{next_text}</p><p><b>현재 운영 C·웹·DTW는 그대로입니다.</b> 승인된3회와 비교·검산까지 완료했으며 추가 학습은 진행하지 않습니다.</p><div class="links"><a href="READY.json">동결 검증</a><a href="COMPARISON_INDEPENDENT.json">전체 비교·검산</a><a href="CONCLUSION.json">판정</a><a href="FROZEN_PROTOCOL.json">학습 전 고정 조건</a></div><p>구현·검산은 같은 Codex가 수행했습니다. 별도 산술 및 텐서 검산이며 독립 사람·에이전트 교차검토로 표현하지 않습니다.</p></section>'''
 css='''*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#203047;font:17px/1.75 "Malgun Gothic",sans-serif}main{max-width:1160px;margin:auto;padding:36px 24px 70px}header{padding:42px;background:#18394f;color:white;border-radius:20px;margin-bottom:28px}h1{font-size:40px;line-height:1.3;margin:18px 0}h2{font-size:26px;margin:0 0 18px}h3{font-size:21px}.eyebrow{font-size:13px;letter-spacing:1px;color:#a8c9e2}.lead{font-size:23px}.tags{display:flex;gap:10px;flex-wrap:wrap}.tags span{background:#2c526c;padding:5px 14px;border-radius:30px;font-size:14px}section{padding:30px;background:white;border:1px solid #dce5ec;border-radius:16px;margin:22px 0}p{margin:12px 0 18px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}.cards article{padding:20px;background:#f3f7fa;border-radius:12px}.callout{background:#eaf3f5;border-left:5px solid #248c86;padding:20px}table{width:100%;border-collapse:collapse;font-size:15px;min-width:560px}th{background:#eaf1f6;text-align:left}td,th{padding:12px;border-bottom:1px solid #dbe3ea}.scroll{overflow-x:auto}img{width:100%;height:auto;margin:22px 0}.links{display:flex;gap:15px;flex-wrap:wrap}a{color:#246583}details{margin:20px 0}summary{cursor:pointer}@media(max-width:700px){main{padding:15px 12px}header,section{padding:23px 18px}h1{font-size:31px}h2{font-size:23px}.lead{font-size:19px}.cards{grid-template-columns:1fr}body{font-size:16px}}'''
 (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T3-F 동결 비교 결과</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf8')
 text=f'''T3-F 비교 완료: {outcome}
sensor 평균 C{s['C']['sensor']['accuracy_mean']:.4f}% / T3{s['T3']['sensor']['accuracy_mean']:.4f}% / T3-F{s['T3F']['sensor']['accuracy_mean']:.4f}%
masked 평균 C{s['C']['masked']['accuracy_mean']:.4f}% / T3{s['T3']['masked']['accuracy_mean']:.4f}% / T3-F{s['T3F']['masked']['accuracy_mean']:.4f}%
3회 sensor정답: {s['T3F']['sensor']['top1_counts']}/각300, masked:{s['T3F']['masked']['top1_counts']}/각300.
완료epoch{epochs},선택{selected}. 실제{totals}batch/{exposures}원본노출. 저장5400예측별도검산.
동결11검사PASS,초기C출력차이0. 실제모든배치C해시/BN/eval보존,매epochTRAIN기준출력변화0. 선택/마지막모델C텐서모두원본일치. 학습가능9600만;기존37636고정.
{gain_text}
미통과항목:{fail}
{next_text}
공개899TRAIN/300DEV만. 사용자/멘토/보류자료미사용,C/웹/DTW변경0. 새추출/다운로드/외부전송/Git0. 처음부터재학습한T3와단일변수인과효과아님. 반복DEV,새사람성능아님. 같은Codex구현+별도산술검산,독립에이전트아님.
준비초기문법/파일줄바꿈복사오류보존후복구,학습조건완화없음.
'''
 (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
 print(text)
if __name__=='__main__':main()
