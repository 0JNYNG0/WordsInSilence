"""Render only independently verified one-time held-out scores."""
from common import *
import html
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
 r=read(U/'COMPARISON_INDEPENDENT.json');assert r['status']=='PASS_SAVED_SCORE_RECOUNT';check(r['pins']);ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins']);vi=read(U/'INPUTS_VERIFIED.json');s=r['summary']
 maintained=r['primary_improvement_maintained'];consistent=r['stronger_consistency_pass']
 verdict='새 공개 자료에서도 평균 개선이 유지됐습니다.' if maintained else '새 공개 자료에서는 사전에 정한 평균 개선 조건을 충족하지 못했습니다.'
 delta={v:s['T3F'][v]['mean']-s['C'][v]['mean'] for v in VIEWS}
 ceiling_pairs=sum(a==b==200 for a,b in zip(s['C']['sensor']['counts'],s['T3F']['sensor']['counts']))
 if not maintained and delta['sensor']>0:
  verdict='새 공개 자료에서 평균 정확도는 높아졌습니다. 사전에 정한 반복 개선 조건은 일부 미충족입니다.'
 ceiling_note=f"주 평가에서 기존 C와 T3-F가 모두 200/200을 맞힌 모델 쌍이 {ceiling_pairs}개입니다. 나머지 변화는 아래 회차별 표에 그대로 표시했습니다. 사전에 정한 ‘3회 중 2회 이상 상승’ 기준은 결과를 보고 바꾸지 않았습니다."
 plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
 fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
 for ax,v,title in zip(axes,['sensor','masked'],['화면 제한 후 검출 · 주 평가','전체 검출 후 범위 제한 · 보조 평가']):
  for arm,offset,color,label in [('C',-.18,'#94b5c6','기존 C'),('T3F',.18,'#207969','T3-F')]:
   values=[n/2 for n in s[arm][v]['counts']]
   bars=ax.bar(np.arange(3)+offset,values,.34,color=color,label=label)
   ax.bar_label(bars,labels=[f'{n:.1f}' for n in values],padding=3,fontsize=9)
  lo=min(s[a][v]['range'][0] for a in ['C','T3F']);hi=max(s[a][v]['range'][1] for a in ['C','T3F'])
  ax.set(xticks=[0,1,2],xticklabels=['1회차','2회차','3회차'],ylim=(max(0,lo-2),min(102,hi+2)),ylabel='정확도 (%) · 세로축 확대',title=title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower left',fontsize=9)
 fig.suptitle('동일한 미사용 공개 영상 200개 · 같은 회차 모델끼리 비교');fig.savefig(U/'comparison.png',dpi=150);plt.close(fig)
 overview=[]
 for v,name in [('sensor','화면 제한 후 검출'),('masked','전체 검출 후 범위 제한')]:
  overview.append([name,f"{s['C'][v]['mean']:.2f}%",f"{s['T3F'][v]['mean']:.2f}%",f'{delta[v]:+.2f}%p'])
 top5rows=[[v,f"{s['C'][v]['top5_mean']:.2f}%",f"{s['T3F'][v]['top5_mean']:.2f}%",f"{s['T3F'][v]['top5_mean']-s['C'][v]['top5_mean']:+.2f}%p"] for v in ['sensor','masked']]
 seedrows=[]
 for i,seed in enumerate(SEEDS):
  seedrows.append([f'{i+1}회차'+(' · 사전 지정 기준' if i==0 else ''),str(seed)]+[f"{s['C'][v]['counts'][i]}/200 → {s['T3F'][v]['counts'][i]}/200 ({r['pairs'][v][i]['delta_pp']:+.2f}%p)" for v in ['sensor','masked']])
 signerrows=[]
 for sid in ['17','18']:
  for v in ['sensor','masked']:
   cc=sum(m['signers'][sid] for m in r['metrics']['C'][v]);ff=sum(m['signers'][sid] for m in r['metrics']['T3F'][v]);signerrows.append([sid,v,f'{cc}/300 ({cc/3:.2f}%)',f'{ff}/300 ({ff/3:.2f}%)',f'{(ff-cc)/3:+.2f}%p'])
 paired=[];lossrows=[]
 for v in ['sensor','masked']:
  a=sum(p['gains'] for p in r['pairs'][v]);b=sum(p['losses'] for p in r['pairs'][v]);paired.append([v,a,b,a-b])
  for p in r['pairs'][v]:
   for x in p['changes']:
    if x['kind']=='loss':lossrows.append([v,SEEDS.index(p['seed'])+1,x['signer_id'],x['label'],x['T3F_prediction']])
 changes=[x for x in r['all_classes'] if any(x[v]['C']!=x[v]['T3F'] for v in VIEWS)]
 change_rows=[[x['label']]+[f"{x[v]['C']} → {x[v]['T3F']}" for v in ['sensor','masked']] for x in changes]
 groupnames={'internal_reentry':'손 관측 중간 재등장','one_hand_never_usable':'한쪽 손 끝까지 미관측','other':'그 외','input_unavailable':'입력 준비 불가'};grows=[]
 for v in ['sensor','masked']:
  for g,n in vi['group_counts'][v].items():
   counts=[sum(x['groups'][g]['correct'] for x in r['metrics'][a][v]) for a in ['C','T3F']]
   grows.append([v,groupnames[g],n]+[f'{c}/{n*3} ({100*c/(n*3):.2f}%)' for c in counts])
 focus=[x for name in ['상처','성남시수어통역센터','수영','마음에들다','백육십이','별'] for x in r['all_classes'] if x['label']==name]
 focusrows=[[x['label']]+[f"{x[v]['C']} → {x[v]['T3F']}" for v in ['sensor','masked']] for x in focus]
 remaining=Counter((x['label'],x['prediction']) for x in r['remaining_errors']);error_rows=[[k[0],k[1],n] for k,n in remaining.most_common()]
 failed_checks=[k for k,v in r['stronger_consistency_checks'].items() if not v]
 check_names={'sensor_mean_higher':'주 평가 평균 개선','two_paired_seeds_improve':'3회 중 2회 이상 개선','masked_mean_preserved':'보조 평가 평균 유지','sensor_top5_preserved':'주 평가 상위 5개 포함률 유지','masked_top5_preserved':'보조 평가 상위 5개 포함률 유지'}
 for v,vname in [('sensor','주 평가'),('masked','보조 평가')]:
  for sid in ['17','18']:check_names[f'{v}_signer_{sid}']=f'{vname} 촬영자 ID {sid} 결과 유지'
  for g,gname in groupnames.items():check_names[f'{v}_group_{g}']=f'{vname} {gname} 결과 유지'
 failed_readable=', '.join(check_names[k] for k in failed_checks) if failed_checks else '없음'
 selected_epochs={a:[ctx['models'][a][str(seed)]['selected_epoch'] for seed in SEEDS] for a in ['C','T3F']}
 next_action='결과를 사용자와 검토한다. 이번 200개는 이제 평가 결과를 본 자료이므로 이후 튜닝 뒤 다시 맹검 결과로 부르지 않는다. 기존 C를 유지하고, 추가 학습이나 운영 교체는 별도 계획 확인 후 진행한다. ID 14~16의 299개는 계속 남겨두며 사용자 촬영은 학습·튜닝·후보 선택에 사용하지 않는다.'
 write(U/'CONCLUSION.json',{'at':now(),'status':'COMPLETE_KEEP_PRODUCTION_C','verdict':verdict,'primary_improvement_maintained':maintained,'stronger_consistency_pass':consistent,'failed_checks':failed_checks,'delta_pp':delta,'summary':s,'new_training':0,'new_videos':200,'signer_ids':[17,18],'models':6,'predictions_verified':2400,'model_state_changed':False,'test_consumed':True,'production_changed':False,'user_mentor_data_used':False,'next':next_action},new=True)
 body=f'''<header><div class="eyebrow">WORDS IN SILENCE · 동결 모델의 미사용 공개 자료 평가</div><h1>C와 T3-F,<br>처음 보는 공개 촬영에서도 달라졌을까?</h1><p>{verdict}</p><div class="tags"><span>100단어 · 200영상</span><span>촬영자ID17·18</span><span>추가 학습0회</span></div></header>
<section><h2>비교 결과</h2>{table(['같은 입력 조건','기존 C 평균','T3-F 평균','차이'],overview)}<p>각 모델의3회 학습 결과를 모두 고정한 채 동일한200영상을 평가했습니다. 아래3회차는 새로 학습한 횟수나 새로운 사람3명을 뜻하지 않습니다.</p><img src="comparison.png" alt="같은 공개 영상에서 C와T3-F의 세 모델별 정확도 비교">{table(['회차','기존 학습 seed','화면 제한 후 검출','전체 검출 후 범위 제한'],seedrows)}<p><b>화면 제한 후 검출(sensor)</b>은 미리 정한 상반신 범위의 RGB만 MediaPipe에 보이는 조건입니다. <b>전체 검출 후 범위 제한(masked)</b>은 전체 영상에서 얻은 관절 중 범위 안 점만 모델에 전달합니다. 실제 사용자의 웹캠 시험은 아닙니다.</p><p>sensor T3-F 범위{s['T3F']['sensor']['range'][0]:.2f}~{s['T3F']['sensor']['range'][1]:.2f}%, masked 범위{s['T3F']['masked']['range'][0]:.2f}~{s['T3F']['masked']['range'][1]:.2f}%. 가장 잘 나온 한 회차만 고르지 않았습니다.</p></section>
<section><h2>어떤 자료를 새로 확인했나</h2>{table(['항목','평가 범위'],[['자료','AI Hub 단어 영상 · 공식 2.Validation의 정면 영상'],['단어','현재 모델의 100단어 모두'],['촬영자 ID','17·18, 각 100영상'],['C/T3-F가 학습한 자료','ID 01~09의 899영상'],['기존 모델 선택에 사용한 자료','ID 11~13의 300영상'],['이번 평가 전 미사용 확인','영상 ID 중복 0 · 소스 해시 중복 0 · 공개 실행 이력 감사'],['계속 남겨 둔 자료','ID 14~16 정상 연결 299개; 기존 식별 불명확 1개 별도 보류']])}<p>파일의 폴더 이름이 Validation이어도 이번 프로젝트에서 모델을 고르는 개발 검증에는 쓰지 않았습니다. 기존의 개발 검증 300개와 이번 200개는 다릅니다. 이전에는 주석·파일 목록으로 보유 여부만 확인했고, 모델의 학습·선택·점수 확인에 사용하지 않았습니다.</p><p>촬영자 구분은 배포 자료의 ID에 근거합니다. 같은 말뭉치 촬영 환경 안에서의 평가이며, 다양한 실사용자·배경·웹캠 조건 전체의 성능을 대표하지 않습니다. 이번에는 정답 수어 영상의 단어 분류를 확인했습니다. 잘못 수행한 동작의 거절 능력이나 DTW 교정 기준을 검증한 것은 아닙니다.</p></section>
<section><h2>모델과 비교 조건은 끝까지 고정</h2>{table(['확인','근거'],[['모델','C 3개 + T3-F 3개 체크포인트 SHA 고정'],['이미 정해진 선택 epoch',f"C {selected_epochs['C']} / T3-F {selected_epochs['T3F']}"],['동결 사전 검사','3쌍 모두 T3-F의 기본 출력 = C, 모델 상태 불변'],['전처리','기존 C 좌우 배정·하단 경계·280채널·미검출/시간 규칙 유지'],['학습 및 선택','추가 학습 0 · 최종 평가로 epoch/seed/조건 선택 0'],['별도 검산',f"관절·가림·채널 {vi['matched_channel_values']:,}개 값 대조, 저장 예측 2,400건 산술 검산"],['입력 실패',f"sensor {len(vi['input_failures']['sensor'])}건 / masked {len(vi['input_failures']['masked'])}건, 실패도 전체 200개에 포함"],['사용자 자료·운영','사용자/멘토 촬영 미접근 · 기존 C/웹/DTW 유지']])}<p>정답 단어와 주석 동작 구간을 보고 입력을 잘라 맞추지 않았습니다. 관측된 손의 시작·끝을 찾는 기존 공통 규칙을 사용하고 중간 미검출 구간과 시간은 유지했습니다. 보이지 않은 손을 만들어 넣거나 경계·가중치를 평가 결과에 맞춰 수정하지 않았습니다.</p></section>
<section><h2>개선과 새 오류를 함께 확인</h2>{table(['입력','C 오답→T3-F 정답','C 정답→T3-F 오답','정답 순증'],paired)}<p>각 조건 200영상 × 3모델 = 600예측입니다. 같은 영상의 반복 평가이며 600개의 독립 촬영이 아닙니다.</p><h3>단어별 정답 합계가 달라진 경우</h3>{table(['단어','sensor C→T3-F /6','masked C→T3-F /6'],change_rows) if change_rows else '<p>단어별 정답 합계 변화 없음.</p>'}<p>각 단어 2영상 × 3모델 = 6회입니다. 단어 합계가 같아도 회차별 개선과 회귀가 상쇄될 수 있습니다.</p><details><summary>C가 맞혔다가 새 후보가 틀린 사례 전체</summary>{table(['입력','회차','촬영자 ID','정답','새 오답'],lossrows) if lossrows else '<p>해당 사례 없음.</p>'}</details></section>
<section><h2>사람별·손 관측 상태별로도 유지됐나</h2>{table(['촬영자 ID','입력','C','T3-F','차이'],signerrows)}<p>촬영자별 분모 300은 같은 100영상 × 3모델의 합계입니다.</p>{table(['입력','관측 상태','서로 다른 영상 수','C 정답','T3-F 정답'],grows)}<p>‘중간 재등장’은 관측이 사라졌다 돌아온 구간입니다. 실제 손 겹침뿐 아니라 내려놓은 반대손이나 끝부분 경계점도 포함합니다. ‘한쪽 손 미관측’도 언어적으로 한 손 수어라는 판정이 아닙니다. 그룹 결과만으로 누락을 복원했다거나 원인을 해결했다고 단정하지 않습니다.</p><h3>상위 5개 후보에 정답이 포함된 비율</h3>{table(['입력','C','T3-F','차이'],top5rows)}<p>상위 5개 포함률에는 1위 정답도 포함됩니다. 1위 정확도와 별개로, 정답이 후보에서 완전히 밀려났는지 확인하는 보조 지표입니다.</p></section>
<section><h2>남은 문제</h2><p>이전에 문제가 됐던 단어도 이번 새 영상에서 따로 확인했습니다. 같은 단어라도 이전 개발 검증의 영상과는 다릅니다.</p>{table(['단어','sensor C→T3-F /6','masked C→T3-F /6'],focusrows)}<details open><summary>T3-F에 남은 혼동</summary>{table(['정답','예측','오답 횟수(두 조건·세 모델 합계)'],error_rows) if error_rows else '<p>이번 범위에서 오답 없음. 일반 사용자 완전 인식을 의미하지 않습니다.</p>'}</details><p>남은 오답 목록은 진단 결과이며, 이 정답에 맞춰 새 가중치나 단어별 예외를 적용하지 않았습니다. T3-F는 기존 C의 정보를 고정해도 추가 점수가 최종 순위를 바꿀 수 있고, 실제 손 좌표 누락이나 의미 단계 자체를 복원하지 않습니다.</p><p><b>판정:</b> {verdict} 보조 조건·촬영자·관측 그룹까지 모두 보존하는 확장 확인: {'통과' if consistent else '일부 미충족'}. 미충족 항목: {failed_readable}. 이 기준은 사전에 정한 서술적 비교이며 통계적 유의성·자동 모델 채택을 뜻하지 않습니다.</p></section>
<section><h2>다음 한 단계</h2><p>{next_action}</p><p>이번 평가 뒤에는 ID 17·18의 200개를 다시 ‘한 번도 보지 않은 시험 자료’로 취급하지 않습니다. 새 사람 일반화를 더 확인하려면 별도 자료와 평가 조건을 먼저 고정해야 합니다.</p><div class="links"><a href="EXPOSURE_AUDIT.json">사용 이력 감사</a><a href="FROZEN_PROTOCOL.json">사전 고정 조건</a><a href="INPUTS_VERIFIED.json">입력 검산</a><a href="COMPARISON_INDEPENDENT.json">저장 점수 검산</a></div><p>구현·실행은 Codex 로컬에서 수행했고, 별도 수식·코드로 입력과 점수를 검산했습니다. 독립 사람·에이전트의 교차 검토로 표현하지 않습니다.</p></section>'''
 css='''*{box-sizing:border-box}body{margin:0;background:#eef3f7;color:#203246;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1180px;margin:auto;padding:30px 25px 80px}header{background:#193c53;color:white;padding:42px;border-radius:20px}h1{font-size:37px;line-height:1.4;margin:15px 0}h2{font-size:27px;margin:0 0 20px}h3{font-size:21px}.eyebrow{font-size:14px;color:#b4d5e8}header p{font-size:24px}.tags{display:flex;gap:10px;flex-wrap:wrap}.tags span{background:#2d586e;padding:6px 16px;border-radius:20px;font-size:14px}section{margin:24px 0;padding:30px;background:white;border-radius:16px;border:1px solid #d9e4ec}p{margin:18px 0}table{width:100%;border-collapse:collapse;font-size:15px;min-width:560px}td,th{padding:12px;border-bottom:1px solid #d8e1e9;text-align:left}th{background:#e9f1f6}.scroll{overflow-x:auto}img{width:100%;height:auto;margin:18px 0}.links{display:flex;gap:20px;flex-wrap:wrap}a{color:#236489}summary{cursor:pointer;font-weight:bold}details{margin:25px 0}@media(max-width:700px){body{font-size:16px}main{padding:14px 12px}header,section{padding:24px 18px}h1{font-size:28px}h2{font-size:23px}header p{font-size:19px}}'''
 body=body.replace('<h2>비교 결과</h2>','<h2>비교 결과</h2><p>'+ceiling_note+'</p>',1)
 body=body.replace('<h2>모델과 비교 조건은 끝까지 고정</h2>','<h2>모델과 비교 조건은 끝까지 고정</h2><p>모든 점수 저장 뒤 진행 상태 파일 갱신에서 Windows 접근 오류가 발생했습니다. 최초 오류와 종료 코드 1을 보존했고, 저장된 2,400예측의 검산으로 완료를 확인했습니다. 모델 계산을 다시 실행하지 않았습니다.</p>',1)
 (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T3-F 미사용 공개자료 평가</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf8')
 text=f'''{verdict}
공식Validation17/18,100단어200개전수. 모델/입력동결,추가학습0. 기존C/웹/DTW유지,사용자촬영미사용.
sensor C{s['C']['sensor']['mean']:.4f}% → T3-F{s['T3F']['sensor']['mean']:.4f}% ({delta['sensor']:+.4f}%p)
masked C{s['C']['masked']['mean']:.4f}% → T3-F{s['T3F']['masked']['mean']:.4f}% ({delta['masked']:+.4f}%p)
각회sensor C{s['C']['sensor']['counts']} /T3-F{s['T3F']['sensor']['counts']},각200. masked C{s['C']['masked']['counts']} /T3-F{s['T3F']['masked']['counts']}.
primary유지={maintained},확장확인={consistent},미충족={failed_checks}.
검산2400예측,입력값{vi['matched_channel_values']},누락원복없음. 같은말뭉치2촬영자ID·세모델반복평가이며새웹캠정확도아님.
{next_action}
'''
 (U/'RESULT_KO.txt').write_text(text,encoding='utf8');print(text)

if __name__=='__main__':main()
