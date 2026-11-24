"""Report the frozen 2x2 comparison; uses saved, independently recounted scores."""
from common import *
import html
from collections import Counter
def label(a):return {'C':'기존 C: 기존 손 배정 + 기존 크기','Q':'Q: 보완 손 배정 + 보완 후 크기','R10':'분리 ①: 보완 손 배정 + 기존 크기','R01':'분리 ②: 기존 손 배정 + 보완 후 크기'}[a]
def fmt(x):return f'{x:+.2f}'
def table(headers,rows):
 esc=html.escape
 return '<table><thead><tr>'+''.join('<th>'+esc(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def main():
 r=read(U/'COMPARISON_INDEPENDENT.json');assert r['status']=='PASS_SAVED_SCORES_AND_ACTUAL_TRACES';assert not (U/'TRAIN_RUNNING.json').exists()
 s=r['summary'];ct=r['contrasts'];inter=r['interaction_Q_minus_R10_minus_R01_plus_C'];seed=[20261013,20261014,20261015]
 definitions={'R10-C':'기존 크기를 유지하고 손 배정만 변경','R01-C':'기존 손 배정에서 크기 기준만 변경','Q-R10':'보완 손 배정에서 크기 기준만 변경','Q-R01':'보완 후 크기를 유지하고 손 배정 변경','Q-C':'손 배정과 크기를 함께 변경'}
 rows=[[label(a),'/'.join(map(str,s[a]['sensor']['top1_counts'])),f"{s[a]['sensor']['accuracy_mean']:.2f}%",f"{s[a]['sensor']['accuracy_range'][0]:.2f}–{s[a]['sensor']['accuracy_range'][1]:.2f}%",f"{s[a]['masked']['accuracy_mean']:.2f}%"] for a in ['C','R10','R01','Q']]
 contrast_rows=[[definitions[k],'/'.join(fmt(x) for x in ct[k]['sensor']['per_seed_pp']),fmt(ct[k]['sensor']['mean_pp'])+'%p',fmt(ct[k]['masked']['mean_pp'])+'%p'] for k in ['R10-C','R01-C','Q-R10','Q-R01','Q-C']]
 gainloss={}
 for k,views in r['paired'].items():
  gainloss[k]={v:{'gains':sum(len(x['gains']) for x in entries),'losses':sum(len(x['losses']) for x in entries)} for v,entries in views.items()}
 result={'status':'COMPLETE_DIAGNOSTIC_ONLY','six_new_runs_complete':True,'summary':s,'contrasts':ct,'interaction':inter,'gain_loss_repeated_300x3':gainloss,'actual_epochs':sum(x['epochs'] for x in r['traces'].values()),'actual_batches':sum(x['batches'] for x in r['traces'].values()),'actual_source_exposures':sum(x['actual_source_exposures'] for x in r['traces'].values()),'automatic_deployment':False,'user_data_used':False}
 write(U/'REPORT_DATA.json',result,new=True)
 conclusion=read(U/'INTERPRETATION.json')
 parts=['손 배정·손 크기 기준 분리 비교 결과','2026-10-19 · 공개 자료만 사용 · 새 학습6회 완료 · 기존 C 유지','',conclusion['summary'],'',
 '비교 구성','C=기존 손 배정+기존 손 크기. Q=보완 손 배정+그 관측에서 다시 계산한 손 크기. 새 분리①(R10)=Q 배정+C 크기, 새 분리②(R01)=C 배정+Q 크기.','손 크기를 계산하는 공식 자체를 바꾼 것이 아니라, 같은 영상에서 손 배정 전후 계산된 분모를 교환했다. 양쪽 분모가 모두 유효한 경우만 교환하고, 한쪽이 계산 불가하면 해당 관측 분기의 원래 마스크와 크기를 유지했다.','',
 '인식 결과 — 각 실행의 공개 검증 300개, 조건마다 같은 세 초기값 사용',*[' | '.join(x) for x in rows],'sensor=상반신 화면만 보고 독립적으로 검출한 조건(주평가). masked=전체 화면 검출 뒤 범위 안의 좌표만 쓰는 조건(보조평가). 새 사람의 실제 웹캠 정확도가 아니다.','',
 '변화를 분리한 효과',*[' | '.join(x) for x in contrast_rows],f"sensor 상호작용(Q−R10−R01+C): {fmt(inter['sensor']['mean_pp'])}%p, seed별 {inter['sensor']['per_seed_pp']}. 단순 효과들을 각각 별개의 전체 원인 비중으로 더하면 안 된다.",'',
 '해석',*conclusion['findings'],'',
 '남은 문제',*conclusion['limits'],'',
 '다음 보완 방향 — 아직 새 실험을 시작하지 않음',*conclusion['next_plan'],'',
 '실행·검산',f"공개 TRAIN899/DEV300,100단어,원래 초기값20261013/14/15. R10/R01각3회 총6회; 실제 {result['actual_epochs']}epoch/{result['actual_batches']}batch/{result['actual_source_exposures']}원본노출. 원본노출은 같은899개를 반복 학습한 횟수이며 새로운영상개수가 아니다.",
 '최대80회·patience12·평균DEV교차엔트로피최소 선택 규칙,원래C와같은학습순서/모델/최적화기를유지. 완료된C/Q3회씩은저장점수재사용,재학습·재추론하지않았다. 새모델결과도선택시저장한점수를검산했으며최종재추론없음.',
 '입력8392개/155821120채널 별도 검산. 손 로컬XY 외 body/global XY·모든관측마스크·시간·동작구간불변. 임의좌표보충/개별단어예외/사용자촬영사용없음. 모델 초기해시·실제배치원본ID·선택epoch·체크포인트·저장logit을별도코드로검산. 같은Codex작성/별도검산이며독립사람·다른에이전트검증으로표현하지않음.',
 '기존C·웹·DTW·사용자/멘토자료는그대로. 원천영상/과거동결자료읽기전용,카메라·다운로드·외부전송·Git변경없음. 이번교차조건은원인진단용이며운영모델자동교체대상이아님.']
 (U/'RESULT_KO.txt').write_text('\n'.join(parts)+'\n',encoding='utf8')
 matrix=table(['손 배정 ↓ / 크기 기준 →','기존 크기','보완 후 계산한 크기'],[['기존 손 배정',f"C {s['C']['sensor']['accuracy_mean']:.2f}%",f"분리② {s['R01']['sensor']['accuracy_mean']:.2f}%"],['보완 손 배정',f"분리① {s['R10']['sensor']['accuracy_mean']:.2f}%",f"Q {s['Q']['sensor']['accuracy_mean']:.2f}%"]])
 def paras(items):return ''.join('<p>'+html.escape(x)+'</p>' for x in items)
 people=[]
 for a in ['C','R10','R01','Q']:
  mm=r['metrics'][a]['sensor'];people.append([label(a)]+[f"{sum(x['signers'][str(sg)] for x in mm)/3:.2f}%" for sg in [11,12,13]])
 group_rows=[]
 for a in ['C','R10','R01','Q']:
  mm=r['metrics'][a]['sensor']
  values=[f"{100*sum(x['groups'][g]['correct'] for x in mm)/sum(x['groups'][g]['total'] for x in mm):.2f}%" for g in ['internal_reentry','one_hand_never_usable','other']]
  group_rows.append([label(a)]+values+[f"{sum(x['top5'] for x in mm)/9:.2f}%"])
 page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 배정·크기 분리 6회 비교</title><style>body{margin:0;background:#edf3f7;color:#163249;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1120px;margin:40px auto;padding:0 24px 70px}header{background:#154357;color:white;border-radius:16px;padding:32px}h1{font-size:32px;line-height:1.45}h2{font-size:23px;line-height:1.5}section{background:white;border-radius:14px;padding:28px;margin-top:22px}p{margin:12px 0}table{width:100%;border-collapse:collapse;font-size:16px}th,td{text-align:left;padding:13px;border-bottom:1px solid #dce4eb}th{background:#eaf3f6}a{color:#006887}.lead{font-size:21px;font-weight:700}@media print{body{background:white;font-size:11pt}main{max-width:none;margin:0}header{color:#163249;background:white;padding:0}section{break-inside:avoid;padding:12px 0}table{font-size:10pt}}</style><main><header><h1>손 배정과 손 크기 기준,<br>무엇이 결과를 바꿨나?</h1><p>승인된6회 비교 완료 · 공개 개발 검증 · 기존 C 유지</p></header>'''
 page+='<section><p class="lead">'+html.escape(conclusion['summary'])+'</p>'+matrix+'<p>표의 값은 화면 안에서만 검출한 조건의3회 평균입니다. 실제 웹캠 사용자 정확도가 아닙니다.</p></section>'
 page+='<section><h2>같은 자료와 같은 시작 조건으로 비교</h2>'+table(['조건','각300개 중 정답(3회)','주평가 평균','실행별 범위','보조평가 평균'],rows)+paras(parts[7:9])+'</section>'
 page+='<section><h2>어떤 변화가 영향을 줬나</h2>'+table(['바꾼 요소','초기값별 차이(%p)','주평가 평균 차이','보조평가 차이'],contrast_rows)+paras(conclusion['findings'])+'</section>'
 page+='<section><h2>촬영자별 결과도 함께 확인</h2>'+table(['조건','촬영자11','촬영자12','촬영자13'],people)+'<p>각 촬영자100개 영상의3회 평균. 세 사람의 개발 검증이며 전체 사용자 집단을 대표하지 않습니다.</p></section>'
 page+='<section><h2>관측 상태별 결과</h2>'+table(['조건','동작 중 재등장 41개','한 손 관측 없음 96개','그 외 163개','전체 상위 5위 포함'],group_rows)+'<p>기존 C의 관측으로 미리 나눈 집합입니다. ‘한 손 관측 없음’은 한 손이 사용 가능한 관절로 잡히지 않았다는 뜻이며, 한 손만 쓰는 수어의 분류표가 아닙니다. 값은 각각 같은 영상에 대한 세 실행의 평균입니다.</p></section>'
 page+='<section><h2>남은 문제와 해석의 한계</h2>'+paras(conclusion['limits'])+'</section><section><h2>다음 보완 방향</h2>'+paras(conclusion['next_plan'])+'</section>'
 page+='<section><h2>검산과 보존</h2>'+paras(parts[-5:])+'<p><a href="COMPARISON_INDEPENDENT.json">저장 점수·추적 검산</a> · <a href="INPUTS_INDEPENDENT.json">입력 검산</a> · <a href="FROZEN_PROTOCOL.json">동결 비교조건</a> · <a href="INTERPRETATION.json">해석·다음 계획</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p></section></main></html>'
 (U/'RESULT_KO.html').write_text(page,encoding='utf8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
