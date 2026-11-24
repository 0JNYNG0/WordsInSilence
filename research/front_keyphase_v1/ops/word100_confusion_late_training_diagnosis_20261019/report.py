from common import *
import html
def esc(x):return html.escape(str(x))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def main():
 a=read(U/'SAVED_ANALYSIS.json');q=read(U/'INPUT_QUALITY.json');m=read(U/'LATE_MECHANISM.json');v=read(U/'VISUAL_REVIEW.json');n=read(U/'NEXT_PLAN.json');verified=read(U/'VERIFICATION.json');assert verified['status'].startswith('PASS')
 summary='반복 오답은 손 미검출만의 문제가 아니었다. 사용하는 손이 계속 관측되는데도 다른 단어를 고른 사례가 있고, 같은 단어라도 특정 촬영자 영상에 오답이 집중됐다. 학습 후반의 흔들림은 기존 C에도 존재하며, 후반 평균은 좋아지면서 회차별 결과가 오르내렸다. 다음은 좌우·크기 규칙을 더 늘리기보다 학습 묶음의 불균형부터 한 조건으로 검증하는 방향을 제안한다.'
 cases=[]
 for x in v['cases']:
  cid=x['clip'];r=a['clips'][cid];st=q['quality'][cid]['views']['sensor']['annotation_core'];det=f"좌 {100*st['left_observed_fraction']:.1f}% / 우 {100*st['right_observed_fraction']:.1f}%"
  cases.append([r['label'],str(r['signer']),f"{r['wrong']}/12",' · '.join(r['wrong_predictions']),det])
 late=[]
 for arm in ['C','Q','R10','R01']:
  rs=[r for r in a['runs'] if r['arm']==arm];lo=min(r['late']['sensor_range'][0] for r in rs)/3;hi=max(r['late']['sensor_range'][1] for r in rs)/3;delta=max(r['late_adjacent_max_change'] for r in rs)/3
  late.append([arm,f'{lo:.2f}–{hi:.2f}%',f'{delta:.2f}%p', ' / '.join(str(r['selected_epoch']) for r in rs)])
 findings=[
 '저장된 C·Q·분리①·분리② 각각 3개, 총 12개 모델의 같은 공개 DEV300 예측을 분석했다. 3,600개 예측 중 오답은 62건이며, 서로 다른 영상은 17개다. 새로운 3,600명 또는 3,600개 원본 시험이 아니다.',
 '오답 62건 중 46건은 촬영자 ID12에서 발생했다. ID11은 15건, ID13은 1건이다. 반복 오답 상위 네 단어의 다른 두 DEV 촬영자 영상은 12개 모델 모두 맞혔다. 단어 전체를 아예 못 배웠다기보다 특정 수행 방식에 대한 일반화가 부족한 패턴이다. 이 숫자만으로 사람의 수어가 잘못됐다고 판단하지 않는다.',
 '오답 62건 중 정답이 2위인 경우는 37건, 상위 5위 안인 경우는 52건이다. 완전히 무관한 결과만 나온 것이 아니라 가까운 후보 사이의 순위가 뒤집히는 경우도 많았다. 후보 점수는 동작의 올바름을 판정하는 확률이 아니다.',
 '백육십이와 흡연실의 문제 영상은 주석 구간에서 사용하는 오른손이 100% 관측됐고 구간도 모두 모델에 들어갔다. 속다는 오른손 45/46프레임이 관측됐으며, 제외된 끝 1프레임은 손이 사라진 시점이다. 관절이 있다는 것은 좌표가 정확하다는 뜻은 아니지만, 전체 오류를 긴 미검출이나 ROI 잘림 탓으로 묶을 수는 없다.',
 '영상 표본에서는 같은 단어라도 손 방향·높이·유지 시간과 교대·반복 흐름이 달랐다. 동대문–이집트의 일부 양손 자세, 숫자 표현의 시작 모양처럼 비슷한 부분과 구별해야 할 뒤쪽 단계가 공존했다. 이는 구분력 보완의 단서이며 단독 원인 확정이나 언어 전문가의 정오 판정은 아니다.'
 ]
 training=[
 '분석 구간을 학습 61~80회로 고정했다. C는 초기값에 따라 이 구간에서 279~295, 287~299, 285~295/300으로 움직였고, 인접 회차의 최대 차이는 13개(4.33%p)였다. 새로운 보완안에만 생긴 현상이 아니다. 고정된 C가 같은 입력에 매번 무작위로 답한다는 뜻도 아니다. 학습 중 가중치·내부 통계가 바뀌며 나타난 변화다.',
 '12개 실행 모두 41~60회보다 61~80회의 평균 정답 수가 높고 평균 검증 손실은 낮았다. 따라서 단순히 오래 학습할수록 계속 나빠지는 과적합으로 설명할 증거는 부족하다. 평균 개선과 단기 흔들림이 함께 존재한다.',
 '같은 초기값에서 C와 다른 조건들의 후반 정확도 곡선 상관은 0.63~0.95였다. 학습 순서·관측 일정·최적화기·정규화처럼 공통된 과정이 영향을 줄 가능성을 뒷받침하지만, 상관관계만으로 어느 요소가 원인인지 분리할 수는 없다.',
 '코드상 학습률은 끝까지 0.001이며 감소 일정이 없다. 실제 프레임만 사용하는 MaskedBatchNorm의 갱신 비율은 매 배치 0.1이다. 현재 28개 배치는 영상 32개, 마지막 배치는 3개인데, 마지막 3개 배치도 내부 통계를 10% 갱신하고 한 번의 최적화 단계를 수행한다. 최근 5개 배치의 통계 가중치는 약 40.95%다. 이는 확인된 계산 구조이고, 정확도 변동의 원인이라는 주장은 아직 가설이다. 패딩이 통계에 섞이는 과거 오류는 현재 코드에서 이미 제외되어 있다.',
 '전체/핵심 구간과 두 관측 조건이 4회 주기로 바뀌지만, 특정 주기 위치가 항상 나쁜 패턴은 세 초기값에서 공통으로 나타나지 않았다. 일정만 원인이라고 단정하지 않는다. 추론 시에는 eval 모드라 dropout이 꺼진다.',
 '모델 선택은 정답 개수 최대가 아니라 두 관측 조건의 평균 교차엔트로피 최소다. 12개 중 6개 실행은 선택된 회차와 후반의 최대 정답 회차가 다르며 차이는 1~3개다. 이번 분석은 선택 규칙의 차이를 설명할 뿐, 과거 정답 수가 높은 회차로 교체하지 않았다. 회차별 개별 영상 예측은 저장되어 있지 않아 어느 영상이 매 회차 뒤집혔는지는 재구성하지 않았다.'
 ]
 plan=[n['priority_1']['reason'],n['priority_1']['candidate'],n['priority_1']['fixed'],n['priority_1']['validation_before_training'],n['priority_1']['after_approval_only'],n['priority_1']['no_promise'],n['priority_2']['proposal'],n['priority_2']['guard'],'추가 학습은 사용자와 이 결과를 검토한 뒤 진행한다. 기존 C·웹·DTW 및 사용자·멘토 촬영은 그대로 유지했다.']
 texts=['공개 자료 반복 혼동·학습 후반 변동 진단','2026-10-19 · 추가 학습 0 · 모델/MediaPipe 추론 0 · 기존 C 유지','',summary,'','반복 혼동',*findings,'',*[' | '.join(row) for row in cases],'','학습 후반',*training,'',*[' | '.join(row) for row in late],'','다음 공통 보완 계획',*plan,'','검증과 한계','공개 관측 자료 345개(DEV300+관련 TRAIN45), RGB25영상의 표본200장, 저장 예측3600개, 학습 기록960회차를 분석. 별도 산술 검산과 모델/근거 해시 확인 완료. 같은 Codex의 별도 검산이며 독립 사람/전문가/다른 에이전트 평가가 아니다. 반복 사용한 DEV에 대한 원인 진단이며 새로운 사용자 일반화나 성능 향상을 입증한 실험이 아니다.']
 (U/'RESULT_KO.txt').write_text('\n'.join(texts)+'\n',encoding='utf8')
 def paras(xs):return ''.join('<p>'+esc(x)+'</p>' for x in xs)
 page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>반복 혼동과 학습 후반 진단</title><style>body{background:#eef3f6;color:#183442;font:17px/1.8 "Malgun Gothic",sans-serif;margin:0}main{max-width:1180px;margin:36px auto;padding:0 24px 70px}header{background:#17495b;color:white;padding:30px;border-radius:16px}h1{font-size:31px;line-height:1.5}h2{font-size:24px}section{background:white;border-radius:14px;padding:28px;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px;border-bottom:1px solid #d6e0e6;text-align:left}th{background:#e7f0f3}img{max-width:100%;height:auto}a{color:#006d89}summary{cursor:pointer;font-weight:700;padding:12px;background:#edf4f7}details{margin:16px 0}.lead{font-size:20px;font-weight:700}.badge{padding:6px 10px;background:#d7edf1;border-radius:7px;display:inline-block;color:#17495b}.note{border-left:4px solid #d09433;padding-left:18px}</style><main><header><h1>왜 같은 동작에서 반복해서 헷갈릴까?<br>왜 학습 후반 결과가 오르내릴까?</h1><p>공개 자료와 기존 저장 기록으로 분석 · 2026-10-19</p></header>'''
 page+='<section><span class="badge">추가 학습·모델 변경 없음</span><p class="lead">'+esc(summary)+'</p></section>'
 page+='<section><h2>1. 반복 오답은 일부 영상에 집중</h2>'+table(['단어','촬영자 ID','오답 / 12개 모델','주요 혼동 후보','핵심 구간 손 관측'],cases)+'<p>동일 영상에 대한 4조건×3초기값이다. ‘손 관측’은 사용 가능한 관절 3개 이상이 있는 프레임 비율이며, 실제 손 모양 정확도가 아니다.</p><img src="repeated_errors.png" alt="반복 오답 상위 5개">'+paras(findings)+'</section>'
 page+='<section><h2>2. 실제 영상과 관절 근거</h2><p>전체 영상을 새로 검출하지 않고, 원본 RGB와 저장 관절을 대조했다. 아래 그림을 누르면 크게 볼 수 있다. 노란 선은 실제 모델 입력 하단 경계다.</p>'
 for case in v['cases']:
  cid=case['clip'];sheets=[s for s in q['sheets'] if s['label']==case['label']];dev=next(s for s in sheets if s['clip']==cid)
  page+='<details><summary>'+esc(case['label'])+' — 관측과 해석</summary>'+paras([case['observed'],case['detection'],case['inference']])+f'<a href="visual/{cid}.jpg"><img src="visual/{cid}.jpg" alt="{esc(case["label"])} 공개 문제 영상"></a><p>'
  for s in sheets:page+=f'<a href="visual/{s["clip"]}.jpg">촬영자 {s["signer"]} 장면</a> · <a href="public_video_cache/{s["clip"]}.mp4">영상</a>　'
  wronglabel=sorted(a['clips'][cid]['wrong_predictions'],key=lambda k:(-a['clips'][cid]['wrong_predictions'][k],k))[0];wrong=next(s for s in q['sheets'] if s['label']==wronglabel and s['reason']=='frequent_confusion_train')
  page+=f'</p><p>주 혼동 참고: <a href="visual/{wrong["clip"]}.jpg">{esc(wronglabel)} TRAIN 장면</a> · <a href="public_video_cache/{wrong["clip"]}.mp4">영상</a></p></details>'
 page+='</section><section><h2>3. 후반 흔들림은 기존 C에도 존재</h2><img src="late_curves.png" alt="고정한 후반20회 학습 곡선"><p>세 그래프는 서로 다른 초기값이다. 정확도 축은 90~100%이며 모델을 바꾸어 적용한 결과가 아니라 저장된 학습 도중 기록이다.</p>'+table(['조건','후반 범위(3회 실행 전체)','인접 회차 최대 변화','선택된 회차(3개)'],late)+paras(training)+'</section>'
 page+='<section><h2>4. 다음은 작은 공통 변경 하나부터</h2><p class="lead">32개×28묶음 + 마지막 3개 → 31개×29묶음</p><p>자료 899개, 학습 단계 29회, 원본 순서는 그대로 두고 마지막 묶음만 유난히 작은 구조를 없애는 비교안이다.</p>'+paras(plan)+'<p class="note">현재는 원인 후보를 좁힌 단계다. 통계 구조를 고치면 정확도가 얼마나 오른다는 기댓값은 아직 측정하지 않았다.</p></section>'
 page+='<section><h2>검증 기록</h2>'+paras(texts[-1:])+'<p><a href="VERIFICATION.json">별도 산술 검산</a> · <a href="NEXT_PLAN.json">다음 계획</a> · <a href="SAVED_ANALYSIS.json">전체 저장 기록 분석</a> · <a href="INPUT_QUALITY.json">관측 검토</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p></section></main></html>'
 (U/'RESULT_KO.html').write_text(page,encoding='utf8')
 write(U/'SUMMARY.json',{'at':now(),'summary':summary,'additional_training':0,'inference':0,'errors':62,'unique_error_clips':17,'id12_errors':46,'focus_errors':38,'C_late_adjacent_max_pp':13/3,'uniform_batch_plan_only':True,'production_changed':False,'user_data_used':False})
 print('REPORT_COMPLETE')
if __name__=='__main__':main()
