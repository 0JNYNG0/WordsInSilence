from pathlib import Path
import json,html
from collections import Counter
U=Path(__file__).resolve().parent
read=lambda n:json.loads((U/n).read_text('utf-8-sig'))
def write(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
s=read('SUMMARY.json');stress=read('STRESS.json');base=read('BASELINE.json')
assert read('VERIFICATION.json')['status']=='PASS' and read('ENVELOPE_VERIFICATION.json')['status']=='PASS'
post={c:sum(r['g1']['status']!=r['fixed_g1']['status'] for r in stress if r['condition']==c and r['removed_frames']>0) for c in ('whole','center10','peak10','peak20')}
write('POSTCHECK.json',dict(dynamic_fixed_status_disagreement=post,
    clarification='Original SUMMARY dynamic_vs_fixed_disagree compares entire reason dictionaries, not just labels; e.g.whole1503 reason differences but0 status differences. Original results preserved; this report uses status differences.',
    effect='No changes to criteria, thresholds, observations or experiment results.',
    one_hand_controls='3 prior reviewed right-hand words,3 DEV signers,2 views,3metrics=54metric-hand checks; not54independent people or videos.',
    repeat_limits='Repeat duplicates whole source observation including idle periods. It is a stress test for sensor absence, not certified permissible repetition of each sign.'))
decision=dict(status='VALIDATED_OFFLINE_QUANTILE_GATE_NOT_ADOPTED',
    retain='C/T3-F, live DTW numbers and web display unchanged',
    proposed_levels=[
        dict(code='NOT_REQUIRED',label='이 손은 비교 대상 아님',criterion='Only existing public reference-declared optional hand; never infer from query absence.'),
        dict(code='INSUFFICIENT',label='비교 근거 부족',criterion='Required/assessed hand lacks a time envelope, has fewer8 observed frames, or fewer5 references meeting original50%/8distinct frame support. Per-hand only when roles unreviewed.'),
        dict(code='PARTIAL',label='일부 구간만 비교 · 전체 수행 판단 보류',criterion='Any unresolved interior evidence gap even when average difference is small; missingness does not mean a sign error.'),
        dict(code='OBSERVED_ENVELOPE',label='관측된 범위 안에서는 공백 없음',criterion='No interior gap; never a complete-action/pass claim. Leading/trailing absence remains unassessed.')],
    quantitative_cutoffs=s['thresholds'],quantitative_gate_adopted=False,
    reason='TRAIN-tail thresholds miss short losses and respond to idle gaps introduced between repetitions. End truncation can shrink the query-derived envelope.',
    next='Before any deployment of a stricter gate, test multiple public TRAIN references to estimate expected observed-action extent independent of query disappearance; distinguish reference-supported boundary rest from unresolved interior gaps. Validate original DEV, one-hand roles, speed/repeat and short losses with conditions frozen. No training or private data.',
    semantic_error_classifier=False,training=0,deployment=False)
write('DECISION.json',decision)
text='''공개자료 DTW 비교 근거 부족 기준 검증

결론: 관측률 하나로 충분/부족을 확정하는 방식은 채택하지 않는다. 관측이 전혀 없거나 기존 최소 비교 조건을 충족하지 못하면 “비교 근거 부족”, 공백이 남으면 “일부 구간만 비교·전체 수행 판단 보류”로 구분하는 방향이 타당하다. 낮은 차이 수치는 이 경고를 해제하지 않는다. 정상 손내림·한 손 동작을 잘못된 수어라고 판정하지 않는다. 이번 단위는 기준 후보의 공개 검증이며 웹·모델 적용과 추가 학습은 하지 않았다.

1. 범위와 검증 방법
공개 AI Hub100단어1199영상: TRAIN899(촬영자01~09)에서만 참고 기준 계산, DEV300(11~13)에서 점검. 기존 DTW 관측/상반신 sensor관측 두 경로. DEV는 이전에도 사용한 개발자료로, 새 최종평가가 아니다. 기존 C/T3-F6가중치·DTW·웹을 보존했다. 사용자/멘토/보류14~18 자료 미접근, 새 추출·학습·분류추론 없음.
G0는 기존 최소 비교 근거(8관측 프레임,5참고자료 등)와 내부공백을 분리한다. G1은 G0에 TRAIN관측비율 하위5%와 가장 긴 공백비율 상위5%를 경계로 추가한 후보다. 두 경로·좌우를 합쳐 항목별 공통값을 만들고 DEV 확인 전에 동결했다. 단어별 임계값이나 DEV결과에 맞춘 재조정은 없다. TRAIN도 정상 수어의 정답 주석은 아니므로 이 경계는 통계적 참고선일 뿐이다.

2. 수치 기준을 그대로 채택하지 않는 이유
시험한 관측비율 하한은 굽힘76.79%, 위치·궤적82.50%; 가장 긴 내부공백/관측범위 길이 상한은 각각13.64%,10.78%였다. 이 숫자를 사용자 합격선으로 권장하지 않는다.
전체손을 가린1773항목·손 조건 모두 부족 처리; 이 중 원래 실제 근거가 있어 제거된1508조건도 모두 부족이다. 그러나 짧은 구간을 놓친 경우에는 고정 비율만으로 충분히 잡지 못했다. 기존 연구에서 손목 이동량이 큰10%구간을 가린 뒤 차이 평균이 오히려 낮아진630항목·손 사례 중, G1은147만 부족,481은일부비교,2는줄어든 관측범위 안 공백없음으로 남았다. 이630은3항목의 합이며630명/영상이 아니다. 이전 부분15각도·고정경로 연구값을 별도 증거로 재사용한 수치이고, 운영19각도 DTW를 재채점하거나 새 인식률을 측정한 결과가 아니다.
예: 공개 ‘순종’ REAL13 오른손의 끝부분 관측7프레임을 가리면, 관측범위가1.67~4.07초에서1.67~3.83초로 짧아졌다. 새 범위 안 관측률은100%지만 원래 범위를 유지하면90.97%다. 이전 연구 궤적 차이는15.06→14.19로 작아졌다. 이것은 잘 따라한 증거가 아니라 끝부분을 빠뜨린 평균의 한계다. 마스크는 합성 검출누락이며 사람이 실제 필수동작을 생략했다고 판정한 것이 아니다.

3. 손내림·한손·속도·반복 확인
준비·마무리 양끝에 미관측 구간을 추가한3600항목·손 대조에서 상태가 유지됐다. 이 경계 처리로 촬영 전후 대기만 늘어도 불리해지는 현상을 막는다. 다만 양끝의 실제 동작 누락까지 ‘정상 휴식’이라고 자동 간주하면 안 된다.
기존 공개 참고에서 오른손만 필요하다고 확인한3단어의9개DEV영상×2경로×3항목=54조건에서 반대손 전체미관측은비교대상아님으로 유지했다. 다른88단어의 필요한 손은 미확인이고, 보수적 양손 수치를 단어 전체의 오답 판정에 쓰지 않았다. 움직임 없는 받침손을 임의 제외하지 않았다.
시간축0.75/1.5배7200항목·손 대조에서 상태가 같았다. 실제 빠른 영상의 MediaPipe검출성능까지 검증한 것은 아니다.
같은 관측을 두 번 이어붙인3600반복 대조에서1244조건이 새로부족이 됐다. 모두sensor경로였고, 두 수행 사이의 손을 내린 구간이 내부공백이 되었기 때문이다. 반복 횟수가 잘못됐다는 판정은 만들지 않았지만, 이 결과는 단순 비율 기준이 휴식에 민감함을 보여준다. 전체 영상을 이어붙인 합성 대조이며 각 수어의 반복허용 여부를 인증한 것은 아니다.
원본 전체화면의 안정된 손내림 추정근거가 내부공백과 함께 있는8영상·손/24항목 중16항목이G1부족이었다. 다른 센서누락도 함께 있을 수 있어 이16을 모두 오탐으로 세지 않는다. 원본 밖 손 정보는 실제 웹캠에는 없으므로 게이트에 입력하지 않고 오프라인 진단에만 썼다.

4. 제안하는 화면 해석
비교 대상 아닌 손: 기존 공개 참고 역할에 근거해 제외. 안 보인다는 이유만으로 제외하지 않는다.
비교 근거 부족: 필요한 손의 해당 정보가 없거나 기존 최소 비교 조건 미충족. 손을 잘못 움직였다는 뜻이 아니라 판단할 자료가 부족하다는 뜻이다.
일부 구간만 비교: 차이 수치를 남기되 공백이 있으므로 전체 동작은 판단 보류. 굽힘·위치·궤적을 따로 처리한다.
관측된 범위 안 공백 없음: 이 범위의 관측이 이어졌다는 뜻일 뿐, 전체 동작이나 필수단계가 완료됐다는 뜻은 아니다. 끝부분 공백은 별도로 남긴다.
어느 상태에도 ‘정답/오답/잘못된반복’ 자동 판정을 붙이지 않는다. 관측률 몇%이면 충분하다는 보장은 아직 없다.

5. 다음 보완
관측이 사라졌다는 이유로 비교 대상 구간까지 줄지 않도록, 여러 사람의 공개 참고에서 공통된 동작 구간 근거를 이용하는 방법부터 별도 검증한다. 정상 준비·마무리/내림으로 확인 가능한 구간과 원인을 모르는 구간은 분리한다. 전 단어 수동 의미주석이나 사용자 맞춤 튜닝을 선행조건으로 만들지 않는다. 현재 결과를 사용자와 검토한 뒤 다음 구현 범위를 정한다. 추가 학습 없음.

검증 담당: Codex 구현, 같은 Codex의 별도 scalar계산/원본시각 적분 검산. 다른에이전트 검토 아님. 28476상태결정 검산,3600원본시간 적분 일치(최대2.85e-14%p),6135입력·보호 해시확인. 통계 검증 통과와 후보 채택은 다르며 G1수치기준은미채택이다.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
trows=[]
for key,label in [('handshape','손가락 굽힘'),('location','손 위치'),('movement','이동 궤적')]:
    t=s['thresholds'][key];trows.append([label,f"{t['coverage_floor']:.2f}%",f"{t['gap_ceiling']*100:.2f}%"])
fallrows=[]
for key,label in [('handshape','굽힘'),('location','위치'),('movement','궤적')]:
    rr=[r for r in stress if r['condition']=='peak10' and r['key']==key and r['removed_frames']>0 and r['witness_score_decreased']];c=Counter(r['g1']['status'] for r in rr)
    fallrows.append([label,len(rr),c['INSUFFICIENT'],c['PARTIAL'],c['OBSERVED_ENVELOPE']])
pages='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DTW 비교 근거 부족 기준 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#162238;font:17px/1.7 "Malgun Gothic",sans-serif}main{max-width:1100px;margin:auto;padding:36px 26px 60px}header{background:#102a43;color:#fff;border-radius:18px;padding:32px}h1{font-size:32px;line-height:1.4;margin:10px 0 18px}h2{font-size:24px;line-height:1.4;margin:0 0 18px}h3{font-size:19px;margin:0 0 12px}p{margin:10px 0}section{background:white;border-radius:16px;padding:28px;margin-top:22px;border:1px solid #dce4ec}.eyebrow{font-size:14px;font-weight:bold;letter-spacing:1px;color:#9fd5ed}.lead{font-size:21px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:#f3f7fb;padding:20px;border-radius:12px}.warn{border-left:5px solid #d59022;background:#fff7e8;padding:18px 22px;border-radius:8px}.tag{display:inline-block;padding:4px 12px;border-radius:8px;background:#f5d79a;color:#533800;font-size:15px;font-weight:bold}.small{font-size:15px;color:#4d6174}table{border-collapse:collapse;width:100%;font-size:16px}td,th{padding:13px 10px;text-align:left;border-bottom:1px solid #dce4ec}th{background:#eef3f8}td:first-child{font-weight:bold}.scroll{overflow-x:auto}a{color:#176f9f}.flow{display:flex;gap:12px;align-items:stretch;margin:18px 0}.flow>div{flex:1;background:#eaf3fa;border-radius:10px;padding:18px}.big{font-size:30px;font-weight:bold;color:#0b577e}.bar{display:flex;height:38px;border-radius:6px;overflow:hidden;margin:16px 0}.bar span{display:flex;align-items:center;justify-content:center;color:white;font-size:15px;font-weight:bold}.number{font-size:25px;font-weight:bold}details{margin-top:20px}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;font:15px/1.8 "Malgun Gothic",sans-serif}footer{margin-top:22px;color:#425770;font-size:15px}@media(max-width:650px){main{padding:18px 12px}header,section{padding:22px 18px}h1{font-size:27px}.grid{grid-template-columns:1fr}.flow{flex-direction:column}table{min-width:570px}.bar span{font-size:12px}.lead{font-size:18px}}
</style><main><header><div class="eyebrow">WORDS IN SILENCE · 공개자료 기준 검증 · 2026.10.20</div><h1>차이가 작아도,<br>충분히 보고 비교했을까?</h1><p class="lead">관측이 없는 경우와 일부만 비교한 경우를 구분합니다.<br>관측률만으로 충분하다고 확정하는 수치 기준은 채택하지 않았습니다.</p><span class="tag">오프라인 검증 완료 · 웹/모델 변경 없음</span></header>
<section><h2>이번 결론</h2><div class="grid"><div class="card"><h3>구분할 수 있는 것</h3><p>비교할 관절·참고자료가 없는 경우, 관측이 끊긴 구간, 참고에서 사용하지 않는 손.</p></div><div class="card"><h3>아직 단정할 수 없는 것</h3><p>보이지 않은 동안 실제 동작이 맞았는지, 끝부분이 정상 손내림인지 필수동작 누락인지.</p></div></div><p class="warn"><b>비교 근거 부족 ≠ 잘못된 수어.</b><br>정상적인 동작이어도 카메라가 못 보면 확인할 수 없습니다. 낮은 차이 점수로 이 한계를 덮지 않습니다.</p></section>
<section><h2>무엇을 검증했나</h2><div class="flow"><div><div class="big">899영상</div>공개 학습 촬영자9명<br>공통 참고 기준 계산</div><div><div class="big">300영상</div>다른 공개 촬영자3명<br>고정한 기준 점검</div><div><div class="big">2경로</div>기존 DTW 관측<br>상반신 sensor 관측</div></div><p>100단어를 대상으로, 사용자 자료 없이 진행했습니다. 개발자료는 이전에도 사용한 자료이므로 새로운 최종 시험 성적은 아닙니다.</p></section>
<section><h2>시험한 수치 기준 — 그대로 사용하지 않음</h2><p>학습 자료에서 관측 비율 하위5%와 가장 긴 공백 비율 상위5%를 공통 경계로 계산했습니다. 단어마다 값을 맞추지 않았고, 개발 결과를 본 뒤 고치지 않았습니다.</p><div class="scroll">'''+table(['항목','관측 비율 하한','긴 공백 / 관측 범위 길이 상한'],trows)+'''</div><p class="small">통계적 참고선을 시험한 것이며 수어 정답·오답의 합격선이 아닙니다.</p></section>
<section><h2>짧은 동작을 놓쳐도 평균 차이는 작아질 수 있음</h2><p>이전 공개 연구의 ‘움직임이 큰10% 구간 가림’에서 차이가 더 작아진630항목·손 사례를 추적했습니다.</p><div class="bar"><span style="width:23.333%;background:#aa650c">부족147</span><span style="width:76.349%;background:#296e95">일부비교481</span><span style="width:.318%;background:#bd3443"></span></div><div class="scroll">'''+table(['항목','차이가 감소한 사례','근거 부족','일부만 비교','범위가 줄어 공백 없음'],fallrows)+'''</div><p><b>2사례는 끝부분을 잃고도 새 관측 범위 안에서 공백이 없는 것으로 남았습니다.</b> 따라서 관측 범위 안100%도 전체 동작 완료를 보장하지 않습니다.</p><p class="small">630은3항목을 합친 검사 수이며630명/영상이 아닙니다. 기존 부분15각도 연구값을 별도 증거로 재사용했습니다. 운영19각도 DTW 재채점·새 인식률이 아닙니다.</p></section>
<section><h2>왜 끝부분 누락을 조심해야 하나</h2><p>공개 ‘순종’ REAL13 오른손의 끝부분7프레임을 가린 대조입니다.</p><div class="grid"><div class="card"><h3>원래 비교 범위를 유지하면</h3><p class="number">1.67~4.07초 · 관측90.97%</p><p>끝의 약0.22초 미관측을 드러낼 수 있습니다.</p></div><div class="card"><h3>새로 보이는 구간만 잡으면</h3><p class="number">1.67~3.83초 · 관측100%</p><p>비교 범위가 줄어 누락이 사라진 것처럼 보입니다.</p></div></div><p>이전 연구 궤적 차이도15.06→14.19로 감소했습니다. 이 감소는 좋아진 동작의 증거가 아닙니다.</p></section>
<section><h2>정상적인 내림·한손·속도·반복은?</h2><div class="scroll">'''+table(['대조','확인한 결과'],[
['준비·마무리 미관측 추가','3,600항목·손 조건에서 상태 유지'],
['기존에 확인한 한손3단어','반대손을 모두 가린54조건에서 비교 대상 아님 유지'],
['시간축0.75/1.5배','7,200조건에서 상태 유지'],
['관측 전체를2회 이어붙임','3,600조건 중1,244조건 새로 부족 → 수치 기준 미채택']])+'''</div><p class="warn">반복 대조에서는 두 수행 사이의 손내림이 내부공백이 되었습니다. 단순 비율 기준이 이런 대기 구간에 민감합니다. 이를 틀린 수어나 잘못된 반복으로 판정하지 않았습니다.</p><p class="small">반복 대조는 합성 연결이며 단어별 반복 허용의 언어적 검증이 아닙니다. 속도 대조도 실제 빠른 촬영의 검출 성능 검증은 아닙니다. 기존 손 역할 기록은12단어이고,88단어는 손별 진단만 했습니다.</p></section>
<section><h2>제안하는 표시 기준</h2><div class="scroll">'''+table(['표시','언제 사용하는가','해석'],[
['이 손은 비교 대상 아님','공개 참고에서 불필요한 손임이 확인됨','안 보인다는 이유만으로 정하지 않음'],
['비교 근거 부족','관측 없음 / 기존 최소 관측·참고 조건 미충족','판단할 자료가 부족함'],
['일부 구간만 비교','계산은 가능하지만 내부공백이 남음','차이 수치와 공백을 함께 보고 전체 판단 보류'],
['관측된 범위 안 공백 없음','해당 범위의 관측이 이어짐','정답·전체 동작 완료 판정은 아님']])+'''</div></section>
<section><h2>다음 보완</h2><p>여러 사람의 공개 참고를 이용해, <b>사용자의 손이 사라져도 비교할 동작 범위가 함께 줄지 않도록</b> 하는 방법부터 검증합니다. 정상 준비·마무리 근거와 원인 불명의 공백을 분리해야 합니다.</p><p>전 단어 수동 검토나 사용자 맞춤 튜닝으로 해결하지 않습니다. 이 결과를 함께 확인한 뒤 다음 구현 범위를 정하고, 추가 학습은 진행하지 않습니다.</p><details><summary>전체 근거와 한계 읽기</summary><pre>'''+html.escape(text)+'''</pre></details></section><footer>Codex 로컬 구현 및 별도 계산 검산 · C/T3-F 및 운영 DTW 보존 · 사용자 촬영 미사용<br>숫자 검산 통과와 개선 후보 채택은 별개입니다.</footer></main></html>'''
(U/'RESULT_KO.html').write_text(pages,'utf-8');print('REPORT_CREATED')
