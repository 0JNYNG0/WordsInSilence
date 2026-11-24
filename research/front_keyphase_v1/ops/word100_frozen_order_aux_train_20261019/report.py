from common import *
from html import escape
import statistics
r=read(U/'COMPARISON_INDEPENDENT.json');a=read(U/'ANALYSIS.json');plan=read(U/'NEXT_PLAN.json')
assert r['status'].startswith('PASS') and not r['primary_review_gate_vs_T3F']['pass']
s=r['summary'];names={'C':'기존 C','T3F':'기존 T3-F','FG16':'새 F-G16'}
modelrows=''.join(f'<tr><th>{names[m]}</th><td>{s[m]["sensor"]["accuracy_mean"]:.2f}%</td><td>{s[m]["masked"]["accuracy_mean"]:.2f}%</td><td>{s[m]["sensor"]["top5_mean"]:.2f}%</td></tr>' for m in names)
seedrows=''.join(f'<tr><td>{i+1}</td><td>{s["T3F"]["sensor"]["top1_counts"][i]} / 300</td><td>{s["FG16"]["sensor"]["top1_counts"][i]} / 300</td><td>0.00%p</td><td>{x["epochs"]} / {x["selected_epoch"]}</td></tr>' for i,x in enumerate(a['rows']))
errorrows=''
for c in a['sensor_error_classes']:
 terms=sorted({p['predicted'] for p in c['FG16_sensor_errors']});n=len(c['FG16_sensor_errors'])
 errorrows+=f'<tr><th>{escape(c["label"])}</th><td>{escape(" · ".join(terms))}</td><td>{n}회 / 9회</td></tr>'
groupnames={'internal_reentry':'중간에 손이 사라졌다 다시 관측된 집합','one_hand_never_usable':'한 손이 계속 미관측인 집합','other':'그 외 관측 집합'}
grouprows=''
for g,gg in r['metrics']['FG16']['sensor'][0]['groups'].items():
 den=gg['total'];n=sum(x['groups'][g]['correct'] for x in r['metrics']['FG16']['sensor'])
 grouprows+=f'<tr><th>{groupnames[g]}</th><td>{den}영상</td><td>{100*n/(3*den):.2f}%</td><td>0.00%p</td></tr>'
conclusion={'at':now(),'status':'COMPLETE_NO_ADDITIONAL_ACCURACY_GAIN_KEEP_C_T3F','new_training_runs':3,
 'sensor_mean_accuracy':s['FG16']['sensor']['accuracy_mean'],'sensor_range':s['FG16']['sensor']['accuracy_range'],
 'masked_mean_accuracy':s['FG16']['masked']['accuracy_mean'],'sensor_delta_vs_T3F_pp':0.0,'sensor_delta_vs_C_pp':r['contrasts']['C']['sensor']['mean_pp'],
 'C_difference_attribution':'Already present in T3-F, not a new GRU gain','newly_correct_vs_T3F':0,'newly_incorrect_vs_T3F':0,
 'incorrect_label_swap':'One seed13 sensor 동대문 remained incorrect, predicted 응접실→칠천; top1 labels are not entirely identical.',
 'epochs':r['completed_epochs']['FG16'],'selected_epochs':r['selected_epochs']['FG16'],
 'late_window_61_80_available':False,'training_seconds_sum':a['training_seconds_sum'],
 'actual_batches':sum(t['actual_batches'] for t in r['traces'].values()),'actual_source_exposures':sum(t['actual_source_exposures'] for t in r['traces'].values()),
 'independent_saved_predictions':5400,'all_frozen_T3F_tensors_preserved':True,'gate_pass':False,'user_recordings_read':0,'production_changed':False,
 'next':'Keep existing C/T3F. Review separate auxiliary learning objective before further training; no automatic reruns.'}
write(U/'CONCLUSION.json',conclusion,new=True)
text=f'''F-G16 순서 기억 보조 3회 학습·비교 결과

결론: 추가 인식률 개선이 확인되지 않아 F-G16은 채택하지 않고 기존 C·T3-F를 유지합니다.

같은 공개 TRAIN899영상(100단어, 촬영자9명) / DEV300영상(다른촬영자3명), 같은3seed와입력일정으로 보조4000변수만학습했습니다. 기존T3-F47236변수와BN는모든학습단계에서고정됐습니다.

3회 평균 DEV정확도:
  화면제한재검출(sensor): C98.67%, T3-F99.22%, F-G16 99.22%.
  전체검출뒤영역적용(masked): C98.44%, T3-F99.00%, F-G16 99.00%.
  sensor상위5위: C99.67%, T3-F99.89%, F-G16 99.89%.
F-G16의sensor3회정답수296/300,299/300,298/300(범위98.67~99.67%)로T3-F와동일합니다. T3-F대비+0.00%p. C대비+0.56%p는기존T3-F의개선을이어받은값이며이번GRU추가효과가아닙니다.

선택체크포인트에서T3-F대비새로맞힌영상0,새로틀린영상0(양view모든seed). 다만seed13동대문오답1건의예측단어가응접실에서칠천으로바뀌었습니다. 점수가완전히동일한것은아닙니다. 실제head가중치/점수는변했고정답순위는그대로였습니다.
sensor남은오답: 속다→외상값,동대문→칠천/이집트,상처→믿다,남해군지부→수어,별→고양,백육십이→육백만. 고유6영상에서3seed합계7오답이며새사람7명이아닙니다. 사람별·관측누락집합별정답수도T3-F와같습니다.
믿다·사육장은공개DEV에서C/T3-F/F-G16모두각3영상×3seed=9/9정답입니다. 이집합에서는이미정답이어서실사용혼동이해결됐다는증거가되지않습니다. 사용자촬영재평가없음.

왜개선이없었는가: 새부분은실제로학습됐지만,첫epoch부터학습손실이약0.000016~0.000031로매우작았습니다. 기존모델이학습단어를이미높은확신으로맞히는상태에서합친점수만단어정답으로학습하므로새보조부분에순서를더구분하도록요구하는신호가약했을가능성이있습니다. 기록에근거한추정이며원인을분리해증명한것은아닙니다. AdamW의정규화도있으므로작은기울기만으로학습이안됐다고단정하지않습니다. 단어정답라벨에는필수단계의생략/순서오류판정이없습니다. GRU가무용하거나수어순서가필요없다는결론도아닙니다.

보조적인점수품질: sensor평균교차엔트로피는0.035206→0.034982로조금낮아졌지만정답수는늘지않았습니다. 1회차는오히려교차엔트로피가소폭나빠졌습니다. 이를성능향상으로포장하지않습니다.

학습종료: 각각13/14/38epoch,선택epoch1/2/26. 미리정한12epoch개선없음조기종료규칙을적용했습니다. 예정61~80epoch구간은없어후반흔들림이줄었다고평가할수없습니다. 실제학습실행합계약{a['training_seconds_sum']:.1f}초. 전처리/검산/보고시간제외. 모든65epoch의1885배치/58435영상노출검산완료.

검증: 저장된5400예측(C/T3-F/F-G16×3seed×2view×300)을별도산술검산,학습배치ID·입력일정·선택epoch·optimizer새4000개·checkpoint·고정tensor검증PASS. 재채점없음. 같은Codex의별도검산코드이며독립연구자검증아님.

다음제안: 기존C/T3-F유지. 새로운보조부분자체도단어를구별하도록별도의보조학습목표를주는설계부터검토합니다. 계수와비교조건은추가학습전에고정하고사용자확인후진행합니다. 별도학습목표역시필수단계완수판정을보장하지않으므로그목적에는전문가가확인한공개단계/생략/순서오류자료가필요합니다.
사용자/멘토촬영읽기·학습·튜닝0,ID14~16미사용유지,기존웹·C·T3-F·DTW변경0.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F-G16 학습 비교 결과</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f9;color:#17253c;font:17px/1.75 'Malgun Gothic',sans-serif}main{max-width:1120px;margin:auto;padding:48px 28px 80px}h1{font-size:38px;line-height:1.3;margin:14px 0 20px}h2{font-size:25px;line-height:1.4;margin:0 0 16px}p{margin:12px 0}.tag{font-weight:700;color:#216d65}.lead{font-size:21px}.notice{background:#fff2d9;border-left:5px solid #d4a14c;padding:18px 24px;margin:26px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card,section{background:white;border:1px solid #dce2ed;border-radius:18px;padding:26px;margin:22px 0}.card{margin:0}.num{font-size:32px;font-weight:800;color:#2058b5}.muted{color:#526078;font-size:15px}table{width:100%;border-collapse:collapse;font-size:16px}th,td{padding:14px 10px;border-bottom:1px solid #dce2ed;text-align:left}thead{background:#edf3fc}tbody th{font-weight:600}tbody tr:last-child{background:#edf6f3}.scroll{overflow-x:auto}li{margin:10px 0}a{color:#2058b5}details{padding:18px;background:#f3f5f9;border-radius:10px;margin-top:20px}summary{cursor:pointer;font-weight:700}.small{font-size:15px}.result{font-weight:700;color:#236c64}.stages{display:grid;grid-template-columns:1fr 1fr;gap:20px}.stages div{padding:18px;background:#edf3fc;border-radius:12px}@media(max-width:700px){main{padding:24px 16px}h1{font-size:29px}.grid,.stages{grid-template-columns:1fr}section,.card{padding:20px}table{font-size:14px}th,td{padding:10px 6px}.scroll table{min-width:500px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid}}
</style><main><div class="tag">WordsInSilence · 2026.10.19 · 공개 자료 3회 비교 완료</div><h1>순서 기억 보조를 학습했지만<br>추가 인식률 개선은 없었습니다</h1><p class="lead">새 후보 F-G16의 정확도는 기존 T3-F와 같았습니다.<br>기존 C·T3-F를 유지하고, 새 후보는 채택을 보류합니다.</p><div class="notice"><b>결과를 세 가지로 구분했습니다.</b><br>구현·학습·동결 검증: 통과 / 기존 정답 보존: 확인 / T3-F보다 더 잘 맞힘: 미확인</div>
<div class="grid"><div class="card"><div class="num">99.22%</div>공개 개발 검증 평균 정확도<br><span class="muted">화면 제한 관측 · 3회 평균</span></div><div class="card"><div class="num">+0.00%p</div>기존 T3-F 대비 변화<br><span class="muted">새 정답0 · 새 오답0</span></div><div class="card"><div class="num">3회 완료</div>새 보조 부분만 학습<br><span class="muted">기존 T3-F 전체 고정</span></div></div>
<section><h2>같은 조건에서 무엇이 달라졌나요?</h2><p>공개 100단어의 <b>학습899영상(9명)과 개발 검증300영상(다른3명)</b>을 사용했습니다. 같은 3개 시작 조건으로 비교했고, 관절 추출·손 배정·화면 범위·정규화·입력 일정은 바꾸지 않았습니다.</p><div class="scroll"><table><thead><tr><th>모델</th><th>화면 제한 관측</th><th>전체 검출 후 범위 적용</th><th>화면 제한 상위5위</th></tr></thead><tbody>MODELROWS</tbody></table></div><p><b>C 대비 +0.56%p는 이미 T3-F에서 얻었던 개선입니다.</b> 이번 GRU 보조 부분이 새로 높인 정확도는 0.00%p입니다.</p><p class="small">‘화면 제한 관측(sensor)’은 공개 영상의 촬영 범위를 제한해 다시 검출한 조건입니다. 실제 사용자의 웹캠 정확도가 아닙니다. ‘전체 검출 후 범위 적용(masked)’은 전체 화면 관측에 입력 범위를 적용한 조건입니다. 같은 개발 자료를 여러 실험에 사용한 결과입니다.</p></section>
<section><h2>3회 모두 정답 수가 같았습니다</h2><div class="scroll"><table><thead><tr><th>실험</th><th>T3-F 정답</th><th>F-G16 정답</th><th>차이</th><th>완료 / 선택 epoch</th></tr></thead><tbody>SEEDROWS</tbody></table></div><p>화면 제한 조건의 정확도 범위는 <b>98.67~99.67%</b>입니다. 전체 검출 조건과 상위5위 정답 수도 기존 T3-F와 같았습니다. 새로 맞힌 영상과 새로 틀린 영상은 양 조건 모두 0건입니다.</p><p class="small">점수 자체가 모두 같은 것은 아닙니다. 첫 실험의 ‘동대문’ 오답은 예측 단어가 ‘응접실’에서 ‘칠천’으로 바뀌었지만 여전히 오답이었습니다.</p></section>
<section><h2>왜 더 좋아지지 않았을까요?</h2><div class="stages"><div><b>확인된 기록</b><p>학습 첫 회부터 손실이 약0.000016~0.000031로 매우 작았습니다. 새 가중치와 보정 점수는 실제로 변했지만 정답 수는 늘지 않았습니다.</p></div><div><b>현재의 해석</b><p>기존 모델이 학습 단어를 이미 높은 확신으로 맞혀, 합친 점수만 채점하는 방식에서는 보조 부분이 순서를 더 구분하도록 배울 신호가 약했을 가능성이 있습니다.</p></div></div><p>지금 학습의 정답은 <b>‘어떤 단어인가’</b>입니다. ‘중간 동작을 빠뜨렸는가’, ‘필수 단계의 순서가 맞는가’는 따로 표시되어 있지 않습니다. 기억장치를 붙였다는 이유만으로 이 기준까지 배우지는 않습니다.</p><p class="small">위 원인은 기록에 근거한 추정입니다. 원인을 분리한 대조 실험은 하지 않았습니다. 작은 기울기만으로 학습이 안 됐다고 단정하거나, GRU가 필요 없다고 결론 내릴 수는 없습니다. 평균 점수 손실은 0.035206→0.034982로 소폭 줄었으나 정답 수의 개선은 아닙니다.</p></section>
<section><h2>남은 오답과 관측 문제</h2><div class="scroll"><table><thead><tr><th>실제 단어</th><th>남은 오답 후보</th><th>오답 횟수</th></tr></thead><tbody>ERRORROWS</tbody></table></div><p class="small">단어당 개발 영상3개 × 실험3회 =9회 기준입니다. 총 고유6영상에서 7회의 오답이 남았습니다. 사람7명이나 새로운 영상7개를 뜻하지 않습니다.</p><div class="scroll"><table><thead><tr><th>화면 제한 관측 집합</th><th>영상 수</th><th>3회 평균 정확도</th><th>T3-F 대비</th></tr></thead><tbody>GROUPROWS</tbody></table></div><p class="small">관측 집합은 입력의 손 검출 상태에 따른 분류입니다. ‘한 손 미관측’을 반드시 한 손만 쓰는 수어라는 의미로 해석하지 않습니다. 사람별 정답 수도 T3-F와 같았습니다.</p><p><b>‘믿다·사육장’은 이 공개 개발 자료에서는 두 모델 모두 정답입니다.</b> 각3영상×3회 모두 맞혔으므로, 이번 결과만으로 실사용 혼동이 해결됐다고 말할 수 없습니다. 사용자 촬영은 다시 평가하지 않았습니다.</p></section>
<section><h2>학습 후반의 흔들림은 비교할 수 없었습니다</h2><p>미리 정한 규칙인 <b>“검증 손실이 12회 연속 개선되지 않으면 종료”</b>에 따라 각각 13·14·38회에서 학습이 끝났습니다. 선택된 시점은 1·2·26회입니다.</p><p>예정한 후반 비교 구간인 61~80회에 도달한 실험이 없습니다. 따라서 <b>후반 흔들림이 줄었다는 결론은 내리지 않습니다.</b> 결과에 맞춰 종료 조건을 바꾸거나 다시 학습하지 않았습니다.</p></section>
<section><h2>다음 보완 방향</h2><ol><li><b>기존 C·T3-F 유지.</b> 추가 정확도 개선 조건을 통과하지 못했으므로 F-G16을 웹에 적용하지 않습니다.</li><li><b>보조 부분의 학습 목표부터 검토.</b> 기존 점수에 조금 더하는 것뿐 아니라 보조 출력 자체도 단어를 구별하도록 별도의 학습 신호를 주는 설계를 검토합니다.</li><li><b>새 학습은 계획 확인 뒤.</b> 손실 계수와 비교 조건을 먼저 고정하고, 모든 실험 결과와 기존 정답이 오답으로 바뀌는지도 보고합니다.</li></ol><p>단어 분류를 보강하는 것과 수어를 빠짐없이 올바르게 수행했는지 판정하는 것은 구분해야 합니다. 후자를 검증하려면 전문가가 확인한 공개 정상·생략·순서 오류 사례가 필요합니다.</p><p class="result">사용자·멘토 촬영 읽기0 / 학습·튜닝0 / 기존 웹·C·T3-F·DTW 변경0</p><details><summary>검증 근거와 상세 기록</summary><p>3회 합계65epoch의 1,885배치에서 실제 학습 영상 ID·입력 일정·58,435회 노출을 검산했습니다. 기존 T3-F 가중치와 정규화 통계는 매 단계 고정됐고, optimizer는 새4,000변수만 갱신했습니다. 기존·새 모델의 저장 예측5,400건을 별도 계산으로 확인했으며 최종 점수를 재추론하지 않았습니다.</p><p>같은 Codex의 별도 검산 코드이며 독립 연구자나 수어 전문가 검증은 아닙니다. ID14~16은 최종평가용으로 계속 미사용 상태를 유지했습니다.</p><p><a href="RESULT_KO.txt">전체 텍스트 보고</a> · <a href="COMPARISON_INDEPENDENT.json">점수·추적 검산</a> · <a href="ANALYSIS.json">기록 분석</a> · <a href="NEXT_PLAN.json">다음 계획</a></p></details></section></main></html>'''
for k,val in [('MODELROWS',modelrows),('SEEDROWS',seedrows),('ERRORROWS',errorrows),('GROUPROWS',grouprows)]:page=page.replace(k,val)
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
print('Report: no additional top1 gain; C/T3F preserved')
