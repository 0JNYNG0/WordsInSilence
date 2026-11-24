from common import *
import html
a=read(U/'ANALYSIS.json');r=read(U/'COMPARISON_INDEPENDENT.json');d=read(U/'DIAGNOSTIC.json');dv=read(U/'DIAGNOSTIC_INDEPENDENT.json');pf=read(U/'PREFLIGHT.json')
assert not r['primary_review_gate_vs_T3F']['pass'] and dv['status']=='PASS'
models=[('C','기존 C'),('T3F','기존 T3-F'),('FG16AUX01','직전 보조 후보'),(ARM,'다중 시점 후보 M3')]
def e(x):return html.escape(str(x))
def percent(model,view):return r['summary'][model][view]['accuracy_mean']
def ranges(rows,cond,field='same_aux_class_pct'):
 vals=[q[field] for q in rows if q['view']=='sensor' and q['condition']==cond]
 return f'{min(vals):.1f}~{max(vals):.1f}%'
table=''.join(f'<tr class="{"accent" if m==ARM else ""}"><td>{name}</td><td>{percent(m,"sensor"):.2f}%</td><td>{percent(m,"masked"):.2f}%</td></tr>' for m,name in models)
seedrows=''.join(f'<tr><td>{i+1}</td><td>{r["summary"]["T3F"]["sensor"]["top1_counts"][i]} / 300</td><td>{r["summary"][ARM]["sensor"]["top1_counts"][i]} / 300</td><td>{r["contrasts"]["T3F"]["sensor"]["per_seed_pp"][i]:+.2f}%p</td><td>{a["selected_epochs"][i]} / {a["completed_epochs"][i]}</td></tr>' for i in range(3))
diagtitles={'swap_first_two_thirds':'앞·중간 교환, 끝은 유지','suffix_third':'마지막 1/3 특징만 남김','reverse_thirds':'끝까지 바뀌는 세 구간 역순'}
diagrows=''.join(f'<tr><td>{name}</td><td>{ranges(d["prior_aux_rows"],cond)}</td><td>{ranges(d["rows"],cond)}</td></tr>' for cond,name in diagtitles.items())
errs=[q for q in a['remaining_errors'] if q['view']=='sensor']
errrows=''.join(f'<tr><td>{e(q["target"])}</td><td>{q["seed"]%100-12}회차</td><td>{e(q["new_prediction"])}</td><td>{q["target_rank"]}위</td><td>{e(q["aux_prediction"])}</td></tr>' for q in errs)
aux_new=[x['views']['sensor']['auxiliary_only']['correct']/3 for x in r['auxiliary_verification']['per_seed']]
aux_old=[21,83/3,136/3]
auxrows=''.join(f'<tr><td>{i+1}</td><td>{aux_old[i]:.2f}%</td><td>{aux_new[i]:.2f}%</td></tr>' for i in range(3))
summary=f'''다중 시점 GRU 보조 후보 F-G16-M3: 구현·사전검증·3회학습·성능비교 완료

결론: 일부 개선은 있으나 재현성과 회귀 기준 미달. 기존 C/T3-F 유지, 웹/DTW교체없음.

변경: 마지막 GRU 기억16개만읽던 보조를 앞·중간·뒤 시간별기억48개를함께읽도록 변경. 기존 T3-F47236개파라미터와BN전체고정,새학습부품7200개(이전4000보다3200증가). 입력280/관측마스크/시간/구간/손실CE(b+a)+.1CE(a)/분할/일정/선택기준 유지. 단어별예외없음.

공개100단어 TRAIN899(ID01~09), DEV300(ID11~13). 3개 초기값 비교, 같은300영상을반복한것이며900독립영상이아님.
sensor평균: C98.67% / T3-F99.22% / 직전보조99.22% / M3 99.33%.
M3각회정답296,300,298/300; 범위98.67~100.00%. T3-F대비+0.11%p, C대비+0.67%p중기존T3F이득은구분해야함.
masked평균: C98.44% / T3-F99.00% / 직전보조99.11% / M3 99.00%. TF대비0.00%p,직전보조대비-0.11%p.
두관측조건top5평균99.89%로TF동일. sensor는공개원본상반신범위재검출,masked는전체검출후범위마스크이며실제웹캠시험아님.

개선:2회차 '별'이 '고양' 대신정답으로양조건에서개선. sensor새오답0.
회귀:2회차masked '성남시수어통역센터'가 '전주시수어통역센터'로오답. 기존정답margin+.380→-.674. masked평균은같아도촬영자13및other집합회귀. 직전보조의3회차masked속다개선도이번에는유지안됨.
채택실패:3회중2회이상개선미달,masked촬영자13/other집합보존미달. 한회100%로새사람100%라고주장하거나그회만선택하지않음.

보조단독sensor정확도 직전21/27.67/45.33%→이번80.67/98.67/92.33%. 배움은크게늘었으나주성능추가개선은작음.
관측된선택epoch1/66/3,종료13/78/15. 총106epoch·3074실제배치·95294실제원본영상노출,학습시간합계116.99초(구현/검산/보고시간제외). 61~80전체창은어느run도없고,2회차61~78에서정답수일정;후반안정성전체개선판정불가.

별도고정공개TRAIN진단:
앞두구간순서교환·끝유지시보조답유지 직전{ranges(d['prior_aux_rows'],'swap_first_two_thirds')}→이번{ranges(d['rows'],'swap_first_two_thirds')}.
마지막1/3만보여줬을때같은보조답 직전{ranges(d['prior_aux_rows'],'suffix_third')}→이번{ranges(d['rows'],'suffix_third')}.
세구간역순에서같은보조답 직전{ranges(d['prior_aux_rows'],'reverse_thirds')}→이번{ranges(d['rows'],'reverse_thirds')}.
후반에만의존하는경향은완화됐다는근거이나모든순서변화에더민감해진것은아님. 역순에서도같은답이더많아짐. 가공특징에TCN문맥포함/올바른수어정답없음/파라미터수와선택epoch차이가있어순서이해나인과단정금지.

남은sensor오답:속다,동대문,상처,백육십이,남해군지부5개영상/반복6건. 남해군지부는보조만은정답이지만기존격차를뒤집지못함;나머지는보조도오답. 임의계수키우기금지.

검산:초기10788입력출력exact,NumPyGRU·구간풀링1.62e-7,패딩/배치3.82e-6이내. 최종저장9000예측+새base/aux3600/실제batch/동결tensor/선택epoch검산PASS. 별도진단훈련된GRU240출력NumPy차3.08e-6. 같은Codex별도수식검산이며독립agent아님. 사전검사도구4오류는근거보존후학습전수정;실제학습오류없음.

다음:이번결과를함께검토. 공개자료의필수단계/정상반복/가림/생략/순서차이를구분할평가기준을먼저정리. 기존100참고영상감사근거재사용,수어의필수단계는수어검토필요. 이후별도순서학습신호는새계획·승인후검토하고자동추가학습/가중치탐색하지않음. 지금특정단어패치나웹교체없음. 사용자·멘토촬영접근/학습/튜닝/후보선택0. 보류ID14~18원시접근0.
'''
(U/'RESULT_KO.txt').write_text(summary,encoding='utf8')
css='''*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:#19313f;font:17px/1.75 "Malgun Gothic",sans-serif}main{max-width:1060px;margin:36px auto;padding:0 24px}header,section{background:#fff;border:1px solid #dce5e9;border-radius:20px;padding:32px;margin:20px 0}header{background:#123547;color:white}h1{font-size:34px;line-height:1.4;margin:8px 0 20px}h2{font-size:24px;line-height:1.45;margin:0 0 18px}h3{font-size:19px;margin:18px 0 4px}p{margin:10px 0}.muted,small{color:#5d6c76}header small{color:#badce4}.badge{display:inline-block;background:#ffedcb;color:#754b10;font-size:14px;font-weight:bold;padding:5px 14px;border-radius:18px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}.card{padding:20px;background:#f1f6f8;border-radius:12px}.num{font-size:30px;color:#077963;font-weight:bold}.callout{padding:17px 20px;border-left:5px solid #0e9e7d;background:#edf8f3;margin:18px 0}.warn{border-color:#d99732;background:#fff7e8}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:16px}th,td{padding:13px 11px;border-bottom:1px solid #dde5e9;text-align:left}th{background:#eef4f6}.accent{background:#eaf7f2;font-weight:bold}.flow{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}.flow>div{padding:20px;border:1px solid #c4d9e3;border-radius:12px;background:#f5fafb}.limit{font-size:15px;background:#f2f5f7;padding:16px;border-radius:10px}a{color:#007492;overflow-wrap:anywhere}summary{font-weight:bold;cursor:pointer}li{margin:8px 0}@media(max-width:700px){main{padding:0 12px;margin:16px auto}header,section{padding:22px 18px}h1{font-size:27px}h2{font-size:22px}.cards,.flow{grid-template-columns:1fr}.scroll table{min-width:560px}body{font-size:16px}}'''
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>앞·중간·뒤 기억을 함께 읽기 — 3회 비교 결과</title><style>{css}</style><main>
<header><small>WORDS IN SILENCE · F-G16-M3 · 2026.10.19</small><h1>앞·중간·뒤 기억을 함께 읽기<br>3회 학습과 비교를 마쳤습니다</h1><p>평균은 조금 올랐지만 개선이 반복되지 않았고,<br>다른 조건에서 새로운 오답이 생겼습니다.</p><span class="badge">채택 기준 미달 · 기존 C / T3-F 유지</span></header>
<section><h2>핵심 결과</h2><div class="cards"><div class="card"><div class="num">99.33%</div>주검증 평균<br>T3-F 99.22% → +0.11%p</div><div class="card"><div class="num">1 / 3회</div>기존보다 정답 수 증가<br>나머지 두 회는 동일</div><div class="card"><div class="num">1건</div>다른 관측 조건에서<br>기존 정답이 새 오답으로</div></div><p>공개 <b>100단어</b>의 학습 영상 899개와 개발 검증 영상 300개를 사용했습니다. 초기값을 바꿔 3회 비교했습니다. 같은 검증 영상 300개를 반복한 것으로, 서로 다른 900개 영상이나 실제 웹캠 이용자의 정확도는 아닙니다.</p><div class="callout warn">기존 모델을 교체할 만큼 안정적인 개선은 확인되지 않았습니다. C·T3-F·웹·DTW는 그대로 유지했습니다.</div></section>
<section><h2>무엇을 바꿨나</h2><div class="flow"><div><b>직전 보조 부분</b><p>순서대로 읽은 뒤<br><b>마지막 기억 하나</b>로 답하기</p></div><div><b>이번 후보 M3</b><p>순서대로 쌓이는 기억을<br><b>앞·중간·뒤에서 각각 요약</b>해 함께 답하기</p></div></div><p>관절 좌표, 손 미검출 표시, 실제 시간, 학습 자료, 학습 신호와 모델 선택 기준은 유지했습니다. 기존 T3-F의 가중치와 정규화 상태는 전부 고정했습니다. 특정 단어의 예외 규칙을 추가하지 않았습니다.</p><p class="limit">새로 학습하는 부품은 4,000 → 7,200개로 늘었습니다. 따라서 변화의 효과를 전부 순서 처리만의 효과라고 단정할 수 없습니다. 앞·중간·뒤는 시간 구간이며, 수어의 필수 동작 단계를 사람이 표시한 것은 아닙니다.</p></section>
<section><h2>전체 성능 비교</h2><div class="scroll"><table><tr><th>모델</th><th>범위 제한 후 재검출<br>주검증 sensor</th><th>전체 검출 후 범위 적용<br>보조 검증 masked</th></tr>{table}</table></div><p>이번 후보는 T3-F 대비 주검증 <b>+0.11%p</b>, 보조 검증 <b>변화 없음</b>입니다. 직전 보조 후보와 비교하면 보조 검증은 <b>−0.11%p</b>입니다. 두 조건 모두 상위 5위 정답률은 평균 99.89%로 T3-F와 같습니다.</p><p>기존 C 대비 주검증은 +0.67%p지만, 이 중 대부분은 이미 T3-F에서 얻은 개선입니다. 이번 변경의 추가 효과와 구분해야 합니다.</p><h3>각 회차의 주검증 결과</h3><div class="scroll"><table><tr><th>회차</th><th>T3-F 정답 수</th><th>M3 정답 수</th><th>차이</th><th>선택 / 종료 학습 회차</th></tr>{seedrows}</table></div><p class="limit">이번 주검증 정확도 범위는 98.67~100.00%입니다. 2회차의 100%를 골라 모든 사용자에게 100%라고 보고하지 않았습니다. 선택 기준은 정답 수가 아니라 사전에 고정한 두 조건의 평균 분류 오차였습니다.</p></section>
<section><h2>좋아진 사례와 새로 생긴 문제</h2><div class="flow"><div><b>개선 · 별</b><p>2회차에서 기존의 ‘고양’ 대신 <b>‘별’</b>을 맞혔습니다. 두 관측 조건 모두 개선됐습니다.</p></div><div><b>회귀 · 성남시수어통역센터</b><p>같은 2회차의 masked 조건에서 기존 정답이 <b>‘전주시수어통역센터’</b>로 바뀌었습니다. 정답은 2위로 밀렸습니다.</p></div></div><p>masked의 전체 정답 수는 같지만, 하나를 고치고 하나를 새로 틀린 결과입니다. 또한 직전 보조 후보가 개선했던 3회차 masked의 ‘속다’는 이번 후보에서 유지되지 않았습니다.</p><div class="callout warn"><b>채택 기준을 통과하지 못한 이유</b><br>① 3회 중 2회 이상 개선되지 않음<br>② masked에서 촬영자 13의 정답 수 감소<br>③ masked의 관측 유형별 한 집합에서 정답 수 감소</div><p>동결한 기준을 결과에 맞춰 완화하지 않았습니다. 주검증에서 새 오답이 없더라도 다른 조건의 회귀를 함께 반영했습니다.</p></section>
<section><h2>앞부분 정보를 더 활용하게 됐나?</h2><p>보조 부분만 따로 단어를 맞히게 했을 때는 크게 좋아졌습니다. 아래는 선택된 모델의 공개 개발 검증 결과입니다.</p><div class="scroll"><table><tr><th>회차</th><th>직전 보조만 사용</th><th>M3 보조만 사용</th></tr>{auxrows}</table></div><p><b>보조 부분의 학습 능력 향상과 최종 인식률 향상은 다릅니다.</b> 기존 T3-F가 이미 맞히는 사례를 보조도 맞히는 것만으로는 최종 오답이 줄지 않습니다.</p><h3>가공된 특징의 순서를 바꿨을 때 같은 보조 답을 낸 비율</h3><div class="scroll"><table><tr><th>진단 조건</th><th>직전 보조</th><th>M3</th></tr>{diagrows}</table></div><p>마지막 특징만 남겼을 때 같은 답이 유지되는 비율은 크게 낮아졌습니다. 앞·중간 교환 시 점수 변화 중앙값도 직전 0.95~4.30%에서 <b>26.36~32.26%</b>로 커졌습니다. 후반부에만 의존하던 경향이 완화됐다는 근거입니다.</p><div class="callout warn"><b>모든 순서 차이에 더 민감해진 것은 아닙니다.</b> 세 구간을 뒤집었을 때는 오히려 같은 답을 낸 비율이 늘었습니다. 이것만으로 수어 순서를 올바르게 이해했다고 결론내릴 수 없습니다.</div><p class="limit">공개 학습 899개 · 두 관측 조건 · 세 모델을 고정해 진단했습니다. 정확도가 아닌 답의 유지 비율입니다. 가공된 특징에는 TCN의 앞뒤 문맥이 들어 있고, 임의로 바꾼 순서에 올바른/틀린 수어 정답은 없습니다. 학습 시점과 부품 크기도 달라 단일 원인의 증명은 아닙니다.</p></section>
<section><h2>아직 남은 오답</h2><p>주검증에서는 서로 다른 영상 5개에서 총 6건의 오답 사건이 남았습니다. ‘동대문’ 한 영상이 두 회차에 포함됩니다.</p><div class="scroll"><table><tr><th>정답</th><th>회차</th><th>최종 1위</th><th>정답 순위</th><th>보조 단독 1위</th></tr>{errrows}</table></div><p>‘남해군지부’는 보조 부분만 보면 맞혔지만, 합친 점수에서는 기존 격차를 뒤집지 못했습니다. 나머지 사례는 보조도 정답을 맞히지 못했습니다. 새 오답 사례도 있으므로 점수 비중을 무작정 높이지 않았습니다.</p></section>
<section><h2>다음 작업은 무엇인가</h2><p><b>이번 결과를 검토한 뒤, 공개 자료에서 동작 단계와 순서 차이를 평가할 기준부터 정리하는 것이 다음 순서입니다.</b></p><ol><li>기존 공개 100단어 참고 영상 감사 자료에서 관측 가능한 동작 변화, 정상 반복, 화면 밖 손, 실제 가림을 구분합니다.</li><li>어떤 동작이 필수인지, 어떤 생략·순서 변경이 실제 수어 오류인지는 수어를 아는 사람의 검토로 확인해야 합니다. 불확실한 사례에 임의 정답을 붙이지 않습니다.</li><li>그 근거를 갖춘 뒤 별도 순서 학습 신호를 검토합니다. 이번 결과를 보고 곧바로 학습 시간·가중치·단어별 규칙을 재조정하지 않습니다.</li></ol><p>기존 관측·시각 감사 근거를 연결해 두었습니다. 언어학적 필수 단계 정답이 새로 확보됐다는 뜻은 아닙니다. 모든 수어에 쓸 공통 방식이 목표지만, 현재 성능 근거는 100단어 개발 검증에 한정됩니다.</p></section>
<section><h2>구현과 검증 기록</h2><p>학습 전 10,788개 입력 조건에서 초기 출력이 T3-F와 정확히 같았습니다. 기존 GRU 초기값도 유지했습니다. 새 시간별 기억·구간 요약을 별도 NumPy 계산으로 검산했고, 미검출 프레임과 실제 시간은 유지했으며 배치용 빈칸만 제외했습니다.</p><p>106회 학습 순회, 실제 배치 3,074개, 원본 영상 노출 95,294건을 기록·검산했습니다. 학습 실행 시간 합계는 약 117초이며 구현·준비·검산·보고 시간은 제외한 값입니다. 기존 모델의 가중치·정규화 상태는 모든 배치에서 유지됐습니다.</p><p>최종 저장 예측 9,000개와 새 후보의 base/aux 예측 3,600개, 선택 시점·체크포인트·학습 추적을 검산했습니다. 최종 점수를 재추론하지 않았습니다. 별도 특징열 진단은 학습된 GRU의 240개 출력에 대해 NumPy 검산을 추가했습니다.</p><details><summary>사전검사 오류와 수정 내역</summary><p>학습 전에 검사 도구에서 네 문제가 발견됐습니다: 복사 시 줄바꿈 형식 차이, JSON 배열과 Python tuple 비교, 검사 함수 import 누락, 부동소수점 순열 계산에 지나치게 엄격한 비트 일치 검사. 최초 증거와 후속 오류를 보존하고 각각 원인을 확인해 수정했습니다. 학습 조건과 모델 설계는 바꾸지 않았습니다. 실제 초기 모델 출력 비교는 계속 정확 일치를 요구했고 통과했습니다. 학습 실행 중 오류는 없었습니다.</p></details><p>구현과 별도 수식 검산은 Codex가 수행했습니다. 독립 에이전트나 수어 전문가의 검증은 아닙니다. 사용자·멘토 촬영은 접근하거나 학습·튜닝·후보 선택에 사용하지 않았고, 보류 ID14~18 원시자료도 사용하지 않았습니다.</p><p><a href="COMPARISON_INDEPENDENT.json">저장 점수·학습 추적 검산</a> · <a href="PREFLIGHT.json">사전검증</a> · <a href="DIAGNOSTIC.json">순서 진단</a> · <a href="DIAGNOSTIC_INDEPENDENT.json">진단 별도 검산</a> · <a href="NEXT_PLAN.json">다음 계획</a> · <a href="EVALUATION_EVIDENCE.json">평가 근거 범위</a> · <a href="RESULT_KO.txt">텍스트 보고</a></p></section></main></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
write(U/'CONCLUSION.json',{'status':'COMPLETE_VERIFIED_SMALL_PRIMARY_GAIN_NOT_PROMOTED','at':now(),'runs':3,
 'sensor_mean':percent(ARM,'sensor'),'masked_mean':percent(ARM,'masked'),'sensor_delta_vs_T3F_pp':r['contrasts']['T3F']['sensor']['mean_pp'],
 'masked_delta_vs_T3F_pp':0.,'masked_delta_vs_AUX01_pp':r['contrasts']['FG16AUX01']['masked']['mean_pp'],
 'sensor_new_correct_vs_T3F':1,'sensor_new_wrong_vs_T3F':0,'masked_new_correct_vs_T3F':1,'masked_new_wrong_vs_T3F':1,
 'gate_pass':False,'failed_checks':r['primary_review_gate_vs_T3F']['failed_checks'],'epochs':a['completed_epochs'],'selected_epochs':a['selected_epochs'],
 'standalone_aux_sensor_accuracy':aux_new,'model_changed':False,'user_data_used':False,'automatic_further_training':False,
 'diagnostic_caution':'Earlier-feature score sensitivity increases and suffix-only dependence decreases, but reverse-third label retention increases. No semantic-order success claim.'},new=True)
print('Report and conclusion saved; no model adoption.')
