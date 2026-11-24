from pathlib import Path
import json,html
U=Path(__file__).resolve().parent
read=lambda n:json.loads((U/n).read_text('utf-8-sig'))
def save(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
s=read('SUMMARY.json');d=read('DIAGNOSTICS.json');v=read('VERIFICATION.json');post=read('POSTCHECK.json');assert v['status']=='PASS'
def med(view,scenario,mode,area,key):return s['localization'][view+'_'+scenario][mode][area][key]['median']*100
table=[]
for scenario,label in [('original','원래 한 번 영상'),('repeat2','전체 영상 2회 연결'),('repeat3','전체 영상 3회 연결')]:
    table.append(dict(label=label,scenario=scenario,single_recall=med('sensor',scenario,'single','core','recall'),multi_recall=med('sensor',scenario,'multi','core','recall'),
        single_precision=med('sensor',scenario,'single','core','precision'),multi_precision=med('sensor',scenario,'multi','core','precision'),
        possible_recall=med('sensor',scenario,'multi','possible_core','recall'),possible_precision=med('sensor',scenario,'multi','possible_core','precision')))
save('REPORT_TABLE.json',table)
for k in ('handshape','location','movement'):
    assert s['lowering'][k]['possible_overlap_s']>=s['lowering'][k]['strong_overlap_s']-1e-8
decision=dict(status='OFFLINE_EVALUATED_REVIEW_REQUIRED_NOT_DEPLOYED',
    reason='Temporal localization and observation sufficiency are distinct from sign correctness. Report repeat-count ambiguity, normal-lowering overlap of possible envelope, and short deletion residuals; no production adoption or DEV retuning.',
    production_changed=False,training=0,private_data_read=0,
    implementation='Codex local separate candidate',verification='Same Codex separate DP/path/vote implementation, not separate agent or linguistic expert',
    next='Review paired results. First test cross-reference support for repeat boundaries and retain ambiguity when costs alone disagree with repetition evidence, rather than forcing a cycle count. Then diagnose short-reference-phase compression versus normal speed/hold/repeats. Preserve normal-lowering and unknown intervals; no per-word fix, user-data tuning or new training without review.')
save('DECISION.json',decision)
lines=[]
for r in table:lines.append(f"{r['label']}: 주석시간포착 {r['single_recall']:.2f}% → {r['multi_recall']:.2f}%, 잡은범위의주석포함 {r['single_precision']:.2f}% → {r['multi_precision']:.2f}%. 불확실범위까지 포함한포착/주석포함 {r['possible_recall']:.2f}%/{r['possible_precision']:.2f}%.")
counts='\n'.join(f"{name}: 선택한참고횟수 분포 {s['cycle_counts']['sensor_'+name]}" for name in ('original','repeat2','repeat3'))
low=s['lowering']['location'];dele=d['deletion_counts']
text=f'''한 번·반복 수행을 함께 고려한 공개 구간 정렬 검증

별도 후보 구현과 공개 비교·독립 구현 검산을 완료했다. 한 번짜리 참고에 전체 영상을 강제로 맞추는 대신, 같은 참고를1·2·3번 이어 놓은 경우를 모두 비교한다. 운영 C·T3-F, 기존 DTW·웹은 그대로 유지했다. 이 결과는 동작을 비교할 시간 범위의 연구 결과이며 단어 인식률을 다시 측정한 것이 아니다.

1. 구현한 내용
전체 영상의 저장된 모든 프레임을 쓴다. 직전 후보의96시점 샘플링을 제거했고, 손이 안 보인다는 이유로 영상의 앞뒤나 중간을 자르지 않는다. 어깨 중심·너비로 정리한 팔꿈치와 몸 관절 손목4점을 사용한다. 손가락21점은 정렬 입력에서 제외해 손가락 미검출이 정렬 범위를 줄이지 않게 했다. 몸 관절 손목도 가려지면 같은 보장을 할 수 없다는 한계는 남는다.
한 번·두 번·세 번 전체 참고의 정렬 비용을 계산한다. 같은 사람을 반복해서 사용해도 참고 사람 수는1명이다. 각 주기에서 실제 팔 관측을 확보한 서로 다른5명 이상이 있어야 비교가 가능하다. 모든 참고의 비용 중앙값으로 횟수 후보를 고르며, 단어마다 규칙을 바꾸지 않았다. 선택한횟수는 정렬 방법일 뿐 올바른 반복횟수 판정이 아니다. 4회 이상은 이번 후보 범위 밖이다.
5명 이상이 동의하는 동작·관측 구간과, 일부 참고에서만 동작이 있는 불확실한 구간을 구분한다. 짧은 구간을 놓치지 않으려고 불확실한 범위도 남기되, 한참고의 가능성만으로 손을 반드시 보여야 한다고 확정하지 않는다. 확인된 손내림 근거와 알려진 비사용손은 별도로 처리한다. 보이지 않는 관절을 합성하지 않는다.

2. 공개 검증 범위
AI Hub100단어, TRAIN899영상(01~09)을 참고, DEV300영상(11~13)을 점검에 사용했다. 기존 관측과 상반신sensor 두 경로로2398입력이며1199개 원본영상이다. DEV는 기존개발자료이고 새최종평가가 아니다. DEV의 주석과 전체원본 손상태는 사후검증에만 사용했으며 정렬 후보에 전달하지 않았다. 추가 추출·학습·분류추론, 사용자/멘토자료 및 보류14~18 접근 없음.
원래영상600경로, 전체영상2회·3회연결 각각600경로, 짧은구간삭제591대조를 계산했다. 전체원본을 반복해 만든 합성시험으로, 사람마다 다른 실제반복이나 모든단어의 반복허용 여부를 인증한 것이 아니다. 모든프레임에서 한 번만 맞춘 대조와 반복후보대조를 함께 계산해 샘플링변경과 반복선택 효과를 구분했다.

3. 구간 정렬 결과 — sensor각300영상의 중앙값
{chr(10).join(lines)}
왼쪽은모든프레임으로한번만정렬한조건,오른쪽은같은입력에서1·2·3회후보를비교한조건이다. 주석시간포착은 주석동작시간 중 후보가 덮는 비율이며, 주석포함은 후보가 잡은 시간 중 주석에 들어가는 비율이다. 둘 다 높아야 범위를 넓게만 잡는 착시를 피할 수 있다. 주석은 수어의 필수 단계 정답표가 아니며 위 수치는 인식률·동작정답률이 아니다.
{counts}
참고마다 원래 동작의 내부 반복이 다를 수 있으므로 횟수 선택 불일치를 곧바로 수어 오류로 부르지 않는다. 동시에 원래 한 번 영상에서 여러 회 후보를 고르는 경우는 과도한 정렬 가능성을 보여주는 진단이다. sensor에서는 원래300영상 중298개가1회,2개가2·3회로 선택됐다. 합성2회는223/300,합성3회는219/300에서 구성한 횟수와 일치했다. legacy경로는각각300/300이었다. 화면 밖 미검출이 반영된sensor에서 반복 선택이 여전히 불안정하다. 비용이 가장 작다는 이유만으로 실제 반복 횟수를 확정하면 안 된다.
sensor3회 대조에서는 강한 비교구간이 비어 있는 경우가 한 번만 정렬한 조건2건,반복후보조건1건이다. 범위정밀도 중앙값은각각298건/299건 기준이고, 주석시간포착은빈구간의0도 포함한300건이다.

4. 짧은 누락과 정상 손내림
손만 가린 7,092개 항목·손 조건에서 강한 비교 범위와 불확실한 범위를 모두 유지하고, 추가 미관측량을 따로 계산했다. 과거 기준에서 놓쳤던 8조건 중 강한 범위로 {d['previous8_revealed_strong']}개, 불확실한 범위를 포함하면 {d['previous8_revealed_possible']}개의 추가 누락을 포착했다. 직전 후보는 이8조건 중4개만 포착했다. 같은 영상·항목·조건 조합이며 8개의 독립 영상이 아니다.
짧은 이동구간 10% 손가림: 실제 근거를 제거한 {d['stress']['peak10']['effective']}개 항목·손 중 강한 범위의 추가 누락 {d['stress']['peak10']['strong_revealed']}개, 불확실한 범위를 포함하면 {d['stress']['peak10']['possible_revealed']}개였다. 불확실한 범위를 넓혀 포착한 것이 구별 동작의 의미를 이해했다는 뜻은 아니다.
공개 원본으로 추정한 안정된 손내림·미관측 시간은 위치 항목 기준 {low['rest_missing_s']:.2f}초였다. 강한 관측 필요 범위와의 겹침은 {low['strong_overlap_s']:.3f}초, 불확실한 범위까지 포함한 겹침은 {low['possible_overlap_s']:.3f}초였다. 직전후보의 강한범위 겹침0초에서 늘어난 회귀다. 저장된경로를 따로 조사하니3개 영상·손에서 발생했고, 그중공개‘실명’영상에서1회보다3회후보를선택하면서0.633초가새로포함됐다. 나머지2건0.100초는1회정렬에도있어전체프레임/정렬경계변경의영향이다. 단어별예외로고치지않고동결결과를보존했다(POSTCHECK).
불확실한 범위의 겹침은 이를 자동 오류 경고로 쓰면 정상 손내림도 걸릴 수 있다는 진단이다. 원본 상태는 기하학적 추정이며 전문가 정답이 아니다. 확실하지 않은 미관측은 동작 오류로 판정하지 않았다.
기존 역할이 확인된 한 손 3단어의 9개 DEV영상×2경로, 총18대조에서 반대손을 가려도 결과가 같았다. 나머지 88단어의 손 역할을 새로 확인한 것은 아니다. 시간값만 0.75/1.5배 또는 앞뒤 다른 속도로 바꾸어도 정렬 입력은 같다. 실제 빠른 촬영의 검출 성능을 검증한 것은 아니다.

5. 영상의 짧은 구간 자체가 빠지면?
591개 공개 영상·손의 이동량이 큰 10% 시간을 영상 관측에서 제거하고 시간을 이어 붙였다. 그중 주석 시간이 실제로 제거된 대조는 {dele['windows_with_annotated_time_removed']}개였다. 삭제 후에도 남은 근거의 강한·불확실 범위에 관측 누락이 없다고 표시된 항목·손은 {dele['after_observed_intervals_only']}/1,773개였고, 주석 시간을 삭제한 경우만 세어도 {dele['after_clear_with_annotated_time_removed']}개였다.
손 센서가 가려진 문제와 동작을 실제로 생략한 문제는 다르다. 남은 프레임끼리 시간 정렬이 가능하면 빠진 동작을 다른 프레임으로 늘여 맞출 수 있다. “관측한 구간에 센서 공백이 없음”을 “필요한 동작을 모두 했다”로 표시해서는 안 된다. 삭제한 구간이 언어적으로 필수라는 전문가 주석은 없으므로 이 숫자를 필수동작 생략 검출 실패율로 보고하지 않는다.

6. 판단과 다음 계획
이번 결과는 운영 적용 없이 사용자 검토를 위해 보존한다. 반복 정렬 개선과 짧은 단계의 의미 검증을 구분해야 한다. 다음에는 우선 정렬 비용만으로 반복 횟수를 확정하지 않고, 여러 참고가 실제 반복 경계를 함께 지지하는지 확인하는 계획을 검토한다. 근거가 엇갈리면 한 가지 횟수를 강제로 고르지 않고 정렬 불확실성으로 남긴다. 이어서 짧은 참고 구간이 몇 개의 관측 프레임에 몰리는 현상을 정상 속도·정지·반복과 대조한다. 손내림 근거와 불확실한 경계 범위를 따로 표시하고 불확실성을 오류로 바꾸지 않는다. 사용자 촬영이나 단어별 예외로 조정하지 않는다. 추가 학습은 결과를 함께 검토한 뒤 결정한다.

검증: 전체입력2398 손21점제거불변, 자체8검사통과. 별도구현으로 {v['rows']}대조의 {v['independent_paths']}정렬경로·횟수선택·투표·시간집계 및7092고정범위검산통과. 별도DP최적비용 {v['independent_dp_optimality_samples']}표본, {v['hash_pins']}해시확인. 같은Codex별도구현검산이며독립에이전트나수어전문가검토는아니다. 보호32파일/C·T3-F6가중치불변.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
tr=''.join(f"<tr><td>{r['label']}</td><td>{r['single_recall']:.2f}% → <b>{r['multi_recall']:.2f}%</b></td><td>{r['single_precision']:.2f}% → <b>{r['multi_precision']:.2f}%</b></td></tr>" for r in table)
detail=''.join('<details><summary>'+html.escape(p.splitlines()[0])+'</summary><div class="detail">'+html.escape('\n'.join(p.splitlines()[1:]))+'</div></details>' for p in text.split('\n\n')[2:])
cards=''.join(f'<div class="metric"><span>{r["label"]}</span><strong>{r["multi_recall"]:.2f}%</strong><div class="track"><i style="width:{r["multi_recall"]}%"></i></div><small>주석 동작시간 포착 중앙값</small></div>' for r in table)
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>한 번·반복 구간 정렬 검증</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f8;color:#173047;font:16px/1.7 'Malgun Gothic',sans-serif}}main{{max-width:1140px;margin:auto;padding:40px 25px 70px}}.tag{{font-size:14px;font-weight:700;color:#267567;letter-spacing:1px}}h1{{font-size:36px;line-height:1.35}}h2{{font-size:23px;margin-top:0}}.lead{{font-size:20px}}section{{background:white;border:1px solid #d8e2ea;border-radius:16px;padding:26px;margin:22px 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}}.metric{{padding:20px;background:#eaf6f1;border-radius:12px}}strong{{font-weight:700}}.metric strong{{display:block;font-size:32px;color:#197362}}small{{font-size:13px;color:#546778}}.track{{height:9px;background:#d4e7e0;border-radius:5px;margin:12px 0;overflow:hidden}}.track i{{display:block;height:100%;background:#2d8f7d}}.warn{{background:#fff5e5;border-color:#edce98}}table{{width:100%;border-collapse:collapse}}th,td{{padding:13px 10px;text-align:left;border-bottom:1px solid #dce4eb}}th{{background:#f3f7fa}}.steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}}.steps div{{background:#f0f4f8;border-radius:10px;padding:17px}}.detail{{white-space:pre-wrap;padding:18px 0}}details{{border-top:1px solid #dae4ec;padding:14px 0}}summary{{cursor:pointer;font-weight:700}}a{{color:#176e61}}@media(max-width:700px){{main{{padding:22px 16px}}h1{{font-size:27px}}.lead{{font-size:18px}}.grid,.steps{{grid-template-columns:1fr}}section{{padding:20px 15px}}td,th{{padding:10px 5px;font-size:13px}}}}</style><main>
<div class="tag">WordsInSilence · 공개 100단어 · 2026.10.20</div><h1>한 번 해도, 반복해도<br>동작 범위를 함께 비교하기</h1><p class="lead">전체 프레임을 유지하고 한 번·두 번·세 번의 참고를 함께 비교하는 별도 후보를 검증했습니다. <b>반복의 시간 정렬과, 필요한 동작을 모두 했는지의 판단은 구분해야 합니다.</b></p>
<div class="grid">{cards}</div><section><h2>1. 이번에 바꾼 방법</h2><div class="steps"><div><b>모든 프레임 유지</b><br>손이 안 보인다는 이유로 구간을 자르지 않습니다.</div><div><b>한 번·반복 함께 정렬</b><br>8~9명의 참고를 각각1·2·3회로 비교합니다. 같은 사람의 반복은 한 명으로 셉니다.</div><div><b>불확실한 구간도 기록</b><br>짧은 동작의 가능성을 남기되, 불확실한 미관측을 수행 오류로 판정하지 않습니다.</div></div></section>
<section><h2>2. 같은 입력에서 반복 정렬의 효과 비교</h2><p>각 조건은 공개300영상의 sensor 관측 결과입니다. 왼쪽은 모든 프레임으로 한 번만 정렬한 대조, 오른쪽은 같은 입력에서 반복 후보까지 비교한 결과입니다.</p><table><thead><tr><th>조건</th><th>주석 동작시간 포착</th><th>잡은 시간의 주석 포함</th></tr></thead><tbody>{tr}</tbody></table><p><b>단어 인식률이 아닙니다.</b> 첫 비율은 동작시간을 덮는 정도, 두 번째는 범위를 불필요하게 넓히지 않았는지 보여줍니다. 전체 영상을 이어 붙인 합성 반복 시험입니다.</p></section>
<section><h2>3. 짧은 누락과 정상 손내림</h2><p>과거 기준에서 공백을 놓쳤던8개 항목·손 조건 중 <b>강한 참고 근거로 {d['previous8_revealed_strong']}개</b>, 불확실한 범위까지 포함하면 <b>{d['previous8_revealed_possible']}개</b>의 추가 누락을 표시했습니다. 직전 후보는 이 중4개를 포착했습니다. 같은 영상의 여러 항목·가림 조합입니다.</p><p>손내림·미관측 시간과 관측을 요구하는 범위의 겹침은 위치 항목 기준 <b>강한 근거 {low['strong_overlap_s']:.3f}초 / 불확실한 범위 포함 {low['possible_overlap_s']:.3f}초</b>였습니다. 원본 상태로 추정한 진단이며, 불확실한 겹침을 곧바로 동작 오류로 표시해서는 안 됩니다.</p><p>역할이 확인된3개 한 손 단어에서는 반대손을 가려도 결과가 같았습니다. 나머지 단어의 손 역할까지 확정한 것은 아닙니다.</p></section>
<section class="warn"><h2>4. 짧은 동작을 실제로 생략한 경우는 별도 문제</h2><p>영상에서 짧은 구간 자체를 제거하면 남은 프레임끼리 늘여 맞출 수 있습니다. 주석시간이 제거된 대조에서도 <b>{dele['after_clear_with_annotated_time_removed']}개 항목·손</b>은 남은 비교 범위에 관측 공백이 없다고 표시됐습니다.</p><p><b>“손이 잘 보였다”는 “필요한 동작을 모두 했다”와 다릅니다.</b> 이 구분을 유지하기 위해 전체수행 합격이나 잘못된 수어 판정은 만들지 않았습니다. 삭제 구간이 언어적으로 필수라는 정답도 없어 이를 필수동작 생략 검출 실패율로 해석하면 안 됩니다.</p></section>
<section><h2>5. 현재 상태와 다음 검토</h2><p><b>C·T3-F, 기존 웹과 DTW 점수는 그대로입니다.</b> 사용자 촬영은 읽거나 학습·튜닝에 사용하지 않았고, 추가 학습도 하지 않았습니다.</p><p>원래 한 번 영상2건에서 반복 후보를 선택했고, 그중1건은 정상 손내림으로 추정된0.633초까지 비교 범위에 넣었습니다. 따라서 비용만으로 횟수를 확정하기보다 <b>여러 참고의 반복 경계 근거를 함께 확인</b>하는 보완이 먼저 필요합니다.</p><p>그다음 짧은 참고 단계가 몇 개의 관측 프레임에 몰려 맞춰지는지 확인합니다. 정상 속도·정지·반복과 구분한 뒤에야 동작 피드백의 근거로 사용할 수 있습니다.</p></section>
<section><h2>상세 근거</h2><p>공개 TRAIN899 / DEV300, 기존 개발자료의 재점검입니다. 별도 구현 검산 {v['rows']}대조·{v['independent_paths']}경로 통과. 같은Codex의 별도 검산이며 수어 전문가 또는 다른 에이전트 검토는 아닙니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="SUMMARY.json">집계</a> · <a href="DIAGNOSTICS.json">누락·손내림 진단</a> · <a href="VERIFICATION.json">별도 검산</a></p>{detail}</section></main></html>'''
(U/'RESULT_KO.html').write_text(page,'utf-8');print('REPORT READY',table,flush=True)
