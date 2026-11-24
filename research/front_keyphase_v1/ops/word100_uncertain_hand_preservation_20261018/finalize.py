"""Close evidence first, then synchronize the three current project records."""
from run import *
import html,re
def main():
    assert not (U/'RUNNING.json').exists()
    for f in ('CONTRACTS.json','INDEPENDENT.json','INTEGRATION_CHECKS.json'):assert read(U/f)['status']=='PASS'
    check(read(U/'CONTEXT.json')['pins']);check(read(U/'OUTPUT_PINS.json'))
    ann=read(U/'ANNOTATED_COMPARISON.json');normal=[x for x in ann if x['verdict']=='normal_identity'];remaining=[x for x in ann if x['verdict']=='error_identity' and x['new_side'] is not None and x['new_side']!=x['actual_side']];uncertain=[x for x in ann if x['verdict']=='uncertain']
    assert len(normal)==441 and all(x['old_side']==x['new_side']==x['actual_side'] for x in normal) and all(x['usable_points']>0 for x in normal)
    assert len(remaining)==9 and all(x['identity_side']==-1 and x['usable_points']>0 for x in remaining)
    assert len(uncertain)==2 and all(x['identity_side']==-1 and x['old_side']==x['new_side'] for x in uncertain)
    changes=read(U/'CHANGES.json');assert len(changes)==1 and changes[0]['clip_id']=='NIA_SL_WORD2029_REAL06_F' and changes[0]['index']==99 and changes[0]['retained_observed_points']==21
    visual={'status':'REVIEWED_ALL_NEW_ASSIGNMENT_CHANGES','reviewer':'Codex, same operator; independent scalar verifier is numerical only','changed_events':1,'records':[{'clip_id':changes[0]['clip_id'],'view':'masked','index':99,'raw_index':1,'actual_side':1,'old_side':0,'new_side':None,'verdict':'correct_duplicate_withhold_observation_preserved','basis':'原本97~101: upper hand attached to anatomical right arm. raw0 and warped raw1 both cover upper same right hand. Anchor raw0 confirmed from97,67ms old; own left confirmed history absent; finger coverage.85. Keep raw0 R and all21 observed raw1 coordinates, withhold raw1 L. Actual lower left hand is not recovered.','image':str(U/'evidence/change_01.jpg')}], 'new_regression_found_in_changed_events':0,'unchanged_remaining_visual_evidence_reused':str(P/'VISUAL_REVIEW.json')}
    write(U/'VISUAL_REVIEW.json',visual)
    proof={'status':'PASS','normal_events':441,'normal_assignment_preserved':441,'normal_valid_points_retained':sum(x['usable_points'] for x in normal),'normal_zero_observation':0,'normal_identity_still_provisional':sum(x['identity_side']<0 for x in normal),'remaining_wrong_hints':len(remaining),'remaining_wrong_hints_not_promoted_to_certain':sum(x['identity_side']<0 for x in remaining),'remaining_observed_points_retained':sum(x['usable_points'] for x in remaining),'remaining':remaining,'uncertain2':uncertain,'crossing315_preserved':sum(x['preserved'] for x in read(U/'CROSSING_COMPARISON.json')),'limitations':['Side-free observed/local outputs are detector-frame observations, not final TCN tensors. Existing learner upper-body ROI mask must still be applied at any later training adapter.','Unknown identity never means no hand or fully correct hand shape.','Possible temporal link edges are hypotheses, not independently verified physical tracking or anatomy.','Candidate outputs did not change production model, scores or frontend.']}
    write(U/'OBSERVATION_PRESERVATION.json',proof)
    # Numerical failure was preserved; recovery only uses the data-type error
    # bound for floating link costs. Boolean decisions and every point are exact.
    recovery={'status':'VERIFIED_RECOVERY','initial_failure':str(U/'FIRST_VALIDATION_FAILURE.json'),'original_verifier':str(U/'verify.py'),'used_verifier':str(U/'verify_recovery.py'),'original_candidate_inputs_outputs_unchanged':True,'change':'Only allclose tolerance for float32-vs-float64 shoulder-scaled link costs:8*float32 epsilon. No confidence, overlap, distance decision, model or data threshold changed. Full exact boolean-link/mask/side and coordinate comparisons passed.','api_numeric_note':'Actual API versus cached output checked assignments, observation masks/coordinates/identities and link edges exactly. Link costs may differ by float32 shoulder-scale rounding, documented above.'}
    write(U/'RECOVERY.json',recovery)
    at=datetime.now(timezone(timedelta(hours=9))).isoformat();summary=read(U/'SUMMARY.json')
    next_plan='손 관측·좌우 확실성 분리와 중복1건 보류 결과를 사용자와 검토한다. 기존C/웹 유지·학습 대기. 남은9건의 틀린 잠정 좌우는 정정되지 않았으며 raw관측72점과 미확정상태로 보존했다. 새로운 관측 묶음은 학습용최종입력이 아니므로 기존상반신ROI/시간/마스크와 어떻게 연결할지, C대비중복보류만/미확정관측보존의 기여를 어떻게 구분할지 조건부터 정해야 한다. 단순마스크추가가 성능을 높인다고 가정하지 않는다. 사용자·멘토자료 학습/튜닝/선택금지. 완료재생·검산 반복불필요.'
    result={**summary,'status':'VALIDATED_OBSERVATION_PRESERVATION_AND_DUPLICATE_GUARD_AWAITING_REVIEW','at':at,'new_withheld':1,'new_reassigned':0,'remaining_wrong_hints':9,'remaining_wrong_hints_certain':0,'normal_valid_points_retained':3330,'normal_uncertain_identity':436,'protected_pins':len(read(U/'CONTEXT.json')['pins']),'independent':'PASS_AFTER_DOCUMENTED_NUMERICAL_VERIFIER_RECOVERY','integration':'PASS','visual':'1 new change reviewed; prior remaining source evidence reused','normal_zero_observation':0,'missing_coordinates_synthesized':0,'training_authorized_now':False,'final_training_tensor_ready':False,'result':str(U/'RESULT_KO.html'),'next_plan':next_plan}
    write(U/'RESULT.json',result);write(U/'NEXT_PLAN.json',{'next_single_action':next_plan,'training':False,'production_change':False})
    text='''손목 신뢰 부족·손 겹침 후속 보완 결과

결론
손이 보인다는 사실과 왼손·오른손을 확신할 수 있는지를 분리하는 연구용 관측 출력 구조를 구현했다. 손목 연결이 불확실해도 관측한 손가락 좌표를 보존한다. 겹침 중복은 최근 확정된 추적과 손가락 영역이 함께 뒷받침될 때만 잘못된 좌우 배정을 보류하도록 보완했다. 무산 영상1프레임의 중복 L배정을 추가로 보류했고, 정상 배정441건과 교차315장면은 보존했다.

손목 신뢰 부족을 어떻게 처리했나
Pose의 손목 연결 신뢰와 Hand의 각 관절 관측 여부를 구분했다. 실제 입력 안이고 Hand 신뢰 기준을 통과한 점은 좌우 미확정이어도 좌표를 유지한다. 손목 자체가 보일 때만 손목 상대 좌표를 계산한다. 손목이 안 보이면 가짜 손목을 만들어 채우지 않고 관측한 절대 좌표만 남긴다.
기존 잠정 좌우는 suggested_side로 보존하고, 독립 근거가 있는 좌우는 identity_side에 따로 기록한다. 미확정을 손 없음이나 실패로 처리하지 않는다. 직전 프레임의 대응 가능한 손바닥 관절도 가능한 연결 후보로만 기록하며, 이것만으로 왼손/오른손을 확정하지 않는다.
공개 전체 관측 중 좌우 근거가 미확정이어도 관절이 보인3692손/프레임을 보존했다. 양쪽 Pose손목을 사용할 수 없어도 Hand 관절을 보존한757손/프레임이 포함된다. 이는 수어 정답률이나 사람 수가 아니다.

중복 검출을 어떻게 보완했나
기존에는 겹침 면적 또는 현재 양팔 비용만으로는 왜곡된 중복 손을 충분히 구분하지 못했다. 이번에는 손가락 영역의 겹침과 최근의 확정된 반대손 추적 근거를 함께 확인한다. 자신의 확정된 추적 이력이 있으면 정상 손 교차일 수 있으므로 보호한다. 확정된 기준 손이 없으면 어느 손을 버릴지 억지로 고르지 않는다.
무산 REAL06 전체화면 검출 f99: 실제 위쪽 오른손을 raw0와 raw1이 함께 덮었다. raw0의 오른손 근거가67ms 전부터 확인됐고, raw1은 왼손 확정 이력이 없었다. raw1 손가락 영역85%가 기준 손 영역과 일치해 잘못된 L배정을 보류했다. raw1의 관측21점은 그대로 보존했다. 가려진 실제 왼손을 복원한 것은 아니다.
단어명·촬영자별 예외 없이 기존 거리·겹침·100ms 기준을 조합했다. 결과를 보고 계수를 바꾸지 않았다.

공개 검증
- TRAIN899영상 ×2검출조건,226362프레임/305602손 후보를 재생했다. 추가 검출이나 분류모델 추론은 하지 않았다.
- 직전 후보 대비1건 보류,0건 좌우 재배정,305601건 기존 배정 유지.
- 알려진 정상441/441,교차315/315 보존. 정상441건의 사용 가능한 관절3330점을 보존했고 손 관측이0점이 된 정상 사례는0건이다.
- 정상441건 중436건은 좌우 근거가 여전히 임시다. 올바른 제안일 수 있다는 것과 확정 근거가 있다는 것은 다르므로 관측과 제안을 보존하면서 불확실성도 그대로 남겼다.
- 전체6292471개 사용 가능한 손 관절 좌표를 원래 관측 값 그대로 검산했다. 이는 모든 검출점이 정확하다는 뜻이 아니라 보존·마스크 계산이 맞다는 뜻이다.
- 합성 계약21항목 통과. 별도 scalar 계산으로 관측 마스크·좌표·손목 상대 좌표·가능한 연결·좌우 확실성·중복 보류를 전수 확인했다.
- 실제 연결 코드와 저장값·손 후보 순서 반전1062프레임,앞부분565프레임,관측 출력 순서 검사8496항목 통과. 원본 배열은 읽기 전용으로 검사했다.

남은 문제
기존 고정감사의13오류 중4건은 보류,9건의 잠정 좌우 오류가 남았다. 별도로 발견한5오류는 앞선 보완을 유지해2정정+3보류다. 누적18손/프레임 사건에서2정정+7보류+9미해결이며 전체 데이터 오류율이 아니다.
남은9건 모두 좌우를 확정하지 않고 관측72점을 보존했다. 일부는 화면 안에2점만 남아 손 전체 모양을 알 수 없다. 관측을 보존한 것을 좌우 정정이나 손모양 복원 성공으로 세지 않는다. 공항의 다른 중복 프레임처럼 확정된 기준 손이 없는 경우도 남아 있다. 기존의 판정 불확실2건은 여전히 불확실로 유지한다.

검산 중 수정한 점
최초 수치 검산은 어깨 폭의float32/float64 반올림 차이 때문에 링크 비용 비교에서 멈췄다. 첫 차이의 상대 크기는약2.68e-8이었다. 최초 오류와 검사 코드를 보존하고, 해당 실수 비용의 수치 비교 오차만float32 정밀도에 맞춰 별도 검사 파일에서 처리했다. 좌우·마스크·연결 여부는 여전히 정확히 일치해야 하며 전수 통과했다. 후보·데이터·결과·판정 기준은 바꾸지 않았다.

적용 범위와 다음 단계
현재 출력은 검출 단계의 관측 보존 구조이며 기존 TCN에 바로 꽂을 최종 학습 입력이 아니다. 전체화면 관측은 이후에도 기존 상반신 학습 범위 마스크를 적용해야 한다. 이번 작업에서 운영 입력이나 모델을 바꾸지 않았다.
알려진 공개 TRAIN 사례를 참고해 설계하고 같은 Codex가 시각검토했으며 수치만 별도 코드로 검산했다. 독립 사람 정답 판정이나 새 사용자 성능 검증은 아니다. 분류 정확도 향상은 아직 알 수 없다.
사용자와 결과 확인 후, 기존C와 비교할 입력 연결·동결조건을 먼저 정한다. 중복 보류 효과와 미확정 관측 보존 효과를 구분하고 추가 정보가 무조건 도움이 된다고 가정하지 않는다.
기존 C/웹/TCN/DTW 유지. 사용자·멘토·DEV·보류 자료 사용0. 학습0,분류추론0,카메라·다운로드·외부전송·Git 변경0.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
    table=''.join(f'<tr><td>{html.escape(x["word"])}</td><td>{x["view"]} · f{x["index"]}</td><td>{x["usable_points"]} / 21</td><td>좌우 미확정 · 제안은 보존</td></tr>' for x in remaining)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 관측 보존·중복 검출 보완</title><style>body{margin:0;background:#f1f5f8;color:#193046;font-family:Malgun Gothic,sans-serif;line-height:1.8}main{max-width:1120px;margin:auto;padding:38px 24px}h1{font-size:32px;line-height:1.4}h2{font-size:24px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}section,.card{background:white;padding:22px;border-radius:12px;margin:18px 0}.num{font-size:32px;color:#156f63;font-weight:700}img{width:100%;height:auto}table{width:100%;border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #dbe4ea;text-align:left}a{color:#126c9a}.flow{display:grid;grid-template-columns:1fr 1fr;gap:18px}.box{background:#eaf4f3;border-radius:10px;padding:18px}.note{border-left:5px solid #d69d2e}details{margin-top:18px}p{white-space:pre-line}@media(max-width:760px){.cards,.flow{grid-template-columns:1fr 1fr}table{font-size:12px}}</style><main>
<h1>손은 보존하고, 좌우의 불확실성은 따로 표시</h1><p>공개 자료로 검증한 연구용 후보 · 기존 C와 웹 유지 · 학습 전 검토</p>
<div class="cards"><div class="card"><div class="num">1건</div>중복 좌우 배정 추가 보류</div><div class="card"><div class="num">441/441</div>정상 배정 보존</div><div class="card"><div class="num">315/315</div>손 교차 보존</div><div class="card"><div class="num">9건</div>잠정 좌우 오류 잔존</div></div>
<section><h2>‘손이 보임’과 ‘어느 손인지 앎’을 구분합니다</h2><div class="flow"><div class="box"><b>관측한 정보</b><p>화면 안에 보이는 관절 좌표와 관측 여부를 보존합니다. 손목 연결 신뢰가 낮다는 이유로 Hand 관절 전체를 버리지 않습니다.</p></div><div class="box"><b>좌우의 근거</b><p>기존 좌우 제안은 남기되, 확정 근거가 없으면 ‘미확정’으로 표시합니다. 손 없음과 구분하고 좌우를 억지로 정하지 않습니다.</p></div></div><p>남은9건의 관측72점도 보존했습니다. 일부는21점 중2점만 실제 입력 안에 있어, 손 전체 모양을 복원했다고 말할 수 없습니다.</p></section>
<section><h2>무산: 같은 오른손의 중복 배정을 막았습니다</h2><p>최근의 확정된 오른손 추적과 손가락 영역이 함께 뒷받침된 경우에만 중복 후보를 보류합니다. 자신의 확정된 추적 이력이 있는 손은 교차 동작일 수 있으므로 보호합니다.</p><img src="evidence/change_01.jpg" alt="무산 실제 오른손 중복 후보의 L배정을 보류하고 관측 좌표를 보존한 원본 비교"><p>변경은 f99의 raw1 한 건입니다. 잘못된 L배정은 보류했지만 관측21점은 남겼습니다. 아래쪽 실제 왼손의 가려진 모양을 복원한 것은 아닙니다.</p></section>
<section><h2>남은9건: 정보 보존과 오류 해결은 다릅니다</h2><table><thead><tr><th>단어</th><th>조건·프레임</th><th>보존한 관절</th><th>처리</th></tr></thead><tbody>'''+table+'''</tbody></table><p>위 사례의 잠정 좌우가 올바르게 정정된 것은 아닙니다. 원래 관측을 보존하고 ‘좌우 미확정’ 정보를 따로 남긴 상태입니다. 정상 사례의 좌표도 동일하게 보존했습니다.</p></section>
<section><h2>검증 범위</h2><p>TRAIN899영상 ×2조건 · 226,362프레임 · 305,602손 후보. 직전 후보 대비305,601건 배정 유지,1건 추가 보류. 신규 좌우 재배정은0건입니다.</p><p>정상441건의 유효 관절3,330점을 보존했으며, 관측이0점으로 사라진 정상 사례는0건입니다. 정상441건 중436건의 좌우 근거는 여전히 임시 상태로 구분했습니다.</p><p>계약21항목,별도 전수 수치검산,실제 연결·순서반전1,062프레임,앞부분565프레임 검증 완료. 변경된 원본 장면도 확인했습니다.</p></section>
<section class="note"><h2>현재 모델 성능이 개선됐다는 결과는 아닙니다</h2><p>새 출력은 검출 단계의 관측 보존 구조입니다. 기존 상반신 영역·시간·마스크 조건에 연결할 최종 학습 입력은 별도 검토가 필요합니다. 이번에는 모델 학습이나 분류 정확도 평가를 하지 않았습니다.</p><p>기존 C·웹·DTW는 유지했습니다. 사용자 촬영은 읽거나 학습·튜닝에 사용하지 않았습니다. 다음에는 결과를 함께 확인하고, 중복 보류와 미확정 관측 보존을 기존 C와 어떻게 비교할지 조건을 정합니다.</p></section>
<details><summary>상세 보고·검증 근거</summary><p><a href="RESULT_KO.txt">상세 보고</a> · <a href="RESULT.json">집계</a> · <a href="OBSERVATION_PRESERVATION.json">관측 보존 검증</a> · <a href="INDEPENDENT.json">독립 수치 검산</a> · <a href="RECOVERY.json">최초 검산의 반올림 오차와 복구 기록</a></p><p>구현·시각검토는 같은 Codex가 수행했고 수치 계산을 별도 코드로 검산했습니다. 알려진 공개 TRAIN 사례를 참고한 개발 진단이며 독립 사람·새 사용자 검증은 아닙니다.</p></details></main></html>'''
    (U/'RESULT_KO.html').write_text(page,encoding='utf8')
    for target in re.findall(r'(?:href|src)="([^"]+)"',page):assert (U/target).exists(),target
    paths=list(U.glob('*.py'))+list(U.glob('*.json'))+[U/'RESULT_KO.txt',U/'RESULT_KO.html']+list((U/'evidence').glob('*.jpg'))
    write(U/'FINAL_AUDIT.json',{'status':'PASS','pins':{str(p):sha(p) for p in paths if p.name!='FINAL_AUDIT.json'},'protected_pins':21643,'output_pins':1798,'known_initial_failure_preserved':str(U/'FIRST_VALIDATION_FAILURE.json'),'recovery':str(U/'RECOVERY.json'),'all_links_exist':True,'training':0,'model_changed':False})
    brief='손 관측·좌우확실성 분리와 최근확정이력+손가락영역 중복보류 별도후보 검증완료. 공개TRAIN899×2조건226362프레임/305602손후보,직전대비무산masked99 raw1의오배정1추가보류·좌표21점유지,305601배정유지. 정상441/441·교차315/315보존,정상유효관절3330점/관측0된정상0. 전체유효관절6292471점보존,좌우미확정관측3692손/프레임·양Pose손목불가에도관측보존757. 원래13오류4보류/9잔존,추가5는2정정/3보류유지. 잔존9의72점보존·확정좌우승격0;정상436도임시근거여서미확정표시. 손형태복원/인식률향상아님. 최초검산float32/64반올림불일치보존후실수비용오차허용만복구;좌우/마스크/연결boolean정확일치전수PASS. 계약21/API1062/prefix565검사PASS,보호21643핀유지,새변경1원본검토. 기존C/TCN/DTW/웹유지,사용자·멘토·DEV·보류사용/학습/분류추론0. 검출출력구조이며최종학습입력미연결,결과검토후학습조건결정.'
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';ctx=p.read_text(encoding='utf8');m=re.search(r'^## 8\..*$',ctx,re.M);ctx=ctx[:m.end()]+'\n\n### 2026-10-18 손 관측 보존·중복 검출 추가 보완 검증 완료\nops/'+U.name+'. '+brief+'\n'+ctx[m.end():];m=re.search(r'^## 9\..*$',ctx,re.M);ctx=ctx[:m.end()]+'\n\n### 2026-10-18 관측 보존 후보 검토·학습 연결 조건 대기\n'+next_plan+'\n'+ctx[m.end():];p.write_text(ctx,encoding='utf8')
    state=read(O/'STATE.json');state['updated_at']=at;state['current_stage']='WORD100_UNCERTAIN_HAND_PRESERVATION_VALIDATED_AWAITING_REVIEW';state['next_single_action']=next_plan;state[U.name]=result;write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

### 현재 목표·마지막 완료
ops/{U.name}. {brief}

### 다음 단일 작업
{next_plan}

### 근거·실행
- CONTEXT: 승인/동결조건/21643보호핀. candidate.py: 별도 관측보존·중복보류. RESULT_KO/RESULT/NEXT_PLAN 보고.
- CHANGES/ANNOTATED_COMPARISON/OBSERVATION_PRESERVATION/VISUAL_REVIEW/INDEPENDENT/INTEGRATION_CHECKS/FINAL_AUDIT 검증.
- 공개재생exec68232 종료0. 최초verify.py 실수비용정밀도assert exit1은FIRST_VALIDATION_FAILURE.json에보존. 후보/입력/결과변경없이verify_recovery.py exec57073 종료0. 정밀도이외좌우/마스크/연결판정정확일치.
- 현재 실행/잠금없음. st-gcn 자동화삭제유지. C/TCN/DTW/웹/원본/W:/sources/과거동결자료유지. 학습·추론·사용자/멘토/DEV/보류사용0.
- 확인된정상손436건도미확정좌우표시지만제안/관측은그대로. 이를정상436건삭제또는오류해결로혼동금지.
- 미확정관측출력은detector범위다. 학습에쓸때기존상반신ROI마스크를추가적용해야하며이번에최종TCN입력은변경하지않았다.
''',encoding='utf8')
    assert read(O/'STATE.json')[U.name]['remaining_wrong_hints']==9
    print('Reports, evidence, context8/9, STATE and handoff synchronized')
if __name__=='__main__':main()
