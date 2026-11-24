from run import *
from datetime import datetime,timezone,timedelta
import html,re
def main():
    assert read(U/'INDEPENDENT.json')['status']=='PASS' and read(U/'INTEGRATION_CHECKS.json')['status']=='PASS' and read(U/'DELTA_INDEPENDENT.json')['status']=='PASS'
    assert not (U/'RUNNING.json').exists()
    at=datetime.now(timezone(timedelta(hours=9))).isoformat();s=read(U/'SUMMARY.json');d=read(U/'REMAINING_DIAGNOSIS.json')
    basis={23:'실제 왼손이 하단 경계로 올라오는 장면. 이전 R은 오류. 양쪽 Pose 손목 신뢰도 부족.',76:'실제 왼손이 막 진입. Hand 손바닥 일부는 입력 아래쪽. 화면 밖 좌표로 좌우를 강제하지 않음.',143:'실제 왼손 진입, 양쪽 Pose 손목 신뢰도 부족. 이전 후보가 더 뒤60/61에서 보류/정정한 영상이며56 오류는 남음.',200:'첫 프레임 휴식 자세의 실제 오른손. 팔 아래 관측 신뢰가 부족한 라벨 단독 L오류.',207:'실제 왼손이 올라오는 장면. 관측 팔꿈치는 있지만 신뢰할 수 있는 손목 연결이 없음.',264:'실제 오른손 진입. 손바닥 대부분이 하단 경계 아래이고 손목 신뢰도 부족.',272:'실제 오른손 진입. 손바닥 일부가 경계 아래, 손목 신뢰도 부족.',294:'실제 오른손을 위에 둔 손 겹침. 두 raw 후보가 위쪽 같은 손을 덮음. 대상 raw1의 양팔 비용이0.285/0.288로 모호하여 현재 팔 근거만으로 안전한 정정이 안 됨.',309:'실제 오른손 하나를 같은 위치에서 두 후보로 검출. 두 후보 모두 임시이며 양쪽 손목 신뢰도 부족. 어느 raw를 버릴지 현재 확정 근거 없음.',409:'휴식 중 실제 오른손 일부가 경계에 보임. 양쪽 손목 신뢰도 부족, 라벨 단독 L오류.', 'additional_7':'실제 오른손. 현재 오른쪽 팔 연결은 유효하고 왼쪽 Pose 손목은 화면 안 좌표지만 신뢰도0.465<0.5. 잘못된 L을 보류. 다음42는 직전 후보부터 중복 보류이고43은 원래 R로 정상, 이번 변경이 정정한 것으로 세지 않음.'}
    reviews=[{**x,'basis':basis[x['id']],'inspected_this_turn':True,'reviewer':'Codex source/adjacent-frame review','new_change':x['id']=='additional_7'} for x in d['events']]
    write(U/'VISUAL_REVIEW.json',{'reviewer':'Same Codex operator, not independent human ground truth','all_remaining_eleven_reviewed':True,'all_new_changes_reviewed':1,'correct_new_withhold':1,'new_identity_regression_found':0,'records':reviews})
    write(U/'CORRECTION.json',{'previous_report':str(T/'RESULT_KO.txt'),'old_statement':'공항에서 반대쪽 손목이 화면 밖이라 양쪽 연결을 사용할 수 없음','corrected':'공항 sensor41: 반대쪽 Pose손목(x1120.4,y889.0)은1920x922 입력 안에 있으나 visibility0.4650577<0.5. 사용불가의 원인은 낮은 신뢰도. 실제 내려놓은 손의 일부는 범위 밖이지만 이것과 Pose손목 좌표 사용불가 원인은 별개.','evidence':str(U/'REMAINING_DIAGNOSIS.json'),'candidate_input_or_result_changed':False})
    next_plan='한쪽 팔 재확인 결과를 사용자와 검토한다. 기존 C/웹과 별도 후보를 보존하고 학습은 대기한다. 남은10중9는 핵심 전·양손목 신뢰부족,1은 실제 동작 중 겹침 중복. 다음 공통 보완은 raw 손 관측을 보존하면서 좌우가 미확정인 정보의 사용과, 확정된 이력으로부터 두 중복 후보의 관절 대응/궤적을 비교하는 계획을 먼저 정한다. 신뢰 기준을 일괄 낮추거나 임시 손을 일괄 삭제하지 않는다. 사용자 촬영 학습/튜닝/선택금지. 이미 완료된 재생/검산 반복 불필요.'
    result={**s,'status':'VALIDATED_ISOLATED_SINGLE_ARM_RECHECK_AWAITING_REVIEW','at':at,'independent':'PASS','delta_independent':'PASS','integration':'PASS','visual_changes_reviewed':1,'visual_remaining_reviewed':11,'new_regression_found':0,'remaining_original_errors':10,'remaining_additional_errors':0,'remaining_errors_before_core':9,'remaining_errors_inside_core':1,'normal_low_confidence_controls':307,'protected_pins':len(read(U/'CONTEXT.json')['pins']),'training_authorized_now':False,'result':str(U/'RESULT_KO.html'),'next_plan':next_plan}
    write(U/'RESULT.json',result);write(U/'NEXT_PLAN.json',{'next_single_action':next_plan,'training':False,'preserve_C':True,'use_user_mentor_data':False,'remaining_work_is_not_declared_solved':True})
    summary='남은 오류 후속: 한쪽 팔 연결 재확인 후보를 공개TRAIN899×2조건226362프레임/305602후보에 검증. 직전후보 대비 공항sensor41의 잘못된L1건 추가보류,305601유지. 정상441/441·교차315/315·직전4변경·기존확정/미배정 모두 보존. 원래13오류는3보류+10미해결,추가5는2정정+3보류로 잘못된배정0. 남은10중9는 핵심동작주석전/양손목신뢰도<.5,1은 무산 동작중 중복. 정상441중307도 양손목신뢰도<.5이므로 일괄삭제/문턱완화하지 않음. 공항 반대손목은 화면밖이 아니라 화면안추정좌표의 낮은신뢰도였다는 이전설명정정. 계약23·전수별도검산·API1010/prefix519/숨긴좌표10630검사PASS,새변경1·남은11원본검토. 보호19835핀유지. 기존C/웹/TCN/DTW유지,사용자·멘토·DEV·보류사용0,학습/분류추론0. 수어인식률향상/전체오류해결아님. 결과검토후학습.'
    text='''남은 좌우 오류 후속 검증

결론
한쪽 팔만 확실히 관측되어도 현재 팔·손 연결을 재확인하는 별도 후보를 만들었다. 직전 후보보다 공항 영상1프레임의 잘못된 왼손 배정을 추가로 보류했다. 정상 배정·교차는 보존했다. 추가 좌우 정정은0건이며, 보류는 잘못된 좌우로 사용하지 않는다는 뜻이지 손을 복원했다는 뜻은 아니다.

무엇을 바꿨나
이전에는 양쪽 팔 연결이 모두 보일 때만 임시 좌우를 재확인했다. 이번에는 현재 손을 뒷받침하는 한쪽 어깨·팔꿈치·손목이 화면 안에서 신뢰 기준을 통과하면 재확인할 수 있다. 반대쪽 팔의 미관측은 모르는 상태로 남긴다. 첫 충돌은 보류, 같은 실제 손에서 두 프레임 연속 근거가 일치할 때만 정정한다. 거리·경쟁 기준,2회/100ms,기존 확정·미배정 보호는 그대로다. 단어/사람별 예외, 미검출 좌표 생성, 미래 프레임은 없다.

검증 결과
공개 TRAIN899영상 ×2검출조건,226362프레임/305602손 후보를 비교했다. 전체화면 검출과 상반신 검출을 구분해서 사용했고 새 MediaPipe 추출은 하지 않았다.
- 직전 후보 대비1건 보류,0건 재배정,305601건 동일.
- 알려진 정상441/441,손 교차315/315,직전 후보에서 바꾼4프레임 모두 보존.
- 기존 확정301908개와 기존 미배정2411개 보호 유지.
- 모든 신규 변경1건과 남은 오류11건의 원본/앞뒤 장면 확인. 새 정상 배정 회귀는 발견하지 못했다.
- 합성 계약23항목,별도 scalar 전수계산,직전결과와 별도배열비교 통과. 실제 연결/순서반전1010프레임,앞부분519프레임,미사용Pose좌표10630점 교란 통과.

남은 문제를 구분했다
현재 남은10오류 중8건은 손이 처음 올라오거나 휴식 자세에서 라벨이 잘못 붙은 경우다. 공항의 이전40프레임은 같은 손 중복검출이고,무산99프레임은 실제 동작 중 겹친 손의 중복검출이다.
남은10중9는 주석상 핵심 수어 동작 시작 전이며,양쪽 Pose손목의 신뢰도가0.5미만이다. 손목 위치 숫자가 출력되어도 믿을 수 있는 관측으로 쓰지 않은 것이다. 이는 오류가 중요하지 않다는 뜻은 아니다. 전체 구간 입력과 이후 추적에 영향이 있을 수 있으며 수어 인식에 미치는 정도는 아직 검증하지 않았다.
무산1건은 실제 핵심 동작 안이다. 같은 위쪽 손을 두 후보가 덮지만,문제 후보의 양팔 연결 비용이0.285/0.288로 비슷하여 현재 팔 근거만으로 어느 것을 제거해야 안전한지 충분히 구분하지 못했다.
정상441건 중307건도 양쪽 손목 신뢰도가 낮았다. 따라서 낮은 신뢰도의 손을 전부 삭제하면 정상 손도 크게 손실된다. 문턱을 임의로 낮춰 추정 좌표를 확정 근거로 만드는 것도 하지 않았다.

누적 오류 수의 정확한 의미
원래456건 감사에서 발견한13오류:3보류,10미해결. 이후 별도로 확인한5오류:2정정,3보류,잘못된 배정0. 서로 다른18개 손/프레임 사건의 누적 진단이다.18개 단어 정확도나 전체 영상의 오류율을 뜻하지 않는다.

이전 설명 정정
앞선 보고에서 공항 사례의 반대쪽 손목이 화면 밖이라고 설명한 부분은 정확하지 않았다. sensor41의 Pose손목 좌표는 입력1920×922 안의(1120.4,889.0)지만 신뢰도0.465가 기준0.5보다 낮았다. 실제 손 일부의 화면 이탈과 손목 추정 좌표의 낮은 신뢰도를 구분해 기록했다. 과거 결과는 덮어쓰지 않고 CORRECTION.json으로 연결했다.

한계와 다음 단계
이번 결과는 알려진 공개 TRAIN 사례를 참고하여 만든 후보의 진단이다. 동일 Codex가 구현·시각검토했고 별도 계산 코드로 검산했다. 독립 사람의 정답 판정이나 새 사람 수어 인식 성능 검증은 아니다.
손의 raw 관측과 좌우 확실성을 분리해 보존하고,겹침 중복은 확정된 이력과 후보별 관절 대응을 추가로 비교하는 방향을 다음 계획으로 남겼다. 현재 검출 근거가 부족한 손을 강제 정정하지 않는다. 결과를 함께 검토한 다음 학습 반영 조건을 정한다.
기존 C/TCN/DTW/웹 유지. 사용자·멘토·DEV·보류 자료 사용0. 학습0,분류추론0,새 촬영/다운로드/외부전송/Git 변경0.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    table=''.join(f'<tr><td>{html.escape(x["word"])}</td><td>{x["view"]} · f{x["index"]}</td><td>{html.escape(x["basis"])}</td><td><a href="remaining/{Path(x["image"]).name}">원본 앞뒤</a></td></tr>' for x in reviews)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>남은 좌우 오류 — 추가 검증</title><style>body{margin:0;background:#f2f5f8;color:#162a3c;font-family:Malgun Gothic,sans-serif;line-height:1.75}main{max-width:1120px;margin:auto;padding:42px 24px}h1{font-size:32px;line-height:1.4}h2{margin-top:36px;font-size:23px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,section{background:white;padding:20px;border-radius:12px;margin:18px 0}.num{font-size:32px;font-weight:bold;color:#186957}p{white-space:pre-line}img{width:100%;height:auto;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dbe3e9;padding:12px;text-align:left}td:first-child{min-width:125px}a{color:#146da0}details{margin:20px 0}.alert{border-left:5px solid #dba437}@media(max-width:700px){.cards{grid-template-columns:1fr 1fr}table{font-size:12px}}</style><main>
<h1>남은 좌우 오류: 공항 1프레임 추가 보류<br>정상 배정과 손 교차 유지</h1><p>현재 팔·손 연결을 이용한 연구용 후보 · 기존 C와 웹은 유지 · 학습 전 검토</p>
<div class="cards"><div class="card"><div class="num">1건</div>잘못된 배정 추가 보류</div><div class="card"><div class="num">441/441</div>정상 배정 보존</div><div class="card"><div class="num">315/315</div>손 교차 보존</div><div class="card"><div class="num">10건</div>알려진 오류 잔존</div></div>
<section><h2>왜 바꿨나</h2><p>오른팔과 손의 연결은 확인되는데 반대쪽 손목이 불확실하다고 해서 재확인을 막을 필요는 없습니다. 보이는 팔의 근거로 다시 확인하고, 한 번 충돌하면 잠시 보류합니다. 같은 손에서 두 번 연속 확인될 때만 좌우를 정정합니다.</p><p>이번 공개 결과는 공항의 잘못된 L 배정 한 프레임을 보류한 것입니다. 추가로 오른손으로 정정한 건수는 0건입니다.</p><img src="evidence/change_01.jpg" alt="공항 실제 오른손의 잘못된 왼손 배정을 보류한 원본 앞뒤 장면"></section>
<section><h2>남은 10건은 같은 문제가 아닙니다</h2><p><b>8건:</b> 손 진입·휴식 자세에서 라벨 오류. 팔꿈치는 보이지만 신뢰할 수 있는 손목 연결이 부족합니다.<br><b>1건:</b> 공항의 이전 프레임에서 같은 손 중복 검출.<br><b>1건:</b> 무산의 핵심 동작 중 손 겹침·중복 검출.</p><p>10건 중 9건은 핵심 동작 주석 시작 전입니다. 그러나 이후 추적과 전체 구간 입력에 영향을 줄 수 있으므로 중요하지 않은 오류로 제외하지 않았습니다.</p><p><b>정상 441건 중 307건도 손목 신뢰도가 낮았습니다.</b> 낮은 신뢰도의 손을 모두 지우는 방식은 정상 손을 함께 잃습니다.</p></section>
<section class="alert"><h2>이전 설명 바로잡기</h2><p>공항의 반대쪽 Pose손목은 화면 밖 좌표가 아니라, 화면 안에 추정 좌표가 있지만 신뢰도 0.465로 기준 0.5를 넘지 못한 경우였습니다. ‘화면 밖’과 ‘신뢰 부족’을 구분해 기록했습니다.</p></section>
<section><h2>원본으로 확인한 남은 사례</h2><p>직전 후보에서 남은 11건을 모두 재검토했습니다. 이 중 공항 f41 한 건을 이번에 보류했습니다. 사진의 노란선은 입력 하단 경계이며, 아래 원본 영역은 검토용으로만 보입니다.</p><table><thead><tr><th>단어</th><th>프레임</th><th>관측 근거와 판정</th><th>확인</th></tr></thead><tbody>'''+table+'''</tbody></table></section>
<section><h2>검증 범위와 다음 단계</h2><p>공개 TRAIN 899영상 ×2조건, 226,362프레임과 305,602손 후보 비교. 직전 후보 대비 305,601건 동일. 기존 보완 4프레임 보존. 계약23항목·별도 전수계산·실제 연결 검사 통과.</p><p>새로운 손 좌표를 만들거나 사용자 촬영을 사용하지 않았습니다. 좌우 처리의 제한적인 개선이며 수어 인식률 향상은 아직 검증하지 않았습니다.</p><p>남은 오류는 원본 손 관측과 좌우 불확실성을 분리해 보존하고, 중복 후보의 관절·추적 근거를 비교하는 방향으로 후속 계획을 정리했습니다. 결과 확인 후 학습 반영 여부를 결정합니다.</p><details><summary>상세 보고와 검증 파일</summary><a href="RESULT_KO.txt">상세 보고</a> · <a href="RESULT.json">수치 결과</a> · <a href="INDEPENDENT.json">전수 검산</a> · <a href="DELTA_INDEPENDENT.json">직전 후보 비교 검산</a> · <a href="CORRECTION.json">설명 정정 근거</a></details></section></main></html>'''
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    paths=list(U.glob('*.py'))+[U/n for n in ('CONTEXT.json','PROTOCOL.json','CONTRACTS.json','SUMMARY.json','RESULT.json','RESULT_KO.html','RESULT_KO.txt','NEXT_PLAN.json','VISUAL_REVIEW.json','REMAINING_DIAGNOSIS.json','VISUAL_EVIDENCE.json','CORRECTION.json','INDEPENDENT.json','DELTA_INDEPENDENT.json','INTEGRATION_CHECKS.json','AUDIT_PATTERN.json','DELTA_PREVIOUS.json','PREVIOUS4.json')]+list((U/'evidence').glob('*.jpg'))+list((U/'remaining').glob('*.jpg'))
    for target in re.findall(r'(?:href|src)="([^"]+)"',page):assert (U/target).exists(),target
    pinscheck(read(U/'CONTEXT.json')['pins']);pinscheck(read(U/'OUTPUT_PINS.json'))
    write(U/'FINAL_AUDIT.json',{'status':'PASS','pins':{str(p):sha(p) for p in paths},'protected_pins':len(read(U/'CONTEXT.json')['pins']),'output_pins':len(read(U/'OUTPUT_PINS.json')),'all_links_exist':True,'first_execution_error':None,'training':0,'production_changed':False})
    state=read(O/'STATE.json');state['current_stage']='WORD100_SINGLE_ARM_RECHECK_VALIDATED_AWAITING_REVIEW';state['next_single_action']=next_plan;state[U.name]=result;write(O/'STATE.json',state)
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8');m=re.search(r'^## 8\..*$',s,re.M);assert m;s=s[:m.end()]+'\n\n### 2026-10-18 残여 좌우 오류·한쪽 팔 재확인 후보 검증 완료\n'+summary+'\n'+s[m.end():]
    s=s.replace('残여 좌우','남은 좌우');m=re.search(r'^## 9\..*$',s,re.M);assert m;s=s[:m.end()]+'\n\n### 2026-10-18 한쪽 팔 재확인 결과 검토·학습 대기\n'+next_plan+'\n'+s[m.end():];p.write_text(s,encoding='utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

### 현재 목표·마지막 완료
ops/{U.name}. {summary}

### 다음 단일 작업
{next_plan}

### 근거·실행
- PROTOCOL/CONTEXT: 동결 조건·보호핀. candidate.py: 별도 후보. RESULT_KO.html/txt, RESULT, NEXT_PLAN 결과.
- DELTA_PREVIOUS/DELTA_INDEPENDENT/INDEPENDENT/INTEGRATION_CHECKS/VISUAL_REVIEW/REMAINING_DIAGNOSIS/CORRECTION/FINAL_AUDIT 근거.
- 공개 재생 exec62650 종료0,독립검산 exec70525 종료0,실제연결/시각화/추가검산 종료0. 현재 실행·잠금없음, st-gcn 자동화삭제유지.
- 기존 후보/원본/W:/sources/동결자료읽기전용. 기존C/TCN/DTW/웹 변경0,학습/추론0,사용자·멘토·DEV·보류자료사용0.
- 1건 추가보류는1개 손/프레임의 오배정을 차단한 것. 인식률향상·전체문제해결로 말하지 않는다.
''',encoding='utf-8')
    assert read(O/'STATE.json')[U.name]['compared_to_previous']['withheld']==1
    print('REPORT, evidence pins, STATE, context8/9 and handoff synced',flush=True)
if __name__=='__main__':main()
