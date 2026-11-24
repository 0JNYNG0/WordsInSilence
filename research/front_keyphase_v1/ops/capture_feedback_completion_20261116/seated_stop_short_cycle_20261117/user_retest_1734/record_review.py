from diagnose import *
from datetime import datetime
def main():
 q=read(D/'DIAGNOSIS.json');rows=q['rows'];at=datetime.now().astimezone().isoformat();ROOT=O.parents[2]
 summary=dict(status='ACTUAL_RETEST_AUTO_END_JITTER_AND_ENDPOINT_UNCONFIRMED',at=at,attempts=[dict(number=r['number'],word=r['word'],stop_reason=r['stop_reason'],overall_pass=r['overall_pass'],observed_repeats=next((c.get('observed_count')for c in r['criteria']if c['id']=='repeat_order'),None),stop_to_browser_ack_seconds=r['stop_to_browser_ack_seconds'])for r in rows],first_thanks=dict(actual_motion_supported=True,auto_stop_failed=True,longest_rest_dwell_seconds=rows[1]['longest_rest_dwell_seconds'],required_rest_dwell_seconds=.75,visually_lowered_interval_seconds=[3.2,6.5],diagnostic_interval_frames=99,hand_active_frames=25,rest_dwell_frames=74,reason='Small intermittent hand-high estimates reset continuous rest interval; not waiting for correct sign result.'),last_thanks=dict(auto_stop_succeeded=True,approach_candidates=2,confirmed_arrivals=0,reason='Saved learned relation stream contains no near endpoint in action; both observed approaches remain endpoint_unconfirmed. Does not prove physical failure to touch.'),stream_frames=sum(r['frames']for r in rows),stream_drops=sum(r['session']['dropped']for r in rows),training=False,tuning=False,candidate_selection=False,new_inference=False,runtime_changed=False)
 write(D/'RESULT.json',summary)
 (D/'RESULT_KO.txt').write_text('''새 실제 촬영3건 진단
사과: 자동 종료, 판정 통과. 첫 감사: 수동 종료, 반복2회 및 손모양·방향 등 판정 통과. 두 번째 감사: 자동 종료, 움직임2후보를 찾았으나 도달 끝 미확인으로 세부판정 보류.
첫 감사는 손을 내린 약3.2~6.5초에도 99관측 중25회 손 활성으로 바뀌었다. 양팔 시작대기와의 거리 조건은 해당 예시에서 충족했는데 손높음 판정이 간헐적으로 끼어0.75초 연속대기를 끊었다. 전체 최장 연속대기 약0.598초. 중간 동작을 더 해야 종료되는 구조가 아니며 손가락 미검출 자체만이 원인은 아니다. 실제 하단 추정점의 불안정이 종료조건에 영향을 준 것으로 확인.
자동 종료와 수어 판정은 별개다. 두 번째 감사는 TCN 감사1위였고 접근 움직임2개가 있었지만 학습된 도달상태를 확정하지 못했다. 원본 장면만 보고 실제 접촉 실패나 사용자 잘못으로 단정하지 않는다. 세부 손모양 출력도 존재하지만 완료구간 부족 때문에 최종 카드가 보류되는 영향이 있다.
548프레임 모두 처리, 처리누락/오류0, 선택적 재검출0. 종료 뒤 실제브라우저확인 약0.703/0.706/0.892초. 긴 기다림은 종료확정 이전이며 계산 지연과 구분한다.
다음 공통 보완: 실제 팔 대기복귀 중 단발성 손점 흔들림과 실제 손 재진입을 구분하는 시간적 안정화 후보를 공개 정면자료에서 재현. 손을 실제 올린 경우·낮은 동작·교차·일시 가림은 종료시키지 않아야 한다. 별도로 도달분류와 움직임연결의 불일치를 공개 정답으로 비교하여 원인을 분리한다. 사용자 촬영으로 임계값이나 가중치를 선택하지 않는다.
이번은 저장 기록·영상 진단만. 기존116핀, 서버와 모델 유지. 추가학습·추론·카메라시작 없음. 실제 촬영 확인은 시도됐지만 안정화 완료/시연 완주로 표시하지 않는다.
''',encoding='utf-8')
 next_action='실제촬영에서 재현된 종료 대기 리셋을 공개 정면자료로 보완: 팔 대기복귀 중 순간 손점 흔들림과 실제 손 재진입을 구분하고 조기종료 대조검증. 별도로 감사 두 접근의 끝 미확인을 관측/학습 도달상태/구간연결로 분리. 사용자자료는 진단만 유지.'
 graph=read(R/'GRAPH.json');graph['status']=summary['status'];graph['latest_diagnosis']=str(D/'RESULT.json')
 for node in graph['nodes']:
  if node['id']=='R1':node['status']='public_passed_actual_stop_jitter_rework'
  if node['id']=='R2':node['status']='public_passed_actual_endpoint_unconfirmed'
  if node['id']=='R4':node['status']='actual_capture_attempted_not_yet_accepted'
 write(D/'GRAPH.json',graph)
 statep=O/'STATE.json';state=read(statep);state['current_stage']=summary['status'];state['next_single_action']=next_action;state['updated_at']=at;w=state['capture_feedback_completion_20261116'];w.update(status=summary['status'],updated_at=at,next_single_action=next_action,current_graph=str(D/'GRAPH.json'),current_work_graph=str(D/'GRAPH.json'),current_checkpoint=str(D/'RESULT.json'))
 w['seated_stop_short_cycle_20261117']['actual_retest_1734']=dict(report=str(D/'RESULT.json'),text=str(D/'RESULT_KO.txt'),auto_end_passed=2,total=3,feedback_supported=2,actual_acceptance_complete=False,training_use=False,tuning_use=False)
 write(statep,state)
 context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
 s=s.replace(h8,h8+f'\n\n### 2026-11-17 17:34 실제 촬영3건 — 종료 흔들림·끝 미확인 남음\n사과자동종료/통과,첫감사수동종료/두반복통과,재감사자동종료/접근2후보의끝미확인. 첫감사손내림3.2~6.5초99관측중손활성25회로0.75초연속대기리셋,최장0.598초. 동작정오를기다리는것아님.548프레임처리·누락/오류0·재검출0;종료후브라우저0.703/0.706/0.892초. 원본진단만,학습/튜닝/후보선택/새추론/실행수정없음,116핀유지. R4실제시험실시됐으나완료아님,R1/R2공개재현후속필요. 근거{D/"RESULT_KO.txt"}.\n',1)
 s=s.replace(h9,h9+f'\n\n### 2026-11-17 실제촬영 진단 후 공통 보완\n{next_action}\n현재웹유지,이미확인한실패를사용자에게계속재촬영요구하지않음.\n',1);context.write_text(s,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {at}
목표 정면웹캠 사과·감사 시범→촬영→자동종료→정상/교정/관측부족→재시도·퀘스트.5방향아바타전용.
현재사용자답변 사과1,감사1종료안돼재감사자동종료확인. 저장3건 진단완료. 사과자동종료/통과;첫감사수동종료/두반복·모양통과;재감사자동종료이나접근2후보끝미확인. TCN세번선택단어1위.548프레임처리/누락·오류0/재검출0. 종료후브라우저0.703/0.706/0.892초.
첫감사하단추정점손활성간헐오판으로대기리셋:영상에서손내림3.2~6.5초99관측중25활성/74대기,최장연속대기0.598초<0.75초. 팔대기복귀거리충족인예시에서손높음만방해. 다시행동해야종료하는구조아님. 재감사는학습도달상태near없음,접근2개는탐지;실제접촉실패/사용자잘못단정금지.
근거 {D/'RESULT_KO.txt'} / RESULT.json / DIAGNOSIS.json / GRAPH.json. 실제영상스냅샷은이폴더로컬진단전용;공개학습자료와혼합금지.
다음 {next_action}
직전공개보완R1/R2/R3통과·웹적용유지하되실제확인실패로R4완료아님. 새일괄학습보다공통관측/시간안정화·도달원인분리필요. 사용자자료튜닝금지.
실행 START_WEB_DEMO.cmd→frontal_demo_launcher.py→{R/'serve.py'},8821기록PID31680. 이번재시작/카메라열기없음. 최신서비스stop_short_cycle_version=seated_rest_dual_window_v1 및 selective_retry_v3. 실제실행상태는다음작업시재확인.
이번학습0/새모델추론0/코드수정0/기존116핀·C·T3-F·100단어본체유지. 사용자기록은읽기진단만. 총학습75변경없음. 공개시험조건과실사용결과구분. 독립전문가검증아님.
''',encoding='utf-8')
 pins=read(R/'web/ADMISSION.json')['pins'];assert all(sha(f)==h for f,h in pins.items());assert all(sha(f)==h for r in rows for f,h in r['source_hashes'].items())
 write(D/'VERIFICATION.json',dict(passed=True,source_files_unchanged=True,runtime_pins_unchanged=len(pins),new_inference=False,training=False,tuning=False,candidate_selection=False,diagnostic_only=True))
 print('Recorded actual retest; runtime unchanged')
if __name__=='__main__':main()
