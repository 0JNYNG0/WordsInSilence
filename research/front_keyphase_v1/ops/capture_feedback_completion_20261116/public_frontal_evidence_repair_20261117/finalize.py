from pathlib import Path
import json,hashlib,datetime,statistics
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;N=G/'frontal_scope_recovery_20261117';ROOT=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
 admission=read(D/'web/ADMISSION.json');assert admission['admitted']and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in admission['pins'].items())
 validation=read(D/'web/VALIDATION.json');http=read(D/'web/HTTP_CHECK.json');assert validation['passed']and http['passed']
 latency=read(D/'web/LATENCY_CHECK.json');capture=read(D/'web/CAPTURE_CHECK.json');fingers=read(D/'checks/PARTIAL_FINGERS.json');repro=read(D/'checks/AGGREGATION_REPRODUCTION.json')
 summary=dict(status='PUBLIC_FRONTAL_EVIDENCE_REPAIRS_APPLIED_USER_RETRY_PENDING',at=now,implementation='Codex local',verification='Same Codex source/control-flow/public regression, frozen-hash, capture/API/JS syntax checks; no independent expert validation',
  normal_public_preserved=32,normal_public_total=32,tcn_ranks_preserved=32,partial_finger_control_cases=len(fingers['rows']),partial_finger_source_label_rows=fingers['public_label_rows'],partial_finger_source_clips=len(set(r['clip']for r in fingers['rows'])),
  overlap_whole_path_cases=len(repro['rows']),short_overlap_recovered=sum(r['kind']=='short_projected_overlap'and r['passed']for r in repro['rows']),overlap_guard_cases=validation['overlap_checks'],unknown_only_cases=16,
  apple_normal_error_temporal_controls=70,diagnostic_only_cases=5,card_quest_cases=7,auto_stop_controls=9,readiness_control_cases=2,
  public_captures=capture['rows'],paired_latency=latency,training_runs=0,weights_changed=False,private_training=False,private_tuning=False,private_candidate_selection=False,private_replay_for_new_candidate=False,prior_frozen_files_preserved=admission['prior_pins_preserved'],active_pins=len(admission['pins']),
  http=http,physical_camera_started=False,physical_user_retry='pending',scope='Front(F) public sources only. Five-camera sources remain avatar-only. Public known source controls and synthetic masks are not unseen-user accuracy.',
  limits=['Brief overlap does not certify hidden finger state; both visible sides must agree, own points present, no contradictory shape, bracketing interval<=250ms. No filled coordinates.',
  'Partial match preserves known contradictions; normal support still requires the full expected visible finger combination.',
  'Auto-end readiness guard prevents starting a recording without a baseline; it does not guess an unseen arm pose or eliminate all future camera detection failures.',
  'One paired timing pass per four clips; public capture timing excludes browser paint and pre-stop rest wait.'])
 write(D/'RESULT.json',summary)
 report=f'''정면 손모양·겹침·자동 종료 보완 — {now}

완료한 것
공개 정면 자료로 세 문제를 재현하고 보완 코드를 검증한 뒤 최신 웹8821에 적용했습니다.
모델 가중치를 다시 학습한 것이 아니라, 기존 모델의 관측을 묶어 판단하는 코드와 촬영 준비 절차를 고쳤습니다.
사용자 촬영은 새 보완의 학습·튜닝·후보선택·재검증 입력에 사용하지 않았습니다. C·T3-F와 기존 가중치는 그대로입니다.

이전 → 현재
1. 짧은 손 겹침만 있어도 정상 모양을 확인 부족으로 막음 → 앞뒤에 같은 모양이 확인되고 실제 누락/모양 변화가 없는 짧은 겹침은, 이미 확인한 특징을 취소하지 않음.
   250ms 이하 앞뒤 연결 예산을 비교 전에 고정했습니다. 가려진 순간의 손가락 상태를 채우거나 정답이라고 확정하지 않습니다. 실제 누락, 긴 가림, 중간/이후 모양 변경은 계속 보류합니다. 좌표 보간은 하지 않습니다.
2. 손가락 하나의 판정이 불확실하면 다른 손가락에서 확인한 오류도 무시 → 확인된 손가락의 차이는 교정 근거로 남기고 모르는 손가락은 별도 보류.
   정상 통과에는 기존처럼 필요한 손가락 전체의 근거가 필요합니다. 모르는 손가락을 틀렸다고 처리하지 않습니다. 공통 함수는 손/단어명에 의존하지 않고 기대 손가락 상태를 입력받습니다.
3. 대기 자세 기준이 없어도 촬영 시작 → 준비가 확인될 때까지 시작 버튼 비활성/안내. 준비되면 같은 화면에서 시작 가능.
   기존3초 카운트다운과 촬영중 RTMW 재사용, 자동 종료의 팔 복귀 기준은 그대로입니다. 카운트다운 직후에도 준비 상태를 확인합니다.

검증 근거
- 기존 공개 정면 사과15·감사17영상의 정상 판정과 TCN 상위 후보32/32 동일.
- 공개 원본으로 검토된 손가락 정답{fingers['public_label_rows']}행을 바탕으로, 확인된 다른 손가락 하나를 불확실하게 만든 조합450개에서 기존 판정 취소 문제를 재현·보완. 이는450개 새 사람/영상의 정확도가 아니라 관측 마스킹 시험입니다.
- 공개 사과7영상에 짧은 겹침/실제 누락/긴 겹침 조건을 각각 주입한21시험: 짧은 겹침7개는 확인 부족→기존 특징 인정, 실제 누락·긴 겹침14개는 확인 부족 유지. 자연 사용자 오류 영상이 아니라 원인 분리를 위한 합성 관측 조건입니다.
- 별도 겹침/모양 변화42조건, 미확정만 있는16조건, 사과 정상/오류/순서/부분/속도70조건 통과. 의미 오류 정답이 없는 중간 자르기5개는 참고 진단으로 남겼습니다.
- 문장·회색/오류 구분·퀘스트7조건 통과. 사과·감사·한손 순종의 종료9조건에서 기존 종료 시점 유지, 짧은 손 누락으로 조기 종료 없음, 대기만 하면 종료 없음.
- 준비 실패시 시작 차단/같은 미리보기에서 준비 회복2조건 통과.
- 실제 공개 영상→RTMW→녹화·관절 재사용→TCN/피드백2시험: 두 단어 자동 종료/정상 판정 유지. 총{sum(r['frames']for r in capture['rows'])}프레임 모두 처리, 재검출0, 종료후 계산 약{min(r['stop_to_result_seconds']for r in capture['rows']):.2f}~{max(r['stop_to_result_seconds']for r in capture['rows']):.2f}초. 실제 사용자 웹캠/브라우저 화면 체감 시간과 다릅니다.
- 같은 입력4개를 같은 실행환경에서 이전/현재 각1회 비교한 계산 중앙값 {latency['before_median']:.3f}→{latency['after_median']:.3f}초. 약간의 실행 변동이 있고 반복 통계 벤치마크는 아니며, 허용 시간 예산을 유지했습니다.
- 운영 HTTP 공개 사과/감사 정상 판정, 최신 모듈/준비 안내JS 전달, 공개 재생은 퀘스트에 영향 없음. 시작 파일 --check-only 통과. 카메라는 켜지 않았습니다.

적용
http://127.0.0.1:8821/ (새로고침)
실행 {D/'serve.py'}, PID {http['server']['pid']}. 기존 START_WEB_DEMO.cmd가 이 경로를 실행합니다.
기존63개 동결 파일 유지; 현재77개 파일/체크포인트 고정. 모델/C·T3-F 변경0, 학습0.
독립 전문가 검증은 아니며 Codex가 구현과 별도 코드·출력·해시 검산을 수행했습니다.

남은 확인
새 촬영에서 같은 문제까지 해결됐다고 아직 단정하지 않습니다. 사용자 직접 정상→손모양 변경→교정 재시도에서 확인해야 합니다. 사용자 자료는 계속 진단 전용입니다.
이번 공개 보완 단계는 완료. 실제 시연의 수용성 확인, HDD 실제 분리 실행, 발표용 정상/교정 시연 고정은 남아 있습니다.
5방향 영상은 아바타 제작 전용이고 피드백 보완에는 쓰지 않았습니다.

작업 기록
초기 검증 스크립트 모듈명 충돌/일부 공개 입력의 선택적 video 필드 누락은 검증 도구에서 수정했습니다. 실행한 보완/회귀 검증은 위와 같고, 실패한 중간 실행을 완료 건수로 계산하지 않았습니다.
통과한 판정·촬영 준비 변경만 웹에 적용했습니다. 기존 동결 산출물은 보존했습니다.
'''
 (D/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 graph=read(N/'GRAPH.json');graph['updated_at']=now;graph['source']=str(N/'GRAPH.json')
 for n in graph['nodes']:
  if n['id']=='F3':n['status']='public_repair_complete_after_user_diagnosis';n['evidence']=str(D/'RESULT.json')
  if n['id']=='F4':n['status']='public_capture_and_web_repair_verified'
  if n['id']=='F5':n['status']='new_user_retry_and_hdd_disconnect_pending';n['prior_diagnosis']=str(N/'seated_user_check/REVIEW.json')
 graph['loop_completed']='F5 diagnosis -> F3 public repair -> F2 regression -> F4 web; next F5 user retry'
 write(D/'GRAPH.json',graph)
 next_action='최신8821에서 사용자가 앉아서 사과·감사를 정상→손모양 변경→교정 재시도하면 원래 결과/자동종료/브라우저대기를 진단한다. 촬영 시작은 사용자만, 학습·튜닝·후보선택 금지. 문제가 있으면 공개 정면 자료로 재현 후 보완. 실제 HDD분리·F6시연 고정은 그다음. 이미 통과한 공개 보완을 이유 없이 재시험/재학습하지 않는다.'
 statepath=O/'STATE.json';state=read(statepath);state.update(current_stage=summary['status'],last_updated=now,next_single_action=next_action)
 task=state['capture_feedback_completion_20261116'];task.update(status=summary['status'],updated_at=now,current_checkpoint=str(D/'RESULT.json'),report=str(D/'RESULT_KO.txt'),current_graph=str(D/'GRAPH.json'),active_feedback_runtime=str(D/'serve.py'),next_single_action=next_action,web_admitted=True)
 task['public_frontal_evidence_repair_20261117']=summary
 task['seated_user_check']['followup_public_repair']=str(D/'RESULT.json')
 write(statepath,state)
 context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf-8-sig')
 new8=f'''\n\n### 2026-11-17 실제 촬영 진단 후 공개 정면3문제 보완 / 웹 적용
정상 손 겹침의 과잉 확인부족·일부 손가락 불확실로 다른 오류까지 무시·자동종료 준비 없는 촬영시작을 공개 정면 자료에서 재현·보완. 신경망 가중치/C·T3-F/100단어 본체 동결,학습0. 손모양 판정 연결 공통 함수와 촬영준비 코드만 수정. 사용자촬영 재생/학습/튜닝/후보선택 없음. 앞뒤 같은 모양+자체 관절 정상+중간 차이 없음+250ms 이내 겹침만 이전 관측 유지,숨은손가락을확정하지않음. 아는손가락차이와모르는손가락분리;정상에는전체필수근거. 대기자세6표본준비전시작버튼비활성/안내,준비회복후시작,카운트다운재확인.
공개32정상/TCN후보동일,정답기반손가락마스킹450·겹침합성21(짧은7복구,실제누락/긴가림14보류유지)·별도겹침42·미확정16·사과회귀70·카드퀘스트7·종료9·준비2통과;의미오류정답없는자르기5진단전용. 새사용자정확도아님. 공개캡처2자동종료/정상유지/재검출0,종료후0.34~0.41초(사용자화면대기아님). 같은4입력계산중앙0.331→0.344초,1회비교범위. 기존63핀보존/현재77핀. 최신8821 PID{http['server']['pid']} {D/'serve.py'},카메라꺼짐,HTTP공개판정/시작파일확인. 보고 {D/'RESULT_KO.txt'},그래프 GRAPH.json. F3/F4공개보완완료→F5새실제촬영/HDD분리→F6남음. 작성/별도검산Codex;독립전문가검증아님.
'''
 h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
 text=text.replace(h8,h8+new8,1).replace(h9,h9+'\n\n### 2026-11-17 공개 정면3문제 보완 적용 이후\n'+next_action+'\n',1);context.write_text(text,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}
목표: 정면 웹캠 사과·감사 촬영→자동종료→빠른 교정/확인부족 구분→재시도/퀘스트. 5방향은아바타전용.
최신완료: {D/'RESULT_KO.txt'} / RESULT.json. 사용자진단에서드러난3코드문제를공개정면자료로재현·검증후웹적용. 모델가중치/C·T3-F/100단어본체그대로,학습0. 학습총이력75유지. 개인촬영이번후보에사용안함(재추론/튜닝/선택없음).
변경: 짧은2D겹침은앞뒤같은상태·자체관절정상·중간모양변경없음·앞뒤250ms이내일때이전관측을취소하지않음;숨은순간정답확정/좌표보간없음. 일부손가락불확실과확인된다른손가락차이분리. 준비6표본없으면촬영시작차단/화면안내,동일미리보기에서준비회복,3초카운트다운끝재확인. 기존자동종료팔복귀조건유지.
검증: 공개32정상/TCN동일,손가락450마스킹조건(실제450영상아님),공개7영상겹침21합성재현(짧은7복구/실제누락·긴가림14보류),겹침42·미확정16·사과70회귀/5진단전용·카드퀘스트7·종료9·준비2통과. 공개캡처사과/감사정상/자동종료/재검출0,종료후0.34~0.41초;브라우저/실제촬영시간아님. 같은4입력계산중앙0.331→0.344초,반복통계실험아님. 현재77핀/이전63핀유지. 작성·별도코드/출력/해시검산Codex,독립전문가검증아님.
웹 {D/'serve.py'},PID{http['server']['pid']},8821,일반환경,카메라꺼짐. START_WEB_DEMO.cmd→frontal_demo_launcher.py갱신/--check-only통과. HTTP공개사과·감사최신판정/JS준비안내확인,공개재생퀘스트진행없음. 학습/평가실행세션없음,서버만유지.
현재그래프 {D/'GRAPH.json'}: F5진단→F3공개보완→F2회귀→F4연결완료;새F5실제촬영/HDD분리,F6남음. {next_action}
이전개인진단: 사과4·감사4,각첫시도정상의도/이후손모양변경,7자동/1수동,첫사과겹침보류·사과마지막손모양보류·마지막감사기준없음. 개인원본과당시결과는보존. 새보완으로개인시도가고쳐졌다고아직주장하지않음. 실제HDD분리미실시. 옆시점학습/후속계획재개금지.
''',encoding='utf-8')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in admission['pins'].items())
 assert read(statepath)['current_stage']==summary['status']
 assert read(D/'GRAPH.json')['nodes'][4]['status']=='new_user_retry_and_hdd_disconnect_pending'
 print('FINALIZED',summary['status'],'PID',http['server']['pid'])
if __name__=='__main__':main()
