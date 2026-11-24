"""Document the requested private diagnostic review; no production/model changes."""
from pathlib import Path
import json, hashlib, statistics
from datetime import datetime, timezone, timedelta

D=Path(__file__).resolve().parent
N=D.parent; G=N.parent; O=G.parent; ROOT=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,o:Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
diag=read(D/'diagnostic_only/FROZEN_DIAGNOSIS.json')
discovery=read(D/'DISCOVERY.json')['rows']
manifest=read(D/'diagnostic_only/FRAME_REVIEW_INPUTS.json')['rows']
assert len(discovery)==len(manifest)==len(diag['auto_end'])==8
observations=[
 ('검지를 편 입 주변 동작 → 편 손으로 반대 주먹 앞을 덮음','정상형으로 보이는 시도. 손 겹침 4프레임 때문에 wipe_shape만 확인 부족. 동작 전체의 전문가 정답 확정은 아님.'),
 ('입 주변에서 오른손을 주먹으로 유지','입 주변 오른손 모양 차이를 검출. 덮는 손 모양에는 별도 확인 부족이 남음.'),
 ('입 주변에서 검지 외 여러 손가락을 함께 폄','입 주변 손모양 차이를 검출. 마지막 도달 상태는 확인 부족.'),
 ('입 주변에서 검지 외 여러 손가락을 함께 폄','중지 등이 펴졌다는 관측이 있었으나 일부 손가락 불확실 때문에 손 전체를 확인 부족으로 처리. 오류 교정 근거를 놓침.'),
 ('양손을 펴고 반복 동작','모든 요구 항목 supported, 정상 통과.'),
 ('양손을 주먹으로 바꾸고 동작','양손 모양 차이 검출. 다른 움직임·반복 항목은 확인 부족이므로 모두 틀렸다고 해석하지 않음.'),
 ('왼손 주먹, 오른손은 편 상태로 동작','왼손 모양 차이 검출. 오른손/반복 항목에는 확인 부족이 남음.'),
 ('오른손 주먹, 왼손은 편 상태로 동작','오른손 모양 차이 검출. 자동 종료 실패 후 수동 종료. 원본에는 손을 내리고 기다린 구간이 있음.'),
]
rows=[];original_hashes={}
for i,(r,m,a,visual) in enumerate(zip(discovery,manifest,diag['auto_end'],observations),1):
 p=Path(r['directory']);saved=read(p/'inference_result.json');q=saved['result']
 compact=r['inference_result.json'];ack=r['browser_ack.json']
 assert sha(m['video'])==m['video_sha256']
 for name in ['inference_result.json','browser_ack.json','recorded.mp4','rtmw/observations.npz','rtmw/SESSION.json']:
  original_hashes[str(p/name)]=sha(p/name)
 assert compact['frontal_module']=='front_observation_20261117'
 assert a['stream_received']==a['stream_completed']==a['observation_rows']==compact['frames']
 assert a['stream_dropped']==0 and not a['stream_errors'] and not a['missing_ranges']
 assert not compact['training_use'] and not compact['tuning_use']
 criteria=q['lesson_acceptance']['criteria']
 rows.append(dict(attempt=i,word=compact['label'],recorded_at_kst=datetime.fromisoformat(compact['recorded_at']).astimezone(timezone(timedelta(hours=9))).isoformat(),
  observed_action=visual[0],review=visual[1],intent_confirmed_by_user=False,
  source=p.as_posix(),original_attempt_id=compact['attempt_id'],stop_reason=a['stop'],auto_end_calibrated=a['calibrated'],
  overall_pass=compact['overall_pass'],tcn_top3=compact['top3'],frames=a['observation_rows'],
  stop_to_result_seconds=compact['stop_to_result'],stop_to_browser_ack_seconds=ack['stop_to_browser_ack'],
  criteria=criteria,diagnostic_contact_sheet=m['image']))
times=[r['stop_to_browser_ack_seconds'] for r in rows]
stats=dict(recordings=8,apple_recordings=4,thanks_recordings=4,automatic_stop=7,manual_stop=1,
 frames_processed=sum(r['frames'] for r in rows),stream_dropped=0,stream_errors=0,
 browser_ack_seconds=dict(min=min(times),max=max(times),median=statistics.median(times)),
 latency_scope='Stop trigger to browser render acknowledgement; excludes the time waiting for the auto-stop trigger and is not a direct perceptual stopwatch measurement.',
 overall_pass_attempts=[r['attempt'] for r in rows if r['overall_pass']])
assert stats['frames_processed']==1453 and stats['overall_pass_attempts']==[5]
f4=next(x for x in diag['checks'] if x['attempt']==4)
assert f4['availability_all']==f4['interval_frames']==37
assert sum(x['frames'] for x in f4['shape_states'] if x['values'][1]==1)==37
f1=next(x for x in diag['checks'] if x['attempt']==1)
w=next(c for c in f1['criteria'] if c['id']=='wipe_shape')
assert w['unobserved_spans']==[[120,123]]
issues=[
 dict(id='overlap_veto',attempt=1,confirmed='Both hands had detected keypoints; projected overlap at frames120..123 changes supported active-palm check to unknown.',
  source=str(G/'public_visibility_roles_20261116/apple_candidate.py'),lines=[34,43,47],
  next='Public front-video reproduction: distinguish the visible covering hand from the covered supporting hand, keep genuine occlusion/shape-change guards. Do not relax a threshold to fit this private clip.'),
 dict(id='partial_finger_evidence_discarded',attempt=4,confirmed='All required keypoints available in37/37frames. Middle-finger prediction extended37/37; an unknown little/ring prediction makes the all-fingers match unknown in36/37frames. Original and frozen rerun criteria identical.',
  source=str(G/'apple_v1/compose.py'),lines=[31,32],
  next='Public labelled finger combinations: preserve confident contradictions separately from unknown fingers, validate tracking errors and transitions so hidden fingers are not guessed.'),
 dict(id='unready_auto_end_baseline',attempt=8,confirmed='auto_end_calibrated=false; ReturnToRest.update cannot end when rest is None. Video shows arms lowered then mouse reach. Preview/calibration frames not saved, so the precise reason calibration failed is unknown.',
  source=str(O/'word100_action_feedback_demo_20261108/auto_end.py'),lines=[21,28],
  next='Public seated-input fixtures: ready-baseline handling before capture/recovery and clear readiness UI, while guarding against ending during short hand occlusion.')]
pins=read(N/'web/ADMISSION.json')['pins'];assert all(sha(p)==h for p,h in pins.items())
report=dict(updated_at=now,status='REVIEW_COMPLETE_THREE_ISSUES_CONFIRMED_NOT_FIXED',scope='User-requested private diagnosis only',
 private_data=True,training_use=False,tuning_use=False,candidate_selection_use=False,production_changed=False,training_runs=0,
 implementer='Codex',verification='Codex source-frame review, original inference/ack cross-check, same frozen observer replay for attempts1/4, deployed63-file hash check. No independent expert or second-agent verification.',
 user_intent='User reported normal and deliberately wrong attempts for both words; exact order/mistakes not yet confirmed. Observed actions above are visual review, not asserted user intent or expert ground truth.',
 stats=stats,rows=rows,issues=issues,all_original_video_hashes_match_frame_review=True,deployed_pins_unchanged=len(pins),
 limitations=['Not a generalization accuracy estimate.','No new capture, camera start, weight/threshold changes, or model candidate selection.','Other unknown movement/terminal criteria are not proof the user omitted the motion.','Physical HDD-disconnected demo and full correction/retry acceptance still pending.'])
write(D/'REVIEW.json',report);write(D/'diagnostic_only/ORIGINAL_FILES_SHA256.json',original_hashes)
body='''사과·감사 실제 촬영 진단 — 2026-11-17

결론
종료 후 결과 표시는 빠르고 감사의 정상/손모양 변경은 일부 구분했습니다.
다만 사과의 정상형 동작이 막히고, 손모양 차이가 확인 부족으로 빠지는 사례가 있어 F5 실제 시연 검증은 아직 통과가 아닙니다.

범위
이번 촬영 8개(사과4, 감사4). 원래 결과/브라우저 표시 확인 기록, 실제 장면을 대조했습니다.
개별 시도의 고의 오류 의도와 순서는 아직 사용자 확인 전입니다. 아래는 원본에서 확인한 모습입니다.
개인 촬영은 진단 전용. 학습·튜닝·후보선택 없음. 운영 코드·가중치·C·T3-F 변경 없음.

촬영 및 속도
8번 중7번은 손을 내린 뒤 자동 종료, 마지막 감사1번은 수동 종료.
전체1453프레임 처리, 입력 누락/처리 오류0. 관절 미검출이 없다는 뜻은 아닙니다.
촬영 종료 신호부터 브라우저 표시 확인까지0.47~0.75초(중앙값0.60초).
손을 내린 뒤 자동 종료를 기다린 시간은 이 수치에서 제외됩니다.

각 촬영
'''
for r in rows:
 body+=f"{r['attempt']}. {r['recorded_at_kst'][11:19]} {r['word']}: {r['observed_action']}\n   {r['review']}\n"
body+='''
확인된 문제와 다음 보완
1. 정상적인 손 겹침: 첫 사과는 손 자체 미검출이 아니라 4프레임의 화면상 겹침 때문에 앞손 모양 전체를 확인 부족으로 바꿨습니다. 앞에서 보이는 손과 가려지는 손의 역할을 공개 정면 영상에서 구분해 검증해야 합니다.
2. 일부 손가락 불확실이 전체 판단을 막음: 네 번째 사과는 중지 등이 펴졌다는 예측이 있는데도 새끼손가락 등의 불확실 때문에 오류 안내를 못 했습니다. 확인된 차이와 모르는 손가락을 따로 처리하는 공개 검증이 필요합니다.
3. 자동 종료 준비 실패: 마지막 감사는 촬영 전 대기 자세 기준이 확보되지 않았습니다. 그 상태로 촬영이 시작되면 현재 종료 코드는 끝낼 수 없습니다. 원본에는 손을 내리고 기다린 장면이 있습니다. 준비단계 원본이 저장되지 않아 기준 확보 실패의 세부 이유는 확정하지 않습니다.

주의할 해석
감사의 잘못된 손모양 시도에서는 해당 손 오류를 잡았습니다. 다른 방향·반복 항목의 확인 부족은 그 동작을 모두 틀렸다는 뜻이 아닙니다.
사과2·3은 손모양 차이를 잡았지만 일부 다른 항목에 확인 부족이 남았습니다.
이번 자료로 정확도 백분율이나 다른 사람에서도 동일한 결과를 보장하지 않습니다.

검증 방법
원래 inference_result.json/browser_ack.json과 RTMW 처리 로그 대조, 영상 장면 확인.
사과1·4만 원래 동결 관측부로 원인 재현; 모든 기준 판정이 원래 저장 결과와 일치.
현재 웹63개 파일 해시 유지. 사용자 영상 해시 유지. 작성·검증 Codex; 독립 전문가 판정 아님.
진단 스크립트 숫자 직렬화 실패는 저장 방식만 고쳐 완료. 운영 판정 수정 아님.

다음 순서
공개 정면 자료로 위3원인 재현 → 공통 코드/관측 보완 → 기존 정상·오류·가림·자동종료·속도 회귀 검증 → 통과분만 웹 연결 → 실제 재시도 및 HDD 분리 시연 → 시연본 고정.
이번 요청에서는 진단까지 마쳤고 보완 코드/모델은 아직 적용하지 않았습니다.
'''
(D/'RESULT_KO.txt').write_text(body,encoding='utf-8')
graph=read(N/'GRAPH.json');graph['updated_at']=now;graph['source_graph']=str(N/'GRAPH.json')
for node in graph['nodes']:
 if node['id']=='F5':node['status']='private_review_complete_three_issues_not_passed';node['review']=str(D/'REVIEW.json');node['hdd_physical_check']='pending'
graph['next_loop']='F5 diagnosis -> F3 public-only reproduction/repair -> F2/F4 regression -> F5 real retry -> F6'
write(D/'GRAPH_REVIEW.json',graph)
stage='FRONTAL_SEATED_REVIEW_COMPLETE_PUBLIC_REPAIR_REQUIRED'
next_action='F5 진단에서 확인된 ①정상 손 겹침의 과잉 확인부족 ②일부 손가락 불확실이 확인된 차이까지 차단 ③대기자세 기준 없이 촬영시작하면 자동종료 불가를 정면 공개자료로 재현·공통 보완한다. 정상/오류/짧은가림/자동종료/속도 검증 뒤 통과분만 연결. 사용자 촬영은 진단 전용, 학습·튜닝·후보선택 금지. 실제 재시도·HDD분리·F6시연고정은 그다음.'
state_path=O/'STATE.json';state=read(state_path)
state.update(current_stage=stage,next_single_action=next_action,last_updated=now)
work=state['capture_feedback_completion_20261116'];work.update(status=stage,updated_at=now,next_single_action=next_action,current_graph=str(D/'GRAPH_REVIEW.json'))
work['seated_user_check'].update(status=report['status'],review=str(D/'REVIEW.json'),report=str(D/'RESULT_KO.txt'),graph=str(D/'GRAPH_REVIEW.json'),user_reply='촬영 완료; normal and deliberate errors, exact per-attempt intent pending',summary=stats,training_use=False,tuning_use=False,candidate_selection_use=False,production_changed=False,next_action=next_action)
write(state_path,state)
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf-8-sig')
new8=f'''### 2026-11-17 실제 앉은 촬영8건 진단 완료 / 공개 재현·보완 필요
사과4·감사4 원래 결과/브라우저 표시기록과 원본 장면을 대조. 자동종료7/8,마지막 감사수동1(촬영전대기자세기준없음). 종료신호→브라우저표시확인0.47~0.75초,중앙0.60초;종료대기시간제외.1453프레임누락/처리오류0. 감사정상1통과·손모양변경3의 해당손차이검출. 사과4개미통과: 정상형1개는4프레임화면겹침으로확인부족;여러손가락폄1개는중지등폄예측이있지만일부손가락불확실로오류를안내못함. 나머지사과2개손모양차이검출,일부끝/손모양확인부족남음. 개별시도의의도는미확인,원본장면검토이며전문가정답아님.
F5실제검증미통과→공개정면자료로3원인재현/보완필요. 이번은진단만;학습/튜닝/후보선택/운영수정0. 동결관측부로사과1·4원인재현(기존판정동일),웹63핀및원본영상유지. 작성/검증Codex,독립전문가/별도검증자없음. C·T3-F유지,5방향은아바타전용. 이전공개F1~F4결과는보존하며새사용자통과로올리지않음. 근거 {D/'RESULT_KO.txt'} 및 REVIEW.json, GRAPH_REVIEW.json.

'''
heading8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n';assert text.count(heading8)==1
text=text.replace(heading8,heading8+new8,1)
heading9='## 9. 다음 단일 작업\n\n';assert text.count(heading9)==1
text=text.replace(heading9,heading9+'### 2026-11-17 실제 촬영 진단 이후\n'+next_action+'\n\n',1)
context.write_text(text,encoding='utf-8')
handoff=f'''# 최신 인수인계 — {now}
목표: 정면 웹캠 사과·감사 촬영→자동종료→빠른 교정/확인부족 구분→재시도/실전퀘스트. 5방향은3D아바타전용.
최신작업: 사용자 실제촬영8개(사과4·감사4) 진단완료. 자동종료7,수동1. 종료신호→브라우저표시확인0.47~0.75초/중앙0.60초(자동종료대기제외).1453프레임처리/큐누락·오류0. 감사정상1통과,손모양변경3에서해당손차이검출. 사과정상형은4프레임손겹침으로확인부족;사과여러손가락폄1개는중지등예측이있는데일부손가락불확실로오류안내누락. 나머지사과2손모양차이검출,일부다른항목불확실. 마지막감사auto_end_calibrated=false로자동종료불가;원본손내림확인,준비단계원본은없어세부실패이유미확정. 개별고의오류의도/순서는사용자답변대기,시각검토를전문가정답으로주장하지않음.
근거: {D/'RESULT_KO.txt'} / REVIEW.json / diagnostic_only/FROZEN_DIAGNOSIS.json. 원래결과/표시기록/영상검토,사과1·4동일동결관측부재현판정일치. 현재웹63핀/원본영상해시유지. 이번학습/튜닝/후보선택/운영변경0. 작성·검증Codex,독립전문가검증아님. 진단스크립트숫자JSON저장오류수정완료. 사용자촬영학습·튜닝·선택절대금지.
상태: F1~F4는이전공개검증완료기록유지,F5실제검증은문제3개로미통과. 최신그래프 {D/'GRAPH_REVIEW.json'}.
다음단일작업: {next_action}
웹 {N/'serve.py'},포트8821,마지막확인PID21244. 이번서버/카메라시작·재시작없음. START_WEB_DEMO.cmd→frontal_demo_launcher.py. 진행중학습/평가없음. 다음작업시서버실행여부다시확인.
이전공개결과: 정면9명114자세정답/3seed검사구간판정부;32정상/TCN유지,237손모양추가오판·손실0,40오류/48시간부분/7문장퀘스트통과. 현재C·T3-F/본체동결. 과거학습72+3=75,이번0. 이전190핀보존. 76선언자산C:검사완료,물리HDD분리는미실시. 옆시점모델실행금지,이전옆시점후속계획재개금지.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
assert all(sha(p)==h for p,h in pins.items())
assert all(sha(p)==h for p,h in original_hashes.items())
assert read(state_path)['current_stage']==stage
assert (D/'RESULT_KO.txt').is_file() and read(D/'REVIEW.json')['stats']['automatic_stop']==7
print(json.dumps(dict(report=str(D/'RESULT_KO.txt'),stats=stats,pins_unchanged=len(pins),original_files_unchanged=len(original_hashes),production_changed=False),ensure_ascii=False,indent=2))
