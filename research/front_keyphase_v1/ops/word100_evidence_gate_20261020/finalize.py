from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;OPS=U.parent;ROOT=U.parents[3]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
assert ROOT.name=='WordsInSilence'
assert not (U/'FINAL_AUDIT.json').exists()
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
for n in ('VERIFICATION.json','ENVELOPE_VERIFICATION.json','BROWSER_QA.json'):assert read(U/n)['status']=='PASS'
f=read(U/'FROZEN.json');s=read(U/'SUMMARY.json');decision=read(U/'DECISION.json')
for k in ('protected','previous','code'):
    for p,h in f[k].items():assert sha(p)==h,p
write(U/'VISUAL_REVIEW.json',dict(status='PASS',screenshots=['QA_DESKTOP.png','QA_RESULTS.png','QA_MOBILE.png'],reviewer='Codex visual inspection',checks=['legible Korean','tables and result labels visible','mobile wraps without page overflow','numeric non-adoption explicit']))
next_action='사용자와 공개 관측부족 기준 결과를 검토한다. 전역 관측률/공백분위 기준 G1은 미채택. 다음은 여러 공개 TRAIN 참고의 동작 구간 근거로 관측종료에 따라 비교범위가 줄어드는 문제를 검증할 계획 확인. C/T3-F/웹/DTW 유지, 사용자자료·추가학습·인식보조 없음.'
title='2026-10-20 비교 근거 부족 기준 공개 검증 완료 — 전역 비율 기준 미채택'
summary=f'''### {title}
갱신 {at}. ops/{U.name}. 사용자 승인 관측부족 기준 검증을 로컬 오프라인으로 완료. 기존 C/T3-F·DTW·웹 보호32파일 보존; 운영 적용/추가학습/사용자자료 접근 없음.
- 공개100단어 TRAIN899×2경로에서만 항목별q05관측비율/q95최장공백비율 산출 후 동결. DEV300×2 점검. 굽힘76.79%/13.64%,위치궤적82.50%/10.78%는 시험한통계경계로 합격선아님/미채택. DEV는 기존개발자료.
- G0: 기존 최소8관측프레임/5지원참고 등의 근거없음은 INSUFFICIENT; 내부공백은 PARTIAL; 알려진비사용손은 NOT_REQUIRED; 범위내공백없음도전체수행완료아님. G1은G0+분위경계. 역할기록12단어중3한손,그외88은손별진단만. 사용자의부재로손제외금지.
- 전체손가림1773항목손조건(실근거제거1508)모두부족. 이전부분15각도연구peak10차이감소630항목손에서G1부족147/일부481/범위축소공백없음2. 운영19각도재채점·630독립영상·새인식률주장금지. 끝부분손실시동적범위축소로누락가림:순종오른손7프레임손실/1.67~4.07→1.67~3.83/관측100%;원범위90.97%. 사용자사례아님.
- 양끝대기추가3600/속도시각변형7200상태동일. 한손3단어9DEV×2경로×3항목54반대손가림비교대상아님. 합성전체시퀀스2회연결3600중1244새부족(sensor중간휴식공백);언어적반복허용검증아님. 원본안정내림추정이내부공백인8영상손24항목중16부족,다른누락공존하므로16오탐주장금지. G1미채택.
- 자체11/별도scalar28476상태검산/원본시각3600적분오차2.85e-14/6135해시검산/보고서desktop·mobileQA PASS. 동일Codex별도구현검산이며다른agent아님. SUMMARY의dynamic_vs_fixed_disagree는reason딕셔너리차이도포함; POSTCHECK에서상태차이만whole0/center10 0/peak10 8/peak20 12로명확화,원결과보존.
- RESULT_KO.html/txt,DECISION,CALIBRATION,BASELINE,STRESS,CONTROLS,LOWERING,VERIFICATION,ENVELOPE_VERIFICATION,POSTCHECK. 추가학습·분류추론·사용자멘토/보류14~18접근0. 웹서버재시작/접근없음.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8-sig')
for marker,entry in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',f'### {title}\n{next_action}\n\n')]:
    assert marker in text;text=text.replace(marker,marker+'\n\n'+entry,1)
context.write_text(text,'utf-8')
state=read(OPS/'STATE.json');state.update(updated_at=at,current_stage='WORD100_EVIDENCE_GATE_REVIEW',activity_state='AWAITING_USER_REVIEW',next_single_action=next_action,resume_command=next_action)
state[U.name]=dict(status=decision['status'],at=at,unit=str(U),report=str(U/'RESULT_KO.html'),summary=s,decision_path=str(U/'DECISION.json'),verification=read(U/'VERIFICATION.json'),production_changed=False,web_restart=False,private_data_read=0,new_training=0,next_single_action=next_action)
write(OPS/'STATE.json',state)
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 관측이 적은데 차이가 작게 보이는 경우의 비교 근거 상태 구분. 마지막 완료: 공개 기준 후보 검증,전역비율 G1미채택. 다음 한 작업: 사용자 결과 검토 및 공개다중참고를 이용한 비교범위 보존 계획 확인.

- 단위 `{U}`. 보고 RESULT_KO.html/txt. DECISION/CALIBRATION/BASELINE/STRESS/CONTROLS/LOWERING 및두검산/FINAL_AUDIT.
- TRAIN899×legacy/sensor로만분위기준계산후동결,DEV300×2점검. q05관측굽힘76.79/위치궤적82.50%,q95긴공백비율13.64/10.78%는미채택후보;합격선아님. 100단어기존DEV이며새최종시험아님.
- 필수또는평가손관측/최소8프레임/5지원참고부족=INSUFFICIENT. 내부공백=PARTIAL. 알려진비사용손=NOT_REQUIRED. 범위내공백없음은전체동작완료판정아님. 알려진역할12단어/3한손;88은손별만. 사용자안보임으로손제외금지.
- 전체손가림1773항목손모두부족(실근거제거1508). 이전연구peak10차이감소630항목손 중G1부족147/일부481/범위축소공백없음2. 이전부분15각도연구증거를재사용;운영19각도재채점/630개독립영상/새인식률아님.
- 양끝대기3600/시간축변형7200상태동일;54비사용손가림무영향. 합성전체시퀀스2회연결3600중sensor1244새부족:중간휴식에민감해G1미채택. 실제언어적반복허용/실제빠른촬영검출성능주장금지.
- 원본안정내림추정근거내부8영상손24항목중16부족:다른누락공존으로16오탐주장금지. full원본근거는오프라인진단만. 끝누락으로관측범위짧아져100%보이는공통문제확인(공개순종/자석등),단어예외없음.
- 자체11/독립scalar28476/원본시간적분3600/6135해시검사PASS,보고서desktop/mobile시각검토PASS. 같은Codex별도구현검산이며다른agent아님. POSTCHECK에SUMMARY reason차이대status차이구별기록,원결과보존.
- C/T3-F6가중치포함32보호파일/과거결과불변. 사용자/멘토/보류14~18자료미접근. 학습/분류추론/운영변경/카메라/서버재시작없음. 기존웹마지막서버기록62068은이번재확인하지않음. 연구실행은모두종료,중복재실행금지.

{next_action}
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
write(U/'STATUS.json',dict(status='COMPLETE_OFFLINE_CRITERION_REVIEW',at=at,next_single_action=next_action))
artifacts={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'}
write(U/'FINAL_AUDIT.json',dict(status='PASS',at=at,scope_complete=True,quantitative_gate_adopted=False,protected_files_unchanged=len(f['protected']),model_weights_unchanged=6,production_changes=0,private_data_read=0,training=0,inference=0,artifact_sha256=artifacts))
assert read(OPS/'STATE.json')[U.name]['status']==decision['status']
assert (U/'RESULT_KO.html').exists() and title in context.read_text('utf-8')
print('FINALIZED',at)
