from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent;OPS=U.parent;ROOT=U.parents[3]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
assert not (U/'FINAL_AUDIT.json').exists()
f=read(U/'FROZEN.json');s=read(U/'SUMMARY.json');d=read(U/'DIAGNOSTICS.json');v=read(U/'VERIFICATION.json')
for name in ('SELFTEST.json','VERIFICATION.json','BROWSER_QA.json','VISUAL_REVIEW.json'):assert read(U/name)['status']=='PASS'
for kind in ('protected','previous','code'):
    for p,h in f[kind].items():assert sha(p)==h,p
assert s['rows']==2391 and s['stress_rows']==7092 and s['lowering_rows']==1800
lower=read(U/'LOWERING.json');stress=read(U/'STRESS.json')
for axis,totals in s['lowering'].items():
    for key,value in totals.items():assert abs(sum(r[key] for r in lower if r['key']==axis)-value)<1e-7
for mode,totals in s['stress'].items():
    rr=[r for r in stress if r['condition']==mode and r['removed_frames']>0]
    assert len(rr)==totals['effective']
    for key in ('old_strong','new_strong','old_possible','new_possible'):assert sum(r[key+'_s']>1e-8 for r in rr)==totals[key]
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
brief='; '.join(f"{name} 기존일치{g['old_matching']}/300→새확정일치{g['confirmed_matching']}/불일치{g['confirmed_different']}/보류{g['abstained']}" for name,g in [(name,s['groups']['sensor_'+sc]) for name,sc in [('원래','original'),('2회연결','repeat2'),('3회연결','repeat3')]])
low=s['lowering']['location']
next_action='여러 참고의 반복경계 합의·횟수 보류 결과를 사용자와 검토한다. 새 후보는 운영 미적용이다. 다음 보완은1회에모두동의해도 다른반복가설의경계근거가있는지,짧은단계가과도하게시간압축되는지 공개대조설계부터다. 귀환자세없는반복/넓어진불확실범위도한계다. 불확실범위를필수동작/오류로표시하지않는다. C·T3-F/웹유지,사용자·멘토촬영학습·튜닝·선택금지,추가학습은결과검토뒤별도승인.'
title='2026-10-20 여러 참고 반복 경계 합의·횟수 보류 후보 공개 검증 완료 — 운영 미적용'
entry=f'''### {title}
갱신 {at}. `ops/{U.name}`. Codex 로컬 구현/공개 비교/별도 구현 검산 완료. C·T3-F/운영 DTW/웹 유지, 새학습 없음.
- 공개100단어 TRAIN899(01~09)참고/DEV300(11~13)평가. 직전2391저장 경로 재사용, 새DTW·MediaPipe·분류추론 없음. 2398관측입력=1199원본의legacy/sensor. DEV주석/full손상태는평가만; 같은DEV후속진단,새최종시험아님.
- 각참고의최소비용횟수(1/2/3)단독선호가5명 이상일치하고 반대선호없을때검토. 반복은동일5사람이모든순서경계의공통query시점을지지해야함. TRAINcore사이에서실관측팔이참고앞뒤준비자세모두에core보다가까워야함(지원≥.5). 손누락/단순DTWwrap/동점은귀환근거아님. 충돌/근거부족은count=null,1회로강제대체안함. 언어적필수반복판정아님.
- 보류시강한범위=viable정렬교집합,불확실범위=합집합. 확정시에도모든가능정렬불확실범위유지. 이전짧은범위축소금지. 단어/사용자예외·결과후계수조정없음.
- sensor각300: {brief}. 보류성공처리금지. 연결횟수맞춤대조이며자연반복정답/TCN정확도아님. legacy결과SUMMARY에별도.
- 남은확정불일치2조건=실명같은영상2/3회연결,참고9명모두1회비용선호. 공동편향가능. 반복합의869대조는팔귀환공동근거도통과; 경계검사추가보류0. 이번오확정감소는의견충돌보류효과,경계검사만의추가개선입증아님. 결과후단어예외/재튜닝안함.
- 위치정상내림추정 {low['rest_missing_s']:.3f}초 중강한범위겹침 {low['old_strong_s']:.3f}→{low['new_strong_s']:.3f}초; 불확실포함 {low['old_possible_s']:.3f}→{low['new_possible_s']:.3f}초. 이를언어오류율/사용자오류로해석금지. 손내림추정은평가전용.
- 7092손근거가림범위불변; 이전8누락항목손조건 강한{d['previous8_strong']}/불확실포함{d['previous8_possible']}. sensorpeak10실근거제거{s['stress']['peak10']['effective']}중불확실포함{s['stress']['peak10']['new_possible']}. 8독립영상아님. 591전체frame삭제대조도완료;관측충분만으로전체동작완료판정금지.
- 한손3단어18경로반대손무영향. 시간표시변경불변성만검사,실제속도센서/자연반복/새사람은미검증. 준비자세로돌아오지않는반복은보류가능.
- 자체12PASS,같은Codex별도구현검산 {v['rows']}대조/{v['reference_evidence']}참고판단/{v['projections_reconstructed']}투표범위/{v['hash_pins']}해시PASS. 다른agent/수어전문가검토아님. 로컬HTML desktop/mobile시각검토PASS.
- C·T3-F6가중치/32보호파일/과거동결자료보존. 사용자·멘토/보류14~18미접근. 학습/운영교체/웹접근/카메라/다운로드/외부전송/Git없음. 완료연구중복실행금지(CURRENT_EXECUTION).

'''
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text('utf-8-sig');assert title not in text
for marker,content in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry),('## 9. 다음 단일 작업',f'### {title}\n{next_action}\n\n')]:
    assert marker in text;text=text.replace(marker,marker+'\n\n'+content,1)
ctx.write_text(text,'utf-8')
state=read(OPS/'STATE.json');state.update(updated_at=at,updated_date_kst='2026-10-20',current_stage='WORD100_REPEAT_BOUNDARY_CONSENSUS_REVIEW',activity_state='AWAITING_USER_REVIEW',next_single_action=next_action,resume_command=next_action)
state[U.name]=dict(status='OFFLINE_VERIFIED_NOT_DEPLOYED',at=at,unit=str(U),report=str(U/'RESULT_KO.html'),summary=s,verification=v,
    diagnostics=str(U/'DIAGNOSTICS.json'),training=0,production_changes=0,private_data_read=0,next_single_action=next_action,execution='All research runs complete; no duplicate execution')
write(OPS/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 여러 참고가 반복경계를 실제로 함께지지하는지확인하고충돌시보류. 마지막완료: 별도후보공개검증/독립구현검산. 다음한작업: 결과와남은한계사용자검토.

- 단위 `{U}`. RESULT_KO.html/txt, SUMMARY/DIAGNOSTICS/VERIFICATION/FROZEN/FINAL_AUDIT 참고.
- TRAIN899/DEV300,공개100단어,사람분리01~09/11~13. 기존2391경로재사용,새추출/DTW/분류/학습없음.2398관측입력1199원본,DEV주석/full상태평가전용. 사용자/멘토/보류14~18미접근.
- 각참고의1/2/3선호가5명일치+반대선호없음. 반복경계모두를동일5사람이지지하고core보다TRAIN앞뒤준비자세와가까운실관측팔귀환근거필요. 손누락/동점/DTWwrap은근거아님. 충돌count=null,강제횟수없음. 언어적반복정답아님.
- 보류강한범위교집합/불확실합집합. 확정이어도모든viable불확실범위보존. 단어예외/DEV재튜닝없음.
- sensor각300: {brief}. 보류를성공으로세지않음. 합성연결대조,자연반복/TCN인식률아님. 불확실범위커져도오류로표시금지.
- 확정불일치2=실명같은영상2/3회연결에서9명모두1회비용선호. 반복합의869대조모두경계근거통과/경계검사추가보류0. 이번개선은충돌보류효과,경계단독효과미입증. 단어예외없음.
- 위치내림추정{low['rest_missing_s']:.3f}초중강한겹침{low['old_strong_s']:.3f}→{low['new_strong_s']:.3f}초; 불확실{low['old_possible_s']:.3f}→{low['new_possible_s']:.3f}초. 7092가림범위불변/이전8조건강한{d['previous8_strong']}/불확실{d['previous8_possible']}. 591전체프레임삭제는완료판정불가유지.
- 한손3단어18경로반대손무영향;시간표시변경만속도대조. 준비자세없는반복/필수단계의미정답/새사용자는미검증.
- 자체12PASS/별도구현검산{v['rows']}대조·{v['reference_evidence']}참고·{v['hash_pins']}해시PASS. 같은Codex별도구현,전문가/타agent아님.
- 기존C/T3-F6가중치/32보호파일/과거자료불변. 운영DTW/웹/모델교체없음. 웹서버접근/재시작/카메라없음. 모든연구실행종료(CURRENT_EXECUTION),재실행금지.

{next_action}
''','utf-8')
cur=read(U/'CURRENT_EXECUTION.json');assert cur.get('evaluation_exit_code')==0 and cur.get('verification_exit_code')==0
cur.update(status='COMPLETE',at=at);write(U/'CURRENT_EXECUTION.json',cur)
write(U/'STATUS.json',dict(status='COMPLETE_OFFLINE_REVIEW',at=at,production_applied=False,next_single_action=next_action))
for path,h in read(U/'EVIDENCE_PINS.json').items():assert sha(U/path)==h
write(U/'FINAL_AUDIT.json',dict(status='PASS',at=at,protected_files_unchanged=len(f['protected']),model_weights_unchanged=6,
    production_changes=0,training=0,private_data_read=0,
    artifacts={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'},
    document_sha256={str(p):sha(p) for p in [ctx,ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']}))
print('FINALIZED',at)
