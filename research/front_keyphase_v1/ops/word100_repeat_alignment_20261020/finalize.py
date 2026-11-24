from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent;OPS=U.parent;ROOT=U.parents[3]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
assert ROOT.name=='WordsInSilence' and not (U/'FINAL_AUDIT.json').exists()
at=datetime.now(timezone(timedelta(hours=9))).isoformat();f=read(U/'FROZEN.json');s=read(U/'SUMMARY.json');d=read(U/'DIAGNOSTICS.json');v=read(U/'VERIFICATION.json');table=read(U/'REPORT_TABLE.json');dec=read(U/'DECISION.json')
for name in ('SELFTEST.json','PREPARED.json','VERIFICATION.json','BROWSER_QA.json','VISUAL_REVIEW.json'):assert read(U/name)['status']=='PASS'
for k in ('protected','previous','code'):
    for p,h in f[k].items():assert sha(p)==h,p
assert s['query_rows']==1800 and s['deletion_rows']==591 and s['stress_rows']==7092
next_action='한 번/반복 구간 정렬의 공개 결과와 정상 손내림·짧은 단계 누락 한계를 사용자와 검토한다. 다음은 비용만으로 횟수를 확정하지 않고 여러 참고의 반복 경계 근거를 함께 확인하며, 엇갈리면 정렬 불확실성으로 남기는 계획 검토다. 이어 짧은 참고 단계의 과도한 시간압축을 정상 속도·정지·반복과 대조한다. C·T3-F/웹 유지, 사용자 촬영 학습·튜닝·선택 금지, 추가 학습은 결과 확인 후.'
title='2026-10-20 한 번·반복 정렬 공개 검증 완료 — 전체 프레임·반복 후보 비교, 운영 미적용'
brief='; '.join(f"{r['label']} 포착 {r['single_recall']:.2f}→{r['multi_recall']:.2f}%, 범위정밀도 {r['single_precision']:.2f}→{r['multi_precision']:.2f}%" for r in table)
low=s['lowering']['location'];de=d['deletion_counts']
entry=f'''### {title}
갱신 {at}. `ops/{U.name}`. Codex 로컬 후보 구현·공개 비교·별도 구현 검산 완료. 기존 C·T3-F/운영 DTW/웹 유지, 새 학습 없음.
- 공개100단어 TRAIN899(01~09) 참고, DEV300(11~13) 점검. legacy/sensor2398입력은1199원본의2경로. 모든 저장프레임 사용(56~250,중앙122),96시점샘플 제거. hand21제거 팔입력2398동일. DEV주석/full상태는 평가만, 새최종시험아님.
- body정규화4팔꿈치/pose손목으로 참고1·2·3회 순회 비교. 각주기8실관측core점/5사람/지원.35/누락1.5/band.2 유지. 모든참고의query균형비용중앙값으로 횟수선택; 같은사람중복투표금지. 5명강한근거/1명이라도지지한불확실범위 구분. 단어예외·오답판정·좌표합성없음.4회이상미검증.
- sensor각300영상의동작주석시간 중앙값: {brief}. 단어인식률아님. 왼쪽은동일fullframe한번정렬,오른쪽은반복후보. 원래영상과합성전체2/3회연결600경로씩총1800; 실제허용반복의전문가검증아님.
- 7092손가림조건범위불변. 이전8누락조건 중강한근거 {d['previous8_revealed_strong']},불확실포함 {d['previous8_revealed_possible']}포착; 독립8영상아님. 짧은peak10실근거제거 {d['stress']['peak10']['effective']}중강한 {d['stress']['peak10']['strong_revealed']}/불확실포함 {d['stress']['peak10']['possible_revealed']}.
- 위치내림미관측 {low['rest_missing_s']:.2f}초 중강한범위겹침 {low['strong_overlap_s']:.3f}초/불확실포함 {low['possible_overlap_s']:.3f}초. 이전0초에서회귀:3영상손중실명1회→3회선택이0.633초,나머지2건0.100초는fullframe한번정렬에도존재(POSTCHECK). sensor원래298/300이1회선택,합성2회223/300·3회219/300횟수일치;legacy각300/300. 이를수어오류판정에쓰지않음. 한손3단어18경로반대손무영향;88역할미확인.
- 짧은전체frame삭제591대조중주석시간삭제 {de['windows_with_annotated_time_removed']}. 삭제후에도관측공백없음 {de['after_observed_intervals_only']}/1773항목손,주석시간삭제인경우도 {de['after_clear_with_annotated_time_removed']}. 관측공백없음을전체동작완료로판정금지. 삭제구간의필수의미정답없음; semantic생략검출실패율아님.
- 자체8/별도검산 {v['rows']}대조·{v['independent_paths']}경로·투표,DP최적비용 {v['independent_dp_optimality_samples']}표본, {v['hash_pins']}해시 PASS. 같은Codex별도구현검산; 다른agent/전문가아님. 보고서desktop/mobile시각검토완료.
- C/T3-F6가중치포함32보호파일/과거동결결과불변. 사용자멘토/보류14~18미접근,학습/분류추론/운영/카메라/다운로드/Git없음. 연구실행종료, CURRENT_EXECUTION참조,완료평가재실행금지.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8-sig');assert title not in text
for marker,new in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry),('## 9. 다음 단일 작업',f'### {title}\n{next_action}\n\n')]:
    assert marker in text;text=text.replace(marker,marker+'\n\n'+new,1)
context.write_text(text,'utf-8')
state=read(OPS/'STATE.json');state.update(updated_date_kst='2026-10-20',updated_at=at,current_stage='WORD100_REPEAT_ALIGNMENT_REVIEW',activity_state='AWAITING_USER_REVIEW',next_single_action=next_action,resume_command=next_action)
state[U.name]=dict(status=dec['status'],at=at,unit=str(U),report=str(U/'RESULT_KO.html'),summary=s,diagnostics=str(U/'DIAGNOSTICS.json'),verification=v,
    production_changes=0,training=0,private_data_read=0,next_single_action=next_action,execution='All research evaluation/verification complete, no duplicate run')
write(OPS/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 한 번·반복을 함께 비교하고 손 미검출로 비교범위가 줄지 않도록 한다. 마지막 완료: 공개 반복정렬 후보 평가/별도검산. 다음 한 작업: 아래 결과의 한계와 후속 계획을 사용자와 검토.

- 단위 `{U}`. RESULT_KO.html/txt, REPORT_TABLE/DIAGNOSTICS/VERIFICATION/DECISION/FINAL_AUDIT 참조.
- TRAIN899/DEV300,100단어,legacy/sensor. 전체2398입력1199원본,모든저장프레임사용. user/mentor/holdout14~18미접근. DEV주석과full원본손상태는평가전용.
- 1·2·3회참고순회중비용중앙값최소 선택,5명/주기8core실관측점,같은사람중복투표없음. hand21정렬안씀/관측앞뒤삭제안함. 5명강한근거와1명이상불확실범위분리; 정상내림근거우선. 언어적반복횟수/동작오류판정없음.
- 동일fullframe한번정렬→반복후보(sensor각300): {brief}. 시간주석중첩이며TCN정확도아님. 실제새사용자/언어적반복허용검증아님.
- 7092가림범위고정. 이전8누락조건강한 {d['previous8_revealed_strong']}/불확실포함 {d['previous8_revealed_possible']}. 내림누락위치 {low['rest_missing_s']:.2f}초 중강한겹침 {low['strong_overlap_s']:.3f}/불확실 {low['possible_overlap_s']:.3f}초. 이전0에서회귀:실명1회→3회선택0.633초,2건fullframe단일에서도0.100초(POSTCHECK). sensor원래1회298/300,합성2회223/300·3회219/300선택일치,legacy각300. 불확실성을동작오류로바꾸지않음. 한손3단어18경로반대손무영향;88역할미확인.
- frame삭제591대조(주석시간포함 {de['windows_with_annotated_time_removed']})에서공백없음 {de['after_observed_intervals_only']}/1773항목손;주석시간삭제에서도 {de['after_clear_with_annotated_time_removed']}. 삭제구간의필수의미는미확인. 남은프레임에늘여맞추므로관측공백없음≠전체단계완료.
- 자체8,독립구현 {v['rows']}대조/{v['independent_paths']}경로·투표,별도DP {v['independent_dp_optimality_samples']}표본/{v['hash_pins']}해시PASS. 같은Codex검산,다른agent/전문가아님. 시각보고QA PASS.
- C·T3-F6가중치/32보호파일/과거결과보존,운영웹·DTW·모델교체/학습/분류추론없음. 웹서버재시작·접근없음;이전PID62068재확인안함. 완료연구재실행금지,CURRENT_EXECUTION참조.

{next_action}
''','utf-8')
write(U/'STATUS.json',dict(status='COMPLETE_OFFLINE_REVIEW',at=at,production_applied=False,next_single_action=next_action))
cur=read(U/'CURRENT_EXECUTION.json');cur.update(status='COMPLETE',exit_code=0,verification_session=59955,verification_exit_code=0,verification_status='PASS',at=at);write(U/'CURRENT_EXECUTION.json',cur)
projection_paths=[r['evidence_path'] for r in read(U/'ROWS.json')+read(U/'DELETION.json')]
assert len(projection_paths)==len(set(projection_paths))==2391
write(U/'PROJECTION_PINS.json',{p:sha(U/p) for p in projection_paths})
artifacts={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'}
write(U/'FINAL_AUDIT.json',dict(status='PASS',at=at,scope='Frozen public comparison complete; no deployment or semantic correctness claim',
    protected_files_unchanged=len(f['protected']),model_weights_unchanged=6,production_changes=0,training=0,private_data_read=0,
    artifact_sha256=artifacts,document_sha256={str(p):sha(p) for p in [context,ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']}))
assert read(OPS/'STATE.json')[U.name]['status']==dec['status'];print('FINALIZED',at)
