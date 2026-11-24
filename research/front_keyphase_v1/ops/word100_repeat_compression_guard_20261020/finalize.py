from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent
OPS=U.parent
ROOT=U.parents[3]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
assert not (U/'FINAL_AUDIT.json').exists(),'Completed; do not repeat finalization'
f=read(U/'FROZEN.json')
s=read(U/'SUMMARY.json')
d=read(U/'DIAGNOSTICS.json')
v=read(U/'VERIFICATION.json')
for name in ('SELFTEST.json','VERIFICATION.json','BROWSER_QA.json','VISUAL_REVIEW.json'):
    assert read(U/name)['status']=='PASS',name
assert len(read(U/'SELFTEST.json')['checks'])==11
for kind in ('protected','previous','code'):
    for p,h in f[kind].items():assert sha(p)==h,p
for p,h in read(U/'OUTPUT_PINS.json').items():assert sha(U/p)==h,p
for p,h in read(U/'EVALUATION_PINS.json').items():assert sha(p)==h,p
assert s['rows']==3291 and s['stress_rows']==7092 and s['lowering_rows']==1800
rows=read(U/'ROWS.json')
assert len(rows)==3291
for group,totals in s['groups'].items():
    view,scenario=group.split('_',1)
    rr=[r for r in rows if r['view']==view and r['scenario']==scenario]
    assert len(rr)==totals['n']==300
    target=int(scenario[-1]) if scenario.startswith('repeat') else 1
    for side in ('old','new'):
        count=[r[side+'_count'] for r in rr]
        assert sum(x==target for x in count)==totals[side+'_correct']
        assert sum(x is not None and x!=target for x in count)==totals[side+'_wrong']
        assert sum(x is None for x in count)==totals[side+'_abstain']
    assert sum(r['old_count']==target and r['new_count']!=target for r in rr)==totals['lost_correct']
lower=read(U/'LOWERING.json')
stress=read(U/'STRESS.json')
for axis,totals in s['lowering'].items():
    for key,value in totals.items():assert abs(sum(r[key] for r in lower if r['key']==axis)-value)<1e-7
for r in stress:assert abs(r['old_possible_s']-r['new_possible_s'])<1e-7
assert d['deleted_signal_count']==d['deleted_join_exact']==14
assert d['deleted_total']==591 and len(d['recovered_unanimous_cases'])==2
assert d['previous8_strong']==4 and d['previous8_possible']==8
assert read(U/'DECISION.json')['status']=='OFFLINE_VERIFIED_NOT_ADOPTED'
assert not list(U.glob('FIRST_ERROR*')),'Preserve and explain errors before finalization'
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
low=s['lowering']['location']
brief='; '.join(
    f"{name}: 일치 {g['old_correct']}→{g['new_correct']}, 불일치 {g['old_wrong']}→{g['new_wrong']}, 보류 {g['old_abstain']}→{g['new_abstain']}"
    for name,g in [(name,s['groups']['sensor_'+sc]) for name,sc in [
        ('원래','original'),('2회 연결','repeat2'),('3회 연결','repeat3'),
        ('빠른 재생','speed075'),('느린 재생','speed150'),('정지 추가','pause')]])
next_action=(
    '공개 반복·압축 보완 후보의 효과와 정상 변형 추가 보류를 사용자와 검토한다. '
    '후보는 진단용으로 보존하고 운영 미적용. 다음 설계는 전체 횟수 불확실성과 국소 관측 부족을 분리하고, '
    '대안 반복의 앞·중간·끝 관측 순서와 범위가 충분한지 확인하는 방향이다. '
    '귀환 자세나 한 시점 압축만으로 생략/반복 오류를 판정하지 않는다. '
    '같은 DEV 결과로 현 동결 조건을 재조정하지 않는다. C·T3-F와 웹 유지, 사용자·멘토 촬영의 학습·튜닝·후보 선택 금지. '
    '추가 학습은 결과 검토 후 별도 승인.')
title='2026-10-20 1회 합의의 대안 반복 근거·짧은 동작 압축 후보 검증 완료 — 정상 속도 추가 보류로 미채택'
entry=f'''### {title}
갱신 {at}. `ops/{U.name}`. Codex 로컬 구현·공개 대조·별도 구현 검산 완료. 기존 모델/웹 미적용, 추가 학습 없음.
- 공개 100단어 TRAIN899(사람01~09) 참고/DEV300(11~13) 평가. 기존 2391개 경로 묶음을 재사용하고 속도·정지 900개 대조만 새 정렬했다. 2398관측입력은1199원본의 legacy/sensor 두 경로. DEV주석/full손상태는 평가 전용, 같은 DEV 후속 분석이며 새로운 최종 평가가 아니다.
- 동결 규칙: 최소비용 선호와 독립적으로 모든 가능한2/3회 정렬의 팔 귀환 근거를 확인. 동일5사람이 모든 경계를 지지하면 기존1회 확정을 보류. TRAIN의 관측된 팔 이동량10구간에서 다수 참고 시점이 query1시점으로 압축되는지 확인; 동일5명 공동근거 필요. 기존 횟수 불명은 모든 viable 가설에서 같은query시점 압축일 때만 신호. 없는 관측은 압축과 구분. 필수 수어 단계10개가 아니다.
- 압축 구간의 강한 근거를 낮추고 가능한 비교 범위는 그대로 보존. semantic_error/complete_performance는 null. 단어별 예외·사용자 자료·DEV 점수에 맞춘 계수 변경 없음.
- sensor 각300개: {brief}. 이는 참고 영상 연결횟수 대조이며 TCN 정확도/수어 정답률이 아니다. 보류를 성공으로 세지 않는다.
- 남은 잘못된1회 확정2건은 같은 실명 영상2회/3회 연결이다. 동일 참고2·3·4·6·8번이 실제 연결 경계를 포함하는 대안 귀환 근거를 지지해 보류했다. 정확한2/3회로 확정한 것은 아니며 경계 후보는 여러 시점에 퍼져 있다.
- 정상 원본 sensor3개, 빠른 재생16개, 느린 재생28개, 정지3개가 추가 보류. 빠른 재생의 한시점 압축16개/느린 재생의 대안 귀환28개가 있어, 신호를 오류로 쓰면 정상 수행을 문제 삼을 위험이 있다. legacy도 원본12/2회1/3회1개 추가 보류. 자동 적용 미채택.
- 짧은구간 전체프레임삭제591대조 중14개에서 압축 신호, 모두 삭제 연결 지점 인접. 삭제 구간의 언어적 필수성 미확정이며 필수동작 누락탐지율로 해석 금지. 대부분 삭제는 이 엄격한 신호로 검출되지 않았고, 정상 빠른재생에서도 같은 신호 발생.
- 정상 위치 손내림 추정{low['rest_missing_s']:.3f}초 중 강한 비교범위 겹침{low['old_strong_s']:.3f}→{low['new_strong_s']:.3f}초, 불확실포함{low['old_possible_s']:.3f}→{low['new_possible_s']:.3f}초. 추가 악화 없지만 이전 모호함 미해결. 7092가림항목 가능범위 불변; 이전8손·항목·조건은 강한4/불확실포함8 보존. 서로 다른8영상이 아니다.
- 속도 대조는 고정 표본 간격에서 길이0.75/1.5배 좌표 보간(양끝 관측 있을 때만, flagsAND), 정지는 중간15프레임 복제. 실제 웹캠 영상·MediaPipe 재추출/자연 반복 시험은 아니다. 팔꿈치·pose손목4점 정렬로 손가락만의 미세 단계는 충분히 검증하지 못함.
- 자체11PASS; 별도 구현 검산 {v['rows']}대조/{v['reference_evidence']}참고근거/{v['new_control_paths']}신규경로/최적경로표본{v['new_path_optimality_samples']}/{v['hash_pins']}해시PASS. 같은Codex의 별도 새규칙 구현이며 이전 검산된 투표/경로 헬퍼는 재사용; 별도 사람/수어전문가 검토 아님. 로컬 보고서 desktop/mobile 시각검토PASS.
- C·T3-F6가중치/32보호파일/과거 동결자료 해시 유지. 사용자·멘토/보류14~18 미접근. 추가 학습/분류추론/운영DTW·웹 교체/카메라/다운로드/외부전송/Git 없음. 평가8583/검산20092 종료0; 완료 실행 중복 금지.

'''
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
text=ctx.read_text('utf-8-sig')
assert title not in text
for marker,content in [
    ('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry),
    ('## 9. 다음 단일 작업',f'### {title}\n{next_action}\n\n')]:
    assert marker in text,marker
    text=text.replace(marker,marker+'\n\n'+content,1)
ctx.write_text(text,'utf-8')
state=read(OPS/'STATE.json')
state.update(updated_at=at,updated_date_kst='2026-10-20',
    current_stage='WORD100_REPEAT_COMPRESSION_GUARD_REVIEW',
    activity_state='AWAITING_USER_REVIEW',next_single_action=next_action,resume_command=next_action)
state[U.name]=dict(status='OFFLINE_VERIFIED_NOT_ADOPTED',at=at,unit=str(U),
    report=str(U/'RESULT_KO.html'),summary=s,verification=v,diagnostics=str(U/'DIAGNOSTICS.json'),
    training=0,production_changes=0,private_data_read=0,next_single_action=next_action,
    execution='Evaluation8583 and verification20092 complete exit0; do not rerun')
write(OPS/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 참고가 모두1회로 선호해도 대안 반복 근거와 짧은 움직임 압축을 검사.
마지막 완료: 별도 후보 구현·공개 비교·별도 구현 검산. 다음 한 작업: 결과와 정상 변형 추가 보류를 사용자와 검토.

- 단위 `{U}`. RESULT_KO.html/txt, SUMMARY, DIAGNOSTICS, VERIFICATION, FROZEN, FINAL_AUDIT 참고.
- TRAIN899/DEV300, 공개100단어. 참고 사람01~09/평가11~13 분리.2391기존경로+900신규속도/정지대조. 같은DEV반복진단,새최종시험아님. 사용자/멘토/보류14~18 미접근.
- 모든 가능한2/3회 정렬의 실제 팔 귀환 근거를 최소비용과 별도로 확인. 동일5명 공동경계이면1회확정보류. TRAIN관측 팔 이동량10부분의 다수시점→query1시점 압축을 공동5명으로 확인. 부분은 언어적 필수단계 아님. 미관측과 압축은 구분.
- 신호 있으면 count=null/국소강한근거 하향, 가능범위는 보존. 의미오류/완료판정null. 단어예외/결과후계수조정 없음.
- sensor각300: {brief}.
- 잘못된1회 확정2건=같은실명영상2/3회연결. 대안경계로 보류했지만 정확한횟수/경계로 확정한 것은 아님. 원본3/빠름16/느림28/정지3개 추가 보류, legacy원본12/반복각1개도 추가보류. 진단용 보존, 자동 적용 미채택.
- 삭제591중14압축신호, 모두삭제연결점인접. 필수동작여부미확정/대부분삭제미검출/정상빠른재생16개도압축신호이므로오류판정근거로부족.
- 정상손내림 강한범위겹침0.100초/불확실포함77.500초 그대로. 기존모호함미해결.7092가림가능범위불변/이전8조건강한4·가능8보존.
- 속도0.75/1.5길이와15프레임정지는 좌표 합성 대조. 실제웹캠/센서/새사람/자연반복미검증. 팔꿈치·pose손목4점으로 손가락만의짧은단계는검증한것아님.
- 자체11PASS; 별도 새규칙검산{v['rows']}대조/{v['reference_evidence']}참고/{v['new_control_paths']}신규경로/{v['hash_pins']}해시PASS. 같은Codex별도구현,기존검산헬퍼재사용. 다른사람/수어전문가아님. 로컬보고서시각QA PASS.
- C/T3-F6가중치·32보호파일 유지. 추가학습/분류추론/운영웹·DTW교체/카메라/다운로드/외부전송/Git 없음.
- 실행8583/20092 종료0. CURRENT_EXECUTION COMPLETE. 완료연구재실행금지.

{next_action}
''','utf-8')
cur=read(U/'CURRENT_EXECUTION.json')
assert cur.get('evaluation_exit_code')==0
assert cur.get('verification_session')==20092
cur.update(status='COMPLETE',at=at,verification_exit_code=0)
write(U/'CURRENT_EXECUTION.json',cur)
write(U/'STATUS.json',dict(status='COMPLETE_OFFLINE_REVIEW',at=at,
    production_applied=False,next_single_action=next_action))
write(U/'FINAL_AUDIT.json',dict(status='PASS',at=at,
    protected_files_unchanged=len(f['protected']),model_weights_unchanged=6,
    production_changes=0,training=0,private_data_read=0,
    artifacts={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'},
    document_sha256={str(p):sha(p) for p in [ctx,ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']}))
print('FINALIZED',at)
