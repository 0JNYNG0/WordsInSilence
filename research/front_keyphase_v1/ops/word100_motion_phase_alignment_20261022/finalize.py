"""Audit saved aggregates and synchronize current research documentation only."""
import json,hashlib,math,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;R=U.parents[3];key=U.name
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
a=read(U/'ANALYSIS.json');v=read(U/'VERIFY.json');f=read(U/'FROZEN.json');r=read(U/'RESULT.json')
review=read(U/'INTERPRETATION_REVIEW.json')
assert v['status']=='PASS' and v['total']==2808
assert not (U/'RUNNING.json').exists()
assert read(U/'REPORT_CHECK.json')['status']=='PASS'
assert {k:len(r[k]) for k in ('known','cross','missing')}=={'known':297,'cross':2376,'missing':135}
for path,pin in {**f['pins'],**f['protected']}.items():assert sha(path)==pin,path
proof=read(U/'VERIFY_RECOVERY_PROOF.json')
for path,pin in {**proof['preserved'],**proof['outputs']}.items():assert sha(path)==pin,path
normal=('fast075','slow15','pace','pause');checks={}
for kind in normal:
    rr=[x for x in r['known'] if x['kind']==kind]
    for side in ('old','new'):
        n=sum(x[side+'_timing']['moving']['n'] for x in rr)
        mean=sum(x[side+'_timing']['moving']['mae']*x[side+'_timing']['moving']['n'] for x in rr)/n
        assert math.isclose(mean,a['known'][kind]['timing']['moving'][side]['mae'],rel_tol=1e-12,abs_tol=1e-12)
        checks[kind+'_'+side]=mean
for group in ('angle','palm','elbow','wrist'):
    rr=[x for x in r['cross'] if x['kind']=='identity']
    for side in ('old','new'):
        for metric in ('warning','comparable','expected','unknown','mismatch'):
            assert sum(x[side][group][metric] for x in rr)==a['cross']['identity']['groups'][group][side][metric]
for folder in ('known','cross'):
    for size in ('skip3','skip20pct'):
        members=a['omissions'][folder][size]['members'];assert len(members)==81
        for side in ('old','new'):
            assert sum(m[side]['inside_raw']>0 for m in members)==a['omissions'][folder][size][side]['variants_any_raw']
        assert sum(m['new']['inside_raw']>0 and m['old']['inside_raw']==0 for m in members)==a['omissions'][folder][size]['newly_detected_variants']
        assert sum(m['old']['inside_raw']>0 and m['new']['inside_raw']==0 for m in members)==a['omissions'][folder][size]['lost_detection_variants']
for x in a['missing'].values():
    assert x['n']==45 and x['all_reference_domain_unchanged'] and x['all_native_expected_not_reduced'] and x['all_forced_native_unknown_preserved']
write(U/'AGGREGATE_CHECK.json',dict(status='PASS',at=now,scope='Parent recombination of saved per-case timing and diagnostic counts; no alignment or inference rerun.',normal_moving_mean=checks,counts={'known':297,'cross':2376,'missing':135},review_present=True))
plots=['timing.png','cross.png','omissions.png','improvement.png','regression.png','missed_omission.png']
write(U/'VISUAL_REVIEW.json',dict(status='PASS',at=now,reviewer='Parent Codex, view_image',plots={n:sha(U/'visuals'/n) for n in plots},observations=['All six published figures visually inspected.','Cross legend and regression x-axis clipping repaired by render-only script and both re-inspected.','No numeric results changed.','Regression plot includes large errors outside measured-changing intervals; reported mean explicitly limited to those intervals.','HTML links checked; no browser rendering or new video inspection this unit.'],browser_render=False))
roots={'WordsInSilence_PROJECT_CONTEXT.md':R/'WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md':R/'NEW_CHAT_HANDOFF.md','STATE.json':U.parent/'STATE.json'}
for name,path in roots.items():assert sha(path)==sha(U/'before'/name),'Concurrent root change: '+name
short=a['omissions']['cross']['skip3'];cp=a['cross']['identity']
summary=dict(scope={'words':3,'people':9,'clips':27,'known':297,'cross':2376,'missing':135},normal_timing={k:a['known'][k]['timing']['moving'] for k in normal},normal_cases={metric:sum(a['known'][k][metric] for k in normal) for metric in ('improved','worsened','unchanged')},identity_cross=cp,short_omission={k:x for k,x in short.items() if k!='members'},missing={k:{p:x for p,x in z.items() if p!='groups'} for k,z in a['missing'].items()})
assert summary['normal_cases']=={'improved':9,'worsened':25,'unchanged':74}
next_action='사용자와 손·팔 위치/방향 보조 DTW 결과를 검토한다. 이번 가중 후보는 개선 미입증으로 운영 미적용. 다음 제안은 남은 짧은 구간 제거 사례에서 구별 정보 자체의 관측 여부, 시간 연결 압축, 다중 참고 범위 흡수를 분리 진단하는 것이다. 그 후 실제 한 사람이 수행한 짧은 연속 동작과 앞뒤 순서를 함께 비교하는 공통 후보의 필요성을 판단한다. 정상 속도·손내림·한 손 동작·교차·반복 대조를 유지하며 특정 단어 예외나 결과를 보고 문턱만 완화하지 않는다. C·T3-F·운영 웹 유지; 사용자·멘토 자료 접근/학습/튜닝/선택 금지. 추가 학습은 결과 공동 확인 후.'
section=f'''### 2026-10-22 손·팔 위치와 방향 정렬 보조 — 계산 검증 완료, 뚜렷한 개선 없어 미채택
- 단위 `{key}`. 공개 TRAIN 수어·후원·멀다 3단어×9명=27영상, 알려진 시간변형297·다른 사람2376·누락 후 재정렬135 비교. 100단어 전체나 새 분류 정확도 평가 아님.
- 기존 DTW 비용에 손바닥·팔꿈치·팔 손목 위치 및 바로 앞뒤 이동 방향을 각각0.25 추가. 긴 동작 단계 기억 기능은 아님. 방향 크기는 무시하고 정지/불확실/관측 끊김 방향을 만들어내지 않음. 기존8명 참고범위·좌표·좌우·정규화·band.2 유지, 결과로 가중치 조정 없음.
- 정상시간변형108: 개선9/악화25/동일74. 관측 변화 구간 평균 시점 오차 fast075 0.182783→0.185328, slow15 0.255115→0.255893, pace 0.161966→0.171875, pause0→0프레임. 변화 작고 두 방식 모두 이 구간2프레임 이내100%; 일반 사용자 정확도 의미 아님.
- 다른 사람 원속도 각도경고163→167/216, 비교가능63.824→63.861%. 기존 비교가능→근거부족5718칸(기존차이304) 별도표시. 주석core밖 연결7.663→7.410%이나 확인손내림→참고활동121→156 손·경로칸으로 증가. 정확한 사람간 단계정답 없어 대리지표로만 보고.
- 다른 사람3프레임제거81변형: 1참고이상 새차이69→72(새9/소멸6),5참고이상38→28,새지속차이56→56,새차이없음12→9. 언어적 필수단계 정답은 아니며 안정적 누락 판정 개선 미입증. 같은원본 새차이80→80/81.
- 손전체45/짧은손45/팔전체45 재정렬통제 통과: 참고 기대범위 고정, 원래촬영 미관측 시간표 보존, 정상판정None. 누락시간이 참고제외에만 연결된 행372→354도 별도보존. 정상원본 손내림·교차좌표 보존은 확인했으나 손내림 시간정렬의 개선은 아님.
- study57045 종료0, 원검산94105 정밀도오류 종료1 보존. float32평균 vs 합계/정수개수의 정밀도 차이 확인(최대9.35e-9→4.44e-16), 허용2e-11 유지. 통과2673재사용+남은135 exec73154 종료0. 복구도중 import경로오류도 보존·명시적경로로수정, 과거20감사파일불변확인. 실험/점수/조건 변경없음.
- 분석51379 종료0, 별도에이전트 비용/DP검산 및 해석 교차검토, 부모집계확인·보호32핀·게시6그림 시각검토·HTML링크검사 통과. 외부 수어전문가 의미검증/브라우저렌더 없음. 추가학습·분류추론·사용자/멘토접근0. C·T3-F·운영DTW/웹 유지, 연구용보존/운영미적용.
'''
evidence_names=['candidate.py','study.py','FROZEN.json','RESULT.json','VERIFY.json','VERIFY_FIRST_ERROR.txt','VERIFY_RECOVERY_PROOF.json','VERIFY_RECOVERY_IMPORT_NOTE.json','VERIFY_IMPORT_OLD_OUTPUT_CHECK.json','ANALYSIS.json','INTERPRETATION_REVIEW.json','DESIGN_REVIEW.json','AGGREGATE_CHECK.json','VISUAL_REVIEW.json','RESULT_KO.html','RESULT_KO.txt','REPORT_CHECK.json']
write(U/'FINAL_AUDIT.json',dict(status='PASS_RESEARCH_VERIFICATION_NOT_ADOPTED',at=now,summary=summary,evidence={n:sha(U/n) for n in evidence_names},protected_files=len(f['protected']),immutable_outputs=2808,first_failures_preserved=True,experimental_changes_after_results=False,executions={'study':57045,'original_verifier':94105,'precision_proof':25302,'remaining_verifier':73154,'analysis':51379},training=0,classifier_inference=0,user_mentor_access=0,running=None,roots_before={n:sha(p) for n,p in roots.items()},next=next_action))
context=roots['WordsInSilence_PROJECT_CONTEXT.md'].read_text('utf-8-sig')
for pattern,content in [(r'(?m)^## 8\.[^\n]*\n',section),(r'(?m)^## 9\.[^\n]*\n','### 2026-10-22 손·팔 위치·방향 정렬 보조 결과 공동 검토\n'+next_action+'\n')]:
    m=re.search(pattern,context);assert m
    context=context[:m.end()]+'\n'+content+'\n'+context[m.end():]
roots['WordsInSilence_PROJECT_CONTEXT.md'].write_text(context,encoding='utf8')
state=read(roots['STATE.json'])
state.update(updated_date_kst=now,updated_at=now,current_stage='WORD100_MOTION_PHASE_ALIGNMENT_RESEARCH_ONLY_NOT_ADOPTED',activity_state='AWAITING_USER_REVIEW',next_single_action=next_action,resume_command=next_action)
state[key]=dict(status='VERIFIED_NO_CLEAR_IMPROVEMENT_NOT_ADOPTED',at=now,summary=summary,report=str(U/'RESULT_KO.html'),verification=str(U/'VERIFY.json'),final_audit=str(U/'FINAL_AUDIT.json'),training=0,classifier_inference=0,user_mentor_access=0,running=None,next=next_action)
roots['STATE.json'].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}

현재 목표: 여러 사람의 공개 자료에서 DTW가 같은 동작 순간을 비교하도록 개선한다. 이번은 손·팔 위치와 방향의 정렬 보조 연구이며 TCN 학습이 아니다.
마지막 완료: `{key}`. 27공개영상 기반2808비교와 검산 완료. 관측부족 보존은 통과했으나 정렬 개선은 입증되지 않아 후보 미채택. C·T3-F·운영웹 유지.
다음 한 작업: {next_action}

{section}

구현/실행: 부모 Codex. 비용·DP·근거 별도검산: phase_design_review. 집계/해석 교차검토: phase_evaluation_review. 최종 집계·그림·동기화: 부모. 수어 전문가 의미검증 아님.
실험조건/결과/모델 수정0. 원검산 정밀도 오류와 복구 import오류는 보존되어 있으나 둘 다 원인확인·복구완료. 현재 실행/잠금 없음. 완료 실험/검산 재실행 금지.
근거: FROZEN/RESULT/VERIFY/VERIFY_RECOVERY_PROOF/ANALYSIS/INTERPRETATION_REVIEW/AGGREGATE_CHECK/VISUAL_REVIEW/FINAL_AUDIT/SYNC_CHECK. 그래프6개 시각확인,HTML정적링크검사,브라우저렌더 없음.
원본/W:/sources/과거동결결과 읽기전용. 사용자·멘토촬영 접근/학습/튜닝/후보선택0. 추가학습은 사용자 공동 확인 전 금지.
'''
roots['NEW_CHAT_HANDOFF.md'].write_text(handoff,encoding='utf8')
assert read(roots['STATE.json'])[key]['summary']['normal_cases']==summary['normal_cases']
assert key in roots['WordsInSilence_PROJECT_CONTEXT.md'].read_text('utf8') and key in roots['NEW_CHAT_HANDOFF.md'].read_text('utf8')
write(U/'SYNC_CHECK.json',dict(status='PASS',at=now,roots={n:sha(p) for n,p in roots.items()},report=str(U/'RESULT_KO.html'),current_stage=state['current_stage'],running=None))
print('AGGREGATES, FINAL AUDIT AND ROOT SYNC PASS')
