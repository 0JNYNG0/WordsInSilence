import json,re
import numpy as np
import study as d
U,O,ROOT=d.U,d.O,d.ROOT
f=d.read(U/'FROZEN.json');v=d.read(U/'VERIFY.json');a=d.read(U/'ANALYSIS.json');res=d.read(U/'RESULT.json');rc=d.read(U/'REPORT_CHECK.json')
assert v['status']==rc['status']=='PASS'
assert not (U/'RUNNING.json').exists() and not (U/'RECOVERY_RUNNING.json').exists()
assert not (U/'RECOVERY_ERROR.txt').exists() and not (U/'VERIFY_FIRST_ERROR.txt').exists()
d.check(f['pins']);d.check(f['protected']);d.check(d.read(U/'RECOVERY_AUTHORIZATION.json')['preserved'])
roots=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in roots:assert d.sha(p)==d.sha(U/'before'/p.name),'Concurrent root edit: '+str(p)
# Aggregate check directly from saved masks, not RESULT means.
for kind in d.KINDS:
    rr=[r for r in res['rows'] if r['kind']==kind];z=[d.load(U/'cases'/(r['name']+'.npz')) for r in rr];g=a['groups'][kind]
    assert len(z)==g['n']==216
    assert sum(bool(s['old_warning'].any()) for s in z)==g['old']
    assert sum(bool(s['persistent'].any()) for s in z)==g['new']
    denominator=sum(np.count_nonzero(s['expected']) for s in z)
    assert denominator==g['expected']==234000
    np.testing.assert_allclose(1-sum(np.count_nonzero(s['unknown']) for s in z)/denominator,g['support'])
    for k in range(3):
        wanted=sum(np.count_nonzero(s['geom_expected'][:,:,k]) for s in z)
        unknown=sum(np.count_nonzero(s['geom_unknown'][:,:,k]) for s in z)
        np.testing.assert_allclose(1-unknown/wanted,g['geometry'][k]['support'])
assert len(list((U/'cases').glob('*.npz')))==1080 and len(list((U/'controls').glob('*.npz')))==135
doc=(U/'RESULT_KO.html').read_text('utf8')
for link in re.findall(r'(?:src|href)="([^"]+)"',doc):assert (U/link).is_file()
at=d.now()
d.write(U/'VISUAL_REVIEW.json',dict(at=at,reviewer='Codex image inspection',charts=['summary.png','position_and_evidence.png','warning_disappears.png'],
 raw=a['raw'],notes=['Summary legend repaired and inspected; labels legible.',
 'Raw two-person6frames show similar gross staged upper-body gesture; exact finger correctness and exact timing not human-annotated.',
 'Position trajectories show different spatial extent; do not label one wrong.',
 'Warning disappears plot exposes missing evidence instead of a low/zero correctness score.',
 'Input/crossing preservation by exact source pins, not a new detector benchmark.'],browser_render=False))
next_action=('사용자와 다른사람 참고 DTW검증 결과를 공동검토한다. 같은원본 경고감소는 다른사람216조합에서 유지되지 않아 '
 '각도범위 경고를 자동오류/교정으로 쓰지 않는다. 다음 제안은 여러 공개 TRAIN사람의 정상차이를 한사람제외 방식으로 분리해 '
 '허용범위·관측부족·짧은차이 보존을 함께 검증하고, 관측된 손위치/팔궤적을 정렬근거로 활용할 별도후보계획을 확정하는 것. '
 '팔은 기존정렬에도사용되며 이번은 별도근거표시/거리검증. 경고를 줄이기만하는 문턱완화·단어예외 금지. '
 '현재3단어의 두검출범위를 사람간 허용범위로혼동금지. C·T3-F·운영웹유지; 사용자/멘토자료접근·학습·튜닝·선택금지. 추가학습은공동확인후.')
entry=f'''### 2026-10-22 다른 사람 참고·손 위치/팔 궤적 — 경고 감소 미입증, 근거 분리 검증
- 단위 `{U.name}`. 공개TRAIN수어·후원·멀다3단어×9명27영상,각다른8명참고216조합×원속도/fast075/slow15/pace/pause=1080. 새RGB추출·100단어전체·TCN성능실험아님.
- 기존DTW입력/가중치/band.2유지. 이전시간압축경고와고정된주변각도범위경고비교. 원속도210→216/216;빠르게213→215;느리게214→216;pace216→216;pause209→216. 정상시간변형合계852→863/864. 다른사람정확한대응/의미오류정답없어오답률아님. 원속도이미새경고216/216포화,추가경고0은개선근거아님.
- 두현재검출의각도일치범위는사람간정상수행허용범위아님. 정상개인차/투영/검출/정렬오차혼합으로사용자교정미채택. 참고8명중5명과경고있는영상도모든조건27/27이나동일시점의미합의아님.
- 신규연구기능:각도·손바닥·팔꿈치·팔손목의거리/근거/시간표분리. 어깨중앙·고정어깨너비정규화,원좌표/좌우유지. 손바닥실제관측점선택,팔은기존팔그룹관측사용;가상위치나모양복원없음. 팔정보원래DTW에있으며이번별도시각화,새정렬가중치없음.
- 원속도비교가능칸비율각도63.03%,손바닥74.23%,팔각99.38%;엄격모든연결관측각도62.27%,손바닥73.60%,팔99.36%. 정확도아님. 팔보임을모양근거로대체안함. 근거없는거리NaN·whole_action_correct=None유지.
- 후원REAL01→REAL05 fast075유일경고소멸은각도관측36.9→21.8%감소. 이전40경고칸중28모름/11차이남지만3연속미달/1범위내. 잘맞음으로처리금지. 합성재표본AND마스크가관측을줄일수있음.
- 고정경로손전체누락45/3frame누락45/팔전체누락45통제135PASS. 참고시간/필수손범위고정;관측없음은모름,다른항목보존. 정상손내림은참고양성근거만제외. 입력핀으로기존교차/정상손/좌우보존;새검출기교차정확도아님.
- 별도scalar각도/기하·부분근거검산1080+135PASS;저장비용독립DP최적검산/집계검산/보호32핀PASS. 원비용산식검증재사용. 동일Codex별도코드검산이며의미전문가아님. 그래프3+공개원본두명6장면시각검토,HTML정적링크,브라우저렌더없음.
- 실행전문법오류정정및run진행파일exclusive저장충돌1회보존. exec57576복구종료0,부분80배열/해시동일확인재사용,제외/문턱변경0. verify10150/analyze47229/report/finish완료,활성실행없음. 기존동결study는이력으로보존,재실행금지.
- C·T3-F·웹/원본유지. 추가학습·분류추론·사용자/멘토자료접근0. 연구용진단으로보존,자동교정/운영미적용. FROZEN/RESULT/VERIFY/ANALYSIS/VISUAL_REVIEW/RESULT_KO.html·txt/RECOVERY_AUTHORIZATION/FINAL_AUDIT/SYNC_CHECK.
'''
summary=dict(scope=dict(words=3,people=9,source_clips=27,ordered_pairs=216,timing_conditions=1080,controls=135),
 groups=a['groups'],warning_reduction_demonstrated=False,TCN_accuracy_measured=False,
 lost_warning=dict(previous_cells=40,now_unknown=28,now_short=11,within_range=1),
 coverage_unit='reference time x required hand x measured item cells; not accuracy')
audit=dict(status='CROSS_PERSON_DIAGNOSTIC_VERIFIED_WARNING_REDUCTION_NOT_DEMONSTRATED',at=at,
 implementer='Codex',verifier=v['verifier'],summary=summary,training=0,classifier_inference=0,user_data_access=0,operational_changes=False,
 protected=len(f['protected']),running=None,next=next_action,report_check=rc,
 first_failure='PREFLIGHT_CORRECTION + FIRST_ERROR preserved; exclusive progress output write failure recovered exact80arrays, exec57576 exit0',
 execution=['freeze exit0','study run exit1 progress-write conflict','recover exec57576 exit0','verify exec10150 exit0','analyze exec47229 exit0','report exit0'],
 artifacts={str(p):d.sha(p) for p in U.iterdir() if p.is_file()})
d.write(U/'FINAL_AUDIT.json',audit)
t=roots[0].read_text('utf-8-sig');mark='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';assert mark in t;t=t.replace(mark,mark+'\n\n'+entry,1)
mark='## 9. 다음 단일 작업';assert mark in t;t=t.replace(mark,mark+'\n\n### 2026-10-22 다른 사람 참고 DTW 결과 공동 검토\n'+next_action+'\n',1);roots[0].write_text(t,encoding='utf8')
s=d.read(roots[2]);s.update(updated_date_kst=at,current_stage='WORD100_CROSS_PERSON_MOTION_SUPPORT_RESEARCH_ONLY',next_single_action=next_action)
s[U.name]=dict(status=audit['status'],at=at,summary=summary,report=str(U/'RESULT_KO.html'),verify=str(U/'VERIFY.json'),audit=str(U/'FINAL_AUDIT.json'),running=None,training=0,user_data_access=0,operational_changes=False,next=next_action)
roots[2].write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
roots[1].write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 다른사람 공개참고에서도 정상속도 경고가줄어드는지 확인하고 손가락/위치/팔궤적과관측부족을분리한다. DTW연구단계,TCN단어인식학습아님.
마지막 완료: `{U.name}`1080조합+135누락통제검증. 다른사람경고감소미입증(원속도216/216새경고). 항목별거리/관측시간분리완료. 연구용보존,C·T3-F·웹유지.
다음 한 작업: {next_action}

{entry}

작업담당 Codex구현·별도산식검산. 사용자/멘토촬영접근·학습·튜닝·후보선택없음. 추가학습공동확인전금지.
모든실행종료. FIRST_ERROR/FAILED_EXECUTION_LOCK은복구전오류근거이며활성잠금아님. RECOVERY_RUNNING없음. 완료계산/검산재실행금지.
보고서는현재단위RESULT_KO.html. 원본/W:/sources/과거동결결과읽기전용. 브라우저렌더미실행,그림시각검토/정적링크검사완료.
''',encoding='utf8')
assert d.read(roots[2])['current_stage']=='WORD100_CROSS_PERSON_MOTION_SUPPORT_RESEARCH_ONLY'
for p in roots:assert U.name in p.read_text('utf-8-sig')
d.write(U/'SYNC_CHECK.json',dict(status='PASS',at=d.now(),root_hashes={str(p):d.sha(p) for p in roots},unit=U.name,running=None))
print('FINAL AUDIT AND ROOT SYNC PASS',flush=True)
