"""Saved-array aggregation audit and authorized completion records only."""
import json,re
from collections import defaultdict
import numpy as np
import study as d

U,O,ROOT=d.U,d.O,d.ROOT
f=d.read(U/'FROZEN.json');v=d.read(U/'VERIFY.json');a=d.read(U/'ANALYSIS.json');r=d.read(U/'RESULT.json')
assert v['status']==d.read(U/'REPORT_CHECK.json')['status']=='PASS'
assert not (U/'FINAL_AUDIT.json').exists()
for name in ('RUNNING.json','FIRST_ERROR.txt','VERIFY_FIRST_ERROR.txt'):
    assert not (U/name).exists(),name
roots=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in roots:assert d.sha(p)==d.sha(U/'before'/p.name),'Concurrent change: '+str(p)
d.check(f['pins']);d.check(f['protected'])

# Recount raw saved masks without study.stats or analyze.aggregate.
fields=['expected','comparable','unknown','reference_insufficient','query_incomplete','query_partial','mismatch','persistent','within_range','base_comparable','base_mismatch','base_persistent']
slices={'angle':slice(0,15),'palm':slice(15,17),'elbow':slice(17,19),'wrist':slice(19,21)}
counts=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
base={};deletions={}
for row in r['rows']:
    z=d.load(U/'cases'/(row['name']+'.npz'))
    if row['kind']=='identity':base[row['query'],row['anchor']]=z
    for group,sl in slices.items():
        c=counts[row['kind']][group]
        for key in fields:c[key]+=int(np.count_nonzero(z[key][:,:,sl]))
        c['warning']+=int(np.any(z['persistent'][:,:,sl]));c['base_warning']+=int(np.any(z['base_persistent'][:,:,sl]));c['any_difference']+=int(np.any(z['mismatch'][:,:,sl]))
        both=z['comparable'][:,:,sl]&z['base_comparable'][:,:,sl]
        c['common']+=int(both.sum())
        c['common_old_difference']+=int((both&z['base_mismatch'][:,:,sl]).sum())
        c['common_new_difference']+=int((both&z['mismatch'][:,:,sl]).sum())
        c['resolved_observed']+=int((both&z['base_mismatch'][:,:,sl]&~z['mismatch'][:,:,sl]).sum())
        c['old_difference_now_unknown']+=int((z['base_mismatch'][:,:,sl]&~z['comparable'][:,:,sl]).sum())
    if row['kind'].startswith('skip'):
        # Original baseline file is already retained in memory by producer order.
        b=base[row['query'],row['anchor']]
        affected=np.unique(b['path'][np.isin(b['path'][:,0],z['deleted']),1])
        domain=np.zeros_like(z['expected']);domain[affected]=z['expected'][affected]
        deletions[row['name']]=(bool(np.any(domain&z['mismatch']&~b['mismatch'])),bool(np.any(domain&z['persistent']&~b['persistent'])))
    assert row['whole_action_correct'] is None
for kind,groups in counts.items():
    for group,c in groups.items():
        target=a['groups'][kind]['groups'][group]
        for key,value in c.items():assert value==target[key],(kind,group,key,value,target[key])
        assert c['expected']==c['unknown']+c['comparable']
        assert c['comparable']==c['within_range']+c['mismatch']
        np.testing.assert_allclose(c['comparable']/c['expected'],target['coverage'])
for x in a['deletion_pairs']:
    assert deletions[x['name']]==(x['new_raw'],x['new_persistent'])
for size in ('skip3','skip20pct'):
    blocks=defaultdict(list)
    for row in r['rows']:
        if row['kind'].startswith(size):blocks[row['query'],row['kind']].append(deletions[row['name']])
    assert len(blocks)==81 and all(len(x)==8 for x in blocks.values())
    s=a['omission'][size]
    raw=[sum(x[0] for x in xx) for xx in blocks.values()]
    pers=[sum(x[1] for x in xx) for xx in blocks.values()]
    assert sum(n>0 for n in raw)==s['variants_any_raw']
    assert sum(n>=5 for n in raw)==s['variants_five_raw']
    assert sum(n==0 for n in raw)==s['variants_no_added_difference']
    assert sum(n>0 for n in pers)==s['variants_any_persistent']
assert len(list((U/'templates').glob('*.npz')))==216
assert len(list((U/'cases').glob('*.npz')))==2376
assert len(list((U/'controls').glob('*.npz')))==135
html=(U/'RESULT_KO.html').read_text('utf8')
for link in re.findall(r'(?:src|href)="([^"]+)"',html):assert (U/link).is_file(),link
at=d.now()
d.write(U/'AGGREGATE_CHECK.json',dict(status='PASS',at=at,cases=2376,templates=216,controls=135,deletion_pairs=1296,approach='Separate direct sums of saved masks; no reuse of summary aggregation helper; same Codex operator.'))
d.write(U/'VISUAL_REVIEW.json',dict(at=at,reviewer='Codex local image inspection',
    published_charts=['summary.png','normal_range.png','omissions.png','missed_all_references.png'],raw='short_omission_raw.jpg',
    notes=['Final summary legend placed below right plot and visually checked.',
    'Range and missing-time charts legible. No whole-action pass implied by green.',
    'Selected all8reference-missed 3frame example has no newly missing cells; cannot attribute all misses to absence of observation.',
    'Public original frames38/40/42 show chest-level hand movement but not expert semantic essentiality truth.',
    'Normal/crossing preservation is exact inputs/identity pins, not new detector correctness.'],browser_render=False))

next_action=('사용자와 다중 사람 참고 범위의 부분적 개선 및 짧은 구간 누락 한계를 공동 검토한다. '
 '다음 제안은 동일 동작 단계의 정렬을 손 위치·팔 궤적·방향 변화로 점검하고, 시점별 최솟값/최댓값과 동작 순서 근거를 분리한 연구 후보를 비교하는 것. '
 '정상 속도·손내림·한 손 동작·교차와 짧은 구간 제거를 함께 검증하며, 누락 뒤 재정렬도 별도 검사한다. '
 '전 단어 수동 규칙이나 단어 예외, 결과를 보고 문턱만 완화하는 방법은 사용하지 않는다. '
 '현재 관측 범위를 수어 정답 허용 범위로 채택하지 않으며 관측 부족/범위 내/범위 밖을 분리한다. '
 'C·T3-F·운영 웹은 유지한다. 사용자·멘토 자료 접근/학습/튜닝/선택 금지, 추가 학습은 공동 확인 후.')
entry=f'''### 2026-10-22 한 사람 제외 다중 참고 범위 — 개인차 경고 감소, 짧은 누락 판단은 미완료
- 단위 `{U.name}`. 공개 TRAIN 수어·후원·멀다 3단어×9명=27영상, 시험 수행자를 제외한 8명 참고로 216개 시간축별 범위. 11조건×216=2376비교, 고정경로 누락통제135. 100단어 전체나 새 분류 정확도 평가 아님.
- 각 시간/손/항목에 완전 관측 최소5명, 8명 최솟값~최댓값. 15각도+손바닥xy+팔꿈치xy+팔손목xy. query 수행자는 모든 범위/참고끼리 정렬 계산에서 제외. 과거 모델/개발에 사용한 공개 사람이라 완전 미사용 신규 사람 평가 아님.
- 기존 좌표/좌우/교차/정렬 가중치/band.2 유지. 엄격 관측 단일참고 통제와 다중범위를 같은 경로에서 비교. 원속도 지속 경고 각도216→163,손바닥215→143,팔꿈치216→182,팔손목216→180 /216. 9명 각각 감소하나 의미오류 정답 없어 오답률/정확도 아님.
- 각도 비교가능149349/234000=63.82%. 기존 범위밖75604칸 중62608은 관측 유지·범위내,5820은 근거부족,7176은 공통관측 차이 유지. 근거부족은 개선으로 세지 않음. 새로 비교가능한 칸의 차이1133 포함 전체새차이8309. 전역 정답판정None.
- 정상시간 각도경고:원216→163,fast075215→160,slow15216→163,pace216→156,pause216→176. 정지추가 때 경고 증가 남음. 시간 재표본화 관측mask AND 영향 존재; 실제 고속 RGB센서 시험 아님.
- 주석 core3frame 제거81변형:1개이상 참고 새차이69,5개이상38,새지속차이56,모든8참고 새차이없음12. core20%제거81:각81/69/75/0. 언어적 필수동작 정답 없음, 기존 원본정렬로 삭제근처를 찾은 대리평가. 누락 검출률로 과장금지. 수어REAL01 skip3_25는 새미관측0인데 새차이0인 한계 시각확인.
- 손전체45/3frame손45/팔전체45 통제135PASS:참고 기대범위 고정,누락은NaN/근거부족,다른항목 유지. 고정경로 통제로 누락 후 재정렬까지 증명하지 않음. 1frame차이 보존·3frame차이·참고4명부족·부분누락 등 fixture6PASS(넓은범위 누락흡수는 알려진 한계를 확인한 시험).
- 사후 원천점검:충분지원각도175603칸 중16392는 다른참고의 주석core밖이 범위끝값 형성. 각도/손바닥의확인된손내림 기여0;팔은내려놓은상태도관측됨. 주석경계차이 가능해 전부오류단정금지. 시점별독립범위가단계/순서를흡수할위험;기준수정없음.
- 범위216/비교2376 독립scalar검산,재사용1080경로동일,새1296저장비용DP최적검산,135통제/보호32핀/별도집계PASS. 동일Codex별도산식이며외부전문가검증아님. 공개원본3장면+게시그래프4개시각검토,HTML정적링크검사;브라우저렌더없음.
- study exec59704/verify79974/analyze49331 모두종료0,에러·잠금없음. 완료추출/계산/검산재실행금지. 보고/최종검산 완료. 추가학습·분류추론·사용자/멘토접근0;C·T3-F·웹유지. 연구용관측범위로만보존,자동정상/교정기준미채택.
'''
summary=dict(scope=dict(words=3,people=9,clips=27,templates=216,cases=2376,controls=135),
    identity=a['groups']['identity'],timing={k:a['groups'][k]['groups']['angle']['warning'] for k in d.KINDS if not k.startswith('skip')},
    omission={k:{kk:vv for kk,vv in value.items() if kk!='members'} for k,value in a['omission'].items()},
    warning_reduction=True,semantic_acceptance_validated=False,TCN_accuracy_measured=False,unseen_classifier_person_evaluation=False)
status='LOPO_RANGE_WARNING_REDUCTION_VERIFIED_SHORT_OMISSIONS_NOT_RELIABLY_DETECTED'
d.write(U/'FINAL_AUDIT.json',dict(status=status,at=at,implementer='Codex',verifier=v['verifier'],
    summary=summary,training=0,classifier_inference=0,user_data_access=0,operational_changes=False,
    protected=len(f['protected']),running=None,next=next_action,
    execution=['study freeze exit0','study run exec59704 exit0','verify exec79974 exit0','range audit exit0','analyze exec49331 exit0','supplement exit0','report exit0','saved aggregation PASS'],
    artifacts={str(p):d.sha(p) for p in U.iterdir() if p.is_file()}))
doc=roots[0].read_text('utf-8-sig')
for marker,addition in [
    ('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry),
    ('## 9. 다음 단일 작업','### 2026-10-22 다중 참고 범위 결과 공동 검토\n'+next_action+'\n')]:
    assert marker in doc;doc=doc.replace(marker,marker+'\n\n'+addition,1)
roots[0].write_text(doc,encoding='utf8')
s=d.read(roots[2]);s.update(updated_date_kst=at,current_stage='WORD100_LOPO_REFERENCE_RANGE_RESEARCH_ONLY',next_single_action=next_action)
s[U.name]=dict(status=status,at=at,summary=summary,report=str(U/'RESULT_KO.html'),verify=str(U/'VERIFY.json'),audit=str(U/'FINAL_AUDIT.json'),running=None,training=0,user_data_access=0,operational_changes=False,next=next_action)
roots[2].write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
roots[1].write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 여러 공개 사람의 동작 차이 범위를 만들고, 한 사람씩 참고에서 제외하여 차이/근거부족/짧은누락을 검증한다. DTW 분석 연구이며 TCN 학습이 아니다.
마지막 완료: `{U.name}`. 공개3단어9명27영상,216참고범위,2376비교+135누락통제 완료. 개인차 경고는 감소했으나 3frame제거81개 중12개는 새차이가 없어 자동정상판정 미채택.
다음 한 작업: {next_action}

{entry}

구현·실행·별도산식검산 담당 Codex. 외부 수어전문가 의미검증 아님. C·T3-F·운영 웹·사용자/멘토 자료 유지. 사용자/멘토 촬영 접근·학습·튜닝·선택0. 추가학습 공동확인 전 금지.
모든 실행 종료, FIRST_ERROR/RUNNING 없음. 완료 계산/검산 재실행 금지. 보고서는 현재단위 RESULT_KO.html/txt. FROZEN/RESULT/VERIFY/ANALYSIS/RANGE_AUDIT/SUPPLEMENT/AGGREGATE_CHECK/VISUAL_REVIEW/FINAL_AUDIT/SYNC_CHECK 근거 유지.
원본/W:/sources/과거 동결 결과 읽기전용. 그래프 시각검토 및 HTML 정적링크 검사 완료, 브라우저 렌더 미실행.
''',encoding='utf8')
assert d.read(roots[2])['current_stage']=='WORD100_LOPO_REFERENCE_RANGE_RESEARCH_ONLY'
for p in roots:assert U.name in p.read_text('utf-8-sig')
d.write(U/'SYNC_CHECK.json',dict(status='PASS',at=d.now(),unit=U.name,root_hashes={str(p):d.sha(p) for p in roots},running=None))
print('FINAL AUDIT AND SYNC PASS',flush=True)
