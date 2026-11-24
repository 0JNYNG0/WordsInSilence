"""Read-only public metadata/code audit; write this planning unit and handoff docs only."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
import hashlib
import json
import shutil

U = Path(__file__).resolve().parent
B = U.parents[1]
ROOT = B.parents[1]
read = lambda p: json.loads(Path(p).read_text('utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', 'utf-8')

assert not (U/'AUDIT.json').exists(), 'Preserve completed audit'
now = datetime.now(timezone(timedelta(hours=9))).isoformat()
statepath = B/'ops/STATE.json'
docs = [ROOT/'WordsInSilence_PROJECT_CONTEXT.md', ROOT/'NEW_CHAT_HANDOFF.md', statepath]
(U/'before').mkdir(exist_ok=True)
for p in docs:
    shutil.copy2(p, U/'before'/p.name)

oldpath = B/'ops/word100_multiref_20261014/BASELINE.json'
rolepath = B/'ops/word100_role_dtw_20261017/ROLE_BANK.json'
old, role = read(oldpath), read(rolepath)
manifest = read(B/'ops/word100_training_20261013/MANIFEST.json')
train = [x for x in manifest if x['role']=='train_candidate']
dev = [x for x in manifest if x['role']=='development_only']
counts = Counter(x['word_code'] for x in train)
assert len(train)==899 and len(dev)==300 and len(counts)==100
assert sorted({x['signer_id'] for x in train})==list(range(1,10))
assert sorted({x['signer_id'] for x in dev})==[11,12,13]

checks=[]
def check(name, value):
    checks.append({'check':name,'passed':bool(value)})
check('legacy bank config hash',sha(oldpath)==read(B/'config/word100_multiref.json')['baseline_sha256'])
check('role bank config hash',sha(rolepath)==read(B/'config/word100_role_dtw.json')['bank_sha256'])
check('role legacy dependency',role['legacy_bank_sha256']==sha(oldpath))
for p,h in {**old['code_sha256'],**role['code_sha256']}.items():
    check('frozen source '+p,sha(p)==h)
check('frozen hand roles',sha(role['roles_path'])==role['roles_sha256'])
manifest_by_id = {x['clip_id']:x for x in manifest}
for code, item in old['words'].items():
    refs=item['references']
    check(code+' nine distinct TRAIN people',len(refs)==9 and sorted(x['signer_id'] for x in refs)==list(range(1,10)) and all(manifest_by_id[x['clip_id']]['role']=='train_candidate' for x in refs))
    check(code+' source and cached feature hashes',all(sha(x['sample_path'])==x['sample_sha256'] and sha(x['observation'])==x['observation_sha256'] for x in refs))
pairs=read(B/'ops/word100_multiref_20261014/TRAIN_PAIRS.json')
check('864 unique ordered pairs; different signers',len(pairs)==864 and len({(x['query_clip'],x['reference_clip']) for x in pairs})==864 and all(x['query_signer']!=x['reference_signer'] for x in pairs))
check('existing independent numeric audits passed',read(B/'ops/word100_multiref_20261014/INDEPENDENT_CHECK.json')['status']=='PASS' and read(B/'ops/word100_role_dtw_20261017/BANK_INDEPENDENT_RECOMPUTE.json')['status']=='PASS')

protected=set(Path(p) for p in {**old['code_sha256'],**role['code_sha256']})
protected.update([oldpath,rolepath,Path(role['roles_path'])])
protected.update((B/'config').glob('word100*.json'))
protected.update([B/'web/practice.html',B/'web/practice.js'])
protected.update(B/'src/wis_lab/service'/f for f in ['word100.py','word100_candidate.py','word100_t3f.py','word100_web_analysis.py'])
for seed in [20261013,20261014,20261015]:
    protected.add(B/f'ops/word100_body_boundary_20261017/training_C_{seed}_best.pt')
    protected.add(B/f'ops/word100_frozen_c_temporal_20261019/T3F_{seed}_best.pt')
protected.update(B/'ops/word100_phase_candidate_review_20261019'/f for f in ['CATALOG.json','RESULT_KO.html','CONCLUSION.json','FINAL_AUDIT.json'])
pins={str(p):sha(p) for p in sorted(protected)}
check('web C checkpoint identity',pins[str(B/'ops/word100_body_boundary_20261017/training_C_20261013_best.pt')]=='283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab')
check('web T3F checkpoint identity',pins[str(B/'ops/word100_frozen_c_temporal_20261019/T3F_20261013_best.pt')]=='b0faa0df5055204295c60b78c596a0b490cd63ef14f30225582a85ea39855846')

audit={
 'status':'PASS_METADATA_AND_STATIC_AUDIT' if all(x['passed'] for x in checks) else 'FINDING_REQUIRES_REVIEW',
 'at':now,'operator':'Codex local; separate metadata/hash check and reuse of earlier numeric audits; no independent agent or KSL expert',
 'current_bank':{'words':len(old['words']),'train_references':sum(len(x['references']) for x in old['words'].values()),'train_people':9,'ordered_pairs':len(pairs),'single_reference_words':len(read(B/'config/word100_analysis_references.json')['items']),'public_dev_previously_evaluated':36,'single_hand_roles':[{'word':k,'label':v['label'],'sides':v['required_sides']} for k,v in role['words'].items() if len(v['required_sides'])==1]},
 'available_data':{'train':len(train),'dev':len(dev),'train_count_histogram':dict(Counter(counts.values())),'all_word_same_label_train_ordered_pairs':sum(n*(n-1) for n in counts.values()),'all_word_dev_to_train_same_label_pairs':3*len(train),'train_people':list(range(1,10)),'dev_people':[11,12,13]},
 'implementation_findings':[
  'Multiple-reference median per hand, then required-hand mean; no nearest-reference-only scoring.',
  'Required-role bank has12words. Three one-hand labels are manually observed public-reference metadata, not an automatic100word role learner.',
  'C/T3F classifier observation and original DTW observation are intentionally different frozen paths. No learned classifier weight distance is calculated by DTW.',
  'Shared monotone DTW path uses arm/shape/orientation/wrist/face/bimanual groups. Bend and velocity alignment weights remain0; bend is nevertheless measured after alignment. This is a design limitation to examine, not evidence of a computation bug.',
  'Sampling up to96 nearest real-time observed frames, band.2, first-to-last ANY-hand observation, internal gaps retained; no semantic phase detector.',
  'MIN_REFS5, MIN_FRAMES8, paired metric support.5, alignment evidence.35. These are computational safeguards, not linguistic correctness boundaries.',
  'Both-hand average may hide a local/one-hand discrepancy; metric support is DTW-pair support, not true-time coverage.',
  'Rest/support/required semantics are not inferable merely from low motion or query absence. Generic auto role classification is not validated.'
 ],
 'checks':checks,'protected':pins,'new_training':0,'new_model_inference':0,'new_dtw_pair_evaluation':0,'user_mentor_recordings_read':0,'heldout14_18_raw_read':0,
 'numerical_validation':'Earlier saved independent audits reused; this turn verifies manifest, identity, source/cache hashes, pairing metadata, and reads code. No new prediction/accuracy claimed.'
}
write(U/'AUDIT.json',audit)
assert all(x['passed'] for x in checks), 'Preserve audit finding; do not update success state'

plan=read(U/'PROPOSED_PLAN.json')
report='''다중 참고·DTW 확인 결과와 다음 계획

결론
전 단어의 필수·선택 동작을 사람이 하나씩 정하는 작업은 보류한다. 현재 기능을 재사용하여 여러 공개 촬영자의 차이와 관측 부족부터 비교하는 것이 다음 순서다. 이번 작업은 감사·계획만이며 학습·모델 추론·웹 변경은 하지 않았다.

현재 확인한 사실
1. 다중 참고 DTW는 이미 있다. 연습12단어 각각 TRAIN9명, 총108영상이다. 한 사람의 동작을 나머지8명과 비교한864방향쌍으로 기준을 만들었다. 동일영상을 자기 자신과 비교해 기준을 낮추지 않았다.
2. 단일 참고 DTW/참고영상 목록은100단어이지만, 다중 참고 기준은12단어다. 여러 참고 중 가장 잘 맞는 한 명만 고르지 않고 손별 차이의 중앙값을 쓴다. 최대점수/최소점수만 고르면 정상 변형이나 오류를 숨길 수 있다.
3. 멀다·공항·별의 오른손 비교는 공개 영상 관찰로 지정한 역할이다. 나머지는 보수적으로 양손을 요구한다. 이것이 모든 수어의 필요손을 자동 판단하는 기능은 아니다. 가만히 있는 받침손도 의미를 가질 수 있다.
4. C/T3-F의 인식과 DTW는 별도 경로다. 현재 DTW는 이전 관측과 고정 참고은행을 유지한다. 따라서 C 인식 입력을 보완했다고 DTW의 관측까지 자동으로 바뀐 것은 아니다. 잘못 연결됐다고 단정할 문제는 아니고, 의도적으로 보존한 경로를 이제 별도로 비교할 지점이다.
5. 손가락 굽힘은 실제 계산되고 있다. 다만 정렬할 때 쓰는 특징 가중치에서 굽힘각·속도는0이고, 팔/손 모양 좌표·방향·위치 등으로 시간을 맞춘 다음 굽힘각 차이를 계산한다. '굽힘을 비교하지 않는다'는 뜻이 아니다. 이 설계가 구별 동작을 충분히 맞추는지는 검증 대상이다.
6. DTW는 시간 순서를 지키지만 한 장면을 다른 영상의 여러 장면과 대응시킬 수 있다. 그러므로 평균 차이가 작아도 필수 동작을 전부 했다는 뜻은 아니다. 현재는 처음~마지막 손 관측 구간을 쓰며 수어 의미 단계 분할은 하지 않는다.

제안하는 작업 순서
1) 공개100단어·899TRAIN영상으로 다중 참고 기준을 연구용 확장한다. 99단어9명/1단어8명. 원본·기존12단어 결과는 재사용·보존한다. 같은 사람을 참고에서 제외한7184방향쌍은 독립7184영상이라는 뜻이 아니다.
2) 기존 관측 경로와 C가 사용하는 관측 경로를 같은 특징/DTW 계산으로 비교한다. 참고와 질의 모두 같은 경로를 써야 한다. C의masked/sensor는 따로 비교한다. 구간·좌우·손 크기·가중치를 한꺼번에 새로 바꾸지 않는다. 관측 경로 효과만 먼저 확인한다.
3) TRAIN9명으로 참고 범위를 만들고 다른 DEV3명300영상으로 비교한다. 손별로 얼마나 측정됐는지, 비교 못한 이유, 사람 간 차이와 큰 국소 차이를 함께 보고한다. 나머지88단어의 필요손을 억지로 확정하지 않고 손별 비교부터 확장한다. 사용자에게 안 보이는 손을 자동 제외하지 않는다.
4) 인식 개선은 별도로 확인한다. C/T3-F의 저장된 상위5후보 각각을 여러 TRAIN 참고와 비교해 실제 정답을 더 잘 구분하는지 먼저 본다. 이후 확인할 하나의 연구 후보는 충분한 공통 근거가 있을 때만 DTW 중앙거리로5후보를 재순위하는 방식이다. 근거가 부족하면 기존순위를 유지한다. 연습으로 선택한 정답은 이 인식 계산에 넣지 않는다. 정답이원래상위5밖이면 이방법으로는구할수없다.
5) 좋아진 오답과 새로 틀린 정답을 모두 세고, 세 촬영자·두 관측 조건에서 회귀를 확인한다. 도움이 없다면 결합하거나 학습을 추가하지 않는다. 추가학습이필요하면 결과를본뒤 한가지변경만 계획하고 사용자확인후시작한다.

어떤 결과를 개선이라고 부를 것인가
- 동작비교: 정상 사람끼리의 차이가 안정적으로 계산되고, 실제 비교한 손/구간 근거가 늘어야 한다. 관측을버려평균만낮춘것은개선이아니다. 최소참고수·8프레임·50%근거등현재안전조건은출발점으로유지한다.
- 인식: 동일영상에서 Top1/Top5, 정확도%·변화%p, 고친오답/새회귀, 촬영자별 결과와 지연시간을 보고한다. 고정3seed 저장결과 각각과 비교하여 좋은seed만 제시하지않는다. 단순 점수상승은정답수증가가아니다.
- 이번DEV는 이미반복사용한개발검증이다. 새사람일반화입증이라고부르지않는다. 미사용ID14~16 유효299영상은최종후보고정후평가용으로남긴다. ID17/18은이미평가에사용했다.
- 정상참고범위는수어의합격선이아니다. 모든단어의수기단계표는필요하지않지만, '필수동작생략/잘못된반복/몇도고쳐라'라는교육용판정에는여전히별도평가근거가필요하다. 대표유형/모호한사례/교정문구위주표본검토로분리한다.

범용성
단어이름을보고규칙을만들거나사용자촬영에맞추지않는다. 라벨과여러사람자료를추가하면같은처리절차를다른단어에도쓸수있게설계한다. 다만현재증거는100단어이며, 지금모델이미학습수어까지알아듣는다는뜻은아니다. 여러참고를활용해도성능향상이자동보장되지는않는다.

이번 확인 범위
기존 코드/설정/공개 저장 결과·해시·자료분리 확인. 이전 수치 검산을 재사용했고 새 성능평가는 하지 않았다. C/T3-F·웹·DTW·이전후보화면 유지, 개인/멘토촬영 미접근, 추가학습0. 담당은Codex이며 별도agent/수어전문가 검증이라고 주장하지 않는다.

상세 조건: PROPOSED_PLAN.json. 실제 확인 근거: AUDIT.json. 먼저1~2단계의공개다중참고·관측경로검증계획을함께확인하고진행한다.

외부 1차 근거
DTW의다대일정렬과제약: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWvariants.html
정상수어동작분포와사람평가의구분: https://arxiv.org/abs/2408.10073
본프로젝트성능보장이나한국수어교정기준으로직접대입하지않는다.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')

next_action=plan['next_single_action']
state=read(statepath)
state.update(current_stage='WORD100_MULTIREF_DTW_AUDITED_PLAN_REVIEW',updated_at=now,updated_date_kst=now[:10],next_single_action=next_action)
state['word100_multiref_dtw_plan_20261019']={
 'at':now,'status':'COMPLETE_AUDIT_PLAN_AWAITING_USER_REVIEW',
 'report':str(U/'RESULT_KO.txt'),'audit':str(U/'AUDIT.json'),'plan':str(U/'PROPOSED_PLAN.json'),
 'current_multiref_words':12,'current_reference_clips':108,'single_reference_words':100,
 'proposed_reference_train':899,'proposed_development':300,'proposed_new_pairs_executed':0,
 'all_words_manual_phase_review':'deferred','new_training':0,'new_model_inference':0,'user_recordings_read':0,
 'existing_models_web_dtw_preserved':True,'next_single_action':next_action}
write(statepath,state)
context=docs[0].read_text('utf-8')
title='### 2026-10-19 공개 다중참고·DTW 감사와 확장 계획 — 전단어 수기단계 보류'
entry=f'''{title}
갱신 {now}. ops/word100_multiref_dtw_plan_20261019. 사용자 요청의 기존기능 감사·구체계획 완료, 실행/추가학습 전 검토대기.
- 다중참고는12연습단어×TRAIN9명=108영상,864방향쌍·자기제외중앙값·DEV36 기존검산재사용. 단일참고목록은100. 역할은공개관찰3단어오른손/그외보수적양손으로100단어자동필요손학습아님.
- 실제C/TF인식관측과DTW원관측은별도고정경로. 입력개선이DTW까지자동전파되지않음. 정렬은팔/모양/방향/위치등,굽힘각은정렬후측정(정렬가중치0). 오류단정/계수변경없음.96시간표본·첫끝관측·내부빈구간유지·DTW대역.2·최소5참고/8프레임/50%항목근거 유지.
- 계획:100단어899TRAIN다중참고→기존관측대C관측(참고/질의동일) 비교→DEV300(ID11~13)정상차이/보류/회귀→저장Top5와공개참고의재순위가능성진단. 인식계산에목표단어입력금지,근거부족시기존순위,단어별예외/가중치탐색없음. 별도후보이며운영연결없음.
- 전단어수기단계보류,자동후보화면보존. 필요손미확정단어는좌우별비교,낮은움직임/사용자미검출을불필요손근거로하지않음. 정상범위≠합격선. 교정평가는대표유형·모호한사례표본으로분리.
- TRAIN99단어9명/1단어8명;예정7184방향쌍/조건,DEV정답참고2697쌍/조건은아직미실행. DEV기사용,ID14~16유효299미사용보존,17/18새맹검아님. C/TF/웹/DTW보호해시유지.학습0/추론0/새DTW0/개인멘토접근0. 별도agent검증아닌Codex정적감사·해시검산. AUDIT/PROPOSED_PLAN/RESULT_KO/FINAL_AUDIT.

'''
for header,insert in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry),('## 9. 다음 단일 작업',title+'\n'+next_action+'\n\n')]:
    assert context.count(header)==1
    context=context.replace(header,header+'\n\n'+insert,1)
docs[0].write_text(context,'utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}

현재: 전단어 수동 단계 검토를 보류하고 기존 다중참고·DTW 감사 및 공개 여러사람 비교 계획을 작성했다. 사용자 계획 확인 대기이며 추가 실험/학습은 시작하지 않았다.

- 현재단위 research/front_keyphase_v1/ops/word100_multiref_dtw_plan_20261019. RESULT_KO.txt/AUDIT.json/PROPOSED_PLAN.json/FINAL_AUDIT.json.
- 기존다중참고12단어108TRAIN/864방향쌍/DEV36. 단일참고100단어.9명중앙값과자기제외기준기구현.3한손단어역할은공개관찰메타데이터,100단어자동역할추정아님.
- C/TF인식관측과DTW원관측별도유지. 이번발견을새버그로단정하지않음. 굽힘각은정렬후측정하며정렬가중치0,순서보존DTW도많은장면을한장면에대응가능. 의미단계/수어합격판정아님.
- 승인받을첫단위:899TRAIN(ID01~09,99단어9명+1단어8명)다중참고확장,기존대C관측은행/질의일치비교,DEV300(ID11~13)정상차이/근거/회귀검증. 예정7184TRAIN방향쌍/조건,2697정답DEV쌍/조건. 아직계산없음.
- 다음인식보조는기존C/TF저장상위5+공개다중참고구별력진단. 목표단어를인식입력으로쓰지않고근거부족시기존결과. 상위5밖정답구제불가. 단어별수기규칙/최소거리한사람골라맞추기/혼합계수탐색금지. 자세한후보계약은2단계결과확인뒤동결,이번은실험READY아님.
- 전단어수기단계표불필요;기존315구간화면은진단용보존. 나머지88단어는손별측정/근거를보고,가만히있거나안보인다고불필요손확정금지. 실제교정의의미정확성은표본/전문근거필요.
- 기존DEV는반복선택에사용,새맹검아님. ID14~16유효299원시미접근/보존. ID17/18은이미평가. C·T3-F모델6개/웹/DTW/원본/과거결과보존. 사용자·멘토는실사용테스터로촬영읽기/학습/튜닝/선택금지.
- 감사방식:Codex정적코드확인+공개기존결과재사용+목록/해시검산. 독립agent/수어전문가검토아님. 새학습0/모델추론0/DTW재평가0/개인접근0. 이번실행은감사기록도구만,계속진행할학습프로세스없음.

다음 한 작업: {next_action}
'''
docs[1].write_text(handoff,'utf-8')
assert all(sha(p)==h for p,h in pins.items())
for p in U.glob('*.json'):read(p)
assert read(statepath)['next_single_action']==next_action
write(U/'FINAL_AUDIT.json',{'status':'PASS_AUDIT_AND_DOCUMENTATION_ONLY','at':now,'checks_passed':len(checks),'protected_count':len(pins),'protected_unchanged':True,'proposal_not_executed':True,'models_trained':0,'inference_runs':0,'dtw_pairs_evaluated':0,'user_recordings_read':0,'documents_synchronized':[str(x) for x in docs],'report':str(U/'RESULT_KO.txt'),'new_agent_review':False})
print(json.dumps({'status':'PASS','checks':len(checks),'protected':len(pins),'report':str(U/'RESULT_KO.txt'),'new_training':0},ensure_ascii=False))
