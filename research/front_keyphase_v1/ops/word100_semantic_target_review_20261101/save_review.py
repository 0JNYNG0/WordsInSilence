from pathlib import Path
import json,hashlib,datetime
N=Path(__file__).resolve().parent;O=N.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=O/'word100_native_motion_recheck_20261101'
audit=read(prior/'full24_native/FULL24_AUDIT.json')
failed=[k for k,v in audit['checks'].items() if not v]
assert failed==['R_hold_eligible_both_classes','L_hold_eligible_both_classes']
support=read(prior/'support_inventory/SUPPORT_RESULT.json')
left=[r for r in support['rows'] if r['target_hand']=='L']
assert len(left)==6 and sum(r['eligible_agreed_absence'] for r in left)==1
unknown=next(r for r in audit['rows'] if r['sample_id']=='N008_W1')['predicates']
assert unknown['L_down']['a'] is None and unknown['L_down']['b'] is None
assert unknown['L_hold']['a']==unknown['L_hold']['b']==1
code=(O/'word100_interval_motion_20261101/comparison.py').read_text(encoding='utf-8-sig')
assert 'def masked_loss(' in code and 'torch.where(mask, raw * w' in code
profiles=read(O/'word100_linguistic_feedback_pipeline_20261027/FEATURE_PROFILES.json')['profiles']
sign=next(r for r in profiles if r['word_id']=='WORD0003')
assert any('엇갈리게 돌린다' in e['description'] for e in sign['candidate_entries'])
protected=read(O/'word100_core_feature_split_20261101/SELECTION.json')['protected']
assert all(sha(p)==h for p,h in protected.items())
sources=[prior/'full24_native/FULL24_AUDIT.json',prior/'support_inventory/SUPPORT_RESULT.json',
 O/'word100_interval_motion_20261101/RUBRIC.json',O/'word100_interval_motion_20261101/PILOT_GATES.json',
 O/'word100_interval_motion_20261101/comparison.py',O/'word100_linguistic_feedback_pipeline_20261027/FEATURE_PROFILES.json',
 O/'word100_linguistic_feedback_pipeline_20261027/source_audit/CURRENT100_SOURCE_CANDIDATES.json',N/'PLAN_KO.txt']
now=datetime.datetime.now().astimezone().isoformat()
out=dict(status='PLAN_REVIEWED_NO_TRAINING_OR_LABEL_CHANGES',at=now,reviewer='Codex root, local code/results and official source review; no independent agent or human validation in this planning turn',
 user_request='Review whether incidental drift is blocking meaningful sign feedback; propose next direction',
 finding='Global readiness for all down/hold-presence heads is an overbroad prerequisite for meaningful handshape/relation/sequence learning. Old failure remains valid for its original task.',
 actual_last_failed_gates=failed,micro_drift_single_case_is_not_direct_last_failure=True,
 existing_partial_label_mask=True,old_frozen_results_preserved=True,protected_model_production_files=32,
 next_action='Prepare a prospective common stage/feature contract tied to verified public descriptions and whole-action context; do not execute old FIT180 micro-down/hold census as default prerequisite.',
 prospective_contract_not_yet_implemented=True,training=0,inference=0,label_changes=0,html_created=False,
 user_private_data_access=False,user_interpretation_used_as_training_label=False,
 source_caveats=['Sobok lexical guide/educational tolerance not verified in available source mapping.',
 'Source description for sign language supports alternating rotation but does not prescribe exactly two cycles.',
 'Physical observation uncertainty and semantic irrelevance are different; missing observations cannot be declared irrelevant.',
 'No new learned performance or human educational correctness established.'],
 evidence={str(p):sha(p) for p in sources},
 web_evidence=dict(title='일상생활 수화(4) 표준화 연구',printed_page=233,pdf_page=247,
 url='https://www.korean.go.kr/common/download.do?book_seq=355&c_file_name=0736b8dc-bde5-435b-84e8-0ad2bc4fea3f_0.pdf&downGubun=bookDataView&file_path=bookData&o_file_name=%ED%8A%B9%EC%88%98%EC%96%B8%EC%96%B4-02-22.pdf',
 direct_dictionary_tool_open='unavailable; primary institutional report text confirmed'))
with (N/'REVIEW.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
state=read(O/'STATE.json');state['current_stage']=out['status'];state['updated_date_kst']=now
state['next_single_action']=out['next_action'];state[N.name]=out
(O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
context=(ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf-8-sig')
insert8='''### 2026-11-01 목표 재정렬 검토 — 미세 이동 보조 판정으로 전체 학습을 막지 않기

`word100_semantic_target_review_20261101`: 사용자소복자연흔들림/수어교대회전지적에코드·저장결과·공식설명검토. 원24최종FAIL은양손hold적격음성0 두조건이며,N008L_down unknown하나가직접원인아님. 다만약0.4초창의미세down/hold네항목준비를전체핵심학습관문으로묶은것은목표대비과도. 코드에는항목별target_mask 제외기능이미있음. 원FAIL/라벨/모델은그대로두고별도다음계약을주요손모양·양손/신체관계·움직임패턴·순서중심으로재설계한다. 직전FIT60/180 micro-down/hold전수판독계획은현재우선순위에서내림. 국립국어원보고서인쇄233쪽은수어의양검지/방향/상하배치/교대회전을설명,정확히2회조건없음. 소복공식허용규칙미확인,사용자의공개영상해석을학습라벨로직접전환X. 새학습/추론/라벨변경/HTML0,보호32불변. PLAN_KO.txt/REVIEW.json에근거·제안·한계구분. 구현·성능개선완료아님.

'''
insert9='''### 2026-11-01 다음 단일 작업 — 주요 단계와 공통 특징의 새 준비계약

사용자검토용계획은 `word100_semantic_target_review_20261101/PLAN_KO.txt`. 다음실행은전체주요동작문맥과자료연결을바탕으로손모양유지/변화,지지손과움직이는손의역할후보,양손·신체관계,이동/회전/교대/반복/순서의공통특징계약부터. 관측/시간문맥/설명근거·미확인을분리하고목표단어로관측정답을강제X. 미확인항목만mask,단충분한정답지원은학습표적별로필수. 모든micro-down/hold반대사례완비를전체선행조건으로X. 원24FAIL·지원12·기존결과불변,새표적/분할/통과조건은학습전고정. 정상흔들림/속도/교차오경고와분명한핵심차이탐지·비교coverage를함께평가. 단순순서삭제시험은기술검증,교육적오답진단과구분. 이번턴은검토/계획만이며새학습·라벨링미실행. C/T3F/운영/개인자료보존.

'''
for heading,entry in [('## 8.',insert8),('## 9.',insert9)]:
    pos=context.index('### ',context.index(heading));context=context[:pos]+entry+context[pos:]
(ROOT/'WordsInSilence_PROJECT_CONTEXT.md').write_text(context,encoding='utf8')
handoff=f'''# WordsInSilence 최신 인수인계

현재 {N.name}, {out['status']}. 최신사용자는소복의자연스러운주먹흔들림/수어양손교대회전을근거로미세판정이핵심학습을막는지검토와다음방향요청. 이번턴은코드·공식문서·저장결과검토/계획기록만. active execution/agent 없음. PLAN_KO.txt/REVIEW.json참조.

핵심판단: 약0.4초창의down/hold존재부재완비를주요손모양·관계·순서학습의필수관문으로묶은것은과도. 기존comparison.py에는항목별mask손실제외기능이이미있다. 미세물리변위unknown은그대로남기되전체핵심학습을차단하지않는새계약을만든다. 단정답품질/각학습표적의양음성지원·분리검증은유지. 지난동결FAIL을PASS로고치거나미확인을0으로전환하지않음. FIT60/180 micro-down/hold전수판독계획은당장실행하지않음.

정확한이전결과: 원native24의최종FAIL은R/L_hold_eligible_both_classes 2개(범위내음성0). 소복N008L_down unknown하나가직접차단원인아님;L_hold=1합의. 원24 96항목75일치/0반대/4한쪽unknown/17양쪽unknown. 별도지원12에서왼손6중유지없음1(수어S037/ID03),유지있음2,unknown·판독차이3. 전체왼손자료가1개라는뜻아님. 기존native검증완료/미학습기록은불변.

공식근거: 국립국어원 '일상생활 수화(4) 표준화 연구' 인쇄233쪽/PDF247쪽 수어설명에양검지·손등방향·상하배치·엇갈린회전. 정확히2회규정없음. 소복의정확한사전동작규칙·허용경계는현재연결자료로미확인. 사용자관찰은문제제기근거이지새학습정답아님. 보이지않는중요동작을작음/모호함이유로무시하지않음.

다음: 전체주요동작문맥에서공통단계/특징계약→해당구간의확실한정답·부분mask→기존SignRep/관절+작은시간별판정부의준비된표적만비교→DTW동일단계대응후핵심차이/정상변이오경고·coverage검증. 실제교육오답과합성기술시험구분. 이번에이구현/학습은시작안함. 새로운모델검색/다운로드가다음작업아님.

C·T3F/운영웹·DTW/원본·기존결과보존,보호32SHA확인. 개인·멘토자료접근/학습/튜닝/선택X. 학습/추론/새라벨0,HTML없음. root만§8§9/STATE/HANDOFF수정.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
assert read(O/'STATE.json')[N.name]['training']==0
assert all(sha(p)==h for p,h in out['evidence'].items())
print(json.dumps(dict(status=out['status'],verified_original_failures=failed,protected_files=32,sync='PASS'),ensure_ascii=False))
