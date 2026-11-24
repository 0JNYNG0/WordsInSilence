import json, hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

key=U.name
assert not (U/'CLOSURE.json').exists(), 'Already closed'
F=read(U/'FROZEN.json');V=read(U/'VERIFIED.json');H=read(U/'HTML_RENDER_VERIFIED.json')
assert V['status']=='PASS_SAVED_RESULTS_AND_CONTRACTS' and H['status']=='PASS'
for p,h in {**F['source_pins'],**F['protected']}.items(): assert sha(p)==h,p
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
summary='''表情ではなく、手・腕と顔部位の関係だけを扱う。'''
summary='''표정은 현재 범위에서 완전히 제외했다. 공개 6표현×TRAIN 기록 ID01/02/03=18영상 2,285프레임의 기존 몸·손·얼굴 관측을 재사용했다. 손끝5·손목(손/몸 모델 분리)·손바닥·팔꿈치·팔 구간의 10종 위치 근거를 이마·좌우 광대 부근·입·턱·코와 연결했다. 비교 대상 자신을 뺀 다른 2명의 참고를 각각 유지하여 얼굴 부근의 투영 겹침이 두 참고에서 지지되는지, 참고끼리 다른지, 둘과 다른지, 관측 부족인지를 구분한다. 화면상 겹침은 실제 접촉이나 잘못된 수어의 확정이 아니며 두 참고의 최소~최대도 정답 허용범위가 아니다.
이전 45프레임은 시각적으로 확인된 45오류가 아니라 주석 구간 불일치였다. 33개는 참고 주석 경계에서 0.1초 이내, 44개는 0.2초 이내. 연속 13구간의 처음·가운데·끝을 검토했고, 믿다 일부의 단계 진행 불일치 의심은 남았다. 주석 경계를 억지로 맞추는 목표로 바꾸지 않았다.
관측된 팔꿈치·손목 이동 방향을 0.25 더한 후보 D를 한 번 비교했다. 36방향쌍의 동일 1,584주석 동작프레임에서 불일치 B143→D141, 누락0; 정확한 시간 대응을 아는 54관절 재표본화 시험의 영상별 평균 오차 평균은 3.23→3.92ms로 소폭 악화(25악화/1개선/28동일). 개선 입증 부족으로 D미채택, 기존 B정렬 유지. TCN 인식률 시험이 아니다.
75출처핀/32운영핀 보존. 저장 관측·마스크·사람 제외·손누락 때 팔근거 분리·얼굴없음·이동미관측간격·64판정조합 및 저장72교차지표/108시간대응지표를 별도 검산했다. 16이미지 보고서 데스크톱/390px 렌더 PASS. 같은 Codex 별도검산이며 독립 사람 정답 검증은 아니다. 실제 좌우/손모양 정답이나 얼굴 가림 원인 검증 완료를 주장하지 않는다. 기존 C·T3-F/운영 웹·DTW 변경0, 학습0, 사용자·멘토자료 접근0. 연구용 관계 측정과 원인 구분은 완료, 자동 수어 교정은 아직 미완료.'''
next_action='''완료한 word100_face_relation_alignment_20261027 입력·측정·B정렬·검산을 재사용한다. 더 많은 검출 모델이나 주석 경계 가중치 탐색을 추가하지 않는다. 손 접근·방향 전환 등 영상에서 확인 가능한 단계 대응을 작은 공개 자료에서 확인하고 정렬이 불확실한 시간대를 분리하는 것이 다음 단일 작업이다. 같은 얼굴 부위 주변의 자연변이와 다른 부위 관계를 구분하되, 시간 정렬/손·얼굴 관측 불확실성을 먼저 표시한다. 두 참고의 범위를 언어적 정답 허용범위로 자동 승격하지 않는다. 검증된 관계 측정의 별도 진단 웹 연결을 계획하되 기존 C·T3-F/운영 교체·TCN재학습은 하지 않는다. 표정 제외. 사용자·멘토촬영 학습·튜닝·선택 금지. 현재 활성 실행 없음.'''
entry=dict(status='PUBLIC_RELATION_DIAGNOSTICS_VERIFIED_DIRECTION_CANDIDATE_NOT_ADOPTED',at=stamp,summary=summary,result=str(U/'RESULT_KO.html'),next=next_action,public_scope=dict(word_types=6,recorded_ids=['01','02','03'],videos=18,frames=2285),alignment='B unchanged; D not adopted',training=False,expressions=False,operating_changed=False,user_or_mentor_access=False,implementation='Codex local',verification='Same Codex separate saved-result/scalar checks; not independent human truth',active_execution=None)
state_path=ROOT/'research/front_keyphase_v1/ops/STATE.json'
state=read(state_path)
assert key not in state
state['updated_date_kst']=stamp;state['current_stage']=entry['status'];state['next_single_action']=next_action;state[key]=entry
dump(state_path,state)
cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
txt=cp.read_text(encoding='utf-8-sig')
assert key not in txt
title='### 2026-10-27 손·팔–얼굴 관계 비교와 동작 시점 재검토'
evidence=f'근거: research/front_keyphase_v1/ops/{key}/RESULT_KO.html, VERIFIED.json, BOUNDARY_AUDIT.json, REFERENCE_VIEWS.json, HTML_RENDER_VERIFIED.json.'
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
    assert txt.count(header)==1
    txt=txt.replace(header,header+'\n\n'+title+'\n'+stamp+'\n'+body+'\n'+evidence+'\n',1)
cp.write_text(txt,encoding='utf-8')
handoff='# WordsInSilence 최신 인수인계\n\n'+stamp+'\n\n'+summary+'\n\n다음 단일 작업: '+next_action+'\n\n구현: research/front_keyphase_v1/src/wis_lab/core/hand_face_evidence.py. 새 모델 다운로드·추론 없이 기존 공개 관측 사용. 원본 보존, 첫 audit numpy.int64 직렬화 실패와 정정은 before/first_failure.txt 및 before/audit_json_failure.py에 보존.\n'+evidence+'\n'
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
review=read(U/'VISUAL_REVIEW.json')
review['reference_examples']=['냉커피: 세 공개 수행 모두 입/코 근처로 오른손 접근. 두 참고 투영 지지라도 세부 손모양 정답을 확정하지 않음.','별: 비교/두 참고의 얼굴 주변 손 위치가 다름. 영상의 단계·수행 차이를 분리하기 전 오답으로 채점하지 않음.','별: 한 수행 손목이 코 앞에 겹치지만 다른 두 수행은 다른 위치. 정렬 오차/수행 차이/검출 영향을 구분하지 못해 review 상태 유지.']
review['render_review']='Codex viewed all3 reference composites and desktop/mobile screenshots. Image loading16 and disclosures/link existence checked by local headless Chrome.'
dump(U/'VISUAL_REVIEW.json',review)
assert read(state_path)[key]['summary']==summary
for p in [cp,ROOT/'NEW_CHAT_HANDOFF.md']: assert '3.23→3.92ms' in p.read_text(encoding='utf-8')
files=[p for p in U.rglob('*') if p.is_file() and 'before' not in p.parts]
files += [ROOT/'research/front_keyphase_v1/src/wis_lab/core/hand_face_evidence.py',cp,ROOT/'NEW_CHAT_HANDOFF.md',state_path]
dump(U/'CLOSURE.json',dict(status='COMPLETE_RESEARCH_UNIT_NO_DEPLOYMENT',at=stamp,protected_pins_verified=len(F['protected']),source_pins_verified=len(F['source_pins']),artifact_and_document_pins={str(p):sha(p) for p in files},active_execution=None,execution='audit.py; run.py; verify.py; report.py; check_report.cjs; close.py; all exit0 after preserved serialization repair',decision='Retain B; use new evidence only as research diagnostics',limitations='No expression, no linguistic correctness/contact certification, no recognition accuracy gain, no independent human annotations'))
print('CLOSURE_PASS',len(files),'artifacts/code/docs')
