"""Trace saved failures and validate the next label contract. No model execution."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;L=U.parent/'word100_event_guide_linkage_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
cases=[]
for seed in [42,43,44]:
 for name in ['NIA_SL_WORD0739_REAL13_F_omit_final_event','NIA_SL_WORD1133_REAL13_F_full_normal','NIA_SL_WORD1133_REAL13_F_reversed_core_auto_order']:
  p=U/'fullpath'/str(seed)/name/'after.json';d=read(p)
  cases.append(dict(seed=seed,case=name,path=str(p),sha256=sha(p),candidate_segments=[{k:s[k] for k in ['a','b','context','confidence','core_state','core_probabilities']} for s in d['triplet_segments']],movement=d['guide_link_direction_checks'],order=d['guide_link_order_checks'],conflict_veto=d['guide_link_vetoes']))
contracts=read(L/'GUIDE_CONTRACTS.json')['words'];labels=read(U/'SEGMENT_LABELS.json')['rows'];primary=read(U/'RESULT.json')['reviewed_segments']['42']
# Declarative relationship, not a word-ID-specific exception.
families={}
for word,cc in contracts.items():
 byid={c['id']:c for c in cc}
 for c in cc:
  root=c.get('parent',c['id']);families.setdefault(root,set()).update(c['contexts'])
families={k:sorted(v) for k,v in families.items()}
affected=[r['clip_id'] for r in labels if {4,5}<={s['context'] for s in r['segments']}]
write(U/'CAUSAL_AUDIT.json',dict(scope='Post-result saved traces, no new training/score tuning; mechanistic observations, not proof of a unique cause.',cases=cases,confirmed=[
 'Old false context at WORD0739 REAL13 source71..82 is still predicted upright-forearm in the new segment head; prior conflict guard keeps it uncertain.',
 '24 visually reviewed windows: old frame-context pooling23; event model22/23/23 across seeds. More temporal context did not establish generalization gain.',
 'Normal and reversed WORD1133 REAL13 both have partial movement candidates and progress-core disagreements; the progress-only veto cannot reliably distinguish them.',
 'All9 FIT examples containing along-forearm and near-wrist have two distinct context classes, although guide marks near-wrist as an optional child of along-forearm.',
 'Core head uses central body/wrist relations only; this is not proof of complete correct handshape or entire movement.'
 ],hypotheses_to_test=[
 'Separate event identity from instantaneous position and train event-family coverage through arrival; current context-class changes may truncate a single action.',
 'Use direct event entry/middle/exit evidence plus normal/partial/reverse contrasts; do not replace completion with core-pose confidence.',
 'For same-body-pose ambiguity, combine observed handshape evidence as a soft clue, preserving attempted-action identity even for wrong handshape; do not require matching expected shape.'
 ],families=families,fit_parent_child_examples=affected,primary_context_errors=primary['errors'],no_private=True))
# A concrete prospective supervision schema; not fabricated annotations or a newly trained model.
def contract(action,phase,observed,direction=None,shape=None):
 return dict(action_family=action,phase=phase,observed=observed,direction=direction,shape=shape)
examples=[contract('along_opposite_forearm','entry',True,1,'open'),contract('along_opposite_forearm','motion',True,1,'open'),contract('along_opposite_forearm','exit',True,1,'open')]
wrongshape=[dict(x,shape='closed') for x in examples];reverse=[dict(x,direction=-1) for x in examples[::-1]];masked=dict(examples[-1],observed=False)
checks=[dict(name='arrival_stays_in_same_event',passed=len({x['action_family'] for x in examples})==1),dict(name='wrong_shape_keeps_attempt',passed=[x['action_family'] for x in wrongshape]==[x['action_family'] for x in examples]),dict(name='reverse_not_forced_correct',passed=[x['phase'] for x in reverse]==['exit','motion','entry']),dict(name='unseen_exit_is_unknown',passed=not masked['observed']),dict(name='parent_derived_from_contract',passed=families['brush_arm']==[4,5]),dict(name='nine_fit_sources',passed=len(affected)==9)]
assert all(x['passed'] for x in checks)
write(U/'NEXT_LABEL_CONTRACT_CHECK.json',dict(passed=True,checks=checks,examples=examples,wrong_shape=wrongshape,reversed=reverse,masked_exit=masked,scope='Schema/unit checks only, not trained feedback performance or new RGB ground truth.'))
plan=[
 dict(step=1,title='자세 이름과 하나의 행동을 분리해 정답 연결 수정',reason='이동 중→손목 도착이 서로 다른 동작 종류로 분리돼 있음. 앞쪽 얼굴 자세와 마지막 팔 자세도 몸 좌표만으로 겹침.',work=['공개FIT에서 행동 단위의 시작/진행/끝 범위와 설명 문장을 연결한다. 순간 위치 표시는 별도 유지한다.','도착·접촉 후보가 앞 행동의 자식/끝 상태인지 공통 관계로 표시한다. 단어ID 예외는 만들지 않는다.','모양과 방향이 틀려도 시도한 행동 자체는 남기는 대조 정답을 둔다. 가림은 확인 불가, 실제 중단은 별도.'],gate='연결한 각 행동에 실제 관측 근거/원본SHA/시점/검토자가 존재. 추정 경계와 실제 확인 경계 구분. 정상·역방향·중단·가림을 같은 정답으로 섞지 않음.'),
 dict(step=2,title='어느 단계에서 막히는지 분리 시험',reason='정확한 구간24개에서도 성능 향상이 없고 자동 구간/완료 판정에도 문제가 함께 남음.',work=['직접 확인한 구간과 자동 구간을 나란히 입력해 행동 식별·방향·끝 도달을 각각 비교한다.','같은 자세라도 들어오는 움직임과 빠져나가는 움직임이 다른 쌍을 검증한다.','관측된 손모양은 보조 증거로 결합하되 목표 모양과 다르다는 이유로 행동 구간을 삭제하지 않는지 검사한다.'],gate='수동구간에서도 틀리면 정답/특징/학습 보강을 우선. 수동만 맞고 자동이 틀리면 구간 연결 수정. 끝 모습만 맞고 순서를 틀리면 사건 관계 손실 보강. 원인별로 필요한 변경만 수행.'),
 dict(step=3,title='공통 신경망을 한 차례 고정 비교로 추가 학습',reason='현재100단어는 공통 보조학습이고, 상세 사건 정답은4표현36영상251파생구간이라 동등하지 않음.',work=['899개 공통학습 유지. 검증된 사건 정답으로 역할·진행·방향·끝 관측을 분리해 학습한다.','원본을 정상 정답으로, 학습용공개자료에서 만든 역방향/중단을 별도 대조로 사용한다. 속도차이는 정상성 유지.','기존 손모양·방향과 C/T3F 고정. 한 번 정한 설정·seed·분모를 평가 결과로 다시 고르지 않는다.'],gate='기존 정상12/속도24/교차3 보존, 기존모양·방향오류9 유지, 남은 역순/누락을 실제 근거와 함께 개선. 같은Codex검산과 사람전문가 검토는 구분.'),
 dict(step=4,title='노출 이력을 확인한 추가 평가 후 화면 적용 판단',reason='이번78조건은 반복 사용한 개발자료이고3seed는 새사용자3명이 아님.',work=['후속 공개수행자/파일SHA의 기존사용·중복 이력을 감사하고 새평가 여부를 정확히 표시한다.','오류 교정과 관측 부족 안내를 구분하고 실제 구간과 수정 내용을 짧게 표시한다.','개선이 없으면 후보를 교체하지 않는다. 정상 회귀 없이 여러 표현에서 이득이 확인된 후보만 별도 웹 검증으로 넘긴다.'],gate='학습/튜닝과 평가 분리, 사용자자료0, 기존운영교체 없음. 전체 인식·세부 피드백·전체지연을 별도 보고.')]
write(U/'NEXT_PLAN.json',dict(status='REVIEWED_NOT_EXECUTED',steps=plan,contract_check=str(U/'NEXT_LABEL_CONTRACT_CHECK.json'),no_additional_training_started=True))
print(json.dumps(dict(cases=len(cases),fit_parent_child=len(affected),contract_checks=len(checks))))
