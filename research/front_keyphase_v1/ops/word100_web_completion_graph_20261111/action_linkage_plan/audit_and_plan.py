"""Audit existing source hierarchy before learning pair relations; preserve prior evidence."""
from pathlib import Path
import sys, hashlib, itertools, collections
W=Path(__file__).resolve().parent; U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 out=W/'COVERAGE_AUDIT.json'; assert not out.exists()
 p=U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json'
 source=read(p); pairs=[]; counts={}
 for r in source['rows']:
  assert r['role']=='fit'; seen=set()
  for s in r['sheets']:
   assert s['viewed'] and sha(s['path'])==s['sha256'];seen.update(s['frames'])
  groups={e:g['group_id'] for g in r['groups'] for e in g['event_ids']}
  points=[]
  for e in r['events']:
   for f in sorted(seen):
    if e['onset'][1]<f<e['offset'][0]:points.append((f,e['event_id'],groups[e['event_id']]))
  for a,b in itertools.combinations(points,2):
   label=0 if a[1]==b[1] else 1 if a[2]==b[2] else 2
   pairs.append(dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],source_sha256=r['source_sha256'],split='probe' if r['person'] in [8,9] else 'train',frames=[a[0],b[0]],events=[a[1],b[1]],groups=[a[2],b[2]],label=label))
 for split in ['train','probe']:
  counts[split]={}
  for c in range(3):
   rr=[r for r in pairs if r['split']==split and r['label']==c]
   counts[split][str(c)]=dict(pairs=len(rr),source_clips=len({r['clip_id'] for r in rr}),words=sorted({r['word_id'] for r in rr}),people=sorted({r['person'] for r in rr}))
 oldp=U.parent/'word100_autonomous_localization_20261111/FIT_LABELS.json';older=read(oldp)
 extra=[dict(clip_id=r['clip_id'],axis=r['axis'],events=len(r['events']),sheet=r['sheet'],limitation='Atomic movements already reviewed, but parent grouping/concurrent feature relation not separately established.') for r in older['rows'] if r['clip_id'].split('_')[2] not in ['WORD0739','WORD1133','WORD1567','WORD2523']]
 write(out,dict(at=stamp(),status='PAIR_THREE_CLASS_TRAINING_NOT_READY',source=str(p),source_sha256=sha(p),class_names=['same atomic motion','different motions in same parent action','different parent actions'],counts=counts,pairs=pairs,additional_existing_sources=extra,older_source_sha256=sha(oldp),training_allowed=False,reason='Repeat-parent class occurs only in WORD1567; distinct-parent class only WORD0739. This is word/class confounding, not proof that a new classifier will generalize. Do not relabel older reviewed atomic events as parent relations without source review.',scope='Counts are pairs of directly reviewed sampled frames in24 source clips, not independent videos. No full/upper duplication, no dense unviewed truth, no required semantic repetition count.',reviewer='root Codex arithmetic/source audit; no independent human/KSL review'))
 nodes=[
 dict(id='A',title='一つの動作の根拠を固定',title_ko='한 행동의 범위와 근거 정리',status='PARTIAL',work=['원본 시간의 시작·진행·전환·끝/유지를 구분하고 불확실 경계는 범위로 보존','같은 움직임 조각 / 반복의 다른 회차 / 별개 행동 / 동시에 가능한 특징 / 불확실을 분리','특정 관계 종류가 특정 단어 하나를 뜻하지 않도록 출처별 분포 검증'],pass_rule='관계 학습에 사용되는 각 정답은 직접 검토 출처가 있고 단어와 정답 종류가 완전히 결박되지 않아야 함. 미확인 관계는 학습에서 제외.',evidence=[str(out)],next='기존 추가6표현의 원본 검토 범위에서 필요한 관계만 보강. 기존 원자 움직임을 임의로 의미상 부모 행동으로 올리지 않음.'),
 dict(id='B',title_ko='행동 종류와 앞뒤 흐름 구분',status='READY',depends=[],work=['기존에 좋아진 판정부는 대조군으로 보존','동일 자세인데 전체 순서 문맥이 바뀌면 오인되는지 원인을 분리','신체 상대관계의 앞·현재·뒤 관측만 사용한 작은 판정부를 같은 학습 정답/최적화 횟수로 비교'],pass_rule='3seed 모두 기존 정상 관측과 주요 손모양/방향 근거를 보존하고, 재배열에서 실제 행동을 유지하면서 다른 행동의 혼입 감소. 표본 정확도만으로 통과시키지 않음.',evidence=[str(U/'episode_stage_contrast/CONTEXT_DIAGNOSTIC.json')],next='relation_context_ablation의 사전 고정 비교. 더 좋은 결과가 나와도 통합 검증 전 웹 미반영.'),
 dict(id='C',title_ko='여러 구간을 설명과 함께 연결',status='PENDING',depends=['A','B'],work=['학습한 행동 근거와 출처가 확인된 관계만으로 후보를 함께 선택','반복은 여러 회차로 보존하고 동시에 가능한 특징은 겹침 허용','서로 다른 행동에 같은 근거를 재사용하는 문제 검사','연결 없음(null)과 대안 후보를 보존하며 목표 순서로 억지 정렬하지 않음'],pass_rule='실제 진행 표본 포함과 다른 행동 혼입을 함께 평가. 보류 증가만으로 성공 금지. 연결 후에 순서/누락을 판정.',evidence=[str(U/'episode_stage_contrast/ASSOCIATION_DIAGNOSIS.json')]),
 dict(id='D',title_ko='오류가 겹쳐도 전체 피드백 검증',status='PENDING',depends=['C'],work=['정상·속도 차이·교차·짧은 변화 유지','순서 교환+누락+관측 부족을 단독/복합 비교','관측 부족과 실제 차이의 문장·구간·색상 일치 확인','통과 시 상위 G3 통합 → G4 미사용 공개8 → G5 웹 → G6 촬영 확인'],pass_rule='상위 GATES.json을 그대로 적용. 미사용 최종자료로 후보/경계/가중치 선택 금지. 웹 촬영은 사용자가 시작.',evidence=[str(U/'GATES.json')])]
 nodes[0].pop('title')
 graph=dict(at=stamp(),parent='G2',parent_status='REWORK',nodes=nodes,parallel_reason='A의 관계학습 정답 보강과 B의 이미 검증된 단계정답 기반 원인분리 비교는 서로 의존하지 않음.',loops=[dict(from_node='D',cause='행동 종류 오인',to_node='B'),dict(from_node='D',cause='옳은 후보가 있으나 잘못 연결',to_node='C'),dict(from_node='D',cause='정답 불확실/편향',to_node='A'),dict(from_node='C',cause='관측 자체 없음',to_node='D',action='숨은 손을 정답으로 만들지 않고 관측 부족 경로 검증')],stop_rules=['완료한 동일 조건의 학습/채점 재실행 금지','새 실패는 원인과 다음 수정 가설을 기록한 뒤 입력/비교 조건을 고정','한 후보 실패는 검증된 다른 부품을 폐기하지 않음','실질적 새 외부자료/모델 도입 및 해결 불가 문제만 사용자 확인'],constraints=['C/T3-F/shared100/runtime frozen','No private/user/mentor data','No word-specific runtime rule','No human-expert claim from Codex audit'])
 write(W/'SUBGRAPH.json',graph)
 print('Relation coverage:',counts,'; three-class training held, independent context diagnostic ready.')

if __name__=='__main__':main()
