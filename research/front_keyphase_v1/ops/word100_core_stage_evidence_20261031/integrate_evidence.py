"""Link saved source/RGB/observed-coordinate evidence; no new model fitting or DTW search."""
from pathlib import Path
import json,hashlib,numpy as np
from collections import Counter
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_semantic_episode_pilot_20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def roles(c):
 labels=[None]*c['frames']
 for s in c['segments']:
  if s['role'] not in {'rest','preparation','core_candidate','return'}:continue
  for i in range(s['core'][0],s['core'][1]+1):
   assert labels[i] is None,(c['alias'],i,'overlapping cores')
   labels[i]=(s['segment_id'],s['role'])
 return labels
def main():
 rows=read(U/'AUTHORIZATION.json')['rows'];fit=read(U/'PHASE_FIT.json');dev=read(U/'PHASE_DEV.json');sources=read(U/'SOURCE_PHASE_CONTRACTS.json')
 labels={c['alias']:c for c in fit['clips']+dev['clips']};expressions={e['word_id']:e for e in sources['expressions']}
 pins={str(p.resolve()):sha(p) for p in [U/'PHASE_FIT.json',U/'PHASE_DEV.json',U/'SOURCE_PHASE_CONTRACTS.json',OLD/'EMPIRICAL_RANGE.json']}
 contract=dict(status='FIXED_DIAGNOSTIC_CONTRACT',new_training=False,new_model_inference=False,new_DTW_search=False,
  projection='Use every saved (query_frame, reference_frame) pair. All reference frames aligned to one query frame must lie in exactly one inclusive annotated core (same segment_id). Any gap, mixed segment, or unknown yields unknown. All three individual fit references must agree in role; otherwise unknown. DEV labels never affect projection.',
  limitations='Draft coarse physical roles, NOT dense native-frame truth, semantic stage identity or educational correctness. Fit/dev granularity differs. Reviewer knew prior context. This is not a fresh held-out evaluation. No mask filtering or query dropping.',pins=pins)
 frozen=U/'FROZEN_DIAGNOSTIC.json'
 if frozen.exists():assert read(frozen)==contract
 else:write(frozen,contract)
 clips=[]
 for r in rows:
  assert sha(r['input'])==r['input_sha256']
  with np.load(r['input'],allow_pickle=False) as z:x=z['x'];mask=z['mask'].astype(bool)&np.isfinite(x);times=z['timestamp_us'];ids=z['frame_id']
  assert np.array_equal(ids,np.arange(r['frames']))
  c={k:r[k] for k in ['alias','label','person','role','word_id','frames']};c.update(seconds=(times/1e6).tolist(),segments=labels[r['alias']]['segments'],source_notes=[],source_candidates=[],rgb_review={k:v for k,v in labels[r['alias']].items() if k!='segments'})
  e=expressions[r['word_id']]
  c['source_notes']=['단계 이름은 표본 영상으로 확인한 물리 역할 초안입니다. 언어적으로 필수인 단계의 정답은 아닙니다.','좌표가 존재하는 비율은 정확도와 다릅니다. 접촉·숨은 손가락·깊이는 보이는 근거가 없으면 확인 불가입니다.']
  for source in e['source_candidates']:
   c['source_candidates'].append(dict(text=source['cached_description'],url=source['source_url']))
  for s in c['segments']:
   lo,hi=s['core'];sl=slice(lo,hi+1);n=hi-lo+1;s['coordinate_support']={};s['source_links']=[]
   for hand,off in [('L',0),('R',28)]:
    counts=dict(position_frames=int(mask[sl,off:off+2].all(1).sum()),palm_frames=int(mask[sl,off+8:off+10].all(1).sum()),arm_frames=int(mask[sl,off+2:off+6].all(1).sum()),angle_cells=int(mask[sl,off+11:off+25].sum()),angle_expected=n*14,frames=n)
    s['coordinate_support'][hand]=counts
   for clause in e['clauses']:
    fids=sorted({a['frame_id'] for ev in clause.get('evidence',[]) if ev['alias']==r['alias'] for a in ev['anchors'] if lo<=a['frame_id']<=hi})
    if fids:s['source_links'].append(dict(clause_id=clause['id'],text=clause['source_span'],url=clause['source_url'],anchor_frames=fids,status=clause['visible_match_status'],training_target=False,scope='listed anchors only; whole span not proven'))
   s['source_uncovered_anchors']=sorted({a['frame_id'] for ev in e.get('visible_source_uncovered_actions',[]) if ev['alias']==r['alias'] for a in ev['anchors'] if lo<=a['frame_id']<=hi})
   p=s['coordinate_support'];s['evidence_summary']=f"원본 {n}프레임: 왼손 위치 {p['L']['position_frames']}/{n}, 오른손 위치 {p['R']['position_frames']}/{n}. 손가락 각도는 왼손 {p['L']['angle_cells']}/{p['L']['angle_expected']}, 오른손 {p['R']['angle_cells']}/{p['R']['angle_expected']}개 값이 있습니다. 존재 여부이며 정확도는 아닙니다."
  if e.get('visible_source_uncovered_actions'):c['source_notes'].append('이 설명으로 다뤄지지 않은 앞선 손동작이 있습니다. 준비 동작으로 지우거나 설명에 맞춰 손 모양을 바꾸지 않습니다.')
  roles(c);clips.append(c)
 by={c['alias']:c for c in clips};emp=read(OLD/'EMPIRICAL_RANGE.json');projections=[]
 for d in emp['development']:
  q=by[d['alias']];n=q['frames'];refresults=[]
  for pair in d['pairs']:
   ref=by[pair['reference']];assert ref['role']=='fit' and ref['word_id']==q['word_id'];rl=roles(ref)
   groups=[[] for _ in range(n)]
   for qi,ri in pair['path_original_frames']:groups[qi].append(ri)
   assert all(groups) and set(v for g in groups for v in g)==set(range(ref['frames']))
   projected=[];mapped=[]
   for g in groups:
    labels_here={rl[i] for i in g};v=next(iter(labels_here)) if len(labels_here)==1 else None
    projected.append(v[1] if v is not None else 'unknown');mapped.append(sorted(set(g)))
   refresults.append(dict(reference=ref['alias'],projected_roles=projected,reference_frames_per_query=mapped))
  assert len(refresults)==3 and {p['reference'] for p in refresults}=={r['alias'] for r in rows if r['role']=='fit' and r['word_id']==q['word_id']}
  consensus=[]
  for i in range(n):
   same={p['projected_roles'][i] for p in refresults};consensus.append(next(iter(same)) if len(same)==1 else 'unknown')
  observed=roles(q);compared=[i for i in range(n) if observed[i] is not None and consensus[i]!='unknown'];agree=[i for i in compared if consensus[i]==observed[i][1]]
  mismatch=[dict(frame=i,projected=consensus[i],rgb_draft=observed[i][1]) for i in compared if i not in agree]
  byrole={role:dict(draft_frames=sum(v is not None and v[1]==role for v in observed),comparable_frames=sum(observed[i][1]==role for i in compared),agreement_frames=sum(observed[i][1]==role for i in agree)) for role in ['rest','preparation','core_candidate','return']}
  diag=dict(alias=q['alias'],frames=n,reference_projections=refresults,consensus_roles=consensus,consensus_frames=sum(x!='unknown' for x in consensus),draft_comparable_frames=len(compared),draft_agreement_frames=len(agree),draft_disagreement=mismatch,unknown_frames=sum(x=='unknown' for x in consensus),native_frames_retained=n,by_draft_role=byrole,not_accuracy=True)
  projections.append(diag);q['projection']=diag
 summary=dict(clips=len(clips),frames=sum(c['frames'] for c in clips),segments=sum(len(c['segments']) for c in clips),role_counts=dict(Counter(s['role'] for c in clips for s in c['segments'])),source_linked_segments=sum(bool(s['source_links']) for c in clips for s in c['segments']),source_anchor_links=sum(len(s['source_links']) for c in clips for s in c['segments']),new_training_runs=0,all_original_frames_retained=True)
 write(U/'PHASE_DATA.json',dict(status='SOURCE_LINKED_PHASE_DRAFT',summary=summary,clips=clips))
 write(U/'STAGE_PROJECTION.json',dict(status='DIAGNOSTIC_ONLY_NOT_ADOPTED',contract=contract,clips=projections,summary={k:sum(d[k] for d in projections) for k in ['frames','consensus_frames','draft_comparable_frames','draft_agreement_frames','unknown_frames','native_frames_retained']},semantic_stage_identity_proven=False))
 print(json.dumps(summary,ensure_ascii=False));print('projection',[(d['alias'],d['consensus_frames'],d['draft_agreement_frames'],d['draft_comparable_frames']) for d in projections])
if __name__=='__main__':main()
