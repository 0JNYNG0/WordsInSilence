"""Separate stored-value reconstruction, not model inference or semantic gold."""
from pathlib import Path
from collections import Counter
import json,hashlib
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
dest=U/'PROCESS_VIEW_TARGETS_INDEPENDENT.json';assert not dest.exists()
t=read(U/'PROCESS_VIEW_TARGETS.json');original={r['id']:r for r in read(U/'PROCESS_TARGETS.json')['rows']}
views={r['view_id']:r for r in read(U/'PARTIAL_VIEWS_MANIFEST.json')['views']}
reviews={r['view_id']:r for fn in ['REVIEW_PARTIAL_0739.json','REVIEW_PARTIAL_1133.json','REVIEW_PARTIAL_1567.json','REVIEW_PARTIAL_2523.json','REVIEW_SENTINEL_PROCESS.json'] for r in read(U/fn)['rows']}
for p,h in t['pins'].items():assert sha(p)==h,p
assert len(t['rows'])==len(views)==217
counts=Counter();relations=intervals=0
for r in t['rows']:
 v=views[r['view_id']];q=original[r['query_id']];fm=v['ordered_frame_map']
 assert r['ordered_frame_map']==fm and r['original_scope_variant']==v.get('original_scope_variant')
 assert r['role']=='fit' and r['recorded_person_id'] in range(1,10)
 if r['view_id'] in reviews:
  d=reviews[r['view_id']];assert r['direct_review']==d
  value=d['observed_process_evidence']
 else:
  chain=q['sequence_links'][v['sequence_link_index']]['visible_chain']
  value=1 if chain=='yes' else None
 assert r['observed_process_evidence']==value
 assert all(r[k] is None for k in ['whole_word_correctness','learner_error','required_repetitions'])
 counts[str(value)]+=1
 by_source={f['source_native_frame']:f for f in fm}
 expected=[]
 for j,a in enumerate(q['relations']):
  f=by_source.get(a['frame'])
  if f is not None and f['pixel_operation']=='copy' and f['image_available']:
   expected.append((j,a['frame'],f['output_index'],a['setup_relation'],a['terminal_relation']))
 got=[(a['source_relation_index'],a['source_frame'],a['output_index'],a['setup_relation'],a['terminal_relation']) for a in r['sparse_relation_targets']]
 assert got==expected
 for a in r['sparse_relation_targets']:assert a['source_annotation']==q['relations'][a['source_relation_index']]
 expected_intervals=[]
 for j,a in enumerate(q['motion_intervals']):
  sf=list(range(a['start'],a['end']+1))
  selected=[f for f in fm if a['start']<=f['source_native_frame']<=a['end']]
  if [f['source_native_frame'] for f in selected]!=sf:continue
  if not all(f['image_available'] and f['pixel_operation']=='copy' for f in selected):continue
  oi=[f['output_index'] for f in selected]
  if oi!=list(range(oi[0],oi[0]+len(oi))):continue
  expected_intervals.append((j,sf,oi,a['label']))
 got_intervals=[(a['source_motion_interval_index'],a['source_frames'],a['output_indices'],a['label']) for a in r['fully_retained_motion_interval_targets']]
 assert got_intervals==expected_intervals
 for a in r['fully_retained_motion_interval_targets']:
  assert a['source_annotation']==q['motion_intervals'][a['source_motion_interval_index']]
  assert a['view_timestamps_us']==[fm[i]['view_timestamp_us'] for i in a['output_indices']]
  assert not any(a['output_indices'][0]<s<=a['output_indices'][-1] for s in r['seam_output_indices'])
 for key in ['bound_setup_witnesses','bound_terminal_witnesses']:
  for w in r[key]:
   f=fm[w['output_index']];assert f['source_native_frame']==w['source_frame'] and f['image_available']
 for w in r['bound_direct_motion_witnesses']:
  if w['visible_contiguous_witness_validated']:
   sf=w['source_frames'];oi=w['output_indices']
   assert sf==list(range(sf[0],sf[-1]+1)) and oi==list(range(oi[0],oi[-1]+1))
   assert all(fm[i]['image_available'] for i in oi)
 relations+=len(expected);intervals+=len(expected_intervals)
assert counts=={'1':79,'0':128,'None':10} and (relations,intervals)==(2360,472)
dest.write_text(json.dumps(dict(status='PASS_SEPARATE_STORED_VALUE_RECONSTRUCTION',views=217,counts=dict(counts),
 sparse_relations=relations,whole_retained_intervals=intervals,all_direct_reviews_preserved=True,
 no_hidden_or_partly_cut_supervision=True,source_target_hash=sha(U/'PROCESS_TARGETS.json'),
 compiled_hash=sha(U/'PROCESS_VIEW_TARGETS.json'),code_hash=sha(__file__),
 independent_human_expert=False,new_inference=0,new_training=0),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Independent stored-value reconstruction PASS217 views/2360 sparse/472 intervals')
