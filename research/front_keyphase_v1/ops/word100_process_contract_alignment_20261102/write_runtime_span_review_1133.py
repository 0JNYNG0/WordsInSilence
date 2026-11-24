"""Record direct DEV visual diagnosis of six frozen first candidates; no labels for training."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
U=Path(__file__).resolve().parent
AB=U.parent/'word100_action_completion_supervision_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
ep=U/'EVALUATION.json';mp=AB/'DEV_PROCESS_REVIEW_MANIFEST.json';tp=AB/'DEV_PROCESS_TARGETS.json'
evaluation=read(ep)
manifest={r['id']:r for r in read(mp)['rows'] if r['word_id']=='WORD1133'}
old={r['id']:r for r in read(tp)['rows'] if r['word_id']=='WORD1133'}
notes={
11:dict(large=[52,57,65],setup=[49,51,52],progress=[52,57],later=65,
 reason='46/47부터 오른손을 올려 왼팔 상부로 가져가고 49–52에는 상부 관계가 보인다. 52–57의 사이 화면에서 팔을 따라 이동하기 시작하지만 마지막 57에서도 오른손은 왼팔의 몸통 가까운 쪽에 있다. 원본의 후속 58–64를 지나 65에서는 오른손이 왼손목 부근의 아래팔 쪽에 위치한다. 선택 구간 자체에 이 후속 도착 화면은 포함되지 않는다.'),
12:dict(large=[63,67,68,77],setup=[63,64,65],progress=[65,67],later=77,
 reason='56/58부터 오른손이 올라와 63–65에 왼팔 상부에 놓인다. 자동 구간 마지막 67과 native 구간 마지막 68을 큰 화면으로 각각 확인했는데, 둘 다 오른손이 아직 왼팔 상부 쪽에 있고 손목 부근까지 이어지는 이후 이동은 포함하지 않는다. 원본 69–76의 연속 후속 장면과 77의 손목 부근 관계는 구간 밖 문맥이다.'),
13:dict(large=[54,59,73],setup=[50,52,54],progress=[54,59],later=73,
 reason='두 입력의 선택 범위는 모두 원본 48–59이다. 오른손을 왼팔 상부에 가져간 50–54와 이후 55–59의 작은 진행은 보이지만 59에서도 오른손은 왼팔 상부 쪽에 남아 있다. 원본 60–72를 따라 더 이동한 뒤 73의 손목 부근 관계가 확인된다. 후속 문맥을 선택 구간의 도착 근거로 가져오지 않았다.')}
rows=[];pins={str(p.resolve()):sha(p) for p in [Path(__file__),ep,mp,tp,AB/'REVIEW_DEV_1133.json']}
selected=[r for r in evaluation['rows'] if r['role']=='development' and r['physical_query']=='right_hand_toward_left_wrist_along_arm' and (r['origin']=='runtime_proposal' or r['kind']=='native')]
assert len(selected)==6
for r in selected:
 m=manifest[r['query_id']];o=old[r['query_id']];n=notes[r['person']]
 assert r['person'] in (11,12,13) and r['source_sha256']==m['source_sha256']
 assert sha(m['source'])==m['source_sha256'];pins[str(Path(m['source']).resolve())]=m['source_sha256']
 cs=r['qualification']['interval_candidates'];assert cs
 assert [(x['end'],x['start']) for x in cs]==sorted((x['end'],x['start']) for x in cs)
 first=cs[0];ii=first['output_indices'];axis=[r['axis'][j] for j in ii]
 assert ii==list(range(first['start'],first['end']+1))
 frames=[a['source_frame'] for a in axis];assert frames==list(range(frames[0],frames[-1]+1))
 fmeta={f['frame']:f for f in m['frames']}
 assert all(a['source_timestamp_us']==fmeta[a['source_frame']]['timestamp_us'] for a in axis)
 pages=[]
 for p in m['pages']:
  assert sha(p['path'])==p['sha256'];pins[str(Path(p['path']).resolve())]=p['sha256']
  pages.append(dict(p,actually_viewed=True,reuse_note='Same source pages directly inspected once; shared by native and automatic selected spans.'))
 assert set(frames)<=set(f for p in pages for f in p['frames'])
 images=[]
 for f in n['large']:
  p=fmeta[f];assert sha(p['path'])==p['sha256'];pins[str(Path(p['path']).resolve())]=p['sha256']
  images.append(dict(p,actually_viewed=True,in_selected_nominal_span=f in frames))
 def witness(f):
  a=next((a for a in axis if a['source_frame']==f),None)
  return dict(source_frame=f,source_timestamp_us=fmeta[f]['timestamp_us'],output_index=a['output_index'] if a else None,
    in_selected_nominal_span=a is not None,image_path=fmeta[f]['path'],image_sha256=fmeta[f]['sha256'])
 progression=list(range(n['progress'][0],n['progress'][1]+1))
 assert set(progression)<=set(frames)
 rows.append(dict(case_id=r['case_id'],query_id=r['query_id'],person=r['person'],role='development',origin=r['origin'],kind=r['kind'],
  physical_query=r['physical_query'],predicate_definitions=o['predicate_definitions'],
  source=m['source'],source_sha256=m['source_sha256'],input_sha256=r['input_sha256'],
  candidate_selection='Frozen first chronological saved candidate; no highest-score selection.',candidate_ordinal=0,candidate_count=len(cs),
  output_indices=ii,source_frames=frames,source_timestamps_us=[a['source_timestamp_us'] for a in axis],selected_axis=axis,
  reviewed_pages=pages,large_existing_images_reviewed=images,all_selected_intermediate_frames_reviewed=True,
  setup_witnesses=[witness(f) for f in n['setup']],
  partial_progress_witnesses=[witness(f) for f in progression],last_selected_frame=witness(frames[-1]),
  last_selected_relation='Right hand remains beside the proximal/upper portion of the left arm in the frontal image, rather than at the distal wrist region.',
  nominal_visible_chain='partial',nominal_query_arrival_observed=False,
  ends_before_later_visible_wrist_region_arrival=True,
  later_context_arrival_witness=witness(n['later']),
  later_witness_is_earliest_arrival=False,later_witness_used_to_label_nominal=False,
  reason=n['reason'],old_full_native_direct_review=dict(visible_chain='yes',sequence=o['sequence_links'],
    source_annotation=o['source_annotation'],source_annotation_sha256=o['source_annotation_sha256']),
  old_whole_positive_copied_to_selected_span=False,
  use_for_training=False,use_for_threshold_tuning=False,training_label=None,
  whole_word_correctness=None,learner_error=None,required_repetitions=None,semantic_sufficient_extent=None,
  exact_completion_boundary=None,verified_3D_contact=None))
for p,h in pins.items():assert sha(p)==h
result=dict(status='DIRECT_RUNTIME_SPAN_DIAGNOSIS_COMPLETE_NOT_TRAINING_SUPERVISION',
 at=datetime.now(timezone.utc).isoformat(),rows=rows,pins=pins,
 scope='DEV WORD1133 IDs11–13, frozen first candidate for native and automatic runtime input only.',
 counts=dict(cases=6,source_clips=3,unique_existing_pages_directly_viewed=6,unique_existing_large_images_directly_viewed=10,
  selected_frame_slots=sum(len(r['source_frames']) for r in rows),unique_selected_source_frames=len({(r['source_sha256'],f) for r in rows for f in r['source_frames']}),
  unique_page_source_frames=sum(len({f for p in m['pages'] for f in p['frames']}) for m in manifest.values()),
  nominal_visible_chain_yes=0,nominal_visible_chain_partial=6,nominal_visible_chain_unknown=0),
 conclusion='All six selected first intervals show only the early/proximal part of this physical query. Their later wrist-region arrival is visible in the source but lies outside each selected interval.',
 interpretation='This identifies a candidate extent problem in these six selected intervals. It does not prove the complete source video is wrong, that every model output fails, or that a sign must sweep an exact anatomical distance.',
 repeated_development=True,independent_test=False,reviewer='Codex direct inspection; not independent human or Korean Sign Language expert.',
 prior_results_known_not_blinded=True,model_scores_used_as_visual_truth=False,new_inference=0,new_training=0,new_media_extraction=0,
 source_readonly=True,existing_labels_modified=False,old_results_recomputed=False,root_documents_modified=False,
 semantic_extent_requires_separate_expert_evidence=True)
out=U/'RUNTIME_SPAN_DIRECT_REVIEW_1133.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),counts=result['counts'])))
