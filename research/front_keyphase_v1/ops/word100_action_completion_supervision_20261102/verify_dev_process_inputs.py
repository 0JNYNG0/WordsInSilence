"""Stored feature/media/clock verification only; never calls a model."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
dest=U/'DEV_PROCESS_INPUTS_VERIFIED.json';assert not dest.exists()
manifest=read(U/'DEV_PROCESS_INPUT_MANIFEST.json');media=read(U/'DEV_PROCESS_VIDEO_MANIFEST.json')
views=read(U/'DEV_PARTIAL_VIEWS_MANIFEST.json')['views']
view_by_key={v['pixel_view_key']:v for v in views}
assert len(views)==len(view_by_key)==80
assert manifest['training_allowed'] is False and media['training_allowed'] is False
old=read(U/'DEV_PROCESS_INPUT_EXECUTION_PLAN.json')['frozen_fit_preservation']
for p,h in old.items():assert sha(p)==h,p
freeze=read(manifest['extraction_freeze'])
assert sha(manifest['extraction_freeze'])==manifest['extraction_freeze_sha256']
for p,h in freeze['pins'].items():assert sha(p)==h,p
video={r['pixel_view_key']:r for r in media['rows']}
assert len(manifest['rows'])==len(video)==80
aliases=[];frames=blank=short=files=raw_mask_entries_at_blank=0;people=Counter()
for r in manifest['rows']:
 q=video[r['pixel_view_key']]
 v=view_by_key[r['pixel_view_key']]
 assert r['source']==v['source_path'] and r['source_sha256']==v['source_sha256']
 assert r['training_allowed'] is False and r['repeated_development'] is True
 assert r['source_frame_indices']==[x['source_native_frame'] for x in v['ordered_frame_map']]
 assert r['source_timestamps_us']==[x['source_native_timestamp_us'] for x in v['ordered_frame_map']]
 assert r['view_timestamps_us']==[x['view_timestamp_us'] for x in v['ordered_frame_map']]
 assert r['image_available']==[x['image_available'] for x in v['ordered_frame_map']]
 assert q['pixel_operations']==[x['pixel_operation'] for x in v['ordered_frame_map']]
 expected_dir=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/development_process_inputs')/r['pixel_view_key']
 expected_video=expected_dir.parent.parent/'development_process_videos'/r['pixel_view_key']/'view.mp4'
 assert Path(r['input']).resolve()==(expected_dir/'input.npz').resolve()
 assert Path(r['video']).resolve()==expected_video.resolve()
 assert r['person']==q['person'] and r['clip_id']==q['clip_id'] and r['person'] in range(11,14) and r['role']=='development'
 assert r['source']==q['source'] and r['source_sha256']==q['source_sha256']
 assert r['video_sha256']==q['video_sha256']==sha(r['video'])
 assert r['frames']==q['frames'] and r['actual_sensor_extent']==q['actual_sensor_extent']
 assert r['source_frame_indices']==q['source_frame_indices'] and r['source_timestamps_us']==q['source_timestamps_us']
 assert r['extraction_freeze_sha256']==manifest['extraction_freeze_sha256']
 for p,h in r['output_pins'].items():
  assert Path(p).resolve().is_relative_to(expected_dir.resolve())
  assert sha(p)==h,p
  files+=1
 assert sha(r['input'])==r['input_sha256']
 with np.load(r['input'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
 n=r['frames'];assert set(a)=={'x','mask','rgb','prelinear','frame_id','timestamp_us','image_available'}
 assert np.array_equal(a['frame_id'],np.arange(n))
 assert a['timestamp_us'].tolist()==q['view_timestamps_us']==r['view_timestamps_us']
 assert np.all(np.diff(a['timestamp_us'])>0) and a['timestamp_us'][0]==0
 assert a['image_available'].tolist()==q['image_available']==r['image_available']
 assert a['mask'].dtype==bool and a['image_available'].dtype==bool
 assert a['x'].shape==a['mask'].shape==(n,69)
 assert a['rgb'].shape==a['prelinear'].shape==(n,768)
 assert all(np.isfinite(a[k]).all() for k in ['x','rgb','prelinear'])
 ctx=read(r['rgb_context']);assert ctx['real_frames']==ctx['returned_rows']==n
 assert ctx['real_frame_id']==a['frame_id'].tolist() and ctx['real_timestamp_us']==a['timestamp_us'].tolist()
 assert ctx['padding_is_observation'] is False and ctx['labels_modified'] is False
 assert ctx['supported'] and ctx['exact_decode_pts_verified']
 if n<16:
  assert ctx['model_window_indices']==[list(range(n))+[n-1]*(16-n)]
  assert ctx['source_mask']==[[True]*n+[False]*(16-n)]
  assert r['internal_padding_count']==ctx['padding_count']==16-n
  assert np.array_equal(a['rgb'],np.repeat(a['rgb'][:1],n,axis=0))
  assert np.array_equal(a['prelinear'],np.repeat(a['prelinear'][:1],n,axis=0))
  short+=1
 else:assert ctx['padding_count']==r['internal_padding_count']==0
 assert r['new_feedback_scoring']==r['new_DTW']==r['new_training']==0
 assert r['device']['actual']==['GPU.0']
 raw_mask_entries_at_blank+=int(a['mask'][~a['image_available']].sum())
 aliases.extend(r['aliases']);frames+=n;blank+=int((~a['image_available']).sum());people[r['person']]+=1
targets=views
assert len(aliases)==len(set(aliases))==80 and set(aliases)=={r['view_id'] for r in targets}
assert short==22
report=dict(status='PASS_STORED_NEW_INPUT_MEDIA_AND_CLOCK_CONTRACT',unique_inputs=80,query_aliases=80,role='development',training_allowed=False,repeated_development=True,independent_final_test=False,
 real_input_frames=frames,blank_frames=blank,internally_padded_short_views=short,output_files_hashed=files,
 inputs_by_recorded_id=dict(people),extra_observation_frames_added=0,old_feedback_reinference=0,new_training=0,
 no_predictions_in_inputs=True,actual_gpu='GPU.0',unknown_policy='Image availability preserved separately from sensor mask; short padding is not observation.',
 raw_sensor_mask_entries_on_unavailable_images=raw_mask_entries_at_blank,
 raw_sensor_mask_on_blank_is_not_direct_observation=True,fit_preservation_pins=old,
 limitation='Structural validation and frozen-path provenance, not detector accuracy, independent human truth or candidate performance.',
 manifest_sha256=sha(U/'DEV_PROCESS_INPUT_MANIFEST.json'),view_manifest_sha256=sha(U/'DEV_PARTIAL_VIEWS_MANIFEST.json'),code_sha256=sha(__file__))
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Saved new inputs PASS80/80 aliases;',frames,'real rows;',short,'short contexts')
