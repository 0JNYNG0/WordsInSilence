from prepare_inputs import *
def main():
 info=read(U/'INPUTS.json');labels=read(U/'LABELS.json');checks=[];all_pairs=set();fit_words=set();fit_people=set();counts={}
 for p,h in info['pins'].items():assert sha(p)==h
 for r in info['rows']:
  assert sha(r['input'])==r['input_sha256'];assert sha(r['source'])==r['source_sha256'];assert sha(r['observation'])==r['observation_sha256']
  key=(r['clip_id'],r['source_sha256']);assert key not in all_pairs;all_pairs.add(key)
  if r['role']=='fit_candidate':fit_words.add(r['word_id']);fit_people.add(r['person'])
  a=dict(np.load(r['input'],allow_pickle=False));o=read_observation(r['observation']);xy=o['xy_px'];b=r['box']
  inside=(xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3]);o['usable'] &= inside
  f=extract_features(o);assert np.array_equal(a['mask'],f['mask']);assert np.array_equal(a['x'],np.nan_to_num(f['x']).astype(np.float32))
  assert np.array_equal(a['frame_id'],o['frame_id']);assert np.max(np.abs(a['timestamp_us']-o['timestamp_us']))<=1
  # Independently reconstruct known label sets instead of calling builder.
  cells={};denied={(c['frame_id'],c['feature']) for c in read(U/'LABEL_REVIEW.json')['uncertain_downgrades'] if c['alias']==r['alias']}
  for st in labels['static']:
   if st['alias']==r['alias']:
    for k,v in st['labels'].items():
     if v is not None:cells.setdefault((st['frame_id'],k),set()).add(v)
  for s in labels['sequences']:
   if s['source_alias']==r['alias']:
    for fr in s['frames']:
     for k,v in fr['direct'].items():
      k=s['side']+'_'+k
      if k in info['targets'] and v is not None:cells.setdefault((fr['frame_id'],k),set()).add(v)
  for i,frame in enumerate(a['frame_id']):
   for j,k in enumerate(info['targets']):
    val=cells.get((int(frame),k),set());expected=len(val)==1 and (int(frame),k) not in denied
    assert a['target_mask'][i,j]==expected
    if expected:assert a['y'][i,j]==next(iter(val))
  assert np.isfinite(a['rgb']).all() and np.isfinite(a['x']).all()
  assert np.array_equal(a['window_starts'],starts(len(a['frame_id'])))
  assert a['window_features'].shape==(len(a['window_starts']),768)
  counts[r['alias']]=int(a['target_mask'].sum());checks.append(r['alias'])
 for r in info['rows']:
  if r['role'] in ['held_person','held_expression_new_person']:assert r['person'] not in fit_people
  if r['role'].startswith('held_expression'):assert r['word_id'] not in fit_words
 save('INPUT_VERIFICATION.json',dict(status='PASS',clips=len(checks),known_cells=sum(counts.values()),per_clip=counts,source_hashes=True,frame_ids_and_timestamps=True,full_native_pose_then_fixed_roi=True,unknown_not_used_as_labels=True,split_separation=True,independent_calculation='Separate verification code, same Codex; not independent human truth',manifest_sha256=sha(U/'INPUTS.json')))
 print('PASS',len(checks),sum(counts.values()))
if __name__=='__main__':main()
