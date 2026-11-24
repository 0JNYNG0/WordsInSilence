import time
from collections import Counter
from common import *
from consumer import compare

def main():
 assert not (U/'DEMO_DATA.json').exists();pins();start=time.perf_counter()
 case=read(O/'word100_action_feedback_casebook_20261101/INPUTS.json')
 old=read(O/'word100_description_evidence_feedback_20261101/REPORT_DATA.json')
 dense=read(U/'DENSE_MANIFEST.json');targets=dense['targets'];clips={}
 for pred in dense['rows']:
  a=pred['alias'];r=case['rows'][a]
  assert sha(pred['path'])==pred['sha256']
  with np.load(pred['path'],allow_pickle=False) as z:p=z['probabilities'].copy();ids=z['frame_id'].copy();ts=z['timestamp_us'].copy();ps=z['new_four_source'].copy()
  with np.load(r['input'],allow_pickle=False) as z:mask=z['mask'].copy()
  with np.load(r['observation'],allow_pickle=False) as z:xy=z['xy_px'].copy();observed=z['observed'].copy()
  valid=observed&np.isfinite(xy).all(-1)
  points=[[[round(float(v),2) for v in xy[i,j]] if valid[i,j] else None for j in range(75)] for i in range(len(ids))]
  gate=np.stack([support(mask,r['feature_names'],t) for t in targets],1)
  probability=[[float(v) if np.isfinite(v) else None for v in row] for row in p]
  media=read(O/f'word100_core_stage_evidence_20261031/media/{a}.json')
  assert len(media['source_pts_seconds'])==len(ids)
  assert np.max(np.abs(np.asarray(media['source_pts_seconds'])*1e6-ts))<1.1
  clips[a]=dict(alias=a,clip_id=r['clip_id'],label=r['label'],recorded_person_id=r['person'],role=r['role'],
    media='/media/'+a,frames=len(ids),frame_ids=ids.tolist(),times=(ts/1e6).tolist(),box=r['box'],
    points=points,probabilities=probability,geometry_support=gate.tolist(),new_four_source=ps.tolist(),annotation_intervals=r['intervals'])
 queries=read(U/'FROZEN.json')['demo']['queries'];pairs=[]
 naturalpairs=sorted(set((n['query'],n['reference']) for n in case['natural']))
 for q,r in naturalpairs:
  assert q in queries
  c=next(c for c in old['cases'] if c['query']==q and c['reference']==r)
  qclip,rclip=clips[q],clips[r];path=c['path']
  mapped=[[] for _ in range(qclip['frames'])]
  for qi,ri in path:
   assert 0<=qi<qclip['frames'] and 0<=ri<rclip['frames'];mapped[qi].append(ri)
  assert all(mapped)
  # Select an existing member (lower median); never synthesize a frame between links.
  rmap=[sorted(rr)[(len(rr)-1)//2] for rr in mapped]
  stages=[]
  for s in c['stages']:
   stages.append({k:s[k] for k in ['id','reference_core','query_frames','path','reference_rgb_description','source_cards','alignment']})
  frames=[]
  for qi,ri in enumerate(rmap):
   rr=compare(qclip['probabilities'][qi],rclip['probabilities'][ri],targets,qclip['geometry_support'][qi],rclip['geometry_support'][ri])
   stages_here=[s['id'] for s in stages if s['reference_core'][0]<=ri<=s['reference_core'][1]]
   frames.append(dict(query_frame=qi,reference_frame=ri,reference_candidates=mapped[qi],stage_ids=stages_here,
     estimates=rr,educational_verdict=None,alignment_verified=False))
  pairs.append(dict(id=q+'_'+r,query=q,reference=r,label=c['label'],path=path,query_to_reference=rmap,
    frames=frames,stages=stages,source_pair=c['source_pair'],source_pair_sha256=c['source_pair_sha256'],
    saved_path_preserved=True,correct_stage_accuracy=None))
 audit=read(U/'AUDIT.json')
 summary=[s for s in audit['pooled'] if s['role']=='development']
 counts=Counter(e['status'] for p in pairs for f in p['frames'] for e in f['estimates'])
 save('DEMO_DATA.json',dict(schema='wis_public_feedback_replay_v1',title='공개 영상으로 확인하는 손동작 비교 데모',
   scope='Already analyzed public videos only. Not live camera analysis, not a validated sign correctness tutor.',
   targets=targets,target_labels=['오른 검지만 펴짐','오른 네 손가락 펴짐','왼 네 손가락 펴짐','오른 네 손가락 접힘','왼 네 손가락 접힘','오른손이 더 위','왼손이 더 위'],
   queries=queries,clips=clips,pairs=pairs,evaluation=summary,seed=42,training=read(U/'TRAINING_DECISION.json'),
   new_inference=dense['new_center_predictions'],new_training=0,model_probabilities_not_calibrated_confidence=True,
   physical_feature_scope='Four fingers exclude thumb; extended finger separation/palm orientation/contact not inferred.',
   guidance='Source descriptions indicate reference context. No missing observation is filled from the expected sign.',
   timing=dict(build_cached_replay_seconds=time.perf_counter()-start,
      head_only_new_centers_seconds=sum(r['inference_seconds'] for r in dense['rows']),
      end_to_end_raw_video_latency=None)))
 save('DEMO_BUILD.json',dict(status='COMPLETE',clips=len(clips),expressions=len(queries),pairs=len(pairs),
   stage_cards=sum(len(p['stages']) for p in pairs),linked_stage_cards=sum(bool(s['source_cards']) for p in pairs for s in p['stages']),
   comparisons=sum(len(p['frames'])*7 for p in pairs),states=dict(counts),
   repeated_frames_not_independent_trials=True,sha256=sha(U/'DEMO_DATA.json'),raw_video_latency_measured=False))
 save('EXECUTION_LEDGER.json',dict(note='Code hashes recorded after execution, inputs/conditions were frozen before execution',
   source_contract_sha256=sha(U/'FROZEN.json'),code_pins={str(p):sha(p) for p in U.glob('*.py')},
   new_classifier_centers=dense['new_center_predictions'],reused_classifier_centers=dense['reused_center_predictions'],
   completed_center_reruns=0,new_training=0,new_rgb_extractions=0,new_detection=0,new_dtw_paths=0))
 print(json.dumps(read(U/'DEMO_BUILD.json'),ensure_ascii=False))
if __name__=='__main__':main()
