"""Compare same-source witnesses after block reorder using already-saved features."""
from pathlib import Path
import sys
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
def main():
 assert not (W/'CONTEXT_DIAGNOSTIC.json').exists();ev={(r['seed'],r['clip_id'],r['kind']):r for r in read(W/'EVALUATION_COMPLETE.json')['rows']};gt={r['clip_id']:r for r in read(U/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']};rows=[]
 for r in read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']:
  if r['clip_id'] not in gt:continue
  cid=r['clip_id'];normal=np.load(ev[42,cid,'full_normal']['source']);permuted=np.load(r['path']);mapping=r['target']['query_to_source_frame_ids'];joins=[j['query_right'] for j in r['target']['splice_joins'] if j['synthetic_discontinuity']]
  for act,j in [('B',0),('C',1)]:
   m=gt[cid]['motions'][act];ns=m['definite_moving_witness']['normal_query_indices'];qs=m['definite_moving_witness']['permuted_query_indices_in_source_order'];assert len(ns)==len(qs)
   for n,q in zip(ns,qs):
    assert mapping[q]==n
    z0,z1=normal['z'][n],permuted['z'][q];scopes={'current_shared_context':(228,292),'current_relations':(292,392),'current_frozen_phase_features':(392,456),'before_context':(0,228),'after_context':(456,684)}
    delta={k:float(np.abs(z1[a:b]-z0[a:b]).mean()) for k,(a,b) in scopes.items()};probs=[]
    for seed in [42,43,44]:
     p0=np.load(ev[seed,cid,'full_normal']['path'])['p'][n,j];p1=np.load(ev[seed,cid,'dynamic_complete_blocks']['path'])['p'][q,j];probs.append(dict(seed=seed,normal=p0.tolist(),permuted=p1.tolist()))
    rows.append(dict(clip_id=cid,action=act,source_frame=n,permuted_frame=q,nearest_splice_frames=min(abs(q-a) for a in joins),position_values_identical=bool(np.array_equal(normal['x'][n,:102],permuted['x'][q,:102])),observation_flags_identical=bool(np.array_equal(normal['x'][n,186:237],permuted['x'][q,186:237])),mean_absolute_feature_difference=delta,stage_probability=probs))
 write(W/'CONTEXT_DIAGNOSTIC.json',dict(at=stamp(),rows=rows,summary=dict(witnesses=len(rows),identical_positions=sum(r['position_values_identical'] for r in rows),identical_flags=sum(r['observation_flags_identical'] for r in rows)),scope='Same-source saved-feature sensitivity diagnostic, not proof a particular encoder caused a semantic error. Short blocks/seams legitimately change temporal evidence; no synthetic joint/contact gold.',inference_calls=0,next='Distinguish changed learned temporal context from conflicting interval reuse before further learning. Do not force expected order or declare absent action present.'))
 print('Saved same-source context diagnostic:',len(rows),'witnesses.')
if __name__=='__main__':main()
