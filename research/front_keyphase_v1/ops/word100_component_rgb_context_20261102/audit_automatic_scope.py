"""Check saved learned-prototype DTW proposals against new sparse witnesses.

No inference, fitting, boundary optimization, or old-result modification.
Sparse witnesses are not certified event start/end or dense frame labels.
"""
from pathlib import Path
import json,hashlib
import numpy as np
T=Path(__file__).resolve().parent;O=T.parent
S=O/'word100_source_component_connector_20261102'
V=O/'word100_learned_body_relation_20261102'
Q=O/'word100_predicate_supervision_20261102'
TARGETS=['right_elbow_lowers_hand_remains_raised','right_hand_traverses_left_arm_toward_wrist','bilateral_height_alternation']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
    with (T/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    rows=[r for r in read(S/'DIRECT_TARGETS.json')['rows'] if r['target'] in TARGETS and r['kind']=='sequence' and r['mask'] and r['y']==1]
    canonical={(r['word_id'],r['target']):r for r in rows if r['role']=='fit' and r['person']==1}
    dev=[r for r in rows if r['role']=='development'];assert len(dev)==9 and len(canonical)==3
    files=[T/'audit_automatic_scope.py',S/'DIRECT_TARGETS.json',Q/'SUPERVISION.json']
    files += [V/'development_feedback'/('rgb_geometry_42_'+r['clip_id']+'.json') for r in dev]
    meta={r['clip_id']:r for r in read(Q/'SUPERVISION.json')['sources']}
    for r in dev:files.append(Path(meta[r['clip_id']]['input']))
    pins={str(p):sha(p) for p in files}
    save('AUTOMATIC_SCOPE_FROZEN.json',dict(
      decision='Audit existing automatic native proposal coverage before any new training or boundary changes',
      canonical_anchors='Only reviewed positive source sequence of FIT ID01; min/max sparse witnesses used as proposal anchors, not exact event boundaries',
      evaluated_records=dev,thresholds_optimized=False,pins=pins,
      metrics='Whether fixed proposal includes each separately reviewed witness frame; not dense IoU, recall of whole actions, or educational correctness',
      existing_paths_reused=True,new_model_inference=False,whole_goal_complete=False))
    out=[]
    for r in dev:
        old=read(V/'development_feedback'/('rgb_geometry_42_'+r['clip_id']+'.json'))
        assert old['path_axes']==['query_native_index','learned_canonical_index']
        assert old['individual_reference_comparisons']==0 and old['learned_prototype_comparisons']==1
        a=np.load(meta[r['clip_id']]['input'],allow_pickle=False)
        c=canonical[(r['word_id'],r['target'])];lo,hi=min(c['native_frames']),max(c['native_frames'])
        path=np.array(old['path'],int);ix=np.unique(path[(path[:,1]>=lo)&(path[:,1]<=hi),0]);assert len(ix)
        full=np.arange(ix.min(),ix.max()+1);frames=a['frame_id'][full]
        witnesses=r['native_frames'];included=[int(f) for f in witnesses if int(f) in set(frames.tolist())]
        missing=[int(f) for f in witnesses if int(f) not in set(frames.tolist())]
        out.append(dict(clip_id=r['clip_id'],role=r['role'],component=r['target'],source_clause_ids=r['source_clause_ids'],
          canonical_anchor_extent=[lo,hi],automatic_native_extent=[int(frames[0]),int(frames[-1])],
          automatic_seconds=(a['timestamp_us'][full[[0,-1]]]/1e6).tolist(),
          reviewed_witness_frames=witnesses,included_witness_frames=included,outside_witness_frames=missing,
          witness_fraction=len(included)/len(witnesses),all_witnesses_included=not missing,
          whole_event_correctness=None,boundary_optimized=False))
    for p,h in pins.items():assert sha(p)==h,p
    save('AUTOMATIC_SCOPE_AUDIT.json',dict(rows=out,total_proposals=len(out),
      all_witnesses_included=sum(r['all_witnesses_included'] for r in out),
      limitation='Existing path on repeatedly exposed DEV; witness inclusion is a diagnostic only. An outside witness is not automatically an essential missing action. High component probability cannot certify the unexamined tail.',
      old_results_unchanged=True,new_model_inference=0,new_training=0))
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
