from pathlib import Path
import sys
G=Path(__file__).resolve().parent;O=G.parent
H=O/'word100_learned_feedback_holdout_20261102';R=O/'word100_learned_rgb_motion_20261102'
V=O/'word100_learned_body_relation_20261102';U=O/'word100_learned_feature_retention_20261102';P=O/'word100_learned_guide_feedback_20261102'
sys.path.insert(0,str(H));sys.path.insert(0,str(R));sys.path.insert(0,str(V))
from prepare_holdout import read,sha,write,exclusive,clock

def frozen():
    assert read(H/'SAVED_VERIFICATION.json')['gate_status']=='PASS_TO_INTEGRATED_DIAGNOSTICS'
    m=read(P/'INPUT_MANIFEST.json'); rows=[r for r in m['rows'] if r['person']==12]; assert len(rows)==4
    targets=read(P/'BLIND_STAGE_ANCHORS.json')['rows']; sources=[]
    for r in rows:
        anchor=[a for a in targets if a['clip_id']==r['clip_id']]
        lo=min(a['frame_id'] for a in anchor)-3;hi=max(a['frame_id'] for a in anchor)+3
        source=U/'video_controls'/f"{r['word_id']}_reencoded_normal.mp4"
        sources.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=12,source=str(source),source_sha256=sha(source),
            input=str(U/'actual_controls'/f"{r['word_id']}_reencoded_normal/analysis.npz"),
            gap_centers=[s['center_frame'] for s in r['labels']],reverse_frames=[lo,hi]))
    plan=dict(at=clock(),role='fixed-candidate development integration diagnostics; no fitting',sources=sources,
        gap_conditions=['pose_only_3','pose_only_7','pose_and_image_7'],primary='rgb_residual_42',
        physical_labels='existing exact center attributes; no propagation from word or guide',
        order_test='Reverse only reviewed main range in actual video; preserve rest. This changes physical order, not automatically every word semantic correctness.',
        adoption='Report every known-center result and every source. A short sensor gap should not be labelled a wrong action; unavailable pixels stay explicitly located. No new confidence cutoffs.',
        no_training=True,no_locked_holdout_tuning=True,no_private_content_access=True,
        pins={str(p):sha(p) for p in [P/'INPUT_MANIFEST.json',P/'BLIND_STAGE_ANCHORS.json',R/'runs/rgb_residual_42/model.pt',H/'EVALUATION_GATE.json']})
    if (G/'FROZEN.json').exists():
        old=read(G/'FROZEN.json');plan['at']=old['at'];assert old==plan
    else:exclusive(G/'FROZEN.json',plan)
    return plan,rows,targets
