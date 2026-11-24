"""Independent mask/source reconstruction, no teacher or student inference."""
from pathlib import Path
import json,hashlib,numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'INPUTS_VERIFIED.json').exists();plan=read(U/'PREPARATION_FREEZE.json');manifest=read(U/'TEACHER_INPUTS.json')
    old=read(O/'word100_guide_conditioned_episode_20261102/CONTEXT_TARGETS.json')['rows']
    prior={r['clip_id']:{p['frame']:(1 if p['action_family']<8 else 2 if p['action_family']==8 else 0) for p in r['points'] if p['mask']} for r in old}
    overlay={r['clip_id']:{p['frame']:p for p in r['points'] if p['provenance']!='existing_derived_FIT_annotation'} for r in read(O/'word100_event_transition_review_20261102/ANNOTATION_OVERLAY.json')['rows']}
    base_metadata=O/'word100_learned_action_process_20261102/PREPARED.json'
    original_by_cache={r['cache']:r for r in read(base_metadata)['rows']}
    count=0;supervised=0;absent=0;seam_skipped=0
    assert len(manifest['rows'])==252 and {r['person'] for r in manifest['rows']}==set(range(1,10))
    for r in manifest['rows']:
        assert sha(r['teacher_path'])==r['teacher_sha256'] and sha(r['input_cache'])==r['input_sha256']
        a=np.load(r['teacher_path']);src=np.load(r['input_cache'])
        mp=np.asarray(original_by_cache[r['input_cache']]['mapping']) if r['index']<216 else src['mapping']
        av=src['available'];n=len(mp)
        assert np.array_equal(mp,a['source_frame_id']) and np.array_equal(av,a['image_available'])
        seam=np.zeros(n,bool)
        for j in np.flatnonzero(np.abs(np.diff(mp))>3)+1:seam[max(0,j-2):min(n,j+3)]=True
        role=np.array([prior[r['clip_id']].get(int(f),-1) for f in mp],np.int64)
        if r['index']<216:role[~av|seam]=-1
        for j,f in enumerate(mp):
            p=overlay[r['clip_id']].get(int(f))
            if p is None or not av[j] or seam[j]:continue
            if r['index']>=216 and p['provenance']!='resolved_native_RGB_correction_0_to_1':continue
            role[j]=p['role']
        assert np.array_equal(role,a['role_label']),(r['clip_id'],r['kind'])
        expected=(role<0)&av&~seam;assert np.array_equal(expected,a['mask'])
        probability=a['probability'];assert probability.shape==(n,2) and np.isfinite(probability).all() and np.allclose(probability.sum(1),1,atol=1e-6)
        assert (probability>=0).all() and (probability<=1).all()
        count+=int(expected.sum());supervised+=int((role>=0).sum());absent+=int((~av).sum());seam_skipped+=int(seam.sum())
    assert count==manifest['soft_frames']==22708
    for p,h in plan['pins'].items():assert sha(p)==h
    result=dict(status='FIT_SOFT_RETENTION_INPUTS_VERIFIED',input_sequences=252,original_public_clips=36,recorded_ids=list(range(1,10)),
        soft_reference_frames=count,direct_supervision_frames_with_augmentation=supervised,unavailable_frames_excluded=absent,edit_seam_frames_excluded=seam_skipped,
        old_prediction_is_not_gold=True,direct_labels_take_precedence=True,new_DEV_or_private_access=False,model_inference_in_verifier=0,
        scope='Separate arithmetic verifies source correspondence/masks/probabilities, not correctness of teacher judgement.',
        preparation_manifest_sha256=sha(U/'TEACHER_INPUTS.json'),whole_goal_complete=False)
    result['source_mapping_manifest_sha256']=sha(base_metadata)
    with (U/'INPUTS_VERIFIED.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
