"""Read-only checks before the locked movement-model comparison."""
from prepare_holdout import H,N,R,O,read,sha,exclusive
import numpy as np

def main():
    frozen=read(H/'FROZEN.json'); manifest=read(H/'INPUT_MANIFEST.json'); rows=manifest['rows']
    assert len(rows)==32 and len({(r['clip_id'],r['kind']) for r in rows})==32
    assert {r['person'] for r in rows}=={14,15}
    assert all(r['role']=='locked_task_evaluation' for r in rows)
    source=read(N/'LOCKED_REVIEW_MANIFEST.json')['rows']; sources={r['clip_id']:r for r in source}
    assert len({r['source_sha256'] for r in source})==8
    old=read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')
    assert not {14,15}&{r['person'] for r in old['rows']}
    targets=read(N/'LOCKED_PHYSICAL_TARGETS.json')['rows']; assert len(targets)==18
    checked=[]
    for r in rows:
        s=sources[r['clip_id']]; assert sha(r['source'])==r['source_sha256']==s['source_sha256']
        assert sha(r['video'])==r['video_sha256'] and sha(r['input'])==r['input_sha256']
        a=dict(np.load(r['input'])); mapping=np.asarray(r['source_frame_indices']); n=len(mapping)
        assert np.array_equal(a['frame_id'],np.arange(n)) and np.all(np.diff(a['timestamp_us'])>0)
        assert mapping.min()==0 and mapping.max()<s['frames'] and np.all(np.diff(mapping)>=0)
        for key in ['x','mask','rgb','image_available','retained_probability','relation_probability']:
            assert len(a[key])==n
        assert a['rgb'].shape==(n,768) and a['retained_probability'].shape==(n,7) and a['relation_probability'].shape==(n,3)
        base=read(r['result']); path=np.asarray(base['path'])
        assert base['word_id']==r['word_id'] and path.ndim==2 and path.shape[1]==2
        assert np.all(np.diff(path,axis=0)>=0) and path[:,0].min()>=0 and path[:,0].max()<n
        for t in targets:
            if t['clip_id']==r['clip_id']:
                ix=np.flatnonzero((mapping>=t['frames'][0])&(mapping<=t['frames'][1])); assert len(ix)>=3
        checked.append(dict(case=r['clip_id']+'_'+r['kind'],frames=n,image_coverage=float(a['image_available'].mean())))
    for p,h in frozen['pins'].items(): assert sha(p)==h
    assert sha(N/'LOCKED_REVIEW_MANIFEST.json')==frozen['source_manifest_sha256']
    assert sha(N/'LOCKED_PHYSICAL_TARGETS.json')==frozen['physical_targets_sha256']
    for p,h in old['source_pins'].items(): assert sha(p)==h
    exclusive(H/'INPUT_VERIFICATION.json',dict(status='PASS_LOCKED32_TIME_SOURCE_AND_MODEL_PINS',rows=checked,
        no_new_training=True,private_content_access=False,physical_targets_are_not_expert_semantic_truth=True))
    print('PASS:32 public inputs; native time mapping and frozen models verified')

if __name__=='__main__':main()
