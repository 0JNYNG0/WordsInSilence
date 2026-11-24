"""Pin frame-processing provenance. No old extraction or score reruns."""
from sequence_model import *
from collections import Counter

def main():
    assert not (J/'PROVENANCE_AUDIT.json').exists()
    old={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']}
    old_paths={r['input']:r for r in old.values()}
    short={r['input']:r for r in read(S/'SHORT_GAP_FIT_MANIFEST.json')['rows']}
    fit=read(B/'PREPARED.json')['rows'];dev=read(B/'EVALUATION_INPUTS.json')['rows'];records=[];pins={}
    for group,rows in [('FIT',fit),('DEV',dev)]:
        for r in rows:
            path=Path(r['input']);a=dict(np.load(path))
            if 'rgb' not in a:
                if r['input'] in short:path=Path(short[r['input']]['source_input'])
                elif r['kind']=='native':path=Path(old[r['clip_id']]['input'])
                else:path=Path(r['axis_path'])
            assert 'rgb' in np.load(path).files
            crop=old_paths.get(str(path));result=path.parent/'RESULT.json'
            # The pixel preparation implementations establish these contracts.
            # Persist actual file provenance as well as the classification.
            contract='legacy_fixed_roi' if crop else 'live_extraction_source'
            record=dict(split=group,clip_id=r['clip_id'],kind=r['kind'],rgb_source=str(path),rgb_sha256=sha(path),
                contract=contract,legacy_roi=crop['box'] if crop else None,
                actual_sensor_extent=read(result).get('actual_sensor_extent') if result.exists() else None)
            records.append(record);pins[str(path)]=record['rgb_sha256']
    diffs=[]
    for r in fit:
        if r['kind']!='original':continue
        source=old[r['clip_id']];new=np.load(r['cache'])['rgb'];before=np.load(source['input'])['rgb']
        assert before.shape==new.shape
        delta=np.abs(new-before)
        diffs.append(dict(clip_id=r['clip_id'],frames=len(new),same_rgb=np.array_equal(new,before),max_abs_difference=float(delta.max()),
            mean_abs_difference=float(delta.mean()),old_input=source['input'],new_input=r['input'],same_source_video_sha256=source['source_sha256']))
    assert len(diffs)==36 and len(records)==304
    evidence=[O/'word100_learned_motion_invariance_20261102/audit_inputs.py',O/'word100_learned_motion_invariance_20261102/INPUT_AUDIT.json',
        O/'word100_stage_feedback_cycle_20261031/review/verify_inputs.py',B/'process_model.py',A/'role_model.py',P/'INPUT_MANIFEST.json']
    for p in evidence:pins[str(p)]=sha(p)
    out=dict(at=clock(),status='MIXED_INPUT_CONTRACT_CONFIRMED_NOT_SOLE_CAUSE_PROVEN',rows=records,
        counts=dict(Counter(r['split']+':'+r['contract'] for r in records)),fit_same_video_differences=diffs,
        earlier_known_note='N INPUT_AUDIT already documented older fixed per-clip RGB boxes versus live whole-sensor extraction. This audit traces that difference through the current B/D/E/F/J caches; it is not newly discovered technology.',
        conclusion='Current native DEV uses legacy ROI RGB/geometry while original FIT uses live full-sensor extraction. Full-sensor reencoded/speed DEV also has failures, so crop mismatch is a confound, not an established explanation for all errors.',
        next='Separate coherent-input experiment: same sensor extent, body-relative coordinate contract, RGB window encoder, timestamp mapping, source availability and checkpoint for both FIT and DEV. New paths/results, old evaluations unchanged. Audit matching before further objective changes.',
        original_scores_revised=False,new_inference=0,new_training=0,pins=pins,whole_goal_complete=False)
    exclusive(J/'PROVENANCE_AUDIT.json',out)
    print(out['counts']);print('same original video RGB differs',sum(not r['same_rgb'] for r in diffs),'/36')
if __name__=='__main__':main()
