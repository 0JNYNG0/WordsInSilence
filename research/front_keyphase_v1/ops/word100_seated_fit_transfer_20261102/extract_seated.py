"""New public FIT pixels through the identical live input pipeline."""
from prepare_seated import *
def main():
    assert not (L/'INPUT_MANIFEST.json').exists();rows=read(L/'CROP_MANIFEST.json')['rows'];assert len(rows)==36
    files=[L/'CROP_MANIFEST.json',L/'PREP_TIMEBASE_FREEZE.json',L/'extract_seated.py',O/'word100_learned_body_relation_20261102/demo_adapter.py',P/'live_adapter.py']
    plan=dict(at=clock(),inputs=36,fit_ids=list(range(1,10)),public_only=True,pins={str(p):sha(p) for p in files},new_training=0,
        contract='Actual cropped sensor source, use all its pixels. No further ROI restriction or extra down-hand requirement. Original source frame/time and immutable supervision retained.',whole_goal_complete=False)
    if (L/'EXTRACTION_FREEZE.json').exists():assert read(L/'EXTRACTION_FREEZE.json')['pins']==plan['pins']
    else:exclusive(L/'EXTRACTION_FREEZE.json',plan)
    sys.path.insert(0,str(O/'word100_learned_body_relation_20261102'));import demo_adapter
    assert Path(demo_adapter.__file__).parent.name=='word100_learned_body_relation_20261102';demo_adapter.U=L
    warm=demo_adapter.warmup();start=time.perf_counter();done=[]
    write(L/'CURRENT_EXECUTION.json',dict(at=clock(),status='EXTRACTING_SEATED_FIT',pid=os.getpid(),warmup=warm,whole_goal_complete=False))
    for row in rows:
        capacity();assert row['person'] in range(1,10) and sha(row['video'])==row['video_sha256'];dest=L/'inputs'/row['clip_id']
        if (dest/'RESULT.json').exists():r=read(dest/'RESULT.json')
        else:r=demo_adapter.analyze(row['video'],dest,row['word_id'])
        a=np.load(dest/'analysis.npz');assert np.array_equal(a['frame_id'],row['source_frame_indices'])
        assert np.max(np.abs(a['timestamp_us']-row['source_timestamps_us']))<=2 and r['actual_sensor_extent']==row['crop']
        assert a['image_available'].all() and a['rgb'].shape==(row['frames'],768) and a['latent'].shape==(row['frames'],64)
        assert np.isfinite(a['x'][a['mask']]).all() and np.isfinite(a['rgb']).all() and np.isfinite(a['latent']).all()
        rec=dict(clip_id=row['clip_id'],word_id=row['word_id'],person=row['person'],kind='seated_fit',source=row['source'],source_sha256=row['source_sha256'],video=row['video'],video_sha256=row['video_sha256'],
            input=str(dest/'analysis.npz'),input_sha256=sha(dest/'analysis.npz'),result=str(dest/'RESULT.json'),frames=row['frames'],source_frame_indices=row['source_frame_indices'],
            actual_sensor_extent=r['actual_sensor_extent'],seconds=r['latency']['total_analysis_seconds'],device=r['device'],cropped_action_evidence=row['cropped_action_evidence'])
        done.append(rec);write(L/'EXTRACT_PROGRESS.json',dict(at=clock(),status='EXTRACTING',pid=os.getpid(),done=len(done),total=36,last=row['clip_id'],seconds=time.perf_counter()-start));print('extracted',len(done),36,row['clip_id'],rec['seconds'],flush=True)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(L/'INPUT_MANIFEST.json',dict(at=clock(),status='ALL36_PUBLIC_SEATED_FIT_INPUTS_EXTRACTED',rows=done,new_training=0,whole_goal_complete=False))
    write(L/'EXTRACT_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,done=36,total=36,seconds=time.perf_counter()-start))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (L/'EXTRACT_FIRST_FAILURE.txt').exists():(L/'EXTRACT_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
