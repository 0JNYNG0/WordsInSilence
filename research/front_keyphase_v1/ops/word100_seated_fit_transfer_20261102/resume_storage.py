"""Resume unfinished public extraction on a separate local derived-data drive."""
from prepare_seated import *

DEST=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')

def main():
    assert not (L/'INPUT_MANIFEST.json').exists()
    old=read(L/'EXTRACTION_FREEZE.json')
    for p,h in old['pins'].items():assert sha(p)==h
    assert DEST.resolve().is_relative_to(Path('W:/WordsInSilence_derived').resolve())
    assert not DEST.resolve().is_relative_to(Path('W:/sources').resolve())
    plan=dict(at=clock(),original_freeze_sha256=sha(L/'EXTRACTION_FREEZE.json'),script_sha256=sha(__file__),destination=str(DEST),
        reason='C drive reserve stopped extraction after19 completed inputs. Preserve all19 and all original files. Only unfinished derived outputs go to W; no deletion/move/network.',
        reserve_bytes=1024**3,fit_only=True,whole_goal_complete=False)
    if (L/'STORAGE_RESUME_FREEZE.json').exists():
        previous=read(L/'STORAGE_RESUME_FREEZE.json')
        assert previous['script_sha256']==plan['script_sha256'] and previous['destination']==plan['destination']
    else:exclusive(L/'STORAGE_RESUME_FREEZE.json',plan)
    DEST.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(O/'word100_learned_body_relation_20261102'));import demo_adapter
    demo_adapter.U=DEST
    start=time.perf_counter();warm=None;done=[];created=0
    write(L/'CURRENT_EXECUTION.json',dict(at=clock(),status='RESUMING_17_ON_LOCAL_W',pid=os.getpid(),destination=str(DEST),whole_goal_complete=False))
    rows=read(L/'CROP_MANIFEST.json')['rows']
    for row in rows:
        assert row['person'] in range(1,10) and sha(row['video'])==row['video_sha256']
        retained=L/'inputs'/row['clip_id'];dest=retained if (retained/'RESULT.json').exists() else DEST/'inputs'/row['clip_id']
        if (dest/'RESULT.json').exists():r=read(dest/'RESULT.json')
        else:
            assert shutil.disk_usage(DEST).free>1024**3
            assert not dest.exists(),'Partial result requires separate audit, not overwrite'
            if warm is None:warm=demo_adapter.warmup()
            r=demo_adapter.analyze(row['video'],dest,row['word_id']);created+=1
        a=np.load(dest/'analysis.npz');assert np.array_equal(a['frame_id'],row['source_frame_indices'])
        assert np.max(np.abs(a['timestamp_us']-row['source_timestamps_us']))<=2 and r['actual_sensor_extent']==row['crop']
        assert a['image_available'].all() and a['rgb'].shape==(row['frames'],768) and a['latent'].shape==(row['frames'],64)
        assert np.isfinite(a['x'][a['mask']]).all() and np.isfinite(a['rgb']).all() and np.isfinite(a['latent']).all()
        assert r['source_sha256']==row['video_sha256']
        rec=dict(clip_id=row['clip_id'],word_id=row['word_id'],person=row['person'],kind='seated_fit',source=row['source'],source_sha256=row['source_sha256'],video=row['video'],video_sha256=row['video_sha256'],
            input=str(dest/'analysis.npz'),input_sha256=sha(dest/'analysis.npz'),result=str(dest/'RESULT.json'),frames=row['frames'],source_frame_indices=row['source_frame_indices'],actual_sensor_extent=r['actual_sensor_extent'],
            seconds=r['latency']['total_analysis_seconds'],device=r['device'],cropped_action_evidence=row['cropped_action_evidence'])
        done.append(rec);write(L/'EXTRACT_PROGRESS.json',dict(at=clock(),status='EXTRACTING',pid=os.getpid(),done=len(done),total=36,new_this_resume=created,last=row['clip_id'],seconds=time.perf_counter()-start))
        print('validated',len(done),36,row['clip_id'],'new_this_resume',created,flush=True)
    for p,h in old['pins'].items():assert sha(p)==h
    exclusive(L/'INPUT_MANIFEST.json',dict(at=clock(),status='ALL36_PUBLIC_SEATED_FIT_INPUTS_EXTRACTED',rows=done,new_training=0,whole_goal_complete=False))
    write(L/'EXTRACT_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,done=36,total=36,new_this_resume=created,seconds=time.perf_counter()-start))
    write(L/'CURRENT_EXECUTION.json',dict(at=clock(),status='EXTRACTION_COMPLETE_PENDING_INPUT_TARGET_AUDIT',active=None,whole_goal_complete=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (L/'STORAGE_RESUME_FIRST_FAILURE.txt').exists():(L/'STORAGE_RESUME_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
