"""Public FIT-only actual local reordering; preserve original/frozen outputs."""
from pathlib import Path
import sys,os,shutil,traceback
import av,numpy as np
T=Path(__file__).resolve().parent;O=T.parent
N=O/'word100_learned_motion_invariance_20261102';V=O/'word100_learned_body_relation_20261102'
sys.path.insert(0,str(N));import prepare_real_fit as util
from prepare_real_fit import read,sha,exclusive,write,clock

def capacity():
    free=shutil.disk_usage(T).free
    if free<1024**3:raise RuntimeError('Preserve1GiB free. No old artifact deletion.')
    return free

def main():
    assert not (T/'LOCAL_FIT_MANIFEST.json').exists(),'Completed extraction must not be repeated'
    rows=read(T/'SOURCE_PLAN.json')['rows'];assert len(rows)==36
    if not (T/'STORAGE_PLAN.json').exists():
        exclusive(T/'STORAGE_PLAN.json',dict(at=clock(),start_free_bytes=capacity(),reserve_bytes=1024**3,
          estimate='Previous36 reverse inputs and videos total195MB; budget350MB for new inputs and sparse review. No old or original deletion.',
          change='Task-local1GiB free reserve replaces earlier2GiB helper, before extraction. No old helper modified.'))
    util.capacity=capacity
    sys.path.insert(0,str(V));import demo_adapter
    demo_adapter.U=T;demo_adapter.warmup();done=[]
    for r in rows:
        assert 1<=r['person']<=9 and sha(r['source'])==r['source_sha256'];capacity()
        key=r['clip_id'];rec=T/'local_records'/f'{key}.json';d=T/'local_inputs'/key
        if rec.exists():
            v=read(rec);assert sha(v['input'])==v['analysis_sha256'];done.append(v);continue
        indices=r['source_frame_indices'];assert sorted(indices)==list(range(r['frames']))
        path=T/'local_videos'/f'{key}.mp4';path.parent.mkdir(exist_ok=True);emap=path.with_suffix('.json')
        if not emap.exists():
            with av.open(r['source']) as c:frames=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
            assert len(frames)==r['frames'];util.encode(frames,indices,path);del frames
            exclusive(emap,dict(sha256=sha(path),source_frame_indices=indices))
        else:assert sha(path)==read(emap)['sha256'] and read(emap)['source_frame_indices']==indices
        result=read(d/'RESULT.json') if (d/'RESULT.json').exists() else demo_adapter.analyze(path,d,r['word_id'])
        with np.load(d/'analysis.npz') as a:
            assert np.array_equal(a['frame_id'],np.arange(r['frames'])) and np.all(np.diff(a['timestamp_us'])>0)
        v=dict(clip_id=key,word_id=r['word_id'],person=r['person'],role='fit',kind='actual_local_reorder',source=r['source'],source_sha256=r['source_sha256'],
          path=str(path),video_sha256=sha(path),source_frame_indices=indices,input=str(d/'analysis.npz'),analysis_sha256=sha(d/'analysis.npz'),seconds=result['latency']['total_analysis_seconds'])
        exclusive(rec,v);done.append(v)
        write(T/'PREP_PROGRESS.json',dict(at=clock(),status='RUNNING',pid=os.getpid(),done=len(done),total=36,free_bytes=capacity()));print('prepared',len(done),36,key,flush=True)
    exclusive(T/'LOCAL_FIT_MANIFEST.json',dict(rows=done,public_fit_only=True,not_automatic_linguistic_errors=True))
    write(T/'PREP_PROGRESS.json',dict(at=clock(),status='COMPLETE',pid=os.getpid(),done=36,total=36,free_bytes=capacity()))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (T/'PREP_FIRST_FAILURE.txt').exists():(T/'PREP_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
