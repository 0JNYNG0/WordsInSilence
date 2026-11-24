"""Actual FIT pixels reversed; temporal RGB features are re-extracted once."""
from pathlib import Path
import sys,os,time,traceback
import av,numpy as np
S=Path(__file__).resolve().parent;O=S.parent;P=O/'word100_learned_guide_feedback_20261102';V=O/'word100_learned_body_relation_20261102'
sys.path.insert(0,str(O/'word100_learned_motion_invariance_20261102'))
from prepare_real_fit import read,sha,exclusive,write,clock,encode,capacity

def main():
    assert not (S/'REVERSE_FIT_MANIFEST.json').exists(),'Completed extraction must not be repeated'
    rows=[r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['role']=='fit'];assert len(rows)==36
    labels=read(S/'PHYSICAL_STATE_TARGETS.json');assert labels['summary']['fit']['known']==262
    sys.path.insert(0,str(V));import demo_adapter
    demo_adapter.U=S;demo_adapter.warmup();done=[]
    for r in rows:
        assert 1<=r['person']<=9 and sha(r['source'])==r['source_sha256'];capacity()
        key=r['clip_id'];meta=S/'reverse_records'/f'{key}.json';d=S/'reverse_inputs'/key
        if meta.exists():
            entry=read(meta);assert sha(entry['input'])==entry['analysis_sha256'];done.append(entry);continue
        path=S/'reverse_videos'/f'{key}.mp4';path.parent.mkdir(exist_ok=True);emap=path.with_suffix('.json')
        indices=list(range(r['frames']-1,-1,-1))
        if not emap.exists():
            with av.open(r['source']) as c:frames=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
            assert len(frames)==r['frames'];encode(frames,indices,path);del frames
            exclusive(emap,dict(sha256=sha(path),source_frame_indices=indices))
        else:assert sha(path)==read(emap)['sha256'] and read(emap)['source_frame_indices']==indices
        result=read(d/'RESULT.json') if (d/'RESULT.json').exists() else demo_adapter.analyze(path,d,r['word_id'])
        with np.load(d/'analysis.npz') as a:
            assert np.array_equal(a['frame_id'],np.arange(r['frames'])) and np.all(np.diff(a['timestamp_us'])>0)
        entry=dict(clip_id=key,word_id=r['word_id'],person=r['person'],role='fit',kind='actual_reversed_pixels',source=r['source'],source_sha256=r['source_sha256'],path=str(path),video_sha256=sha(path),source_frame_indices=indices,input=str(d/'analysis.npz'),analysis_sha256=sha(d/'analysis.npz'),seconds=result['latency']['total_analysis_seconds'])
        exclusive(meta,entry);done.append(entry)
        write(S/'PREP_PROGRESS.json',dict(status='RUNNING',pid=os.getpid(),at=clock(),done=len(done),total=36,last=key));print('prepared',len(done),36,key,flush=True)
    exclusive(S/'REVERSE_FIT_MANIFEST.json',dict(rows=done,scope='Physical-state labels follow original frame mapping. Not all reversed signs are linguistically wrong.',public_fit_only=True))
    write(S/'PREP_PROGRESS.json',dict(status='COMPLETE',pid=os.getpid(),at=clock(),done=36,total=36))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (S/'PREP_FIRST_FAILURE.txt').exists():(S/'PREP_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
