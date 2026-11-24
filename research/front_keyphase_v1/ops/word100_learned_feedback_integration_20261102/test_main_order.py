"""Actual main-action reversal with native-frame provenance, no tuning."""
from fractions import Fraction
import os,traceback,time
import av,numpy as np
from common import G,R,V,P,read,sha,write,exclusive,frozen,clock
import demo_adapter as extraction
from rgb_runtime import RGBMotionObserver

def encode(source,dest,mapping):
    # Four short public clips only, sequential processing, no original edits.
    assert not dest.exists();dest.parent.mkdir(exist_ok=True)
    with av.open(source) as container:frames=[f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
    assert len(frames)==len(mapping) and sum(f.nbytes for f in frames)<1200*1024**2
    h,w=frames[0].shape[:2]
    with av.open(str(dest),'w') as out:
        s=out.add_stream('libx264',rate=30);s.width=w;s.height=h;s.pix_fmt='yuv420p';s.options={'crf':'18','preset':'fast'}
        for i,src in enumerate(mapping):
            frame=av.VideoFrame.from_ndarray(frames[src],format='rgb24');frame.pts=i;frame.time_base=Fraction(1,30)
            for packet in s.encode(frame):out.mux(packet)
        for packet in s.encode():out.mux(packet)

def main():
    plan,_,anchors=frozen();assert not (G/'MAIN_ORDER_RESULTS.json').exists()
    extraction.U=G;extraction.warmup();motion=RGBMotionObserver('rgb_residual',42);guides=read(P/'GUIDE_SUPPORT.json')['guides'];guideids=[g['canonical_stage_id'] for g in guides]
    if not (G/'ORDER_RUN_STARTED.json').exists():exclusive(G/'ORDER_RUN_STARTED.json',dict(pid=os.getpid(),at=clock()))
    rows=[]
    for s in plan['sources']:
        word=s['word_id'];original=dict(np.load(s['input']));n=len(original['frame_id']);lo,hi=s['reverse_frames'];assert 0<=lo<hi<n
        mapping=np.arange(n);mapping[lo:hi+1]=mapping[lo:hi+1][::-1]
        dest=G/'order_videos'/f'{word}_reverse_main.mp4';meta=dest.with_suffix('.json')
        if meta.exists():assert read(meta)['sha256']==sha(dest) and read(meta)['source_frame_indices']==mapping.tolist()
        else:
            assert sha(s['source'])==s['source_sha256'];encode(s['source'],dest,mapping)
            exclusive(meta,dict(source=s['source'],source_sha256=s['source_sha256'],sha256=sha(dest),source_frame_indices=mapping.tolist(),
                reversed_frames=[lo,hi],meaning='Reordered physical samples, not an independent human sign-error judgment'))
        directory=G/'order_inputs'/word
        base=read(directory/'RESULT.json') if (directory/'RESULT.json').exists() else extraction.analyze(dest,directory,word)
        a=dict(np.load(directory/'analysis.npz'));assert len(a['frame_id'])==n and np.array_equal(a['frame_id'],np.arange(n))
        feedback=G/'order_feedback'/f'{word}.json'
        if feedback.exists():out=read(feedback)
        else:
            out=motion.attach(base,a,a['retained_probability'],a['relation_probability']);exclusive(feedback,out)
        baseline=read(R/'feedback'/f'rgb_residual_42_{word}_reencoded_normal.json');scores=[]
        for anchor in anchors:
            if anchor['clip_id']!=s['clip_id']:continue
            native=anchor['frame_id'];ix=int(np.flatnonzero(mapping==native)[0]);allowed=anchor['allowed_same_physical_configuration']
            before=guideids[int(original['stage_probability'][native].argmax())];after=guideids[int(a['stage_probability'][ix].argmax())]
            scores.append(dict(source_frame=native,edited_frame=ix,expected_stage=anchor['stage_id'],equivalent_stages=allowed,
                baseline_stage=before,edited_stage=after,baseline_configuration_correct=before in allowed,edited_configuration_correct=after in allowed,
                edited_confidence=float(a['stage_probability'][ix].max())))
        rows.append(dict(word=word,source=s['source'],source_sha256=s['source_sha256'],video=str(dest),video_sha256=sha(dest),
            input=str(directory/'analysis.npz'),result=str(feedback),source_frame_indices=mapping.tolist(),reverse_frames=[lo,hi],
            baseline_motions=baseline['motion_observations'],motions=out['motion_observations'],
            baseline_stages=[dict(guide_id=z['guide_id'],status=z['status']) for z in baseline['stages']],
            stages=[dict(guide_id=z['guide_id'],status=z['status'],agreement=z['neural_stage_agreement']) for z in out['stages']],
            physical_stage_checks=scores,latency=base['latency'],whole_sign_error_gold=False))
        write(G/'ORDER_PROGRESS.json',dict(done=len(rows),total=4,word=word,pid=os.getpid(),at=clock()));print('actual main reversal',word,flush=True)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(G/'MAIN_ORDER_RESULTS.json',dict(rows=rows,no_training=True,frame_order_changed=True,whole_sign_error_labels=False,
        scope='Existing physical configuration labels move with source frames. Transition images can look ambiguous when reversed; not expert pedagogic truth.'))
    write(G/'ORDER_PROGRESS.json',dict(status='COMPLETE',done=4,total=4,pid=os.getpid(),at=clock()))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (G/'ORDER_FIRST_FAILURE.txt').exists():(G/'ORDER_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
