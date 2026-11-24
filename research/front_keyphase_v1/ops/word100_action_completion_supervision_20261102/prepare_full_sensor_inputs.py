"""Prepare consistent full-frame FIT sensor inputs; no feedback-head scoring.

Public raw videos only. Reuse exact completed per-clip outputs after verifying
their hashes; never silently restart a partial clip or rescore an old test.
"""
from pathlib import Path
import os,sys,json,hashlib,time,traceback,importlib.util
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;AA=O/'word100_partial_order_live_20261102'
V=O/'word100_learned_body_relation_20261102'
OUT=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/full_sensor_inputs')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def now():return datetime.now(timezone.utc).isoformat()
def main():
    assert not (U/'FULL_SENSOR_INPUT_MANIFEST.json').exists(), 'Completed preparation must not be rerun'
    review=read(U/'PROCESS_INPUT_REVIEW.json');assert review['prepare_new_full_frame_inputs'] is True
    source_rows=read(U/'PROCESS_REVIEW_MANIFEST.json')['rows'];sources={r['clip_id']:r for r in source_rows}
    assert len(sources)==36 and {r['person'] for r in sources.values()}==set(range(1,10))
    frozen_path=U/'FULL_SENSOR_EXTRACTION_FROZEN.json'
    if not frozen_path.exists():
        parent=read(AA/'EXECUTION_FROZEN.json')
        for p,h in parent['pins'].items():assert sha(p)==h,p
        pins=dict(parent['pins']);pins.update({str(p):sha(p) for p in [AA/'EXECUTION_FROZEN.json',Path(__file__).resolve(),U/'PROCESS_INPUT_REVIEW.json',U/'PROCESS_REVIEW_MANIFEST.json']})
        freeze=dict(at=now(),purpose='New coherent full-original FIT inputs for new direct process supervision; not old benchmark rerun.',
            pins=pins,sources={cid:dict(source=r['source'],sha256=r['source_sha256'],person=r['person'],frames=r['source_frames']) for cid,r in sources.items()},
            backend='Same full-frame MediaPipe joint_v3 + face geometry + GPU SignRep feature extraction as AA; no feedback model head or DTW call.',
            no_private=True,new_training=0,new_feedback_scoring=0,max_new_sources=36)
        write(frozen_path,freeze)
    freeze=read(frozen_path)
    for p,h in freeze['pins'].items():assert sha(p)==h,p
    sys.path.insert(0,str(V))
    spec=importlib.util.spec_from_file_location('_process_full_sensor',V/'demo_adapter.py')
    adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
    shared=adapter.shared
    compiled,device=shared.compile_rgb()
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for cid,q in sorted(sources.items()):
        d=(OUT/cid).resolve();assert d.is_relative_to(OUT.resolve())
        source=Path(q['source']);assert sha(source)==q['source_sha256']
        if (d/'COMPLETE.json').exists():
            record=read(d/'COMPLETE.json');assert record['extraction_freeze_sha256']==sha(frozen_path)
            assert record['source_sha256']==q['source_sha256'] and sha(record['input'])==record['input_sha256']
            rows.append(record);continue
        assert not d.exists(), f'Partial source requires inspection before retry: {d}'
        d.mkdir();write(d/'STARTED.json',dict(at=now(),pid=os.getpid(),source=str(source),source_sha256=q['source_sha256']))
        try:
            start=time.perf_counter();ids,ts,w,h=shared.source_axis(source)
            assert len(ids)==q['source_frames'] and np.array_equal(ids,np.arange(len(ids)))
            with ThreadPoolExecutor(max_workers=2) as pool:
                rgb_job=pool.submit(shared.extract_rgb,source,ids,ts,w,h,compiled,None)
                quality_job=pool.submit(adapter.image_information,source,ids)
                tick=time.perf_counter();ob,face=shared.base._capture(source,d,None);mp_seconds=time.perf_counter()-tick
                rgb,pre,rgb_seconds=rgb_job.result();available=quality_job.result()
            assert np.array_equal(ids,ob.frame_id) and np.array_equal(ts,ob.timestamp_us)
            observed={k:getattr(ob,k).copy() for k in ['xy_px','usable','observed','frame_id','timestamp_us']}
            observer=shared.base._module('_process_body_geometry',O/'word100_guide_video_feedback_20261028/motion_observer.py')
            feat=observer.extract_features(observed)
            arrays=dict(x=np.nan_to_num(feat['x']).astype(np.float32),mask=feat['mask'],rgb=rgb,prelinear=pre,
                frame_id=ids,timestamp_us=ts,image_available=available)
            assert arrays['x'].shape==(len(ids),69) and rgb.shape==pre.shape==(len(ids),768)
            assert np.isfinite(arrays['x']).all() and np.isfinite(rgb).all() and np.isfinite(pre).all()
            dest=d/'input.npz';np.savez_compressed(dest,**arrays)
            record=dict(clip_id=cid,word_id=q['word_id'],person=q['person'],role='fit',source=str(source),source_sha256=sha(source),
                input=str(dest),input_sha256=sha(dest),frames=len(ids),source_frame_indices=ids.tolist(),native_timestamps_us=ts.tolist(),
                actual_sensor_extent=[0,0,w,h],image_available_count=int(available.sum()),parallel_observers=True,
                seconds=time.perf_counter()-start,mediapipe_seconds=mp_seconds,rgb_seconds=rgb_seconds,device=device,
                extraction_freeze_sha256=sha(frozen_path),new_feedback_scoring=0,new_DTW=0,new_training=0,
                normalization='Existing body-relative feature definition, no extra crop or lower-bound mask.',
                source_scope='Public FIT registered ID; not private/mentor or DEV')
            assert record['source_sha256']==q['source_sha256'];write(d/'COMPLETE.json',record);rows.append(record)
            write(U/'FULL_SENSOR_PROGRESS.json',dict(at=now(),status='RUNNING',pid=os.getpid(),done=len(rows),total=36,current=cid,device=device))
            print('full sensor',len(rows),cid,round(record['seconds'],3),flush=True)
        except BaseException:
            (d/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    for p,h in freeze['pins'].items():assert sha(p)==h,p
    write(U/'FULL_SENSOR_INPUT_MANIFEST.json',dict(at=now(),status='COHERENT_PUBLIC_FIT_INPUTS_PREPARED',rows=rows,
        registered_ids=list(range(1,10)),source_clips=36,new_training=0,new_feedback_scoring=0,new_DTW=0,
        extraction_freeze=str(frozen_path),extraction_freeze_sha256=sha(frozen_path),whole_goal_complete=False))
    write(U/'FULL_SENSOR_PROGRESS.json',dict(at=now(),status='COMPLETE',active=None,done=36,total=36))
    print('Completed36 new coherent FIT inputs; no feedback heads or DTW evaluated',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'SENSOR_PREPARATION_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
