"""Encode new public development view pixels losslessly with their explicit view PTS.

Never reorders cached model features. No learned head, detector or training is
called. Completed media are immutable and reusable only after hash checks.
"""
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
import av,numpy as np,json,hashlib,shutil,os,traceback,time
U=Path(__file__).resolve().parent
OUT=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/development_process_videos')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rgbsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def now():return datetime.now(timezone.utc).isoformat()
def capacity():
    free=shutil.disk_usage(OUT.parent).free
    if free<2*1024**3:raise RuntimeError('Preserve2GiB disk margin; no files deleted.')
    return free

def encode(path,arrays,pts):
    assert not path.exists() and len(arrays)==len(pts) and len(arrays)>1
    period=int(np.median(np.diff(pts)));rate=Fraction(1000000,period)
    with av.open(str(path),'w',options={'video_track_timescale':'1000000'}) as c:
        s=c.add_stream('libx264rgb',rate=rate);s.width=arrays[0].shape[1];s.height=arrays[0].shape[0]
        s.pix_fmt='rgb24';s.time_base=Fraction(1,1000000);s.codec_context.time_base=Fraction(1,1000000)
        s.options={'crf':'0','preset':'ultrafast','bf':'0'}
        for a,t in zip(arrays,pts):
            f=av.VideoFrame.from_ndarray(a,format='rgb24');f.pts=int(t);f.time_base=Fraction(1,1000000)
            for packet in s.encode(f):c.mux(packet)
        for packet in s.encode():c.mux(packet)

def main():
    assert not (U/'DEV_PROCESS_VIDEO_MANIFEST.json').exists(), 'Do not repeat completed media preparation'
    m=read(U/'DEV_PARTIAL_VIEWS_MANIFEST.json');verified=read(U/'DEV_PROCESS_CODE_INPUT_AUDIT.json')
    assert verified['files'][str(U/'DEV_PARTIAL_VIEWS_MANIFEST.json')]==sha(U/'DEV_PARTIAL_VIEWS_MANIFEST.json')
    assert verified['counts']['view_aliases']==80 and verified['counts']['unique_pixel_time_views']==80
    assert read(U/'RGB_PTS_CODEC_PROBE.json')['status']=='PASS_SYNTHETIC_PIXEL_EXACT_AND_MICROSECOND_PTS'
    allowed={f'NIA_SL_{word}_REAL{person:02d}_F'
             for word in ('WORD0739','WORD1133','WORD1567','WORD2523') for person in (11,12,13)}
    public_root=U.parent/'word100_multipeople_live_feedback_20261101/data_audit/videos'
    unique={}
    for v in m['views']:
        assert v['role']=='development' and v['recorded_person_id'] in [11,12,13] and v['labels'] is None
        assert v['training_allowed'] is False
        assert v['clip_id'] in allowed and v['clip_id'].endswith(f"_REAL{v['recorded_person_id']:02d}_F")
        assert Path(v['source_path']).resolve()==(public_root/(v['clip_id']+'.mp4')).resolve()
        assert len(v['pixel_view_key'])==64 and all(x in '0123456789abcdef' for x in v['pixel_view_key'])
        maps=v['ordered_frame_map']
        assert 2<=len(maps)<=1200
        assert [e['output_index'] for e in maps]==list(range(len(maps)))
        assert all(b['source_native_frame']>a['source_native_frame'] for a,b in zip(maps,maps[1:]))
        assert all(e['pixel_operation'] in ['copy','blackout'] and e['image_available']==(e['pixel_operation']=='copy') for e in maps)
        unique.setdefault(v['pixel_view_key'],v)
    assert len(m['views'])==len(unique)==80 and {v['clip_id'] for v in unique.values()}==allowed
    freeze_path=U/'DEV_PROCESS_VIDEO_FROZEN.json'
    pins={str(U/n):sha(U/n) for n in ['DEV_PARTIAL_VIEWS_MANIFEST.json','DEV_PROCESS_CODE_INPUT_AUDIT.json','DEV_PARTIAL_CONTRACT.json','DEV_PARTIAL_VIEWS_INTENT.json','RGB_PTS_CODEC_PROBE.json','dev_materialize_process_videos.py','PROCESS_VIDEO_MANIFEST.json','PROCESS_INPUT_MANIFEST.json']}
    if not freeze_path.exists():
        write(freeze_path,dict(at=now(),pins=pins,views=80,unique_media=80,
            codec='libx264rgb lossless RGB24, microsecond PTS, no B frames',new_inference=0,new_training=0,
            role='development',recorded_ids=[11,12,13],training_allowed=False,
            repeated_development=True,independent_final_test=False,
            no_orig_or_frozen_outputs_changed=True,purpose='New exact retained/masked pixels; extraction must not see removed clean RGB context.'))
    frozen=read(freeze_path);assert frozen['pins']==pins
    for p,h in frozen['pins'].items():assert sha(p)==h,p
    OUT.mkdir(parents=True,exist_ok=True);rows=[];t0=time.perf_counter()
    for cid in sorted({v['clip_id'] for v in unique.values()}):
        group=[v for v in unique.values() if v['clip_id']==cid];q=group[0];arrays=None
        assert all(v['source_path']==q['source_path'] and v['source_sha256']==q['source_sha256'] for v in group)
        assert sha(q['source_path'])==q['source_sha256']
        wanted={e['source_native_frame'] for v in group for e in v['ordered_frame_map']}
        pixels={};times={};origin=None
        with av.open(q['source_path']) as c:
            s=c.streams.video[0]
            for i,f in enumerate(c.decode(video=0)):
                assert f.pts is not None
                if origin is None:origin=f.pts
                if i in wanted:
                    pixels[i]=f.to_ndarray(format='rgb24');times[i]=round(float((f.pts-origin)*s.time_base)*1e6)
        assert set(pixels)==wanted
        for v in group:
            d=(OUT/v['pixel_view_key']).resolve();assert d.is_relative_to(OUT.resolve());end=d/'COMPLETE.json'
            if end.exists():
                r=read(end);assert r['freeze_sha256']==sha(freeze_path) and sha(r['video'])==r['video_sha256']
                assert r['pixel_view_key']==v['pixel_view_key'] and r['clip_id']==cid and r['role']=='development'
                assert Path(r['video']).resolve()==(d/'view.mp4').resolve()
                assert r['source_sha256']==q['source_sha256'] and r['source_vs_decoded_rgb_exact']
                assert r['source_frame_indices']==[e['source_native_frame'] for e in v['ordered_frame_map']]
                assert r['view_timestamps_us']==[e['view_timestamp_us'] for e in v['ordered_frame_map']]
                rows.append(r);continue
            assert not d.exists(),f'Inspect partial media without COMPLETE before retry: {d}'
            capacity();d.mkdir();write(d/'STARTED.json',dict(at=now(),pid=os.getpid(),pixel_view_key=v['pixel_view_key']))
            maps=v['ordered_frame_map'];arrays=[]
            for e in maps:
                f=e['source_native_frame'];assert times[f]==e['source_native_timestamp_us']
                a=pixels[f] if e['pixel_operation']=='copy' else np.full_like(pixels[f],128)
                assert e['pixel_operation'] in ['copy','blackout'] and e['image_available']==(e['pixel_operation']=='copy')
                arrays.append(a)
            pts=[e['view_timestamp_us'] for e in maps];assert pts[0]==0 and all(b>a for a,b in zip(pts,pts[1:]))
            video=d/'view.mp4';encode(video,arrays,pts)
            decoded_hashes=[];decoded_times=[]
            with av.open(str(video)) as c:
                for i,f in enumerate(c.decode(video=0)):
                    a=f.to_ndarray(format='rgb24');ts=round(float(f.pts*f.time_base)*1e6)
                    assert i<len(arrays) and ts==pts[i] and np.array_equal(a,arrays[i]),(v['view_id'],i,ts)
                    decoded_hashes.append(rgbsha(a));decoded_times.append(ts)
            assert len(decoded_hashes)==len(maps)
            r=dict(pixel_view_key=v['pixel_view_key'],clip_id=cid,person=v['recorded_person_id'],role='development',
                source=q['source_path'],source_sha256=q['source_sha256'],video=str(video),video_sha256=sha(video),
                aliases=[x['view_id'] for x in m['views'] if x['pixel_view_key']==v['pixel_view_key']],
                frames=len(maps),source_frame_indices=[e['source_native_frame'] for e in maps],
                source_timestamps_us=[e['source_native_timestamp_us'] for e in maps],view_timestamps_us=decoded_times,
                image_available=[e['image_available'] for e in maps],pixel_operations=[e['pixel_operation'] for e in maps],
                decoded_rgb_sha256=decoded_hashes,source_vs_decoded_rgb_exact=True,actual_sensor_extent=[0,0,arrays[0].shape[1],arrays[0].shape[0]],
                freeze_sha256=sha(freeze_path),inference=0,training=0,labels_assigned=0,
                training_allowed=False,repeated_development=True,independent_final_test=False)
            write(end,r);rows.append(r)
            write(U/'DEV_PROCESS_VIDEO_PROGRESS.json',dict(at=now(),status='ENCODING_AND_PIXEL_PTS_VERIFYING',pid=os.getpid(),done=len(rows),total=80,
                elapsed_seconds=time.perf_counter()-t0,free_bytes=capacity()))
            print('process media',len(rows),'/80',cid,len(maps),flush=True)
        assert sha(q['source_path'])==q['source_sha256']
        del pixels,arrays
    assert len(rows)==80
    for p,h in pins.items():assert sha(p)==h,p
    write(U/'DEV_PROCESS_VIDEO_MANIFEST.json',dict(at=now(),status='ACTUAL_PUBLIC_VIEW_PIXELS_PREPARED_AND_EXACTLY_VERIFIED',rows=rows,
        media=80,aliases=80,source_clips=12,freeze_sha256=sha(freeze_path),new_inference=0,new_training=0,
        training_allowed=False,repeated_development=True,independent_final_test=False,whole_goal_complete=False))
    write(U/'DEV_PROCESS_VIDEO_PROGRESS.json',dict(at=now(),status='COMPLETE',active=None,done=80,total=80,elapsed_seconds=time.perf_counter()-t0))
if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'DEV_PROCESS_VIDEO_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
