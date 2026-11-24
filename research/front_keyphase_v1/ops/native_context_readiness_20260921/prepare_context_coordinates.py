"""Acquire surrounding public coordinates with existing reader; reuse approved middle frames."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent; LAB=HERE.parents[1]; PILOT=LAB.parent/'sign_model_pilot_v1'
sys.path[:0]=[str(PILOT),str(PILOT/'src')]
import expand_build as source_reader
from pilot.data import build_channels

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def new_json(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    started=time.monotonic()
    eligibility=read(HERE/'eligibility.json')
    entries=[r for r in eligibility['rows'] if r['status']=='ELIGIBLE_BY_SOURCE_ANNOTATION']
    assert len(entries)==298 and Counter(r['split_role'] for r in entries)=={'train':224,'validation':74}
    cache=LAB/'runs/native_full_context_v1'; cache.mkdir(exist_ok=False)
    index_dir=HERE/'new_frame_indexes'; index_dir.mkdir(exist_ok=False)
    source_pins={str(p):sha(p) for p in (Path(source_reader.__file__),Path(source_reader.PREP)/'zip_probe.py',PILOT/'src/pilot/data.py',HERE/'eligibility.json')}
    new_json(HERE/'coordinate_plan.json',{
        'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'source_pins':source_pins,
        'inputs':[{'clip_id':r['clip_id'],'signer_id':r['signer_id'],'split_role':r['split_role'],'source_samples':r['source_samples']} for r in entries],
        'selection_rule':'All298 metadata-eligible existing public samples; no score or word filtering',
        'output':str(cache),'whole_input_definition':'All available indexed source frames, never assume native index0/video origin agreement',
        'reuse':'108 saved full-frame indexes; original annotated coordinates reused from hashed native caches for every clip',
        'new_reads':'One targeted directory pass per allowed signer for190 clips with unsaved full-frame metadata; decode only frames outside existing annotation caches, verifying member CRC with existing reader',
        'changes':'Operational acquisition only; production/code readers and original caches stay unchanged; no training or inference',
        'missingness':'No interpolation, smoothing, inferred points, hand removal, or filled source gaps',
        'forbidden_signers':[10,14,15,16,17,18,25],
    })
    grouped=defaultdict(list)
    for r in entries:grouped[r['signer_id']].append(r)
    rows=[]; archive_results=[]; failures=[]
    for signer, clips in sorted(grouped.items()):
        info=eligibility['index_provenance'][str(signer)]
        archive=Path(info['archive']); st=archive.stat()
        assert (st.st_size,st.st_mtime_ns)==(info['archive_bytes'],info['archive_mtime_ns'])
        assert sha(info['path'])==info['sha256']
        old=read(info['path']); by_clip=defaultdict(list)
        wanted={r['clip_id'] for r in clips}
        for member in old['members']:
            if member['clip_id'] in wanted:by_clip[member['clip_id']].append(member)
        missing=wanted-set(by_clip)
        scans=0; directory_stats=None; new_index=None
        if missing:
            members,directory_stats=source_reader.selected_directory(archive,missing)
            scans=1
            assert all(m['clip_id'] in missing for m in members)
            for member in members:by_clip[member['clip_id']].append(member)
            new_index=index_dir/f'keypoint_index_{signer:02}.json'
            new_json(new_index,{'archive':str(archive),'archive_stat':info,'requested_clip_ids':sorted(missing),
                                'members':members,'stats':directory_stats})
        decoded_count=reused_count=0
        with archive.open('rb') as handle:
            for item in clips:
                ident=item['clip_id']; original=item['source_samples'][0]
                row={'clip_id':ident,'signer_id':signer,'split_role':item['split_role'],
                     'reference_label':item['reference_label'],'status':'ERROR'}
                try:
                    frame_map={}
                    for member in by_clip[ident]:
                        match=source_reader.FRAME_RE.search(member['name'])
                        assert match is not None
                        frame=int(match.group(1)); assert frame not in frame_map
                        frame_map[frame]=member
                    frames=sorted(frame_map); assert frames
                    assert sha(original['existing_npz'])==original['npz_sha256']
                    with np.load(original['existing_npz'],allow_pickle=False) as saved:
                        old_idx=saved['frame_index'].copy(); old_xy=saved['xy_native'].copy()
                        old_conf=saved['confidence_native'].copy(); old_present=saved['coordinate_present'].copy()
                        old_ts=saved['timestamp_nominal_us'].copy()
                    assert set(map(int,old_idx))<=set(frames)
                    existing={int(fi):i for i,fi in enumerate(old_idx)}
                    xycs=[]; reused=decoded=0
                    for fi in frames:
                        if fi in existing:
                            oi=existing[fi]
                            a=np.concatenate([old_xy[oi],old_conf[oi,:,None]],axis=1)
                            reused+=1
                        else:
                            a=source_reader.decode(handle,frame_map[fi]);decoded+=1
                        xycs.append(a)
                    arr=np.stack(xycs); indices=np.array(frames,dtype=np.int64)
                    present=np.isfinite(arr).all(axis=-1)&~np.all(arr[...,:2]==0,axis=-1)
                    ts=np.rint(indices/source_reader.FPS*1e6).astype(np.int64)
                    take=np.searchsorted(indices,old_idx)
                    assert np.array_equal(arr[take,:,:2],old_xy)
                    assert np.array_equal(arr[take,:,2],old_conf)
                    assert np.array_equal(present[take],old_present)
                    assert np.array_equal(ts[take],old_ts)
                    built=build_channels(arr[...,:2],present,ts)
                    assert built['x'].shape==(len(frames),280) and np.isfinite(built['x']).all()
                    dest=cache/f'{ident}_all_available.npz'
                    with dest.open('xb') as f:
                        np.savez_compressed(f,xy_native=arr[...,:2],confidence_native=arr[...,2],
                            coordinate_present=present,frame_index=indices,timestamp_nominal_us=ts,
                            annotation_source_rows=take,x=built['x'])
                    row.update(status='PREPARED',npz=str(dest),npz_sha256=sha(dest),bytes=dest.stat().st_size,
                        source_archive=str(archive),full_index_source=info['path'] if ident not in missing else str(new_index),
                        original_annotation_npz=original['existing_npz'],original_annotation_sha256=original['npz_sha256'],
                        frames=len(frames),first_frame=frames[0],last_frame=frames[-1],
                        annotation_frames=len(old_idx),annotation_source_bounds=[int(old_idx[0]),int(old_idx[-1])],
                        before_annotation_frames=int((indices<old_idx[0]).sum()),after_annotation_frames=int((indices>old_idx[-1]).sum()),
                        gaps=[{'before_frame':int(indices[i]),'after_frame':int(indices[i+1])} for i in np.flatnonzero(np.diff(indices)!=1)],
                        original_annotation_values_preserved=True,new_crc_checked_frames=decoded,reused_annotation_frames=reused,
                        source_duration_s=item['duration_s'],last_index_seconds=frames[-1]/source_reader.FPS,
                        native_time_origin='UNVERIFIED_AGAINST_VIDEO',full_action_annotation_completeness='UNVERIFIED',
                        all_frames_any_native_hand_fraction=float(present[:,25:67].any(axis=1).mean()),
                        note='Provided native coordinate presence, not MediaPipe detection or execution correctness')
                    decoded_count+=decoded;reused_count+=reused
                except Exception as exc:
                    row.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
                    failures.append({'clip_id':ident,'error':row['error']})
                rows.append(row)
        after=archive.stat(); assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        archive_results.append({'signer_id':signer,'central_directory_scans':scans,'directory_stats':directory_stats,
            'old_index_clips_reused':len(wanted-missing),'new_index_clips':len(missing),
            'new_crc_checked_frames':decoded_count,'reused_annotation_frames':reused_count,'archive_unchanged':True})
        print(json.dumps({'signer':signer,'prepared':sum(r['status']=='PREPARED' for r in rows if r['signer_id']==signer),
                          'new_frames':decoded_count,'reused_frames':reused_count}),flush=True)
    for path,digest in source_pins.items():assert sha(path)==digest,path
    ready=[r for r in rows if r['status']=='PREPARED']
    result={'status':'PREPARED_PENDING_INDEPENDENT_AUDIT' if len(ready)==298 else 'INCOMPLETE',
        'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'plan_sha256':sha(HERE/'coordinate_plan.json'),
        'script_sha256':sha(__file__),'rows':rows,'archives':archive_results,'failures':failures,
        'counts':{'planned':298,'prepared':len(ready),'errors':len(failures),
                  'by_role':dict(Counter(r['split_role'] for r in ready)),
                  'total_frames':sum(r['frames'] for r in ready),'new_crc_checked_frames':sum(r['new_crc_checked_frames'] for r in ready),
                  'reused_annotation_frames':sum(r['reused_annotation_frames'] for r in ready),'bytes':sum(r['bytes'] for r in ready),
                  'clips_with_gaps':sum(bool(r['gaps']) for r in ready)},
        'elapsed_seconds':round(time.monotonic()-started,3),'training':0,'classifier_inference':0,
        'new_videos_or_mediapipe':0,'user_arrays_read':0,'reserved_arrays_read':0,'production_changes':0}
    new_json(HERE/'coordinates.json',result)
    print(json.dumps(result['counts']),flush=True)
    return 0 if not failures and len(ready)==298 else 1


if __name__=='__main__':
    raise SystemExit(main())
