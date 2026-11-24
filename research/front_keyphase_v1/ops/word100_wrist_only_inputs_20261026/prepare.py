from common import *
import os,time,zipfile,traceback,shutil

def run(stage,limit):
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);fh=sha(U/'FROZEN.json')
    assert not (U/'RUNNING.json').exists(),'Inspect actual execution before any recovery; never remove live lock.'
    if stage=='all':assert read(U/'SMOKE_REVIEW.json')['ready_for_full'] is True
    import cv2,torch,mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    helper=module('wrist_reused_detector_helpers',V/'prepare.py')
    exp=module('wrist_yolo_image_detector',OLD/'experiment.py')
    selective=module('wrist_selective_rule',Y/'candidate.py')
    loader=module('wrist_restricted_loader',U/'model_io_local.py');ym,environment=loader.load_model()
    opts=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(ROOT/'Tools/models/hand_landmarker.task')),
        running_mode=vision.RunningMode.IMAGE,**f['mp_options'])
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid(),stage=stage,limit=limit,context_sha256=fh))
    if not (U/'ENVIRONMENT.json').exists():write(U/'ENVIRONMENT.json',environment)
    zips={};done=0;start=time.perf_counter()
    rows=[r for r in f['rows'] if stage=='all' or r['uid'] in f['smoke_uids']]
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for row in rows:
            uid=row['uid'];cid=row['clip_id'];rp=U/'records'/(uid+'.json')
            if rp.exists():
                prior=read(rp);assert prior['context_sha256']==fh;check(prior['outputs']);continue
            if done>=limit:break
            assert not (U/'starts'/(uid+'.json')).exists(),'Preserve partial attempt; audited recovery needed.'
            check(row['input_pins']);write(U/'starts'/(uid+'.json'),dict(at=now(),uid=uid))
            info=row['info'];arr=info['source_arrays'];native=load(arr['capture']);raw=load(arr['raw']);ass=load(Path(arr['capture']).parent/'assignment.npz')['sides']
            base=load(arr['observation']);oldch=load(arr['channels']);n=info['frames'];box=info['original_roi'];w,h=info['canvas']
            assert box[:2]==[0,0] and equal(raw['frame_id'],native['frame_id'])
            targets={q['index']:q['sides'] for q in row['targets']}
            cache={};video=U/'sources'/(cid+'.mp4');source_seconds=0
            if cid in f['cache']:
                cr=f['cache'][cid];assert sha(cr['file'])==cr['sha256'];saved=read(cr['file'])
                assert saved['source_sha256']==row['source']['sha256'];video=Path(cr['video'])
                if saved['canvas']==[w,h]:cache=saved['frames']
            if targets:
                t0=time.perf_counter()
                if not video.exists():
                    arc=row['source']['resolved_archive']
                    if arc not in zips:zips[arc]=zipfile.ZipFile(arc,'r')
                    zi=zips[arc].getinfo(row['source']['member']);assert zi.CRC==row['source']['crc32'] and zi.file_size==row['source']['bytes']
                    with zips[arc].open(zi) as a,video.open('xb') as b:shutil.copyfileobj(a,b,1024*1024)
                assert sha(video)==row['source']['sha256'];source_seconds=time.perf_counter()-t0
            out={k:np.stack([base[k].copy() for _ in range(3)]) for k in ('xy_px','observed','usable','conf','hand_score')}
            fullxy=np.stack([native['xy_px'].copy() for _ in range(3)]);fullok=np.stack([native['usable'].copy() for _ in range(3)])
            wrists=np.full((n,2,2),np.nan,np.float32);wv=np.zeros((n,2),bool)
            out.update(frame_id=native['frame_id'],timestamp_us=native['timestamp_us'],roi_box=np.asarray(box),
                point_origin=np.where(out['usable'],1,0).astype(np.uint8),position_only=np.zeros((n,2),bool),
                timing_ms=np.zeros((n,6)),mp_called=np.zeros(n,bool),yolo_called=np.zeros(n,bool),cache_counts=np.zeros((n,4),np.int32))
            records=[];cap=None;next_frame=0
            if targets:cap=cv2.VideoCapture(str(video));assert cap.isOpened()
            for i in sorted(targets):
                target=int(native['frame_id'][i]);ts=time.perf_counter()
                while next_frame<=target:ok,bgr=cap.read();assert ok;next_frame+=1
                # A camera crop is a different available image, not a post-inference filter.
                if row['view']=='sensor':bgr=np.ascontiguousarray(bgr[0:h,0:w])
                assert bgr.shape[:2]==(h,w)
                out['timing_ms'][i,0]=(time.perf_counter()-ts)*1000
                p=native['xy_px'][i,33:].reshape(2,21,2);v=native['usable'][i,33:].reshape(2,21)
                pose=raw['pose_xy_px'][i];pu=raw['pose_usable'][i]
                boxes=selective.trigger(p,v,pose,pu,w,h,box);assert sorted(boxes)==sorted(targets[i])
                savedframe=cache.get(str(target),{});calls=[]
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                def detect(kind,side=None,b=None):
                    entry=savedframe.get(kind,{});key='full' if b is None else ','.join(map(str,b))
                    available='full' in entry if b is None else key in entry.get('crops',{})
                    if available:
                        ds=restore(entry['full'] if b is None else entry['crops'][key]);ms=0
                        origin=entry['source'];out['cache_counts'][i,(0 if kind=='mp' else 2)+(b is not None)]+=1
                    else:
                        if kind=='mp':ds,ms=helper.detect_mp(lm,rgb if b is None else rgb[b[1]:b[3],b[0]:b[2]],mp,b)
                        else:ds,ms=exp.ydetect(ym,bgr if b is None else bgr[b[1]:b[3],b[0]:b[2]],b)
                        origin='new_current_available_image'
                    if side is not None:
                        for d in ds:d['requested_side']=side
                    calls.append(dict(detector=kind,box=b,cached=available,source=origin,elapsed_ms=ms))
                    out['timing_ms'][i,(1 if kind=='mp' else 3)+(b is not None)]+=ms
                    return ds
                out['mp_called'][i]=True;full=detect('mp');crop=[]
                for s,b in boxes.items():crop.extend(detect('mp',s,b))
                ts=time.perf_counter();old,sides=helper.retained_old(raw,ass,i,p,v);dec=rule.merge(full,crop,old,sides,pose,pu,w,h)
                pp=p.copy();vv=v.copy();org=np.where(v,1,0).astype(np.uint8)
                for q in dec:
                    if not q['accepted']:continue
                    s=q['side']
                    if s not in boxes or v[s].any():q.update(accepted=False,reason='not_missing_target');continue
                    good=np.asarray(q['joint_supported'],bool);pp[s]=np.where(good[:,None],points(full[q['index']]),np.nan);vv[s]=good;org[s]=np.where(good,2,0)
                out['timing_ms'][i,5]+=(time.perf_counter()-ts)*1000
                fullxy[1,i,33:]=pp.reshape(42,2);fullok[1,i,33:]=vv.reshape(42)
                yboxes=selective.trigger(pp,vv,pose,pu,w,h,box);yf=[];yc=[];yd=[];fp=pp.copy();fv=vv.copy();fo=org.copy()
                if yboxes:
                    out['yolo_called'][i]=True;yf=detect('yolo')
                    for s,b in yboxes.items():yc.extend(detect('yolo',s,b))
                    ts=time.perf_counter();nn=int(raw['raw_count'][i]);_,_,fp,fv,yd=selective.supplement(yf,yc,pp,vv,raw['hand_xy_px'][i,:nn],ass[i,:nn].tolist(),pose,pu,w,h,yboxes)
                    out['timing_ms'][i,5]+=(time.perf_counter()-ts)*1000
                    for q in yd:
                        if q['accepted']:
                            s=q['side'];fo[s]=np.where(np.asarray(q['joint_supported']),3,0);out['position_only'][i,s]=q['position_only']
                fullxy[2,i,33:]=fp.reshape(42,2);fullok[2,i,33:]=fv.reshape(42)
                for k,(a,b,origin,decisions,fa,ca) in enumerate(((pp,vv,org,dec,full,crop),(fp,fv,fo,yd,yf,yc)),1):
                    a,b=roi(a,b,box);sl=slice(33,75);keep=base['usable'][i,sl].reshape(2,21)
                    assert b[keep].all() and equal(a[keep],base['xy_px'][i,sl].reshape(2,21,2)[keep])
                    out['xy_px'][k,i,sl]=a.reshape(42,2);out['observed'][k,i,sl]=b.reshape(42);out['usable'][k,i,sl]=b.reshape(42)
                    out['point_origin'][k,i,sl]=np.where(b,origin,0).reshape(42)
                    if k==2:out['conf'][k,i]=out['conf'][1,i];out['hand_score'][k,i]=out['hand_score'][1,i]
                    for q in decisions:
                        if not q['accepted']:continue
                        s=q['side'];hs=slice(33+21*s,54+21*s);confidence=min(float(fa[q['index']]['score']),float(ca[q['crop_index']]['score']))
                        out['conf'][k,i,hs]=np.where(out['usable'][k,i,hs],confidence,0);out['hand_score'][k,i,s]=confidence
                for s in (0,1):
                    j=33+21*s
                    if not v[s].any() and out['usable'][2,i,j]:wrists[i,s]=out['xy_px'][2,i,j];wv[i,s]=True
                added=out['usable'][2,i]&~base['usable'][i]
                if wv[i].any():assert cv2.imwrite(str(U/'frames'/f'{uid}_{i:04d}.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
                records.append(exp.serializable(dict(i=i,frame_id=target,timestamp_us=int(native['timestamp_us'][i]),canvas=[w,h],boxes=boxes,old_sides=sides,
                    mp_full=full,mp_crop=crop,mp_decisions=dec,yolo_boxes=yboxes,yolo_full=yf,yolo_crop=yc,yolo_decisions=yd,calls=calls,
                    added_roi_points=int(added.sum()),wrist_sides=np.flatnonzero(wv[i]).tolist())))
            if cap:cap.release()
            out.update(full_xy_px=fullxy,full_usable=fullok,wrist_xy=wrists,wrist_usable=wv)
            features,changed=wrist_channels(info,base,oldch,wrists,wv)
            for k in (1,2):
                keep=base['usable'];assert out['usable'][k][keep].all() and equal(out['xy_px'][k][keep],base['xy_px'][keep])
            op=U/'outputs'/(uid+'.npz');np.savez_compressed(op,**out)
            xp=U/'features'/(uid+'.npz');np.savez_compressed(xp,**features)
            outputs={str(p):sha(p) for p in (op,xp)};check(row['input_pins'])
            write(rp,dict(at=now(),uid=uid,clip_id=cid,view=row['view'],context_sha256=fh,outputs=outputs,frames=records,
                source_video=str(video) if targets else None,source_sha256=row['source']['sha256'],source_seconds=source_seconds,
                mp_frames=int(out['mp_called'].sum()),yolo_frames=int(out['yolo_called'].sum()),cache_counts=out['cache_counts'].sum(0).tolist(),
                new_wrists=int(wv.sum()),changed_feature_wrists=changed))
            done+=1;event(dict(stage='INPUT_DONE',uid=uid,total=len(list((U/'records').glob('*.json'))),wrists=int(wv.sum()),cached=out['cache_counts'].sum(0).tolist(),elapsed=round(time.perf_counter()-start,1)))
    for z in zips.values():z.close()
    check(f['pins']);check(f['protected']);event(dict(stage='BATCH_COMPLETE',batch=stage,newly_done=done,total=len(list((U/'records').glob('*.json'))),elapsed_seconds=time.perf_counter()-start))
    (U/'RUNNING.json').unlink()
if __name__=='__main__':
    try:run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 1798)
    except Exception:
        write(U/('ERROR_'+now().replace(':','_')+'.json'),dict(error=traceback.format_exc()));raise
