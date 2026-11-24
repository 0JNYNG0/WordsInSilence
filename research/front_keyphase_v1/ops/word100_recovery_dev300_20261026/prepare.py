from common import *
import os,time,zipfile,traceback,shutil

def freeze():
    p=read(U/'PREFLIGHT.json');check(p['pins']);check(p['protected']);assert not (U/'FROZEN.json').exists()
    # All runtime config stays in this unit. Existing checkpoint and source metadata are read-only.
    source=(OLD/'model_io.py').read_text('utf8').replace("p=U/'yolo_hand_best.pt'",f"p=Path({str(OLD/'yolo_hand_best.pt')!r})").replace("(U/'MODEL_SOURCE.json')",f"(Path({str(OLD/'MODEL_SOURCE.json')!r}))")
    (U/'model_io_local.py').write_text(source,encoding='utf8')
    paths=list(U.glob('*.py'))+[OLD/'experiment.py',OLD/'model_io.py',OLD/'MODEL_SOURCE.json',OLD/'yolo_hand_best.pt',
        ROOT/'Tools/models/hand_landmarker.task',O/'word100_yolo_selective_wrist_20261025/candidate.py']
    for name in ('C','T3F'):
        seeds=(20261013,) if name=='C' else (20261013,20261014,20261015)
        for s in seeds:
            paths+=[(B/f'training_C_{s}_best.pt') if name=='C' else T/f'T3F_{s}_best.pt',
                    (B/f'training_C_{s}_RESULT.json') if name=='C' else T/f'T3F_{s}_RESULT.json']
    pins=p['pins']|{str(q):sha(q) for q in paths}
    write(U/'FROZEN.json',dict(at=now(),rows=p['rows'],pins=pins,protected=p['protected'],counts=p['counts'],
        authorization='Persistent autonomous public TRAIN/DEV recognition and feedback improvement. No user/mentor/fresh199/200 inputs, no operating replacement.',
        protocol='Same frozen300 DEV IDs11/12/13, every clip/failed input retained. Reuse original full-source native detection and current upper-body ROI. Only whole missing hand with visible arm wrist insideROI triggers same-frame full IMAGE MP + corresponding crop using existing constants. Keep all original points. Then only still-missing eligible side triggers saved selective YOLO rule. No thresholds/word-specific changes, no HaMeR fills. Preserve actual full-source detections and merge before pointwise ROI.',
        modes=['native full-source -> original ROI','selective MP retry -> same ROI','MP then selective YOLO -> same ROI'],
        verification='Reconstruct existing input exactly; verify every accepted arm/duplicate decision from saved raw output, coordinates and original masks. All newly ROI-used hand frames get image-review queue before effect interpretation; normal/crossing old points remain exact. Wrist-only never shape. No step claiming hidden-finger truth. Runtime missing implies unknown, never normal.',
        inference='After full inputs verified, frozen C13/T3F3seeds/E3 once per distinct changed input; identical input uses original masked logits. Keep scores cached. Include fixed original span/size diagnostics separately. No model training or candidate-selection tuning on DEV scores during this frozen comparison.',
        acceptance='Report added evidence, regressions and total/triggered component time. Any malformed inputs stops inference; visual uncertainty stays explicit and prevents operational adoption, not silent exclusion from denominator. Positive project improvement requires gains without concealed losses; accuracy-neutral result remains neutral.',
        evaluation_limit='DEV is already used for model development and is not independent final validation. Source signerIDs not biometric identity. Same Codex numerical/visual verification not independent human ground truth.',
        prohibited=['camera','downloads','external','Git','operating edits','user/mentor/199/200','training','forced identity','per-word rules']))
    for name in ('sources','outputs','records','frames','starts'): (U/name).mkdir()
    event(dict(stage='FROZEN',clips=300,target_frames=2563))

def detect_mp(lm,rgb,mp,box=None):
    h,w=rgb.shape[:2];t=time.perf_counter();r=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
    out=[];off=np.asarray(box[:2] if box else [0,0])
    for pp,ss in zip(r.hand_landmarks,r.handedness):
        xy=np.array([[p.x*w,p.y*h] for p in pp])+off
        d=dict(xy=xy,label=ss[0].category_name,score=float(ss[0].score))
        if box:d['box']=box
        out.append(d)
    return out,(time.perf_counter()-t)*1000

def retained_old(raw,ass,i,hand,valid):
    n=int(raw['raw_count'][i]);old=list(raw['hand_xy_px'][i,:n]);sides=ass[i,:n].tolist()
    for s in (0,1):
        if valid[s].any() and s not in sides:old.append(hand[s]);sides.append(s)
    return old,sides

def run(limit=300):
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    import cv2,torch,mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    exp=module('dev300_yolo_detect',OLD/'experiment.py')
    selective=module('dev300_selective',O/'word100_yolo_selective_wrist_20261025/candidate.py')
    loader=module('dev300_model_io',U/'model_io_local.py');ym,environment=loader.load_model()
    opts=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(ROOT/'Tools/models/hand_landmarker.task')),
        running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid(),limit=limit))
    if not (U/'ENVIRONMENT.json').exists():write(U/'ENVIRONMENT.json',environment)
    done=0;zips={};started=time.perf_counter()
    with vision.HandLandmarker.create_from_options(opts) as lm:
        black=np.zeros((1080,1920,3),np.uint8);detect_mp(lm,black,mp)
        for _ in range(3):exp.ydetect(ym,black)
        for row in f['rows']:
            cid=row['clip_id'];rp=U/'records'/(cid+'.json')
            if rp.exists():
                old=read(rp);assert old['context_sha256']==sha(U/'FROZEN.json');check(old['outputs']);continue
            if done>=limit:break
            assert not (U/'starts'/(cid+'.json')).exists(), 'Partial prior attempt requires explicit audited recovery'
            write(U/'starts'/(cid+'.json'),dict(at=now(),clip_id=cid))
            rec=read(row['record']);native=load(rec['native_capture']['candidate']);raw=load(rec['native_capture']['raw']);ass=load(Path(rec['native_capture']['raw']).parent/'assignment.npz')['sides']
            n=row['frames'];box=row['box'];source=row['source'];video=U/'sources'/(cid+'.mp4')
            t0=time.perf_counter()
            if not video.exists():
                arc=source['archive']
                if arc not in zips:zips[arc]=zipfile.ZipFile(arc,'r')
                with zips[arc].open(source['member']) as a,video.open('xb') as b:shutil.copyfileobj(a,b,1024*1024)
            assert sha(video)==source['sha256'];source_seconds=time.perf_counter()-t0
            out={};base=load(rec['views']['masked']['observation'])
            for key in ('xy_px','observed','usable','conf','hand_score'):
                out[key]=np.stack([base[key].copy() for _ in range(3)])
            fullxy=np.stack([native['xy_px'].copy() for _ in range(3)]);fullok=np.stack([native['usable'].copy() for _ in range(3)])
            out.update(frame_id=native['frame_id'],timestamp_us=native['timestamp_us'],roi_box=np.asarray(box),
                point_origin=np.where(out['usable'],1,0).astype(np.uint8),position_only=np.zeros((n,2),bool),
                timing_ms=np.zeros((n,6)),mp_called=np.zeros(n,bool),yolo_called=np.zeros(n,bool))
            records=[];cap=cv2.VideoCapture(str(video));assert cap.isOpened();target_index={q['index']:q['sides'] for q in row['recovery_target_frames']}
            frame_id=0
            for i,target in enumerate(native['frame_id']):
                ts=time.perf_counter()
                while frame_id<=int(target):ok,bgr=cap.read();assert ok;frame_id+=1
                out['timing_ms'][i,0]=(time.perf_counter()-ts)*1000
                p=native['xy_px'][i,33:].reshape(2,21,2);v=native['usable'][i,33:].reshape(2,21)
                pose=raw['pose_xy_px'][i];pu=raw['pose_usable'][i];w,h=int(raw['width'][i]),int(raw['height'][i]);assert bgr.shape[:2]==(h,w)
                boxes=selective.trigger(p,v,pose,pu,w,h,box)
                assert sorted(boxes)==sorted(target_index.get(i,[]))
                if not boxes:continue
                out['mp_called'][i]=True;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);full,ms=detect_mp(lm,rgb,mp);out['timing_ms'][i,1]=ms;crop=[]
                for side,b in boxes.items():
                    ds,ms=detect_mp(lm,rgb[b[1]:b[3],b[0]:b[2]],mp,b);out['timing_ms'][i,2]+=ms
                    for q in ds:q['requested_side']=side
                    crop.extend(ds)
                t0=time.perf_counter();old,sides=retained_old(raw,ass,i,p,v);dec=rule.merge(full,crop,old,sides,pose,pu,w,h)
                pp=p.copy();vv=v.copy();orig=np.where(v,1,0).astype(np.uint8)
                for q in dec:
                    if not q['accepted']:continue
                    s=q['side']
                    if s not in boxes or v[s].any():q.update(accepted=False,reason='not_missing_target');continue
                    good=np.asarray(q['joint_supported'],bool);pp[s]=np.where(good[:,None],points(full[q['index']]),np.nan);vv[s]=good;orig[s]=np.where(good,2,0)
                out['timing_ms'][i,5]+=(time.perf_counter()-t0)*1000
                fullxy[1,i,33:]=pp.reshape(42,2);fullok[1,i,33:]=vv.reshape(42)
                yboxes=selective.trigger(pp,vv,pose,pu,w,h,box);yfull=[];ycrop=[];ydec=[];fp=pp.copy();fv=vv.copy();forig=orig.copy()
                if yboxes:
                    out['yolo_called'][i]=True;yfull,ms=exp.ydetect(ym,bgr);out['timing_ms'][i,3]=ms
                    for side,b in yboxes.items():
                        ds,ms=exp.ydetect(ym,bgr[b[1]:b[3],b[0]:b[2]],b);out['timing_ms'][i,4]+=ms
                        for q in ds:q['requested_side']=side
                        ycrop.extend(ds)
                    t0=time.perf_counter();nn=int(raw['raw_count'][i]);_,_,fp,fv,ydec=selective.supplement(yfull,ycrop,pp,vv,raw['hand_xy_px'][i,:nn],ass[i,:nn].tolist(),pose,pu,w,h,yboxes)
                    out['timing_ms'][i,5]+=(time.perf_counter()-t0)*1000
                    for q in ydec:
                        if q['accepted']:
                            s=q['side'];forig[s]=np.where(np.asarray(q['joint_supported']),3,0);out['position_only'][i,s]=q['position_only']
                fullxy[2,i,33:]=fp.reshape(42,2);fullok[2,i,33:]=fv.reshape(42)
                for k,(a,b,org) in enumerate(((pp,vv,orig),(fp,fv,forig)),1):
                    a,b=roi(a,b,box);sl=slice(33,75);oldvalid=base['usable'][i,sl].reshape(2,21)
                    assert b[oldvalid].all() and same(a[oldvalid],base['xy_px'][i,sl].reshape(2,21,2)[oldvalid])
                    out['xy_px'][k,i,sl]=a.reshape(42,2);out['observed'][k,i,sl]=b.reshape(42);out['usable'][k,i,sl]=b.reshape(42)
                    out['point_origin'][k,i,sl]=np.where(b,org,0).reshape(42)
                    if k==2:
                        out['conf'][k,i]=out['conf'][1,i];out['hand_score'][k,i]=out['hand_score'][1,i]
                    for q in (dec if k==1 else ydec):
                        if not q['accepted']:continue
                        s=q['side'];hs=slice(33+21*s,54+21*s);kept=out['usable'][k,i,hs]
                        fa,ca=(full,crop) if k==1 else (yfull,ycrop)
                        confidence=min(float(fa[q['index']]['score']),float(ca[q['crop_index']]['score']))
                        out['conf'][k,i,hs]=np.where(kept,confidence,0);out['hand_score'][k,i,s]=confidence
                new=out['usable'][2,i]&~out['usable'][0,i]
                if new.any():
                    assert cv2.imwrite(str(U/'frames'/f'{cid}_{i:04d}.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
                records.append(exp.serializable(dict(i=i,frame_id=int(target),timestamp_us=int(native['timestamp_us'][i]),boxes=boxes,old_sides=sides,
                    mp_full=full,mp_crop=crop,mp_decisions=dec,yolo_boxes=yboxes,yolo_full=yfull,yolo_crop=ycrop,yolo_decisions=ydec,
                    added_roi_points=int(new.sum()))))
            cap.release();assert n==len(native['frame_id']);out.update(full_xy_px=fullxy,full_usable=fullok)
            for k in (1,2):
                keep=base['usable'];assert out['usable'][k][keep].all() and same(out['xy_px'][k][keep],base['xy_px'][keep])
            op=U/'outputs'/(cid+'.npz');np.savez_compressed(op,**out)
            outputs={str(op):sha(op)}
            new=int((out['usable'][2]&~out['usable'][0]).sum());mpnew=int((out['usable'][1]&~out['usable'][0]).sum())
            write(rp,dict(at=now(),clip_id=cid,context_sha256=sha(U/'FROZEN.json'),frames=records,outputs=outputs,
                source_video=str(video),source_sha256=source['sha256'],source_seconds=source_seconds,mp_calls=int(out['mp_called'].sum()),yolo_calls=int(out['yolo_called'].sum()),mp_added=mpnew,yolo_added=new-mpnew))
            done+=1;event(dict(stage='CLIP_DONE',clip_id=cid,total_done=len(list((U/'records').glob('*.json'))),mp_new=mpnew,yolo_new=new-mpnew))
    for z in zips.values():z.close()
    check(f['protected']);check(f['pins']);total=len(list((U/'records').glob('*.json')))
    event(dict(stage='BATCH_COMPLETE',newly_done=done,total_done=total,elapsed_seconds=time.perf_counter()-started));(U/'RUNNING.json').unlink()
    if total==300:write(U/'PREP_DONE.json',dict(at=now(),clips=300,status='PREPARED_PENDING_INDEPENDENT_INPUT_AND_VISUAL_VERIFICATION'))

if __name__=='__main__':
    try:
        if sys.argv[1]=='freeze':freeze()
        else:run(int(sys.argv[2]) if len(sys.argv)>2 else 300)
    except Exception:
        p=U/('ERROR_'+now().replace(':','_')+'.txt');p.write_text(traceback.format_exc(),encoding='utf8');raise
