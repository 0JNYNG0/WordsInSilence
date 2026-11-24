"""Read-only AIHub REAL02 coordinate audit. No model training/inference."""
from pathlib import Path
import sys,json,hashlib,struct,re,zlib,zipfile,time,traceback
from datetime import datetime,timezone
import numpy as np
import cv2

sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
B=O/'word100_body_boundary_20261017'; OLD=O/'word_keypoint_archive_proposal_20260922'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,v):
    p=U/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def now():return datetime.now(timezone.utc).isoformat()
def stat(p):
    s=Path(p).stat();return [s.st_size,s.st_mtime_ns]
def stats(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=int(a.size),median=float(np.median(a)) if a.size else None,p90=float(np.quantile(a,.9)) if a.size else None,max=float(a.max()) if a.size else None)

def freeze():
    assert not (U/'FROZEN.json').exists()
    previous=read(O/'word100_train_reference_dev_replay_20261026/FROZEN.json')
    check(previous['protected'])
    context=read(B/'PREP_CONTEXT.json'); rows=[]; pins={}
    for r in sorted(context['ordered_rows'],key=lambda r:r['clip_id']):
        if r['signer_id']!=2:continue
        assert r['role']=='train_candidate'
        rp=B/'data/records'/(r['clip_id']+'.json');rec=read(rp)
        cap=rec['native_capture'];check(cap['pins'])
        z=np.load(cap['baseline']);xy=z['xy_px'][:,33:75].reshape(-1,2,21,2);obs=z['observed'][:,33:75].reshape(-1,2,21)
        wrists=xy[:,:,0];valid=obs[:,:,0];dist=np.linalg.norm(wrists[:,0]-wrists[:,1],axis=1)
        speed=np.linalg.norm(np.diff(wrists,axis=0),axis=2)
        motion=stats(speed)['p90'] or 0
        # Selection solely from existing MP, BEFORE inspecting provided coordinate quality.
        row=dict(clip_id=r['clip_id'],label=r['canonical_label'],word_code=r['word_code'],signer_id=2,role=r['role'],intervals=r['intervals'],record=str(rp),record_sha=sha(rp),source=rec['source'],baseline=cap['baseline'],candidate=cap['candidate'],baseline_sha=sha(cap['baseline']),candidate_sha=sha(cap['candidate']),sidecar=read(Path(cap['baseline']).with_suffix('.sidecar.json')),missing_rank=float(1-valid.mean()),cross_rank=float(np.sum((dist<100)&valid.all(axis=1))),speed_rank=motion,retained_video=r.get('retained_video'))
        rows.append(row);pins[str(rp)]=sha(rp);pins.update(cap['pins'])
    assert len(rows)==100
    selected=[]; reasons={}
    for i in [0,19,39,59,79,99]:
        c=rows[i]['clip_id'];selected.append(c);reasons[c]=['systematic_vocabulary_sample']
    for rank in ['missing_rank','cross_rank','speed_rank']:
        for r in sorted(rows,key=lambda r:(-r[rank],r['clip_id'])):
            if r['clip_id'] not in selected:
                selected.append(r['clip_id']);reasons[r['clip_id']]=[rank]
                if sum(rank in v for v in reasons.values())==2:break
    assert len(selected)==12
    ap=read(OLD/'INSPECTION_PLAN.json');arc=Path(ap['archive'])
    assert stat(arc)==[ap['archive_bytes'],ap['archive_mtime_ns']]
    for p in [B/'PREP_CONTEXT.json',OLD/'INSPECTION_PLAN.json',OLD/'INSPECTION_RESULT.json',OLD/'CLIP_DIRECTORY.json',Path(__file__)]:pins[str(p)]=sha(p)
    write('AUTHORIZATION.json',dict(at=now(),request='Inspect existing downloaded AIHub coordinates and compare quality. Additional downloads only if suitability established.',training=False,model_change=False,user_data_access=False,originals_read_only=True))
    write('FROZEN.json',dict(at=now(),rows=rows,visual_ids=selected,visual_selection_reasons=reasons,archive=str(arc),archive_stat=stat(arc),central=ap,protected=previous['protected'],pins=pins,scope='100 TRAIN REAL02 front clips structural audit; 12 MP-selected detailed visual clips; one performer only',protocol={'correspondence':'exact clip ID and source frame index; no fitted spatial/time transform','coordinates':'AIHub source pixel XY versus saved full-frame MP source pixel XY, anatomical left/right','heldout':'no DEV/test/personal/mentor data','valid_point':'finite XY, confidence>0, XY not both zero; not confidence calibration','visual_frames':'first,last,core25/50/75%,most overlap,most missing,largest wrist step; unique capped8','quality':'agreement is not groundtruth; image review is Codex visual review, not expert 21-point groundtruth','download_gate':'representative image agreement plus absence of unresolved systematic coordinate/time issues; otherwise no large download','failure':'preserve first exception; no silent sample exclusions','training':False}))
    print('FROZEN',len(rows),'visual',len(selected),flush=True)

def index():
    f=read(U/'FROZEN.json');assert stat(f['archive'])==f['archive_stat'];ap=f['central'];wanted={r['clip_id'] for r in f['rows']}
    selected=[];h=hashlib.sha256();pattern=re.compile(rb'(NIA_SL_WORD\d+_REAL02_F)_(\d+)_keypoints\.json$')
    with open(f['archive'],'rb',buffering=1024*1024) as z:
        z.seek(ap['directory_offset'])
        for _ in range(ap['directory_entries']):
            head=z.read(46);assert head[:4]==b'PK\x01\x02'
            nl,el,cl=struct.unpack_from('<3H',head,28);name=z.read(nl);extra=z.read(el);comment=z.read(cl);h.update(head+name+extra+comment)
            m=pattern.search(name)
            if not m or m[1].decode() not in wanted:continue
            flags,method=struct.unpack_from('<2H',head,8);crc,cs,us=struct.unpack_from('<3L',head,16);off=struct.unpack_from('<L',head,42)[0]
            pos=0
            while pos+4<=len(extra):
                tag,size=struct.unpack_from('<2H',extra,pos);e=extra[pos+4:pos+4+size];pos+=4+size
                if tag==1:
                    q=0
                    if us==0xffffffff:us=struct.unpack_from('<Q',e,q)[0];q+=8
                    if cs==0xffffffff:cs=struct.unpack_from('<Q',e,q)[0];q+=8
                    if off==0xffffffff:off=struct.unpack_from('<Q',e,q)[0]
            selected.append(dict(clip_id=m[1].decode(),frame=int(m[2]),name=name.decode('utf-8'),flags=flags,method=method,crc32=crc,compressed=cs,uncompressed=us,offset=off))
        assert z.tell()==ap['directory_offset']+ap['directory_bytes']
    assert h.hexdigest()==read(OLD/'INSPECTION_RESULT.json')['directory_sha256']
    write('INDEX.json',dict(archive=f['archive'],directory_sha=h.hexdigest(),members=selected,clips=len({m['clip_id'] for m in selected})))
    print('INDEX',len(selected),flush=True)

def payload(z,m):
    z.seek(m['offset']);a=struct.unpack('<IHHHHHIIIHH',z.read(30));assert a[0]==0x04034b50 and a[3]==m['method']
    name=z.read(a[-2]).decode('utf-8');z.read(a[-1]);assert name==m['name']
    raw=zlib.decompress(z.read(m['compressed']),-15);assert len(raw)==m['uncompressed'] and zlib.crc32(raw)&0xffffffff==m['crc32']
    return json.loads(raw),hashlib.sha256(raw).hexdigest()

def process():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);ind=read(U/'INDEX.json');groups={}
    for m in ind['members']:groups.setdefault(m['clip_id'],[]).append(m)
    results=[];zips={}
    try:
        with open(f['archive'],'rb') as z:
            for ri,r in enumerate(f['rows']):
                cid=r['clip_id'];out=U/'records'/(cid+'.json')
                assert not out.exists(),'Do not overwrite partial output'
                base=np.load(r['baseline']);mp=np.load(r['candidate']);n=len(mp['frame_id']);assert np.array_equal(base['frame_id'],mp['frame_id'])
                frames=mp['frame_id'];assert np.array_equal(frames,np.arange(n));assert np.all(np.diff(mp['timestamp_us'])>0)
                hands=np.full((n,2,21,3),np.nan,dtype=np.float64);pose=np.full((n,25,3),np.nan);xyz=np.full((n,2,21,4),np.nan);hashes={};extras=[];types=set();cam=0
                entries=sorted(groups.get(cid,[]),key=lambda m:m['frame']);missing=sorted(set(range(n))-{m['frame'] for m in entries})
                for m in entries:
                    obj,h=payload(z,m);t=m['frame'];hashes[str(t)]=h
                    person=obj['people'];types.add(type(person).__name__)
                    if isinstance(person,list):assert len(person)==1;person=person[0]
                    if t>=n:extras.append(t);continue
                    for side,key in enumerate(['hand_left_keypoints','hand_right_keypoints']):
                        a=np.asarray(person[key+'_2d']);assert a.size==63;hands[t,side]=a.reshape(21,3)
                        b=np.asarray(person.get(key+'_3d',[]))
                        if b.size:assert b.size==84;xyz[t,side]=b.reshape(21,4)
                    a=np.asarray(person['pose_keypoints_2d']);assert a.size==75;pose[t]=a.reshape(25,3)
                    cam+=bool(obj.get('camparam'))
                valid=np.isfinite(hands).all(axis=-1)&(hands[...,2]>0)&np.any(hands[...,:2]!=0,axis=-1)
                xy=mp['xy_px'][:,33:75].reshape(n,2,21,2);mv=mp['observed'][:,33:75].reshape(n,2,21)&np.isfinite(xy).all(axis=-1)
                bv=base['observed'][:,33:75].reshape(n,2,21);by=base['xy_px'][:,33:75].reshape(n,2,21,2)
                d=np.linalg.norm(hands[...,:2]-xy,axis=-1);d[~(valid&mv)]=np.nan
                bd=np.linalg.norm(hands[...,:2]-by,axis=-1);bd[~(valid&bv)]=np.nan
                swapped=np.linalg.norm(hands[:,::-1,:,:2]-xy,axis=-1);swapped[~(valid[:,::-1]&mv)]=np.nan
                w=r['sidecar']['width'];height=r['sidecar']['height'];outside=valid&((hands[...,0]<0)|(hands[...,0]>=w)|(hands[...,1]<0)|(hands[...,1]>=height))
                step=np.linalg.norm(np.diff(hands[...,:2],axis=0),axis=-1);step[~(valid[1:]&valid[:-1])]=np.nan
                repeated=int(sum(np.array_equal(hands[t],hands[t-1],equal_nan=True) for t in range(1,n)))
                palm=np.linalg.norm(hands[:,:,0,:2]-hands[:,:,9,:2],axis=-1);palm[~(valid[:,:,0]&valid[:,:,9])]=np.nan
                lag=[]
                for k in range(-5,6):
                    a=np.arange(n);b=a+k;ok=(b>=0)&(b<n);a=a[ok];b=b[ok];v=valid[a,:,0]&mv[b,:,0]
                    dd=np.linalg.norm(hands[a,:,0,:2]-xy[b,:,0],axis=-1);lag.append(dict(shift=k,**stats(dd[v])))
                source=r['source'];arc=source['archive'];zs=zips.setdefault(arc,zipfile.ZipFile(arc)) if arc not in zips else zips[arc]
                zi=zs.getinfo(source['member']);assert zi.CRC==source['crc32'] and zi.file_size==source['bytes']
                rawvideo=zs.read(zi);assert hashlib.sha256(rawvideo).hexdigest()==source['sha256']
                video=None;video_check=None
                if cid in f['visual_ids']:
                    vp=U/'clips'/(cid+'.mp4');vp.parent.mkdir(exist_ok=True);vp.write_bytes(rawvideo);video=str(vp)
                    cap=cv2.VideoCapture(video);fps=cap.get(cv2.CAP_PROP_FPS);reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));decoded=0
                    while True:
                        ok,frame=cap.read()
                        if not ok:break
                        assert frame.shape[:2]==(height,w);decoded+=1
                    cap.release();video_check=dict(fps=fps,reported_frames=reported,decoded_frames=decoded,exact_mp_frames=decoded==n,exact_official_frames=decoded==len(entries),timestamp_max_error_us=float(np.max(np.abs(mp['timestamp_us']-frames*1e6/fps))))
                ap=U/'arrays'/(cid+'.npz');ap.parent.mkdir(exist_ok=True)
                np.savez_compressed(ap,official=hands,official_pose=pose,official_xyz=xyz,official_valid=valid,mp_xy=xy,mp_valid=mv,baseline_xy=by,baseline_valid=bv,frame_id=frames,timestamp_us=mp['timestamp_us'],distance=d)
                res=dict(clip_id=cid,label=r['label'],frames=n,official_members=len(entries),missing_frames=missing,extra_frames=extras,crc_checked=len(entries),people_container_types=sorted(types),camera_records=cam,official_valid_points=int(valid.sum()),total_hand_points=int(valid.size),mp_valid_points=int(mv.sum()),baseline_valid_points=int(bv.sum()),provided_but_mp_absent_points=int((valid&~mv).sum()),confidence_exactly_one=int(((hands[...,2]==1)&valid).sum()),outside_image_points=int(outside.sum()),repeated_hand_frame_pairs=repeated,palm_length_px=stats(palm),wrist_disagreement_px=stats(d[:,:,0]),all_point_disagreement_px=stats(d),baseline_wrist_disagreement_px=stats(bd[:,:,0]),swapped_wrist_disagreement_px=stats(swapped[:,:,0]),point_step_px=stats(step),lag_diagnostic_only=lag,video=video,video_check=video_check,arrays=str(ap),arrays_sha=sha(ap),payload_sha256=hashes,source_sha=source['sha256'])
                write('records/'+cid+'.json',res);results.append(res);print('AUDIT',ri+1,cid,'missing',len(missing),'median_wrist',res['wrist_disagreement_px']['median'],flush=True)
        assert stat(f['archive'])==f['archive_stat'];check(f['protected'])
        write('DONE.json',dict(status='STRUCTURAL_AUDIT_COMPLETE_VISUAL_REVIEW_REQUIRED',rows=results,clips=len(results),frames=sum(r['frames'] for r in results),crc_checked=sum(r['crc_checked'] for r in results),training=0,new_detection=0,personal_data_access=0))
    finally:
        for z in zips.values():z.close()

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
