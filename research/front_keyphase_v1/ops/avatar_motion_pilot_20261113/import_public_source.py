"""Import exactly one frozen public clip from a user-located AIHub ZIP.

This does not approve source geometry or choose a camera convention. It produces
two explicitly labelled projection diagnostics for later same-view RGB review.
"""
from pathlib import Path
import argparse,hashlib,json,re,zipfile,traceback
import numpy as np
from geometry_core import camera,project

U=Path(__file__).resolve().parent
CID='NIA_SL_WORD1567_REAL02'
MATCH=re.compile(r'(?:^|/)('+CID+r'_([FLRUD]))/(?:'+CID+r'_\2)_(\d+)_keypoints\.json$')


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(data):return hashlib.sha256(data).hexdigest()
def stats(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(x.max()))if len(x)else dict(n=0)


def array(person,key,points,dim):
    x=np.asarray(person.get(key,[]),dtype=float)
    if x.size!=points*dim:
        raise ValueError(f'{key}: {x.size} values, expected {points*dim}; no fill or invented body3D')
    return x.reshape(points,dim)


def main(path):
    archive=Path(path)
    if not archive.is_file():raise FileNotFoundError(archive)
    source_stat=[archive.stat().st_size,archive.stat().st_mtime_ns]
    assert not (U/'SOURCE_PIN.json').exists(),'Source import already started; inspect evidence before resuming'
    write(U/'SOURCE_PIN.json',dict(path=str(archive.resolve()),stat=source_stat,clip=CID,
          frozen_sha256=sha((U/'FROZEN.json').read_bytes()),quality_checked=False))
    old_record=read(U.parent/'word100_aihub_coordinate_quality_20261026/records'/f'{CID}_F.json')
    out=U/'source';out.mkdir(exist_ok=True)
    groups={s:[]for s in 'FLRUD'}
    records=[]
    with zipfile.ZipFile(archive)as z:
        for info in z.infolist():
            m=MATCH.search(info.filename)
            if m:groups[m[2]].append((int(m[3]),info))
        if any(not v for v in groups.values()):raise ValueError('All five same-clip keypoint views are required')
        summaries={}
        for view,items in groups.items():
            items.sort(key=lambda v:v[0]);ids=[v[0]for v in items]
            if ids!=list(range(len(ids))):raise ValueError(f'{view}: source frame gaps or repeated IDs; no renumbering')
            hands2=[];hands3=[];pose2=[];pose3=[];ks=[];ms=[];ds=[]
            for frame,info in items:
                raw=z.read(info) # ZipFile checks CRC; only exact REAL02 selected payloads.
                obj=json.loads(raw)
                person=obj['people']
                if isinstance(person,list):
                    if len(person)!=1:raise ValueError('Expected one annotated person')
                    person=person[0]
                k,m,d=camera(obj['camparam'])
                hands2.append(np.stack([array(person,f'hand_{side}_keypoints_2d',21,3)for side in ['left','right']]))
                hands3.append(np.stack([array(person,f'hand_{side}_keypoints_3d',21,4)for side in ['left','right']]))
                pose2.append(array(person,'pose_keypoints_2d',25,3));pose3.append(array(person,'pose_keypoints_3d',25,4))
                ks.append(k);ms.append(m);ds.append(d)
                digest=sha(raw)
                if view=='F' and str(frame)in old_record['payload_sha256']:
                    if digest!=old_record['payload_sha256'][str(frame)]:raise ValueError('Recovered front source differs from preserved original '+str(frame))
                dest=out/f'{CID}_{view}_{frame:012d}_keypoints.json';dest.write_bytes(raw)
                records.append(dict(member=info.filename,view=view,frame=frame,sha256=digest,crc32=info.CRC,bytes=len(raw),path=str(dest.resolve())))
            h2,h3,p2,p3=np.asarray(hands2),np.asarray(hands3),np.asarray(pose2),np.asarray(pose3)
            ks,ms,ds=np.asarray(ks),np.asarray(ms),np.asarray(ds)
            cache=out/f'{CID}_{view}.npz'
            np.savez_compressed(cache,hand2d=h2,hand3d=h3,pose2d=p2,pose3d=p3,K=ks,extrinsic=ms,distortion=ds,frame_id=ids)
            diag={}
            for mode in ['camera','world']:
                errors=[]
                for i in range(len(ids)):
                    xyz=np.concatenate([p3[i,:,:3],h3[i,:,:,:3].reshape(42,3)])
                    uv=np.concatenate([p2[i,:,:2],h2[i,:,:,:2].reshape(42,2)])
                    statuses=np.concatenate([p3[i,:,3]>0,h3[i,:,:,3].reshape(42)>0])
                    projected,positive=project(xyz,ks[i],ms[i],ds[i],coordinate_mode=mode)
                    mask=statuses&positive&np.isfinite(uv).all(-1)
                    errors.extend(np.linalg.norm(projected[mask]-uv[mask],axis=-1).tolist())
                diag[mode]=stats(errors)
            summaries[view]=dict(frames=len(ids),cache=str(cache.resolve()),cache_sha256=sha(cache.read_bytes()),
                                 all_xyz_finite=bool(np.isfinite(h3[...,:3]).all()and np.isfinite(p3[...,:3]).all()),
                                 upper_body_status_positive=bool((p3[:,[1,2,3,4,5,6,7],3]>0).all()),
                                 camera_constant=bool(np.allclose(ks,ks[0])and np.allclose(ms,ms[0])and np.allclose(ds,ds[0])),
                                 projections=diag)
    assert [archive.stat().st_size,archive.stat().st_mtime_ns]==source_stat
    write(U/'SOURCE_IMPORTED.json',dict(status='IMPORTED_PENDING_CAMERA_CONVENTION_RGB_AND_GEOMETRY_VALIDATION',clip_id=CID,
          views=summaries,source_members=records,raw_source_unchanged=True,
          no_real_animation_yet=True,coordinate_mode_selected=None,physical_unit_confirmed=False,
          warning='Low supplied2D reprojection error can be self-consistent bad3D; original5view visual validation remains required.'))
    print('Imported exact public clip; still no geometry acceptance or real retargeting.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True);args=parser.parse_args()
    try:main(args.archive)
    except BaseException:
        p=U/'FIRST_IMPORT_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
