"""Diagnose source compatibility; no correction, fitted transform, or learning."""
import study as S
import visuals as V
import numpy as np, cv2, zipfile, hashlib, collections
from PIL import Image,ImageDraw

U=S.U
def run():
    f=S.read(U/'FROZEN.json');done=S.read(U/'DONE.json');index=S.read(U/'INDEX.json')
    # Deliberate diagnostic additions, not replacements for the preselected sample.
    ids=['NIA_SL_WORD1290_REAL02_F','NIA_SL_WORD1844_REAL02_F']
    if (U/'DIAGNOSTIC_ADDENDUM.json').exists():
        prior=S.read(U/'DIAGNOSTIC_ADDENDUM.json');assert prior['ids']==ids and not prior['spatial_correction'] and not prior['time_correction']
        assert not (U/'DIAGNOSTICS.json').exists()
    else:S.write('DIAGNOSTIC_ADDENDUM.json',dict(at=S.now(),ids=ids,reason='Large displacement / truncated coordinate tail diagnosed after structural audit; not an unbiased accuracy sample',spatial_correction=False,time_correction=False))
    groups=collections.defaultdict(list);rows=[]
    byid={r['clip_id']:r for r in f['rows']}
    with open(f['archive'],'rb') as fp:
        for r in done['rows']:
            cid=r['clip_id'];m=next(m for m in index['members'] if m['clip_id']==cid and m['frame']==0)
            obj,_=S.payload(fp,m);cam=obj['camparam'];K=np.fromstring(cam['Intrinsics']['data'],sep=' ').reshape(3,3)
            distortion=np.fromstring(cam['Distortion']['data'],sep=' ');z=np.load(r['arrays']);xyz=np.ascontiguousarray(z['official_xyz'][0,:,:,:3].reshape(-1,3),dtype=np.float64);uv=z['official'][0,:,:,:2].reshape(-1,2)
            p=xyz@K.T;projected=p[:,:2]/p[:,2,None]
            # OpenCV projectPoints ignores intrinsic skew; this auxiliary diagnostic
            # uses that convention. The pinhole projection above includes skew.
            distorted,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,distortion);distorted=distorted[:,0]
            # Report both diagnostics without selecting either as a corrected label.
            g=cam['Intrinsics']['data'];groups[g].append(cid)
            row=dict(clip_id=cid,label=r['label'],camera=g,first_left_wrist_z=float(xyz[0,2]),provided_2d_vs_pinhole_px=S.stats(np.linalg.norm(projected-uv,axis=1)),provided_2d_vs_cv_distorted_px=S.stats(np.linalg.norm(distorted-uv,axis=1)),median_wrist_disagreement=r['wrist_disagreement_px']['median'],missing=len(r['missing_frames']),extra=len(r['extra_frames']))
            if r['missing_frames']:
                present=sorted(int(t) for t in r['payload_sha256'] if int(t)<r['frames'])
                row['missing_is_contiguous_tail']=present==list(range(len(present)))
                # Explicitly test whether the shorter label sequence could be uniformly subsampled.
                row['index_vs_3_stride_diagnostic']=[]
                for factor in (1,3):
                    a=np.array(present);b=a*factor;v=b<r['frames'];a=a[v];b=b[v]
                    mask=z['official_valid'][a,:,0]&z['mp_valid'][b,:,0]
                    delta=np.linalg.norm(z['official'][a,:,0,:2]-z['mp_xy'][b,:,0],axis=-1)
                    row['index_vs_3_stride_diagnostic'].append(dict(factor=factor,**S.stats(delta[mask])))
            rows.append(row)
    visual=[]
    for cid in ids:
        r=next(r for r in done['rows'] if r['clip_id']==cid);source=byid[cid]['source']
        with zipfile.ZipFile(source['archive']) as arc:raw=arc.read(source['member'])
        assert hashlib.sha256(raw).hexdigest()==source['sha256']
        vp=U/'clips'/(cid+'.mp4')
        with vp.open('xb') as out:out.write(raw)
        z=np.load(r['arrays']);last=max(int(t) for t in r['payload_sha256'] if int(t)<r['frames'])
        mid=int(np.argmin(abs(z['timestamp_us']/1e6-np.mean(byid[cid]['intervals'][0]))))
        targets=sorted(set([0,last,min(last+1,r['frames']-1),mid,r['frames']-1]))
        cap=cv2.VideoCapture(str(vp));frames={};n=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if n in targets:frames[n]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            n+=1
        cap.release();assert n==r['frames']
        sheet=Image.new('RGB',(1440,90+len(targets)*340),(245,248,252));draw=ImageDraw.Draw(sheet)
        draw.text((18,10),r['label']+' | '+cid,font=V.font(24),fill=(20,35,55))
        draw.text((18,48),'추가 진단 표본 · 원본 / AI Hub 제공 좌표 / 저장 MediaPipe',font=V.font(19),fill=(40,55,70))
        items=[]
        for i,t in enumerate(targets):
            rgb=frames[t];ah=z['official'][t,:,:,:2];av=z['official_valid'][t];mp=z['mp_xy'][t];mv=z['mp_valid'][t]
            points=np.concatenate([ah[av],mp[mv]],axis=0);box=(max(0,int(points[:,0].min())-70),max(0,int(points[:,1].min())-70),min(1920,int(points[:,0].max())+70),min(1080,int(points[:,1].max())+70))
            panels=[Image.fromarray(rgb),V.overlay(rgb,ah,av),V.overlay(rgb,mp,mv)]
            for j,panel in enumerate(panels):
                cut=panel.crop(box);cut.thumbnail((464,280));sheet.paste(cut,(j*480+8+(464-cut.width)//2,90+i*340+45))
                draw.text((j*480+12,90+i*340+12),['원본 RGB','AI Hub 좌표' if av.any() else '해당 프레임 좌표 파일 없음','MediaPipe 저장 관측'][j],font=V.font(18),fill=(20,35,55))
            draw.text((18,90+i*340-10),f'frame {t} / {t/30:.3f}s',font=V.font(16),fill=(70,80,90))
            items.append(t)
        fn='images/'+cid+'_diagnostic.jpg';sheet.save(U/fn,quality=94)
        visual.append(dict(clip_id=cid,label=r['label'],frames=items,image=fn,video_sha=S.sha(vp),decoded_frames=n))
    out=dict(rows=rows,camera_groups=[dict(intrinsics=k,clips=v) for k,v in groups.items()],extra_visuals=visual,missing_contiguous_tails=sum(r.get('missing_is_contiguous_tail',False) for r in rows),correction_applied=False)
    S.write('DIAGNOSTICS.json',out);print('DIAGNOSTICS',len(rows),len(visual),flush=True)

if __name__=='__main__':run()
