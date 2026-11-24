from common import *
import cv2
f=read(U/'FROZEN.json');old={r['clip_id']:r for r in f['old_rows']};ix={r['clip_id']:r for r in read(OLD/'INDEX.json')}
scores={(r['id'],r['condition']):r for r in read(OLD/'SCORED_ROWS.json')}
LINKS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
TRIPLETS=[(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]
def angle(p,t):
    a,b,c=p[list(t)];u=a-b;v=c-b;den=np.linalg.norm(u)*np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/den,-1,1)))) if den>1e-5 and np.isfinite(den) else None
def skeleton(im,p,color):
    im=im.copy();v=np.isfinite(p).all(-1)
    for a,b in LINKS:
        if v[a] and v[b]:cv2.line(im,tuple(p[a].astype(int)),tuple(p[b].astype(int)),color,2)
    for j in np.flatnonzero(v):cv2.circle(im,tuple(p[j].astype(int)),4 if j==0 else 2,color,-1)
    return im
(U/'audit_visual').mkdir();out=[]
for sid in ['R01','R02','R14','R21','R22']:
    rr=scores[sid,'Y'];cid=rr['clip_id'];i=rr['frame_id'];s=rr['side'];r=old[cid];z=load(r['arrays']);rec=read(ix[cid]['records'])['frames'][i]
    good,scale=rule.context(z['raw_pose_xy_px'][i],z['raw_pose_usable'][i],1920,1080)
    ds=[(j,d) for j,d in enumerate(rec['full']) if rec['y_sides'][j]==s];assert len(ds)==1
    j,d=ds[0];p=points(d);crop=[]
    for k,c in enumerate(rec['crop']):
        if c['requested_side']!=s:continue
        b=points(c);dist=np.linalg.norm(p-b,axis=1)/scale
        ad=[abs(a-v) if a is not None and v is not None else None for a,v in [(angle(p,t),angle(b,t)) for t in TRIPLETS]]
        crop.append({'index':k,'arm':rule.arm(b,z['raw_pose_xy_px'][i],good,scale),'valid_joints':int(np.isfinite(b).all(-1).sum()),
          'wrist_delta_pixels':float(np.linalg.norm(p[0]-b[0])) if np.isfinite(b[0]).all() else None,
          'wrist_delta_shoulders':float(dist[0]) if np.isfinite(dist[0]) else None,
          'point_distances_shoulders':[float(v) if np.isfinite(v) else None for v in dist],
          'matching_palm_indices':[k for k in rule.PALM if np.isfinite(dist[k]) and dist[k]<=rule.PARAMS['point_agreement']],
          'angle_differences_degrees':ad,'note':'Two-model-scale consistency, NOT accuracy against true finger labels.'})
    row={'id':sid,'clip_id':cid,'frame_id':i,'side':s,'label':r['display_label'],'shoulder_scale_pixels':scale,
         'full_index':j,'full_confident_joints':int(np.isfinite(p).all(-1).sum()),'MP_M1_status':scores[sid,'M1']['status'],
         'YOLO_full_wrist_status':rr['status'],'crop_candidates':crop,'strict_original_reason':rec['merge'][j]['reason']}
    out.append(row)
    cap=cv2.VideoCapture(r['rgb_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();cap.release();assert ok
    cx,cy=p[0];x0=max(0,min(1920-640,int(cx)-320));y0=max(0,min(1080-480,int(cy)-240))
    ims=[('Original pixels',im),('YOLO full frame',skeleton(im,p,(30,230,60)))]
    if not crop:ims.append(('Crop: no hand detected',im))
    for c in crop:ims.append((f'YOLO crop {c["index"]}, arm {c["arm"]}',skeleton(im,points(rec['crop'][c['index']]),(30,170,255))))
    tiles=[]
    for title,im2 in ims:
        tile=im2[y0:y0+480,x0:x0+640];tile=cv2.copyMakeBorder(tile,36,0,0,0,cv2.BORDER_CONSTANT,value=(20,25,35));cv2.putText(tile,title,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);tiles.append(tile)
    while len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(U/'audit_visual'/f'{sid}.jpg'),np.vstack([np.hstack(tiles[k:k+2]) for k in range(0,len(tiles),2)]))
write(U/'FIVE_SCENE_AUDIT.json',out)
for r in out:print(r['id'],r['MP_M1_status'],'full',r['full_confident_joints'],'crops',[(c['arm'],c['wrist_delta_pixels'],c['matching_palm_indices']) for c in r['crop_candidates']])
