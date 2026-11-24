"""Prepare fixed public FIT RGB review sheets and saved-observation PTS audit.
No keypoint/model inference. No motion signs or automatic primitive labels.
"""
from pathlib import Path
import hashlib,json,math
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
SPECS=[('1567',2,list(range(40,85,2))+[88,96]),('1567',4,list(range(62,107,2))+[112,120]),
       ('1567',6,list(range(48,85,2))+[88,92]),('0003',1,list(range(31,58))+[59,63]),
       ('0003',3,list(range(53,88))+[90]),('1290',1,list(range(43,68))+[73,79]),('1290',3,list(range(54,80)))]
def main():
 assert not(U/'REVIEW_MANIFEST.json').exists()and not(U/'sheets').exists()
 sources={}
 paths=[G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',
        O/'word100_observed_event_boundaries_20261110/REVIEW_INPUTS.json',
        O/'word100_episode_joint_linkage_20261111/REVIEW_INPUTS.json',
        O/'word100_shared_motion_learning_20261109/READY.json',
        G/'hierarchical_relation_learning/parent_consistency_repair/FROZEN_V2.json',
        G/'action_linkage_plan/MULTILEVEL_RELATION_TARGETS.json',Path(__file__)]
 for p in paths[:3]:
  for r in read(p)['rows']:sources[r['clip_id']]=r
 ready={r['clip_id']:r for r in read(paths[3])['rows']}
 oldsplit={r['clip_id']:r['split']for r in read(paths[5])['rows']}
 rows=[];pins={str(p):sha(p)for p in paths}
 for word,person,frames in SPECS:
  cid=f'NIA_SL_WORD{word}_REAL{person:02d}_F';s=sources[cid];r=ready[cid]
  assert r['role']=='fit'and person in [1,2,3,4,6,7]and person not in [5,8,9]
  assert oldsplit.get(cid,'train')=='train'
  assert r['source_sha256']==s['source_sha256']and sha(s['source'])==s['source_sha256']
  assert sha(r['path'])==r['sha256']and sha(r['channel'])==r['channel_sha256']
  pins[s['source']]=s['source_sha256'];pins[r['path']]=r['sha256'];pins[r['channel']]=r['channel_sha256']
  rows.append(dict(clip_id=cid,word_id='WORD'+word,person=person,split='train',
                   split_basis='Existing pair-head train person set; 0003 ID03 has no earlier eligible pair, not a probe reassignment.',
                   source=s['source'],source_sha256=s['source_sha256'],raw=r['path'],raw_sha256=r['sha256'],
                   channel=r['channel'],channel_sha256=r['channel_sha256'],review_frames=frames))
 frozen=dict(status='FIXED_TRAIN_ONLY_RGB_REVIEW_NO_MODEL',rows=rows,pins=pins,
             excluded_people=[5,8,9],no_dev=True,no_new_labels_yet=True,
             visual_contract='Source RGB only, constant display crop. Performers anatomical R/L, not viewer side. No skeleton/prediction/motion-sign overlays. Hidden/ambiguous/hold remain distinct.',
             source_contract='Decode original existing files; exact source frame_id/relative PTS checked against saved raw observations. No time interpolation.',
             annotation_contract='New candidate labels only after actual RGB review, independently reviewable. Existing onset/offset/witness are provenance, not automatically copied primitive truth.')
 write(U/'REVIEW_FROZEN.json',frozen);(U/'sheets').mkdir();(U/'frames').mkdir()
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
 for r in rows:
  z=np.load(r['raw']);ch=np.load(r['channel']);fid=z['frame_id'];ts=z['timestamp_us'];want=set(r['review_frames']);decoded=[];saved={};full={}
  with av.open(r['source'])as container:
   for idx,f in enumerate(container.decode(video=0)):
    pts=round(float(f.pts*f.time_base)*1e6);decoded.append(pts)
    if idx in want:
     im=f.to_image();full[idx]=im
     # Same fractional crop throughout a source: face, shoulders, forearms and hands.
     w,h=im.size;box=(round(w*.27),round(h*.12),round(w*.74),round(h*.80));crop=im.crop(box)
     saved[idx]=crop;dest=U/'frames'/f"{r['clip_id']}_{idx:04d}.png";crop.save(dest)
  decoded=np.asarray(decoded,np.int64);relative=decoded-decoded[0]
  assert np.array_equal(fid,np.arange(len(decoded)))and len(ts)==len(decoded)
  err=np.abs(relative-ts);assert int(err.max())<=1,(r['clip_id'],int(err.max()))
  assert set(saved)==want and np.all(np.diff(ts)>0)
  x=ch['full'];assert len(x)==len(ts)and x.shape[1]==280
  # Canonical body points: L/R shoulder1/2, L/R elbow3? Wrist indices are audited by point contract below.
  # Store ALL first9 body point masks; do not infer motion or label observability from model confidence.
  masks=(x[:,186:195]>.5)
  maskpath=U/f"{r['clip_id']}_OBSERVATIONS.npz"
  np.savez_compressed(maskpath,frame_id=fid,timestamp_us=ts,body_point_mask=masks)
  pages=[];frame_list=r['review_frames']
  for pg,start in enumerate(range(0,len(frame_list),12)):
   fs=frame_list[start:start+12];cw,ih=360,340;hh=50;canvas=Image.new('RGB',(cw*4,hh+len(range(0,len(fs),4))*(ih+30)),(242,244,247));draw=ImageDraw.Draw(canvas)
   draw.text((8,5),f"{r['clip_id']} | TRAIN ONLY | page {pg+1}",font=font,fill='black')
   draw.text((8,28),'Source RGB; anatomical sides; no predicted labels; chronological row-major',font=small,fill='black')
   for j,idx in enumerate(fs):
    col,row=j%4,j//4;im=saved[idx].copy();im.thumbnail((cw-6,ih));xx=col*cw+(cw-im.width)//2;yy=hh+row*(ih+30)
    canvas.paste(im,(xx,yy));draw.text((col*cw+5,yy+ih+4),f"frame {idx} | {ts[idx]/1e6:.6f}s",font=font,fill='black')
   dest=U/'sheets'/f"{r['clip_id']}_p{pg+1:02d}.png";canvas.save(dest);pages.append(dict(path=str(dest),sha256=sha(dest),frames=fs))
  # Full frame at a fixed middle review point makes the crop independently checkable.
  mid=frame_list[len(frame_list)//2];dest=U/'sheets'/f"{r['clip_id']}_FULL_CONTEXT.png";full[mid].save(dest)
  r.update(pages=pages,full_context=dict(path=str(dest),sha256=sha(dest),frame=mid),
           source_frames=len(ts),original_size=list(full[mid].size),display_crop_fraction=[.27,.12,.74,.80],
           maximum_pts_error_us=int(err.max()),observation_array=str(maskpath),observation_sha256=sha(maskpath),
           body_point_mask_columns=list(range(9)),review_status='PREPARED_NOT_YET_VISUALLY_REVIEWED')
  print('Prepared',r['clip_id'],len(frame_list),'RGB frames',len(pages),'pages, PTS max error',int(err.max()),flush=True)
 for p,h in pins.items():assert sha(p)==h
 write(U/'REVIEW_MANIFEST.json',dict(status='PREPARED_FOR_ACTUAL_RGB_REVIEW',rows=rows,pins=pins,
       no_model_inference=True,no_training=True,no_new_primitive_labels=True,raw_sources_unchanged=True,
       training_sources=7,probe_sources_reviewed=0,review_frame_count=sum(len(r['review_frames'])for r in rows)))
if __name__=='__main__':main()
