"""Prepare all frames in predefined hold ranges, with a fixed person split."""
from pathlib import Path
import hashlib,json,math
from datetime import datetime,timezone
import cv2
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent
G=P.parents[2]
ANN=[G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
 assert not(P/'PREPARED.json').exists()
 selected=sorted([(a,r)for a in ANN for r in read(a)['rows']if r['person']in[4,5,6,7,8,9]],key=lambda ar:(ar[1]['person'],ar[1]['clip_id']))
 assert len(selected)==24
 P.mkdir(parents=True,exist_ok=True)
 plan={'at':datetime.now(timezone.utc).isoformat(),'train_people':[4,5,6,7],'probe_people':[8,9],'probe_upstream_exposed':True,'not_final_evaluation':True,'base_review_sha256':sha(P.parent/'REVIEW_COMPLETE.json'),'source_annotations':{str(a):sha(a)for a in ANN},'selected':[{k:r[k]for k in['clip_id','person','source','source_sha256','hold_sample_frames']}for a,r in selected]}
 if (P/'FROZEN_REVIEW_SCOPE.json').exists():
  frozen=read(P/'FROZEN_REVIEW_SCOPE.json');assert frozen['selected']==plan['selected'] and frozen['source_annotations']==plan['source_annotations'] and frozen['base_review_sha256']==plan['base_review_sha256']
 else:(P/'FROZEN_REVIEW_SCOPE.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);rows=[]
 for a,r in selected:
  src=Path(r['source']);assert sha(src)==r['source_sha256'];lo,hi=min(r['hold_sample_frames']),max(r['hold_sample_frames']);frames=list(range(lo,hi+1));dst=P/r['clip_id'];dst.mkdir(exist_ok=True)
  cap=cv2.VideoCapture(str(src));fps=cap.get(cv2.CAP_PROP_FPS);assert 29<fps<31 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==r['frames'];cells=[];records=[]
  for i in range(hi+1):
   ok,bgr=cap.read();assert ok
   if i<lo:continue
   im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));fp=dst/f'frame_{i:04d}.png'
   if fp.exists():assert Image.open(fp).tobytes()==im.tobytes()
   else:im.save(fp)
   pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.;w,h=im.size;box=[int(w*.22),int(h*.05),int(w*.78),int(h*.88)];crop=im.crop(box);crop.thumbnail((350,330));cell=Image.new('RGB',(360,366),'white');cell.paste(crop,((360-crop.width)//2,32));ImageDraw.Draw(cell).text((9,4),f'Frame {i}  {pts:.3f}s',font=font,fill='#182b42');cells.append(cell);records.append(dict(frame=i,time_seconds=pts,timestamp_source='OpenCV decoded CAP_PROP_POS_MSEC',path=str(fp),sha256=sha(fp)))
  cap.release();assert [r['frame']for r in records]==frames
  sheet=Image.new('RGB',(1440,math.ceil(len(cells)/4)*366+52),'#dfe5eb');ImageDraw.Draw(sheet).text((12,12),r['clip_id']+f' | ALL frames {lo}..{hi} | original RGB',font=font,fill='#182b42')
  for i,c in enumerate(cells):sheet.paste(c,((i%4)*360,52+(i//4)*366))
  sp=dst/'all_frames.jpg'
  if not sp.exists():sheet.save(sp,quality=95)
  g=max(r['groups'],key=lambda g:g['offset'][1]);split='train'if r['person']in[4,5,6,7]else'probe'
  rows.append(dict(clip_id=r['clip_id'],person=r['person'],role='public_fit_upstream',head_split=split,upstream_exposed=True,source=str(src),source_sha256=r['source_sha256'],source_annotation=str(a),source_annotation_sha256=sha(a),fps=fps,annotation_nominal_fps=r['fps'],frames=frames,original_hold_samples=r['hold_sample_frames'],family=g['family'],source_group=g,frame_records=records,contact_sheet=str(sp),contact_sheet_sha256=sha(sp),contact_sheet_crop_box=box,all_frames_decoded=True,visual_inspection_completed=False,semantic_completion=None))
  print(r['clip_id'],len(frames),split,flush=True)
 out={'at':datetime.now(timezone.utc).isoformat(),'status':'PREPARED_NOT_REVIEWED','clips':24,'frames':sum(len(r['frames'])for r in rows),'train_clips':16,'probe_clips':8,'frozen_scope_sha256':sha(P/'FROZEN_REVIEW_SCOPE.json'),'rows':rows}
 (P/'PREPARED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('Ready',out['frames'],flush=True)
if __name__=='__main__':main()
