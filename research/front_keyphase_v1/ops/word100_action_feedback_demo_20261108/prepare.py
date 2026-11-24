from task import *
import numpy as np,torch,cv2,shutil
from runtime_candidate import Candidate

def cards():
 # Description-level events; arrival holds are optional, reference repetition count is not a rule.
 return {
 'WORD0739':[
  dict(id='open_front',title='손 펴서 얼굴 앞에',instruction='오른손을 펴서 얼굴 앞에 세워 보세요.',groups=[[0]],hands=['R'],features={'1':1},reference=[46,58]),
  dict(id='close_chest',title='손 닫으며 가슴 쪽으로',instruction='손을 닫으면서 가슴 쪽으로 내려 보세요.',groups=[[1]],hands=['R'],features={'3':1},reference=[61,71]),
  dict(id='raise_forearm',title='오른팔 다시 세우기',instruction='닫힌 손을 유지하며 오른팔을 다시 세워 보세요.',groups=[[2]],hands=['R'],features={'3':1},reference=[73,89])],
 'WORD1133':[
  dict(id='brush_arm',title='펴진 손으로 팔 따라 쓸기',instruction='오른손을 펴고 왼팔을 따라 손목 쪽으로 쓸어 보세요.',groups=[[3]],hands=['R','L'],features={'1':1,'4':1},reference=[46,63]),
  dict(id='reach_wrist',title='손목 쪽까지 이어가기',instruction='쓸어내리는 동작을 손목 쪽까지 이어 보세요.',groups=[[4]],hands=['R','L'],features={'4':1},reference=[65,79],arrival_not_required_hold=True)],
 'WORD1567':[
  dict(id='alternating',title='양손 번갈아 올리기',instruction='오른손과 왼손을 번갈아 위로 올려 보세요.',groups=[[5,7],[6,8,9]],hands=['R','L'],features={'3':1,'4':1},reference=[51,96],repeat_count_required=None,unverified='엄지 모양은 별도 확인이 필요해요.')],
 'WORD2523':[
  dict(id='gather_hands',title='양손 손끝 모으기',instruction='양손의 손끝을 가운데로 모아 보세요.',groups=[[10]],hands=['R','L'],features={},reference=[29,39],unverified='정확히 맞닿았는지는 이 화면만으로 확정하지 않아요.'),
  dict(id='spread_hands',title='양손을 양옆으로 벌리기',instruction='모았던 두 손을 양옆으로 벌려 보세요.',groups=[[11,12]],hands=['R','L'],features={'1':1,'2':1},reference=[41,55])]
 }

def main():
 assert not (U/'PREPARED.json').exists();f=read(U/'FROZEN.json');torch.set_num_threads(4)
 model=Candidate();stats=model.stats;guides=read(G/'GUIDE_SUPPORT.json')['guides'];manifest={r['clip_id']:r for r in read(G/'INPUT_MANIFEST.json')['rows']};rows=[];assets={}
 for r in read(J/'feedback_viewfix/FEEDBACK_READY.json')['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);meta=manifest[r['clip_id']];wi=r['word'];n=len(z['x']);data={k:z[k].copy() for k in ['raw_x','y','ym','frame_id','timestamp_us']}
  for view in ['full','upper']:
   with torch.inference_mode():o=model.feedback(torch.from_numpy(z['x_'+view][None]),torch.tensor([n]),torch.tensor([wi]))
   data['h_'+view]=o['latent'][0].numpy();data['p_'+view]=o['attributes'][0].sigmoid().numpy()
  sy=z['sy'].copy();sw=z['sw'].copy()
  # Class0 = outside the existing broad main-action interval, not a semantic claim about any unlabelled interior gap.
  yy=np.where(sy>=0,sy+1,-1);weights=sw.copy();ts=z['timestamp_us']/1e6
  if r['role']=='fit':
   outside=np.ones(n,bool)
   for lo,hi in meta['intervals']:outside&=((ts<lo-.1)|(ts>hi+.1))
   yy[outside]=0;weights[outside]=.2
  else:assert np.all(yy==-1)
  data.update(stage_y=yy,stage_w=weights);path=U/'inputs'/f"{r['clip_id']}.npz";path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**data)
  rows.append({**r,'path':str(path),'sha256':sha(path),'intervals':meta['intervals'],'source':meta['source'],'source_sha256':meta['source_sha256']})
  if r['person']==1:
   source=Path(meta['source']);assert sha(source)==meta['source_sha256'];dest=U/'media'/f"{f['words'][wi]}.mp4";dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
   rawrec=read(J/'records'/f"{r['clip_id']}.json");raw=np.load(rawrec['raw_path']);assets[f['words'][wi]]=dict(video=str(dest),sha256=sha(dest),clip_id=r['clip_id'],width=int(raw['source_width']),height=int(raw['source_height']),timestamp_us=raw['timestamp_us'].tolist(),xy=np.nan_to_num(raw['xy_px'][:,:,:2],nan=-1).tolist())
   gg=[q for q in guides if q['word_id']==f['words'][wi]];cap=cv2.VideoCapture(str(source));imgs=[]
   for q in gg:
    fid=sum(q['frames'])//2;cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
    h,w=im.shape[:2];im=cv2.resize(im,(480,round(h*480/w)));cv2.putText(im,q['canonical_stage_id']+' f'+str(fid),(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,220),1);imgs.append(im)
   cap.release();cols=2;hh,ww=imgs[0].shape[:2];canvas=np.full((((len(imgs)+1)//2)*hh,ww*2,3),245,np.uint8)
   for i,im in enumerate(imgs):canvas[(i//2)*hh:(i//2+1)*hh,(i%2)*ww:(i%2+1)*ww]=im
   path=U/'audit'/f"{f['words'][wi]}.jpg";path.parent.mkdir(exist_ok=True);cv2.imwrite(str(path),canvas)
 write(U/'CARDS.json',dict(words=cards(),provenance='Camera-visible description drafts mapped to prior saved guide spans; not expert sign-correctness labels',guides=guides,frame_interval_semantics='Reference demonstration spans; user times are inferred separately',duration_requirement=False,exact_repetition_requirement=False))
 write(U/'REFERENCES.json',assets)
 write(U/'PREPARED.json',dict(rows=rows,guide_ids=stats['guide_ids'],guide_words=stats['guide_words'],words=f['words'],baseline_checkpoint=str(J/'feedback_viewfix/feedback_runs/pose_transfer_42/model.pt'),label_origin='Existing reviewed centres, canonical spans, and explicitly weak transferred spans; outside main-action background added from existing intervals',development_training_labels_used=False,source_manifest_sha256=sha(J/'feedback_viewfix/FEEDBACK_READY.json')))
 event('prepared',clips=len(rows),fit=sum(r['role']=='fit' for r in rows),dev=sum(r['role']=='development' for r in rows))
if __name__=='__main__':main()
