from pathlib import Path
import sys,json,hashlib,datetime
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;P=G/'public_visibility_roles_20261116';H=G/'public_shape_arrival_repair_20261116'
sys.path.insert(0,str(P))
from thanks_candidate import ThanksCandidate
from audit_public import adapter,torch,np,V,B
import cv2
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
 torch.set_num_threads(2);net=ThanksCandidate()
 rows=read(H/'PUBLIC_INPUTS.json')['rows']+read(H/'public_views/INPUTS.json')['rows']
 truth=read(B/'source_corrected/SOURCE_TRUTH.json')['rows'];base={r['clip_id']:r for r in read(B/'INPUTS.json')['rows']}
 picked=[r for r in rows if r['id']in['NIA_SL_WORD1290_REAL07_F','NIA_SL_WORD1290_REAL11_F','NIA_SL_WORD1290_REAL01_L','NIA_SL_WORD1290_REAL03_L','NIA_SL_WORD1290_REAL07_L','NIA_SL_WORD1290_REAL13_R']]
 write(D/'CONTRACT.json',dict(created_at=datetime.datetime.now().astimezone().isoformat(),scope='Public-only native RGB/keypoints cause split, then FIT-only fixed training if justified; no private clips or criteria search.',preserve=['C/T3-F','frozen100 backbone','apple normal/auto-stop/latency','known individual finger evidence'],gate='No additional wrong or lost previously correct public anchor; no normal or incomplete-action regression; final public unselected check before web.',candidate_selection='Known DEV is development not untouched holdout. Record any rejected candidate.',training_labels='Only reviewed public FIT RGB; not auto-transferred from normal class or adjacent unknown/hidden fingers. No forced completion.'))
 allrows=[]
 for r in picked:
  assert 'Temp'not in r['raw']and 'user_'not in r['raw'];raw=dict(np.load(r['raw']));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  q=net.analyze(raw);sh=net.shape.sequence(ch,'R');f,quality,_=net.feature(ch);m=net.motion.sequence(ch)
  probs=[]
  for model,mean,sd in net.relation.models:
   with torch.inference_mode():probs.append(model(torch.tensor(((f-mean)/sd)[None])).softmax(-1)[0].numpy())
  prob=np.array(probs);prior=[t for t in truth if t['person']==r['person']];ids=sorted({t['frame']for t in prior if t['state']>0}|set(range(45,70,4)))
  details=[]
  for i in ids:
   if i>=len(f):continue
   details.append(dict(frame=i,relation=int(q['relation_states'][i]),motion=int(m[i]),quality=bool(quality[i]),relation_prob=prob[:,i].tolist(),shape=int(sh['states'][i]),fingers=sh['fingers'][i].tolist(),available=sh['available'][i].tolist(),geometry=sh['geometry_reasons'][i].tolist(),reason=sh['finger_reasons'][i].tolist()))
  write(D/'audit'/f'{r["id"]}.json',dict(source=r,output=q,details=details));allrows.append(dict(source=r,criteria={c['id']:c['status']for c in q['criteria']},details=details))
  video=r.get('video')or base[r['id']]['video'];cap=cv2.VideoCapture(video);tiles=[]
  for i in ids:
   if i>=len(f):continue
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(raw['frame_id'][i]));ok,im=cap.read();assert ok
   xy=raw['xy_px'][i].copy()*np.array([im.shape[1]/int(raw['source_width']),im.shape[0]/int(raw['source_height'])]);hands=xy[91:133];lo=np.maximum(np.nanmin(hands,0)-100,0).astype(int);hi=np.minimum(np.nanmax(hands,0)+100,[im.shape[1],im.shape[0]]).astype(int);im=im[lo[1]:hi[1],lo[0]:hi[0]];sc=min(330/im.shape[1],220/im.shape[0]);im=cv2.resize(im,None,fx=sc,fy=sc);t=np.full((260,350,3),255,np.uint8);t[30:30+im.shape[0],:im.shape[1]]=im;cv2.putText(t,f'F{i} S{q["relation_states"][i]} M{m[i]} H{sh["states"][i]}',(4,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(10,20,30),1);tiles.append(t)
  cap.release()
  while len(tiles)%4:tiles.append(np.full_like(tiles[0],255))
  cv2.imwrite(str(D/'audit'/f'{r["id"]}.jpg'),np.vstack([np.hstack(tiles[i:i+4])for i in range(0,len(tiles),4)]))
  print(r['id'],len(q['events']),{c['id']:c['status']for c in q['criteria']},flush=True)
 write(D/'AUDIT.json',dict(rows=allrows,private_data=False))
if __name__=='__main__':main()
