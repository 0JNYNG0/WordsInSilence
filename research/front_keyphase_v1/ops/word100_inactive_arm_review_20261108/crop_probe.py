"""New public camera-framing condition, not a user-frame test or model retuning."""
from review import *
import cv2,av
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'));from stream_runtime import Engine
def main():
 assert not (U/'CROP_PROBE.json').exists();torch.set_num_threads(4);f=read(J/'FROZEN.json');rows=[r for r in read(G/'INPUT_MANIFEST.json')['rows'] if r['person']==11];plan=[]
 for r in rows:
  rec=read(J/'records'/f"{r['clip_id']}.json");z=np.load(rec['raw_path']);xy=z['xy_px'][:12];body=np.nanmedian(xy,0);s=float(np.linalg.norm(body[5]-body[6]));top=max(0,int(body[0,1]-.7*s));bottom=min(int(z['source_height']),int(max(body[7,1],body[8,1])+.15*s));height=bottom-top;width=int(height*4/3);cx=(body[5,0]+body[6,0])/2;left=max(0,min(int(z['source_width'])-width,int(cx-width/2)));assert width>80 and height>60
  plan.append(dict(**r,raw=rec['raw_path'],crop=[left,top,left+width,bottom],crop_from='first12 public baseline shoulder/nose/elbow coordinates; fixed before viewing new outputs'))
 write(U/'CROP_FROZEN.json',dict(rows=[{k:r[k] for k in ['clip_id','word_id','source','source_sha256','raw','crop','crop_from']} for r in plan],condition='Simulate a closer4:3 upper-body camera ending just below elbows; entire resulting image goes to RTMW',output_size=[640,480],model='same RTMW-l and fixed TCN',training=False,private_used=False,score_selection=False))
 engine=Engine();model=Candidate();out=[]
 for r in plan:
  assert sha(r['source'])==r['source_sha256'];old=np.load(r['raw']);l,t,rr,b=r['crop'];coords=[];scores=[];ts=[];panels=[];lo,hi=r['intervals'][0];sample_ids=[int(np.argmin(abs(old['timestamp_us']-(lo+(hi-lo)*p)*1e6))) for p in [.2,.5,.8]]
  with av.open(r['source']) as v:
   for i,fr in enumerate(v.decode(video=0)):
    original=fr.to_ndarray(format='bgr24');im=cv2.resize(original[t:b,l:rr],(640,480));ans=engine.infer(im);coords.append(ans['xy']);scores.append(ans['score']);ts.append(round(float(fr.pts*fr.time_base)*1e6))
    if i in sample_ids:
     clean=im.copy();draw=im.copy()
     for side,color,start,arm in [('L',(180,70,180),91,[5,7,9]),('R',(80,180,0),112,[6,8,10])]:
      edges=list(zip(arm[:-1],arm[1:]))
      for digit in [1,5,9,13,17]:edges+=[(start,start+digit)]+[(start+k,start+k+1) for k in range(digit,digit+3)]
      for a,bb in edges:
       p,q=ans['xy'][a],ans['xy'][bb]
       if np.isfinite([p,q]).all() and 0<=p[0]<640 and 0<=q[0]<640 and 0<=p[1]<480 and 0<=q[1]<480:cv2.line(draw,tuple(p.astype(int)),tuple(q.astype(int)),color,2)
     cv2.putText(clean,f"{r['word_id']} f{i} cropped RGB",(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,220),2);cv2.putText(draw,'purple=LEFT green=RIGHT',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,220),2);panels.append(np.concatenate([clean,draw],1))
  xy=np.stack(coords);score=np.stack(scores);ts=np.asarray(ts);fid=old['frame_id'];assert len(xy)==len(fid);np.savez_compressed(U/(r['clip_id']+'_crop.npz'),xy_px=xy,score=score,timestamp_us=ts,frame_id=fid,source_width=640,source_height=480)
  answer=None;error=None
  try:
   a=A.convert(xy,score,ts,fid,640,480);x=a['tcn280']
   with torch.inference_mode():logits=model.tcn(torch.from_numpy(x[None]),torch.ones(1,len(x)))[0].numpy()
   answer=ranks(logits,f,f['classes'].index(r['word_id']))
  except Exception as exc:error=repr(exc)
  stats={}
  for name,start in [('L',91),('R',112)]:
   original=old['xy_px'][:,start:start+21];outside=((original[:,:,0]<l)|(original[:,:,0]>=rr)|(original[:,:,1]<t)|(original[:,:,1]>=b)).sum(1)>=18;now=xy[:,start:start+21];inside=(np.isfinite(now).all(2)&(now[:,:,0]>=0)&(now[:,:,0]<640)&(now[:,:,1]>=0)&(now[:,:,1]<480)).sum(1)>=18
   stats[name]=dict(baseline_hand_outside_crop_frames=int(outside.sum()),new_inframe_hand_in_those_frames=int((outside&inside).sum()),proxy_not_visibility_gold=True)
  sheet=U/(r['clip_id']+'_crop.jpg');cv2.imwrite(str(sheet),np.concatenate(panels,0));out.append(dict(clip_id=r['clip_id'],label=r['label'],crop=r['crop'],frames=len(xy),new_tcn=answer,error=error,geometric_comparison=stats,sheet=str(sheet)));print(json.dumps(out[-1],ensure_ascii=False),flush=True)
 write(U/'CROP_PROBE.json',dict(rows=out,private_used=False,training=False,actual_webcam=False,manual_visibility_review_pending=True))
if __name__=='__main__':main()
