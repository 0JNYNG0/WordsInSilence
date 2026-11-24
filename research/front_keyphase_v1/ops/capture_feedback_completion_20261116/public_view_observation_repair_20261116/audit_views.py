from view_common import *
import cv2

def main(specs=None,output_name='audit'):
 torch.set_num_threads(2);cv2.setNumThreads(2);net=PalmMotionLesson();by={r['id']:r for r in sources()};out=J/output_name;out.mkdir(exist_ok=True);details=[]
 specs=specs or {'01_L':[45,47,53,57],'01_R':[45,47,53,57],'03_L':[53,56,65,77],'03_R':[53,56,65,77],'07_L':[45,48,57,65],'07_R':[45,48,57,65],'11_R':[49,52,55,61,63,66],'15_R':[52,57,62,65,68,71],'13_L':[53,55,57,59,62,67]}
 for suffix,frames in specs.items():
  cid='NIA_SL_WORD1290_REAL'+suffix;r=by[cid];z=dict(np.load(r['raw']));ch=channels(z);q=net.analyze(z);cap=cv2.VideoCapture(r['video']);tiles=[];rr=[]
  for i in frames:
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(z['frame_id'][i]));ok,im=cap.read();assert ok
   xy=z['xy_px'][i]*np.array([im.shape[1]/int(z['source_width']),im.shape[0]/int(z['source_height'])]);hands=xy[91:133];lo=np.maximum(np.nanmin(hands,0)-90,0).astype(int);hi=np.minimum(np.nanmax(hands,0)+90,[im.shape[1],im.shape[0]]).astype(int);crop=im[lo[1]:hi[1],lo[0]:hi[0]].copy();marked=crop.copy()
   for start,color in [(91,(200,100,20)),(112,(30,170,220))]:
    pts=xy[start:start+21]-lo
    for k in range(21):
     if np.isfinite(pts[k]).all():cv2.circle(marked,tuple(pts[k].astype(int)),3,color,-1)
    for ids in [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]:
     for a,b in zip(ids,ids[1:]):
      if np.isfinite(pts[[a,b]]).all():cv2.line(marked,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),color,1)
   pair=np.hstack([crop,marked]);scale=min(600/pair.shape[1],255/pair.shape[0]);pair=cv2.resize(pair,None,fx=scale,fy=scale);tile=np.full((290,620,3),255,np.uint8);tile[30:30+pair.shape[0],:pair.shape[1]]=pair;cv2.putText(tile,f'{suffix} f{i} plane R{q["plane_states"]["R"][i]} L{q["plane_states"]["L"][i]}',(5,20),0,.55,(10,20,30),1);tiles.append(tile)
   rr.append(dict(frame=i,planes={h:q['plane_states'][h][i]for h in['R','L']},form={h:q['hand_form_evidence'][h]['states'][i]for h in['R','L']},availability={h:net.shape.observations[h]['available'][i].tolist()for h in['R','L']}))
  cap.release()
  while len(tiles)%2:tiles.append(np.full_like(tiles[0],255))
  cv2.imwrite(str(out/(cid+'.jpg')),np.vstack([np.hstack(tiles[i:i+2])for i in range(0,len(tiles),2)]));write(out/(cid+'.json'),dict(source=r,details=rr,output=q));details.append(dict(id=cid,source=r['video'],details=rr))
 write(J/(output_name+'_INDEX.json'),dict(rows=details,private_data=False));print('AUDITED',len(details),flush=True)
if __name__=='__main__':main()
