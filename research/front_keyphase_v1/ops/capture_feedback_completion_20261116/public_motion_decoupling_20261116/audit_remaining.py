from common import *
import cv2

def main():
 torch.set_num_threads(2)
 from semantic_conflict import SemanticLesson
 net=SemanticLesson();by={r['id']:r for r in sources()};base={r['clip_id']:r for r in read(B/'INPUTS.json')['rows']}
 for suffix,frames in [('13_L',list(range(51,60))),('11_F',list(range(56,71,2)))]:
  cid='NIA_SL_WORD1290_REAL'+suffix;r=by[cid];z=dict(np.load(r['raw']));ch=channels(z);q=net.analyze(z);cap=cv2.VideoCapture(r.get('video')or base[cid]['video']);tiles=[];details=[]
  for i in frames:
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(z['frame_id'][i]));ok,im=cap.read();assert ok
   h=ch['body_relative_xy'][i];xy=z['xy_px'][i]*np.array([im.shape[1]/int(z['source_width']),im.shape[0]/int(z['source_height'])]);hands=xy[91:133];lo=np.maximum(np.nanmin(hands,0)-65,0).astype(int);hi=np.minimum(np.nanmax(hands,0)+65,[im.shape[1],im.shape[0]]).astype(int);im=im[lo[1]:hi[1],lo[0]:hi[0]];im=cv2.resize(im,None,fx=min(410/im.shape[1],260/im.shape[0]),fy=min(410/im.shape[1],260/im.shape[0]));t=np.full((300,430,3),255,np.uint8);t[35:35+im.shape[0],:im.shape[1]]=im;cv2.putText(t,f'{cid} frame{i}',(3,23),0,.6,(20,30,40),1);tiles.append(t)
   details.append(dict(frame=i,hand={hand:{k:v[i]for k,v in q['hand_form_evidence'][hand].items()if isinstance(v,list)}for hand in['R','L']}))
  cap.release()
  while len(tiles)%3:tiles.append(np.full_like(tiles[0],255))
  dest=E/'audit_remaining';dest.mkdir(exist_ok=True);cv2.imwrite(str(dest/(cid+'.jpg')),np.vstack([np.hstack(tiles[i:i+3])for i in range(0,len(tiles),3)]));write(dest/(cid+'.json'),dict(source=r,details=details))
if __name__=='__main__':main()
