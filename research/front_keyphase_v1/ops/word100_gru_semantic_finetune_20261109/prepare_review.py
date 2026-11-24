from common import *
import cv2,shutil
def main():
 assert not (U/'REVIEW_INPUTS.json').exists()
 (U/'before').mkdir(exist_ok=True)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
 manifest=read(L/'INPUT_MANIFEST.json');rows=[]
 for r in manifest['rows']:
  if r['person'] not in [1,4,7,11,12,13]:continue
  rec=read(J/'records'/f"{r['clip_id']}.json");z=np.load(rec['raw_path']);ts=z['timestamp_us'];lo,hi=r['intervals'][0];ids=np.flatnonzero((ts>=lo*1e6)&(ts<=hi*1e6))
  samples=np.unique(np.rint(np.linspace(max(0,ids[0]-3),min(len(ts)-1,ids[-1]+3),12)).astype(int))
  xy=z['xy_px'];stable=xy[:,[0,5,6,7,8,9,10]];p=stable[ids];good=np.isfinite(p).all(2);points=p[good];h,w=int(z['source_height']),int(z['source_width'])
  left=max(0,int(points[:,0].min()-w*.07));right=min(w,int(points[:,0].max()+w*.07));top=max(0,int(points[:,1].min()-h*.06));bottom=min(h,int(points[:,1].max()+h*.07));box=[left,top,right,bottom]
  cap=cv2.VideoCapture(r['source']);ims=[]
  for i in samples:
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok,(r['clip_id'],i)
   im=im[top:bottom,left:right];im=cv2.resize(im,(250,250));cv2.rectangle(im,(0,0),(250,28),(245,245,245),-1);cv2.putText(im,f'f{i}  {ts[i]/1e6:.2f}s',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1);ims.append(im)
  cap.release();canvas=np.full((750,1000,3),245,np.uint8)
  for k,im in enumerate(ims):canvas[k//4*250:(k//4+1)*250,k%4*250:(k%4+1)*250]=im
  p=U/'review'/(r['clip_id']+'.jpg');p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),canvas)
  rows.append(dict(clip_id=r['clip_id'],role=r['role'],word_id=r['word_id'],person=r['person'],source=r['source'],source_sha256=r['source_sha256'],frames=samples.tolist(),sheet=str(p),crop=box,raw=str(rec['raw_path']),raw_sha256=sha(rec['raw_path'])))
 write(U/'REVIEW_INPUTS.json',dict(rows=rows,model_predictions_hidden=True,reviewer='Codex RGB review; not independent expert gold',scope='Sparse reviewed anchors and endpoint movement, not every intervening frame or educational correctness'))
 write(U/'PLAN.json',dict(authorization='User 2026-11-09: strengthen description-video labels and fine-tune current temporal convolution+GRU; compare public data; preserve C/T3-F, no personal data.',
  split='FIT existing01-09 (36clips); DEV11-13 (12clips). New RGB review FIT01/04/07 and DEV11/12/13, before predictions. Existing DEV exposed historically, not fresh test.',
  architecture='Keep PoseGuideNet base. Add small pair-of-time direction and order readouts. No new pretrained download, no RAG, no RGB runtime.',
  compare=['unchanged original attributes and stages / prior geometric direction','frozen existing base + trained readouts','fine-tuned existing base + same readouts'],
  seeds=[42,43,44],selection='Fixed42 primary; all three reported, no best-seed/epoch selection.',
  outcomes=['Reviewed point handshape','Reviewed pair direction','Reviewed semantic stage/order','synthetic reversal/order and normal speed controls','unknown masking','CPU latency'],
  limitations=['Sparse same-Codex RGB observations are not independent expert correction labels.','Synthetic changes remain separate from natural error accuracy.','Pair tasks use reviewed windows: not automatically found event accuracy.'],
  runtime_change=False,private_used=False,download=False))
 event('review_prepared',clips=len(rows))
if __name__=='__main__':main()
