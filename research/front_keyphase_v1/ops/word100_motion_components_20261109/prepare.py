from common import *
import cv2,shutil
def main():
 assert not (U/'REVIEW.json').exists()
 (U/'before').mkdir(exist_ok=True)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
 rows=[]
 for r in read(L/'INPUT_MANIFEST.json')['rows']:
  cid=r['clip_id'];rec=read(J/'records'/f'{cid}.json');z=np.load(rec['raw_path']);ts=z['timestamp_us'];lo,hi=r['intervals'][0];ids=np.flatnonzero((ts>=lo*1e6)&(ts<=hi*1e6))
  if r['person'] in [2,3,5,6,8,9]:
   samples=np.unique(np.rint(np.linspace(max(0,ids[0]-3),min(len(ts)-1,ids[-1]+3),12)).astype(int));kind='movement'
  elif r['word_id']=='WORD2523':
   # Later part including return, avoiding truncating possible final handshape.
   samples=np.unique(np.rint(np.linspace(ids[len(ids)//3],min(len(ts)-1,ids[-1]+5),12)).astype(int));kind='late_hands'
  else:continue
  h,w=int(z['source_height']),int(z['source_width']);xy=z['xy_px'];points=xy[samples][:,[0,5,6,7,8,9,10]].reshape(-1,2);points=points[np.isfinite(points).all(1)]
  box=[max(0,int(points[:,0].min()-.08*w)),max(0,int(points[:,1].min()-.08*h)),min(w,int(points[:,0].max()+.08*w)),min(h,int(points[:,1].max()+.08*h))]
  cap=cv2.VideoCapture(r['source']);canvas=np.full((840,1120,3),245,np.uint8)
  for k,i in enumerate(samples):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok
   l,t,rr,b=box;im=cv2.resize(im[t:b,l:rr],(280,280));cv2.rectangle(im,(0,0),(280,26),(245,245,245),-1);cv2.putText(im,f'f{i} {ts[i]/1e6:.2f}s',(5,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1);canvas[k//4*280:k//4*280+280,k%4*280:k%4*280+280]=im
  cap.release();p=U/'review'/f'{cid}.jpg';p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),canvas)
  rows.append(dict(clip_id=cid,person=r['person'],role=r['role'],word_id=r['word_id'],source=r['source'],source_sha256=sha(r['source']),raw_path=rec['raw_path'],raw_sha256=sha(rec['raw_path']),frames=samples.tolist(),sheet=str(p),kind=kind))
 write(U/'REVIEW.json',dict(rows=rows,scope='public sparse RGB review; no model predictions; same Codex, not expert truth'))
 write(U/'PLAN.json',dict(goal='General body-relative motion + required-component coverage, no word-specific numerical rules',
  authorization='2026-11-09 continue motion correction; check Egypt final fingertip gathering; user explicitly requires general components',
  design=['No word/gloss/person input in direction measurement or learned direction classifier.','Hand/arm endpoints and relation names supplied through reusable contracts.','Keep current handshape/phase network intact; separate measured motion from static latent.','A learned open-hand label cannot verify fingertip gathering. Required unsupported component remains unknown.'],
  split='FIT01-09, DEV11-13, historical DEV not fresh generalization; no private learning/tuning/selection',
  conditions=['previous latent-only direction head (saved baseline)','generic geometry measurements','geometry-informed shared direction classifier'],
  seeds=[42,43,44],primary=42,policy='No best-seed or threshold selection from DEV; raw data preserved. Public decision before private recheck.',
  web='Only proven generic evidence/completeness fix may enter research demo; no C/T3-F replacement or camera opening',download=False))
 event('review_prepared',clips=len(rows))
if __name__=='__main__':main()
