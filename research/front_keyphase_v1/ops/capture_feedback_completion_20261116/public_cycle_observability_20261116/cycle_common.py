from pathlib import Path
import sys
M=Path(__file__).resolve().parent;L=M.parent/'public_temporal_visibility_20261116'
sys.path.insert(0,str(L))
from final_temporal import *
import copy

def all_sources():return current_sources()+read(L/'confirmation/INPUTS.json')['rows']
def baseline(cid):
 p=L/'qualified_checks'/(cid+'.json')
 return read(p if p.exists()else L/'verified_extra'/(cid+'.json'))['after']

def prepare():
 torch.set_num_threads(2);out=M/'audit';out.mkdir(exist_ok=True);by={r['id']:r for r in all_sources()}
 write(M/'CONTRACT.json',dict(goal='Separate merged repeated actions from weak terminal observations; retain real uncertainty and public normal/error performance.',graph=['A source-native motion/keypoint/observer audit','B FIT-only scale calibration and common episode association','C normal/error/partial/missing/speed public comparison; repair root cause if regressions','D only qualified runtime activation and records'],gate='No new normal warning/lost supported, preserve known error differences, no success for partial/hold/missing, no meaningful wait regression.',forbidden=['Private recordings for any learning/tuning/selection','Word/person exceptions','Expected-count-driven invented cycles','Change C-T3F or frozen100 baseline','Hidden-joint interpolation as truth'],baseline=str(L/'RESULT.json'),verification_actor='Codex source and separate executable/serialized checks; no independent KSL expert'))
 # Same17 source-reviewed FIT near frames, recomputed in a non-collapsing
 # projected palm scale. No new DEV label or threshold search.
 values=[];truth=[]
 for r in read(V/'target_region/SOURCE_TRUTH.json')['rows']:
  ch=channels(dict(np.load(by[r['clip_id']]['raw'])));d=palm_diameter_distance(ch);i=int(np.flatnonzero(ch['frame_id']==r['frame'])[0]);assert np.isfinite(d[i]);values.append(float(d[i]));truth.append(dict(r,diameter_distance=float(d[i])))
 write(M/'CALIBRATION.json',dict(rows=truth,limit=max(values),fit_only=True,source=str(V/'target_region/SOURCE_TRUTH.json'),scope='Observed 2D region, no physical contact or hidden fingers. Max pairwise observed palm landmark distance replaces a foreshortenable single axis.'))
 examples={'06_L':[40,42,44,46,48,50,52,54,56],'06_R':[40,42,44,46,48,50,52,54,56],'13_R':[53,56,59,61,63,65,67,69,71],'16_L':[45,49,51,54,57,59,61,63,66],'17_R':[82,86,89,92,95,98,100,103,106]}
 for suffix,frames in examples.items():
  cid='NIA_SL_WORD1290_REAL'+suffix;r=by[cid];video=r.get('video')
  if not video:continue
  cap=cv2.VideoCapture(video);tiles=[]
  for i in frames:
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read()
   if not ok:raise RuntimeError((cid,i))
   h,w=im.shape[:2];im=im[int(.22*h):int(.84*h),int(.24*w):int(.76*w)];im=cv2.resize(im,(420,340));cv2.putText(im,f'{suffix} f{i}',(6,22),0,.65,(255,255,255),2);tiles.append(im)
  cap.release();cv2.imwrite(str(out/(cid+'.jpg')),np.vstack([np.hstack(tiles[i:i+3])for i in range(0,9,3)]))
 print('FIT_LIMIT',max(values),'N',len(values),flush=True)
if __name__=='__main__':prepare()
