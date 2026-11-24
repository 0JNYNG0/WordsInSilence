from visible_lesson import *
sys.path.insert(0,str(V.parent))
from diagnose_terminal import sources
import cv2

def main(output_name='visible_checks',factory=VisibleLesson):
 torch.set_num_threads(2);out=V/output_name;out.mkdir(exist_ok=True);assert not(out/'RESULT.json').exists()
 write(out/'FROZEN.json',dict(code={p.name:sha(p)for p in[V/'visible_lesson.py',V/'view_conditioned_shape.py',V/'target_region.py']},
  scope='Known public 14-18; six native requirements. Derived controls, not recorded learner errors. No camera/private footage.',
  gate='Normal must retain all visible criteria, fast/slow/scale preserved; missing/static/first-only must not pass. No threshold selection from results.'))
 net=factory();results=[];tiles=[]
 for r in sources():
  raw=dict(np.load(r['raw_path']));n=len(raw['frame_id'])
  for kind in ['normal','fast','slow','scale','first_only','constant','missing']:
   z={k:v.copy()for k,v in raw.items()}
   if kind=='fast':z['timestamp_us']=(z['timestamp_us']*.65).astype(np.int64)
   if kind=='slow':z['timestamp_us']=(z['timestamp_us']*1.5).astype(np.int64)
   if kind=='scale':z['xy_px']=z['xy_px']*.75+np.array([90,40],np.float32)
   end={14:54,15:58,16:51,17:90,18:60}[r['person']]
   if kind=='first_only':
    for k in['xy_px','score','timestamp_us','frame_id']:z[k]=z[k][:end+1]
   if kind=='constant':z['xy_px']=np.repeat(z['xy_px'][end:end+1],n,axis=0)
   if kind=='missing':z['xy_px'][:,91:]=np.nan
   q=net.analyze(z);supported=all(c['status']=='supported'for c in q['criteria'])
   expected=kind in['normal','fast','slow','scale']
   results.append(dict(person=r['person'],kind=kind,supported=supported,expected=expected,check=supported==expected,
    events=q['events'],criteria=q['criteria'],seconds=q['seconds']))
   if kind=='normal':
    write(out/f'p{r["person"]}.json',q)
    cap=cv2.VideoCapture(r.get('video',r.get('path')))
    for e in q['events']:
     fid=e['end'];cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
     h,w=im.shape[:2];im=im[int(h*.12):int(h*.66),int(w*.25):int(w*.75)]
     scale=min(460/im.shape[1],370/im.shape[0]);im=cv2.resize(im,None,fx=scale,fy=scale)
     tile=np.full((410,480,3),255,np.uint8);tile[36:36+im.shape[0],:im.shape[1]]=im
     cv2.putText(tile,f'P{r["person"]} F{fid}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,40,60),2);tiles.append(tile)
    cap.release()
 write(out/'RESULT.json',dict(rows=results,passed=sum(r['check']for r in results),total=len(results),web_admitted=False))
 if tiles:cv2.imwrite(str(out/'native_endpoints.jpg'),np.vstack([np.hstack(tiles[i:i+2])for i in range(0,len(tiles),2)]))
 for r in results:
  if r['kind']=='normal'or not r['check']:print(r['person'],r['kind'],r['supported'],{c['id']:c['status']for c in r['criteria']})
 print('CHECKS',sum(r['check']for r in results),len(results))
if __name__=='__main__':main()
