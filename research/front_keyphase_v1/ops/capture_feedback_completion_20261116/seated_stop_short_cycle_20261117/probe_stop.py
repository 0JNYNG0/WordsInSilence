from probe_public import *
import av,cv2,importlib.util
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'));import server
from stream_runtime import Engine
from auto_end import ReturnToRest as Before
sys.path.insert(0,str(D));from return_rest import ReturnToRest as After

def run_stop(kind,raw):
 obj=kind();xy=raw['xy_px'];w=int(raw['source_width']);h=int(raw['source_height'])
 for _ in range(12):obj.calibrate(xy[0],w,h)
 obj.begin();first=None
 for i,p in enumerate(xy):
  if obj.update(p,w,h,int(raw['timestamp_us'][i])) and first is None:first=i
 return dict(calibrated=obj.rest is not None,first_stop=first,diagnostic=obj.diagnostic()if hasattr(obj,'diagnostic')else None)

def extract(row,engine,border,aspect=False):
 z=dict(np.load(row['raw']));p=z['xy_px'][0];scale=np.linalg.norm(p[5]-p[6]);bottom=int(max(p[7,1],p[8,1])+scale*border)
 original_h=int(z['source_height']);original_w=int(z['source_width']);top=int(original_h*.08);left=int(original_w*.25);right=int(original_w*.75)
 if aspect:
  center=float(p[[5,6],0].mean());half=(bottom-top)*2/3;left=round(center-half);right=round(center+half)
 name=row['id']+'_'+str(border).replace('.','p')+('_aspect'if aspect else '');cache=D/'probe'/f'{name}.npz'
 if cache.exists():return dict(np.load(cache))
 coords=[];scores=[];pictures=[]
 with av.open(row.get('video',row.get('path')))as f:
  for i,fr in enumerate(f.decode(video=0)):
   im=fr.to_ndarray(format='bgr24');im=cv2.resize(im[top:bottom,left:right],(640,480));q=engine.infer(im)
   coords.append(q['xy']);scores.append(q['score'])
   if i in [0,len(z['xy_px'])//2,len(z['xy_px'])-1]:
    for j in [5,6,7,8,9,10]+list(range(91,133)):
     pt=q['xy'][j]
     if np.isfinite(pt).all():cv2.circle(im,tuple(pt.astype(int)),2,(20,180,70),-1)
    pictures.append(im)
 raw=dict(xy_px=np.array(coords+coords[-1:]*35),score=np.array(scores+scores[-1:]*35),frame_id=np.arange(len(coords)+35),timestamp_us=np.arange(len(coords)+35)*33333,source_width=np.array(640),source_height=np.array(480))
 np.savez_compressed(cache,**raw)
 if pictures:cv2.imwrite(str(D/'probe'/f'{name}.jpg'),np.hstack(pictures))
 write(D/'probe'/f'{name}_source.json',dict(public_source=row,crop=[left,top,right,bottom],unchanged_video=True,tail='repeat last observed frame for stop dwell only'))
 return raw

def main():
 torch.set_num_threads(2);cv2.setNumThreads(2);engine=Engine();rows=[]
 for r in sources():
  if r['person']!=1:continue
  for border in [.18,.35]:
   z=extract(r,engine,border);a=run_stop(Before,z);b=run_stop(After,z)
   row=dict(id=r['id'],border=border,before=a,after=b)
   rows.append(row);print(r['id'],border,'before',a['calibrated'],a['first_stop'],'after',b['calibrated'],b['first_stop'],flush=True)
 write(D/'probe/STOP_FIT.json',dict(rows=rows,private_data=False,physical_camera=False))
if __name__=='__main__':main()
