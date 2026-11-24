from apple_io import *
sys.path.insert(0,str(V));from view_conditioned_shape import ViewConditionedShape
import torch,adapter
from core import Features
def main():
 torch.set_num_threads(2);net=ViewConditionedShape();fe=Features();rows=read(U/'INPUTS.json')['rows'];truth=read(U/'SOURCE_TRUTH.json')['rows'];out=[]
 for r,t in zip(rows,truth):
  raw=dict(np.load(r['raw']));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  sh={h:net.sequence(ch,h)for h in['L','R']};f,q,_=fe(ch)
  # Native x right-positive, y downward-positive; body-relative coords too.
  xy=ch['body_relative_xy'];av=ch['point_available'];lp=xy[:,[91,96,100,104,108]].mean(1);rp=xy[:,[112,117,121,125,129]].mean(1)
  mouth=xy[:,[71,74,77,80]].mean(1);d=rp-lp;dx=rp-mouth
  out.append(dict(person=r['person'],role=r['role'],mouth=[dict(frame=i,fingers=sh['R']['fingers'][i].tolist(),d=dx[i].round(3).tolist())for i in t['mouth_anchors']],
   pair=[dict(frame=i,R=int(sh['R']['states'][i]),L=int(sh['L']['states'][i]),Rf=sh['R']['fingers'][i].tolist(),Lf=sh['L']['fingers'][i].tolist(),d=d[i].round(3).tolist())for i in t['pair_anchors']]))
  dest=U/'features';dest.mkdir(exist_ok=True)
  np.savez_compressed(dest/f'p{r["person"]:02d}.npz',features=f,quality=q,pair=d,mouth=dx,**{f'{h}_{k}':sh[h][k]for h in['L','R']for k in['states','fingers','available']})
 write(U/'BASELINE.json',dict(rows=out))
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
