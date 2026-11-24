from pathlib import Path
import sys,json,time
import numpy as np
import cv2,av
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;P=G/'selective_rtmw_retry_20261117';N=G/'frontal_scope_recovery_20261117'
sys.path.insert(0,str(P))
from probe import FeedbackConsumer,Engine,close,read,write,require_front
from selective_retry import requests,VideoFrames,visible
import torch

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    records=read(N/'INPUTS.json')['rows'];failures=[r for r in read(P/'PUBLIC_VALIDATION.json')['controls']if r['kind']in ['missing_one_finger','missing_hand']and r['unknown_after']]
    net=FeedbackConsumer();engine=Engine();rows=[];sheets=[]
    try:
        for cid in dict.fromkeys(r['id']for r in failures):
            r=next(r for r in records if r['id']==cid);require_front(r);z=dict(np.load(r['raw']));video=r.get('video',r.get('path'))
            normal=net.raw(z,r['word']);guide=next(c for c in normal['cards']if c['id']==('right_shape'if r['word']=='WORD1290'else'index_shape'));a,b=guide['query_spans'][0];mid=(a+b)//2;lo,hi=mid-2,mid+2
            raw={k:v.copy()for k,v in z.items()};raw['xy_px'][lo:hi+1,125:129]=np.nan;raw['score'][lo:hi+1,125:129]=0;initial=net.raw(raw,r['word']);candidates,rejected=requests(raw,initial)
            reader=VideoFrames(video,z);begin=time.perf_counter();images=reader.read(list(range(lo-1,hi+2)),time.perf_counter()+10);decode=time.perf_counter()-begin
            w=int(z['source_width']);h=int(z['source_height']);parts=[];overlap=[];metrics=[]
            for i in range(lo-1,hi+2):
                p=z['xy_px'][i];own=p[112:133].astype('float32');other=p[91:112].astype('float32')
                ar=0
                if visible(own,w,h).all()and visible(other,w,h).all():ar=float(cv2.intersectConvexConvex(cv2.convexHull(own),cv2.convexHull(other))[0])
                overlap.append(dict(frame=i,actual_raw_hull_overlap=ar))
            if candidates:
                c=candidates[0];box=np.array(c['box'],np.float32);center=(box[:2]+box[2:])/2;extent=(box[2:]-box[:2])*1.1;second=np.r_[np.maximum(center-extent/2,0),np.minimum(center+extent/2,[w,h])]
                for i in c['frames']:
                    outputs=[];timings=[]
                    for bb in [box,second]:
                        t=time.perf_counter();rgb=cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB);convert=time.perf_counter()-t;t=time.perf_counter();kp,score=engine.pose(rgb,bboxes=[bb]);duration=time.perf_counter()-t;outputs.append(kp[0]);timings.append(dict(convert=convert,pose=duration))
                    delta=np.linalg.norm(outputs[0][112:133]-outputs[1][112:133],axis=1)/c['palm'];bad=np.flatnonzero(delta>.35).tolist()
                    metrics.append(dict(frame=i,mean=float(delta.mean()),maximum=float(delta.max()),largest_joints=np.argsort(delta)[-6:].tolist(),above_max=bad,deltas=delta.tolist(),timings=timings))
                    np.savez_compressed(D/f'{cid}_f{i}_poses.npz',one=outputs[0],two=outputs[1],original=z['xy_px'][i],box=box,second=second)
            for i in [lo-1,(lo+hi)//2,hi+1]:
                im=images[i];p=z['xy_px'][i];pts=p[np.r_[5:11,91:133]];pts=pts[visible(pts,w,h)];left=max(0,int(pts[:,0].min()-60));right=min(w,int(pts[:,0].max()+60));top=max(0,int(min(pts[:,1].min()-60,p[0,1]-20)));bottom=min(h,int(pts[:,1].max()+60));crop=im[top:bottom,left:right];scale=min(520/crop.shape[1],360/crop.shape[0]);small=cv2.resize(crop,None,fx=scale,fy=scale);tile=np.full((410,540,3),245,np.uint8);tile[40:40+small.shape[0],10:10+small.shape[1]]=small;cv2.putText(tile,f'{cid} f{i}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1);parts.append(tile)
            sheet=np.concatenate(parts,axis=1);path=D/f'{cid}_source.jpg';cv2.imwrite(str(path),sheet);sheets.append(str(path))
            row=dict(id=cid,span=[lo,hi],rejected=rejected,candidate=candidates[:1],decode_seconds=decode,decoded_frames=reader.decoded,observed_overlap=overlap,pose_comparison=metrics,source=video,sheet=str(path))
            rows.append(row);print('DIAG',cid,'decode',round(decode,3),'reject',rejected,'disagree',[(x['frame'],x['above_max'],round(x['mean'],3))for x in metrics],flush=True)
        write(D/'DIAGNOSIS.json',dict(rows=rows,private_data=False,training=False,source='Known public frontal footage and controlled gaps; fresh visual review required'))
    finally:close(net)
if __name__=='__main__':main()
