"""Public-only reproduction; never reads personal diagnostic or camera data."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,time
import numpy as np
import cv2

V=Path(__file__).resolve().parent; U=V.parent; O=U.parent
sys.path.insert(0,str(U))
from coach import Coach

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run_case(coach,name,a):
    coach.reset(); cats=Counter(); reasons=Counter(); rows=[]; stage_times=[None,None]
    for i,t in enumerate(a['timestamp_us']):
        ob=coach.update(a['xy_px'][i],a['score'][i],int(t),int(a['source_width']),int(a['source_height']))
        cat=ob['diagnostic']['category']; cats[cat]+=1
        dd=ob.get('action_direction_check',[]); dd=[dd] if isinstance(dd,dict) else dd
        for d in dd:
            reasons[str((d.get('reason'),d.get('eligible'),d.get('state')))]+=1
        for j,c in enumerate(ob['completed']):
            if c and stage_times[j] is None:stage_times[j]=i
        if cat!='progress':
            rows.append(dict(i=i,t=int(t),stage=ob['stage_index'],diagnostic=ob['diagnostic'],direction=dd))
    result=dict(name=name,completed=coach.completed,stage_received_frame=stage_times,counts=dict(cats),direction_reasons=dict(reasons))
    write(V/('PUBLIC_TRACE_'+name+'.json'),rows)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

def main():
    assert not (V/'PUBLIC_PROBE.json').exists()
    plan=dict(source=str(U/'source/reference.npz'),private_inputs=False,training=False,
      cases=['native_cache','public_video_640_letterbox','public_video_upper_body_640',
             'delayed_sweep_start','delayed_repeat_start'],
      intent='Keep the same public action order/shapes. Test camera-size redetection and long observable pre-movement waits; these are controlled reproductions, not new-person accuracy.',
      contract='Do not lower thresholds or train; retain first failures. Any change must pass prior positive/negative regression before live admission.')
    write(V/'PUBLIC_PLAN.json',plan)
    with np.load(U/'source/reference.npz') as z:raw={k:z[k].copy() for k in z.files}
    coach=Coach(); results=[run_case(coach,'native_cache',raw)]
    # Genuine re-extraction on public pixels at a camera-sized resolution.
    sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
    from stream_runtime import Engine
    engine=Engine()
    source=O.parent/'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'
    for name in ['public_video_640_letterbox','public_video_upper_body_640']:
        cap=cv2.VideoCapture(str(source)); points=[];scores=[]; ms=[]; i=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if name.endswith('letterbox'):
                im=np.zeros((480,640,3),np.uint8);im[60:420]=cv2.resize(bgr,(640,360))
            else: im=cv2.resize(bgr[60:780,480:1440],(640,480))
            out=engine.infer(im);points.append(out['xy']);scores.append(out['score']);ms.append(out['compute_ms'])
            if i==51:cv2.imwrite(str(V/(name+'.jpg')),im)
            i+=1
        cap.release()
        a=dict(xy_px=np.array(points),score=np.array(scores),timestamp_us=raw['timestamp_us'],frame_id=raw['frame_id'],source_width=np.int64(640),source_height=np.int64(480))
        np.savez_compressed(V/(name+'.npz'),**a)
        result=run_case(coach,name,a);result['rtmw_median_ms']=float(np.median(ms));results.append(result)
    for name,frame in [('delayed_sweep_start',35),('delayed_repeat_start',51)]:
        # 16 seconds of a real public starting pose, then the original action.
        # Small fixed shoulder-relative rigid motion avoids an unrealistic exact-equality-only plateau.
        a={k:v.copy()for k,v in raw.items()};idx=np.r_[np.arange(frame+1),np.repeat(frame,480),np.arange(frame+1,len(raw['timestamp_us']))]
        a['xy_px']=raw['xy_px'][idx].copy();a['score']=raw['score'][idx].copy()
        for j in range(frame+1,frame+481):a['xy_px'][j,:,1]+=np.sin(j*.17)*.5
        a['frame_id']=np.arange(len(idx));a['timestamp_us']=np.rint(a['frame_id']*1e6/30).astype(np.int64)
        results.append(run_case(coach,name,a))
    write(V/'PUBLIC_PROBE.json',dict(results=results,private_data=False,training=False,
       coach_sha256=hashlib.sha256((U/'coach.py').read_bytes()).hexdigest(),
       action_sha256=hashlib.sha256((U/'action_runtime.py').read_bytes()).hexdigest()))

if __name__=='__main__':main()
