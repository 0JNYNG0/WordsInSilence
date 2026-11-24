from common import *
from candidate import trigger,supplement
import sys,os,time,traceback,datetime
import cv2
def run():
    f=read(U/'FROZEN.json')
    for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
    assert not (U/'RUNNING.json').exists() and not (U/'INDEX.json').exists()
    # Own runtime configuration; historical results/checkpoint remain read-only.
    source=(OLD/'model_io.py').read_text('utf8').replace("p=U/'yolo_hand_best.pt'",f"p=Path({str(OLD/'yolo_hand_best.pt')!r})").replace("(U/'MODEL_SOURCE.json')",f"(Path({str(OLD/'MODEL_SOURCE.json')!r}))")
    (U/'model_io_local.py').write_text(source,encoding='utf8')
    loader=module('selective_model_io',U/'model_io_local.py');import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    exp=module('selective_shared_detect',OLD/'experiment.py');model,meta=loader.load_model()
    for _ in range(3):exp.ydetect(model,np.zeros((1080,1920,3),np.uint8))
    write(U/'RUNNING.json',{'pid':os.getpid(),'at':datetime.datetime.now().astimezone().isoformat()})
    write(U/'EXECUTION_CONTEXT.json',{'code_sha256':sha(Path(__file__)),'loader_sha256':sha(U/'model_io_local.py'),'environment':meta,'pid':os.getpid()})
    (U/'outputs').mkdir();(U/'records').mkdir();index=[]
    for r in f['old_rows']+f['new_rows']:
        z=load(r['arrays']);mprecs=read(r['source_assets']['detections'])['frames'];n=r['frames'];cid=r['clip_id']
        base=z['redetected_roi_xy_px'][:,33:].reshape(n,2,21,2).copy();bv=z['redetected_roi_usable'][:,33:].reshape(n,2,21).copy()
        arr={'s_xy':base.copy(),'s_valid':bv.copy(),'p_xy':base.copy(),'p_valid':bv.copy(),'called':np.zeros(n,bool),'called_sides':np.zeros((n,2),bool),
             'yolo_ms':np.zeros(n),'geometry_ms':np.zeros(n),'frame_id':z['frame_id'],'position_only':np.zeros((n,2),bool)}
        recs=[];cap=cv2.VideoCapture(r['rgb_path']);assert cap.isOpened()
        for i in range(n):
            ok,bgr=cap.read();assert ok
            p,v=native_m1(z,i,mprecs[i]);pose=z['raw_pose_xy_px'][i];pu=z['raw_pose_usable'][i];w=int(z['width'][i]);h=int(z['height'][i]);boxes=trigger(p,v,pose,pu,w,h,z['roi_box'])
            if not boxes:continue
            arr['called'][i]=True;arr['called_sides'][i,list(boxes)]=True
            full,ms=exp.ydetect(model,bgr);crop=[]
            for side,b in boxes.items():
                ds,dt=exp.ydetect(model,bgr[b[1]:b[3],b[0]:b[2]],b);ms+=dt
                for d in ds:d['requested_side']=side
                crop.extend(ds)
            t=time.perf_counter();nr=int(z['raw_raw_count'][i])
            sp,sv,pp,pv,dec=supplement(full,crop,p,v,z['raw_hand_xy_px'][i,:nr],z['assignment_sides'][i,:nr].tolist(),pose,pu,w,h,boxes)
            arr['s_xy'][i],arr['s_valid'][i]=roi(sp,sv,z['roi_box']);arr['p_xy'][i],arr['p_valid'][i]=roi(pp,pv,z['roi_box']);arr['geometry_ms'][i]=(time.perf_counter()-t)*1000
            arr['yolo_ms'][i]=ms
            for q in dec:
                if q['accepted'] and q['position_only']:arr['position_only'][i,q['side']]=True
            for pref in ('s','p'):
                assert np.array_equal(arr[pref+'_valid'][i][bv[i]],bv[i][bv[i]])
                assert np.allclose(arr[pref+'_xy'][i][bv[i]],base[i][bv[i]],rtol=0,atol=.0002)
                # Make inherited coordinates bit-identical, not just equivalent reconstruction.
                arr[pref+'_xy'][i][bv[i]]=base[i][bv[i]]
            recs.append(exp.serializable({'i':i,'boxes':boxes,'full':full,'crop':crop,'decisions':dec}))
        cap.release()
        path=U/'outputs'/f'{cid}.npz';np.savez_compressed(path,**arr)
        rp=U/'records'/f'{cid}.json';write(rp,{'clip_id':cid,'frames':recs})
        index.append({'clip_id':cid,'cohort':'extension07' if r['signer_id']==7 else 'known39','frames':n,'arrays':str(path),'arrays_sha256':sha(path),'records':str(rp),'records_sha256':sha(rp)})
        print('DONE',len(index),46,cid,'calls',int(arr['called'].sum()),'positiononly',int(arr['position_only'].sum()),flush=True)
    for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
    write(U/'INDEX.json',index);(U/'RUNNING.json').unlink();print('COMPLETE',flush=True)
if __name__=='__main__':
    try:run()
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'error':traceback.format_exc()})
        raise
