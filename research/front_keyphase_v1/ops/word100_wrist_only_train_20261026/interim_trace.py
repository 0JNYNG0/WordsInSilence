"""Trace saved detector wrists into candidate channels; no inference or training."""
from pathlib import Path
import json, hashlib, datetime, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
J=Path('C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1/ops/word100_wrist_only_train_20261026')
I=J.parent/'word100_wrist_only_inputs_20261026'
U=J/'interim_visual_v2'
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def main():
    import cv2
    assert not (U/'COORDINATE_TRACE.json').exists()
    idx=read(U/'INDEX.json'); frozen=read(I/'FROZEN.json'); info={r['uid']:r['info'] for r in frozen['rows']}
    result=[]; pins={str(U/'INDEX.json'):sha(U/'INDEX.json')}; feature_checks=0
    for row in idx['rows']:
        uid=row['uid']; i=row['index']; s=row['side']; f=info[uid]
        rp=I/'records'/(uid+'.json'); rec=read(rp); assert sha(rp)==idx['record_pins'][str(rp)]
        z=load(row['output']); assert sha(row['output'])==row['output_sha256']
        fr=next(x for x in rec['frames'] if x['i']==i)
        prefix='mp' if row['origin']==2 else 'yolo'
        q=next(q for q in fr[prefix+'_decisions'] if q['accepted'] and q['side']==s)
        full=np.asarray(fr[prefix+'_full'][q['index']]['xy']); crop=np.asarray(fr[prefix+'_crop'][q['crop_index']]['xy'])
        saved=z['wrist_xy'][i,s]; assert np.array_equal(full[0].astype(np.float32),saved)
        base=load(f['source_arrays']['observation']); old=load(f['source_arrays']['channels']); xp=I/'features'/(uid+'.npz'); new=load(xp)
        assert sha(xp)==rec['outputs'][str(xp)]
        for temporal in ('whole','context'):
            lo,hi=f['original_spans'][temporal]
            if lo<=i<=hi and base['usable'][i,[11,12]].all():
                bmid=base['xy_px'][i,[11,12]].sum(0)/np.float32(2)
                expected=(saved-bmid)/f['original_scales'][temporal]['body']; j=9+21*s
                assert np.array_equal(expected,new['x_'+temporal][i-lo,2*j:2*j+2])
                assert new['x_'+temporal][i-lo,186+j]==1
                feature_checks+=1
        pins[str(rp)]=sha(rp); pins[str(xp)]=sha(xp)
        result.append(dict(review_index=row['review_index'],uid=uid,index=i,side=s,source=prefix,
            full_point0=full[0].tolist(),crop_point0=crop[0].tolist(),stored_point0=saved.tolist(),
            float32_rounding_px=float(np.linalg.norm(full[0]-saved)),full_crop_disagreement_px=float(np.linalg.norm(full[0]-crop[0])),
            original_pose_wrist=base['xy_px'][i,15+s].tolist() if base['usable'][i,15+s] else None,
            decision=q['reason'],geometry_mapping_pass=True))
    # Four representative location disagreements: do not re-detect or estimate unseen anatomy.
    temporal=[]
    for number in (32,33,35,37):
        row=idx['rows'][number]; uid=row['uid']; i=row['index']; s=row['side']; f=info[uid]
        rec=read(I/'records'/(uid+'.json')); video=Path(rec['source_video']); assert sha(video)==rec['source_sha256']
        out=load(row['output']); frames=out['frame_id']; wanted=[max(0,min(len(frames)-1,i+d)) for d in (-3,-1,0,1,3)]
        wanted=sorted(set(wanted)); ids={int(frames[j]):j for j in wanted}; images={}; cap=cv2.VideoCapture(str(video)); assert cap.isOpened()
        for fid in range(max(ids)+1):
            ok,im=cap.read(); assert ok
            if fid in ids: images[ids[fid]]=im
        cap.release()
        sheet=np.full((430,len(wanted)*360,3),245,np.uint8); bx,by=row['wrist']; rad=180
        ww,hh=f['canvas']; x0=max(0,int(bx)-rad); x1=min(ww,int(bx)+rad); y0=max(0,int(by)-rad); y1=min(hh,int(by)+rad)
        for k,j in enumerate(wanted):
            im=images[j][:hh,:ww]; raw=im[y0:y1,x0:x1].copy(); cw,ch=raw.shape[1],raw.shape[0]; scale=min(340/cw,330/ch)
            marked=cv2.resize(raw,(round(cw*scale),round(ch*scale)))
            def point(p,col,r):
                cv2.circle(marked,(round((p[0]-x0)*scale),round((p[1]-y0)*scale)),r,col,2)
            if out['usable'][0,j,15+s]: point(out['xy_px'][0,j,15+s],(255,200,0),5)
            if out['usable'][0,j,33+21*s]: point(out['xy_px'][0,j,33+21*s],(255,255,255),6)
            if out['wrist_usable'][j,s]: point(out['wrist_xy'][j,s],(0,0,255),6)
            matching=[x for x in result if x['uid']==uid and x['index']==j and x['side']==s]
            if matching: point(matching[0]['crop_point0'],(0,190,230),4)
            xx=k*360+10; sheet[55:55+marked.shape[0],xx:xx+marked.shape[1]]=marked
            cv2.putText(sheet,f'index {j} / frame {int(frames[j])}',(xx,45),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        title=f'#{number} {uid} side {s}: red recovered / yellow crop / cyan pose / white old hand'
        cv2.putText(sheet,title,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1)
        cv2.putText(sheet,'No point is anatomical ground truth. Same original source frames; no detector rerun.',(10,414),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,0,0),1)
        dst=U/f'temporal_{number}.jpg'; assert not dst.exists(); assert cv2.imwrite(str(dst),sheet,[cv2.IMWRITE_JPEG_QUALITY,96])
        pins[str(dst)]=sha(dst); temporal.append(dict(review_index=number,source=str(video),source_sha256=rec['source_sha256'],indices=wanted,path=str(dst)))
    for p,h in pins.items(): assert sha(p)==h
    output=dict(at=datetime.datetime.now().astimezone().isoformat(),status='SAVED_COORDINATE_MAPPING_PASS_ANATOMY_UNCERTAIN',
        selected_wrist_frames=len(result),feature_insertions_verified=feature_checks,rows=result,temporal=temporal,pins=pins,
        finding='Stored wrists equal full-image detector landmark0 after float32 rounding; candidate wrist channels use the original fixed body scale. No crop coordinate offset or point-index substitution found in these40 selected frames.',
        limitation='This verifies data mapping, not anatomical accuracy. RGB review and current pose disagreement can reveal detector localization uncertainty; neither another detector nor pose wrist is ground truth.',ready_for_training=False,detector_calls=0,training_steps=0)
    with (U/'COORDINATE_TRACE.json').open('x',encoding='utf8') as f: json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:output[k] for k in ('status','selected_wrist_frames','feature_insertions_verified','detector_calls','training_steps')}))
if __name__=='__main__': main()
