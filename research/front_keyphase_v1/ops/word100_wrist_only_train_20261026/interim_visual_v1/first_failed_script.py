"""Read only completed records; early visual triage cannot authorize training."""
from pathlib import Path
import json,sys,hashlib,datetime,math,importlib.util
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
J=Path(__file__).resolve().parent
I=J.parent/'word100_wrist_only_inputs_20261026'
U=J/'interim_visual_v1'
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in ('xy_px','usable','wrist_xy','wrist_usable','point_origin')}
def main():
    import cv2
    assert not U.exists();U.mkdir()
    spec=importlib.util.spec_from_file_location('wrist_review_render',J/'review_full.py');render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    events=[json.loads(x) for x in (I/'EVENTS.jsonl').read_text('utf8').splitlines() if x.strip()]
    done=sorted({r['uid'] for r in events if r['stage']=='INPUT_DONE'})
    f=read(I/'FROZEN.json');infos={r['uid']:r['info'] for r in f['rows']}
    rows=[];record_pins={}
    for uid in done:
        rp=I/'records'/(uid+'.json');rec=read(rp);record_pins[str(rp)]=sha(rp)
        if rec['new_wrists']==0:continue
        path=I/'outputs'/(uid+'.npz');assert sha(path)==rec['outputs'][str(path)]
        z=load(path);box=infos[uid]['original_roi']
        for i,s in zip(*np.where(z['wrist_usable'])):
            p=z['wrist_xy'][i,s];base=z['xy_px'][0,i];valid=z['usable'][0,i]
            scale=float(np.linalg.norm(base[11]-base[12])) if valid[[11,12]].all() else float(infos[uid]['original_scales']['whole']['body'])
            scale=max(scale,1)
            opposite=54 if s==0 else 33
            distance=float(np.linalg.norm(p-base[opposite])/scale) if valid[opposite] else None
            border=float(min(p[0]-box[0],box[2]-p[0],p[1]-box[1],box[3]-p[1])/scale)
            pose=float(np.linalg.norm(p-base[15+s])/scale) if valid[15+s] else None
            rows.append(dict(uid=uid,index=int(i),side=int(s),origin=int(z['point_origin'][2,i,33+21*s]),wrist=p.tolist(),
                frame_id=next(r['frame_id'] for r in rec['frames'] if r['i']==i),opposite=distance,border=border,pose=pose,
                image=str(I/'frames'/f'{uid}_{i:04d}.jpg'),output=str(path),output_sha256=sha(path)))
    # Stable source/frame/side de-duplication for preview only; full final review still includes both views.
    unique={}
    for r in rows:
        key=(r['uid'].split('__')[0],r['frame_id'],r['side'])
        if key not in unique or (r['origin']==3 and unique[key]['origin']!=3):unique[key]=r
    pool=list(unique.values());chosen={}
    def select(seq,reason,limit):
        for r in seq[:limit]:
            key=(r['uid'],r['index'],r['side'])
            if key not in chosen:chosen[key]=dict(r,reasons=[])
            chosen[key]['reasons'].append(reason)
    select(sorted([r for r in pool if r['origin']==3],key=lambda r:hashlib.sha256(str((r['uid'],r['index'],r['side'])).encode()).hexdigest()),'YOLO deterministic',16)
    for field,reverse in [('opposite',False),('border',False),('pose',True)]:
        select(sorted([r for r in pool if r[field] is not None],key=lambda r:(r[field],r['uid'],r['index']),reverse=reverse),field+' rank',8)
    chosen=list(chosen.values());cache={};pins={str(I/'FROZEN.json'):sha(I/'FROZEN.json')}
    W,H=640,320
    for k,r in enumerate(chosen):
        im=cv2.imread(r['image']);assert im is not None;pins[r['image']]=sha(r['image']);pins[r['output']]=r['output_sha256']
        if r['uid'] not in cache:cache={r['uid']:load(Path(r['output']))}
        z=cache[r['uid']];i,s=r['index'],r['side'];x,y=r['wrist'];r['review_index']=k
        panel=np.full((H,W,3),248,np.uint8);context,scale,dx,dy=render.letterbox(im,315,175);panel[34:209,0:315]=context
        cv2.circle(panel,(round(x*scale+dx),round(y*scale+dy+34)),4,(0,0,255),2)
        rad=max(70,round(im.shape[1]*.055));x0=max(0,round(x)-rad);x1=min(im.shape[1],round(x)+rad);y0=max(0,round(y)-rad);y1=min(im.shape[0],round(y)+rad)
        crop=im[y0:y1,x0:x1];marked,ss,xx,yy=render.letterbox(crop,300,260)
        cv2.circle(marked,(round((x-x0)*ss+xx),round((y-y0)*ss+yy)),5,(0,0,255),2)
        for j,col in [(15+s,(255,200,0)),(54 if s==0 else 33,(40,180,40))]:
            if z['usable'][0,i,j]:
                a,b=z['xy_px'][0,i,j];cv2.circle(marked,(round((a-x0)*ss+xx),round((b-y0)*ss+yy)),4,col,1)
        panel[34:294,330:630]=marked;raw,_,_,_=render.letterbox(crop,315,101);panel[214:315,0:315]=raw
        title=f"#{k} {r['uid'].replace('NIA_SL_','')} t{i} {'L' if s==0 else 'R'} src{r['origin']}"
        cv2.putText(panel,title,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(15,15,15),1,cv2.LINE_AA)
        cv2.putText(panel,'Red recovered / cyan pose / green opposite',(4,29),cv2.FONT_HERSHEY_SIMPLEX,.36,(40,40,40),1,cv2.LINE_AA)
        if k%8==0:sheet=np.full((H*4,W*2,3),255,np.uint8)
        q=k%8;sheet[(q//2)*H:(q//2+1)*H,(q%2)*W:(q%2+1)*W]=panel;r['page']=k//8
        if q==7 or k==len(chosen)-1:assert cv2.imwrite(str(U/f'preview_{k//8:03d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
    for p,h in pins.items():assert sha(p)==h,p
    result=dict(at=datetime.datetime.now().astimezone().isoformat(),status='INTERIM_VISUAL_PREVIEW_NOT_READY',
        snapshot_completed_views=len(done),snapshot_event=events[-1],additions=len(rows),unique_scene_hands=len(pool),
        origin_counts=dict(Counter(r['origin'] for r in rows)),selected=len(chosen),pages=math.ceil(len(chosen)/8),rows=chosen,
        pins=pins,record_pins=record_pins,policy='16 deterministic YOLO plus8 nearest opposite/ROI edge/largest pose distance; de-duplicate same source-frame-side for preview. Selection has no word label or classification score. Full input replay and final broader visual review still required.',
        all_input_numeric_verified=False,ready_for_training=False,training=0)
    with (U/'INDEX.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','snapshot_completed_views','additions','origin_counts','selected','pages')}),flush=True)
if __name__=='__main__':main()
