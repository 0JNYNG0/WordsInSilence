"""Incremental visual review of completed public inputs. No inference or readiness grant."""
from pathlib import Path
import sys,json,hashlib,datetime,importlib.util,math
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf8')
J=Path('C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1/ops/word100_wrist_only_train_20261026')
I=J.parent/'word100_wrist_only_inputs_20261026'; U=I/'incremental_yolo_v1'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in ('xy_px','usable','wrist_xy','wrist_usable','point_origin')}
def main():
    import cv2
    assert not U.exists(); U.mkdir()
    spec=importlib.util.spec_from_file_location('review_render',J/'review_full.py');render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    old=read(J/'interim_visual_v2/INDEX.json'); prior={(r['uid'],r['index'],r['side']) for r in old['rows']}
    assert read(J/'interim_visual_v2/REVIEW.json')['reviewed_frames']==len(prior)
    events=[json.loads(t) for t in (I/'EVENTS.jsonl').read_text('utf8').splitlines() if t.strip()]
    done=sorted({r['uid'] for r in events if r['stage']=='INPUT_DONE'}); rows=[]; pins={};outputs={}
    for uid in done:
        rp=I/'records'/(uid+'.json');rec=read(rp)
        if not rec['new_wrists'] or not rec['yolo_frames']:continue
        op=I/'outputs'/(uid+'.npz');assert sha(op)==rec['outputs'][str(op)];z=load(op)
        for i,s in zip(*np.where(z['wrist_usable'])):
            key=(uid,int(i),int(s));j=33+21*s
            if z['point_origin'][2,i,j]!=3 or key in prior:continue
            fr=next(r for r in rec['frames'] if r['i']==i)
            rows.append(dict(uid=uid,index=int(i),side=int(s),frame_id=fr['frame_id'],wrist=z['wrist_xy'][i,s].tolist(),origin=3,
                image=str(I/'frames'/f'{uid}_{i:04d}.jpg'),output=str(op),output_sha256=sha(op),record=str(rp),record_sha256=sha(rp)))
    rows.sort(key=lambda r:(r['uid'],r['index'],r['side'])); W,H=640,320;cache={}
    for k,r in enumerate(rows):
        im=cv2.imread(r['image']);assert im is not None; pins[r['image']]=sha(r['image']);pins[r['record']]=r['record_sha256'];outputs[r['output']]=r['output_sha256']
        if r['uid'] not in cache:cache={r['uid']:load(r['output'])}
        z=cache[r['uid']];i,s=r['index'],r['side'];x,y=r['wrist'];panel=np.full((H,W,3),248,np.uint8)
        context,sc,dx,dy=render.letterbox(im,315,175);panel[34:209,:315]=context;cv2.circle(panel,(round(x*sc+dx),round(y*sc+dy+34)),4,(0,0,255),2)
        rad=max(70,round(im.shape[1]*.055));x0=max(0,round(x)-rad);x1=min(im.shape[1],round(x)+rad);y0=max(0,round(y)-rad);y1=min(im.shape[0],round(y)+rad)
        crop=im[y0:y1,x0:x1];marked,sc,dx,dy=render.letterbox(crop,300,260)
        cv2.circle(marked,(round((x-x0)*sc+dx),round((y-y0)*sc+dy)),5,(0,0,255),2)
        for j,col in [(15+s,(255,200,0)),(54 if s==0 else 33,(40,180,40))]:
            if z['usable'][0,i,j]:
                a,b=z['xy_px'][0,i,j];cv2.circle(marked,(round((a-x0)*sc+dx),round((b-y0)*sc+dy)),4,col,1)
        panel[34:294,330:630]=marked;raw,*_=render.letterbox(crop,315,101);panel[214:315,:315]=raw
        cv2.putText(panel,f"#{k} {r['uid'].replace('NIA_SL_','')} t{i} {'L' if s==0 else 'R'}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(15,15,15),1)
        cv2.putText(panel,'Red YOLO wrist / cyan pose / green opposite',(4,29),cv2.FONT_HERSHEY_SIMPLEX,.36,(40,40,40),1)
        if k%8==0:sheet=np.full((H*4,W*2,3),255,np.uint8)
        q=k%8;sheet[q//2*H:(q//2+1)*H,q%2*W:(q%2+1)*W]=panel;r['review_index']=k;r['page']=k//8
        if q==7 or k==len(rows)-1:assert cv2.imwrite(str(U/f'page_{k//8:03d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
    for p,h in {**pins,**outputs}.items():assert sha(p)==h
    result=dict(at=datetime.datetime.now().astimezone().isoformat(),status='COMPLETED_RECORDS_YOLO_REVIEW_PENDING',snapshot_views=len(done),latest_event=events[-1],selected=len(rows),pages=math.ceil(len(rows)/8),rows=rows,pins=pins,output_pins=outputs,prior_review=str(J/'interim_visual_v2/REVIEW.json'),prior_index_sha256=sha(J/'interim_visual_v2/INDEX.json'),training_ready=False,policy='All YOLO added wrists in completed records, except exact uid/index/side already viewed in interim40. No label/score selection, input mutation, inference or training.')
    with (U/'INDEX.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','snapshot_views','selected','pages')}))
if __name__=='__main__':main()
