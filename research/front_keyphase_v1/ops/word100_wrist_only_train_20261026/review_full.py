"""Label-blind review sampling of ALL saved wrist additions. No input filtering or scoring."""
from pathlib import Path
import sys,json,hashlib,datetime,math,re
from collections import Counter,defaultdict
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def letterbox(im,w,h):
    import cv2
    scale=min(w/im.shape[1],h/im.shape[0]);nw=max(1,round(im.shape[1]*scale));nh=max(1,round(im.shape[0]*scale));out=np.full((h,w,3),245,np.uint8)
    dx=(w-nw)//2;dy=(h-nh)//2;out[dy:dy+nh,dx:dx+nw]=cv2.resize(im,(nw,nh));return out,scale,dx,dy
def main(unit):
    import cv2
    U=Path(unit).resolve();v=read(U/'ALL_VERIFIED.json');f=read(U/'FROZEN.json')
    assert v['status']=='PASS' and v['frozen_sha256']==sha(U/'FROZEN.json') and not (U/'RUNNING.json').exists()
    assert not (U/'RISK_INDEX.json').exists()
    info={r['uid']:r['info'] for r in f['rows']};groups=defaultdict(list)
    for item in v['review']:groups[item['uid']].append(item)
    rows=[]
    for uid,items in sorted(groups.items()):
        out=load(U/'outputs'/(uid+'.npz'));box=info[uid]['original_roi'];records=read(U/'records'/(uid+'.json'));frs={r['i']:r for r in records['frames']}
        for it in items:
            i=it['index'];s=it['side'];p=np.asarray(it['wrist']);base=out['xy_px'][0,i];ok=out['usable'][0,i];shoulder=np.linalg.norm(base[11]-base[12]) if ok[[11,12]].all() else None
            if shoulder is None or not np.isfinite(shoulder) or shoulder<1:shoulder=max(1,float(info[uid]['original_scales']['whole']['body']))
            pose_d=float(np.linalg.norm(p-base[15+s])/shoulder) if ok[15+s] else None
            other=54 if s==0 else 33
            opposite=float(np.linalg.norm(p-base[other])/shoulder) if ok[other] else None
            previous=None
            for k in range(i-1,max(-1,i-11),-1):
                if out['usable'][0,k,33+21*s]:
                    previous=float(np.linalg.norm(p-out['xy_px'][0,k,33+21*s])/shoulder/(i-k));break
            boundary=float(min(p[0]-box[0],box[2]-p[0],p[1]-box[1],box[3]-p[1])/shoulder)
            decisions=frs[i]['mp_decisions'] if it['origin']==2 else frs[i]['yolo_decisions']
            accepted=[q for q in decisions if q['accepted'] and q['side']==s]
            assert accepted
            row=dict(review_index=len(rows),**it,pose_distance_shoulders=pose_d,opposite_distance_shoulders=opposite,previous_motion_per_sample=previous,boundary_distance_shoulders=boundary,
                recording_id=int(re.search(r'REAL(\d+)',uid).group(1)),view=uid.rsplit('__',1)[1],review_reasons=[])
            rows.append(row)
    selected=set()
    def include(items,why):
        for r in items:selected.add(r['review_index']);r['review_reasons'].append(why)
    include([r for r in rows if r['origin']==3],'all_YOLO_wrist_additions')
    buckets=defaultdict(list)
    for r in rows:buckets[(r['recording_id'],r['view'])].append(r)
    for key,items in buckets.items():
        ranked=sorted(items,key=lambda r:hashlib.sha256(f"{r['uid']}:{r['index']}:{r['side']}".encode()).hexdigest())
        include(ranked[:2],'two_deterministic_examples_per_recording_and_view')
    for field,reverse in [('pose_distance_shoulders',True),('opposite_distance_shoulders',False),('previous_motion_per_sample',True),('boundary_distance_shoulders',False)]:
        ranked=sorted([r for r in rows if r[field] is not None],key=lambda r:(r[field],r['review_index']),reverse=reverse)
        include(ranked[:24],'rank24_'+field)
    chosen=[r for r in rows if r['review_index'] in selected];dest=U/'full_visual';dest.mkdir(exist_ok=True)
    W,H=640,320;pp=8;cache={}
    for k,r in enumerate(chosen):
        im=cv2.imread(r['image']);assert im is not None
        if r['uid'] not in cache:cache={r['uid']:load(U/'outputs'/(r['uid']+'.npz'))}
        z=cache[r['uid']];i=r['index'];s=r['side'];x,y=r['wrist']
        panel=np.full((H,W,3),248,np.uint8);context,scale,dx,dy=letterbox(im,315,175);panel[34:209,0:315]=context
        cv2.circle(panel,(round(x*scale+dx),round(y*scale+dy+34)),4,(0,0,255),2)
        radius=max(70,round(im.shape[1]*.055));x0=max(0,round(x)-radius);x1=min(im.shape[1],round(x)+radius);y0=max(0,round(y)-radius);y1=min(im.shape[0],round(y)+radius)
        crop=im[y0:y1,x0:x1];marked,ss,xx,yy=letterbox(crop,300,260)
        cv2.circle(marked,(round((x-x0)*ss+xx),round((y-y0)*ss+yy)),5,(0,0,255),2)
        for j,color in [(15+s,(255,200,0)),(54 if s==0 else 33,(40,180,40))]:
            if z['usable'][0,i,j]:
                a,b=z['xy_px'][0,i,j];cv2.circle(marked,(round((a-x0)*ss+xx),round((b-y0)*ss+yy)),4,color,1)
        panel[34:294,330:630]=marked;raw,_,_,_=letterbox(crop,315,101);panel[214:315,0:315]=raw
        title=f"#{r['review_index']} {r['uid'].replace('NIA_SL_','')} t{i} {'L' if s==0 else 'R'} src{r['origin']}"
        cv2.putText(panel,title,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.40,(15,15,15),1,cv2.LINE_AA)
        cv2.putText(panel,'Red recovered / cyan pose / green opposite',(4,29),cv2.FONT_HERSHEY_SIMPLEX,.36,(40,40,40),1,cv2.LINE_AA)
        if k%pp==0:sheet=np.full((H*4,W*2,3),255,np.uint8)
        q=k%pp;sheet[(q//2)*H:(q//2+1)*H,(q%2)*W:(q%2+1)*W]=panel
        r['visual_page']=k//pp
        if q==pp-1 or k==len(chosen)-1:assert cv2.imwrite(str(dest/f'risk_{k//pp:03d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
    result=dict(at=now(),verified_sha256=sha(U/'ALL_VERIFIED.json'),script_sha256=sha(__file__),all_additions=len(rows),selected=len(chosen),pages=math.ceil(len(chosen)/pp),origin_counts=dict(Counter(r['origin'] for r in rows)),rows=rows,
        policy='AllYOLO +2deterministic per recording/view +24nearest opposite +24near boundary +24largest pose disagreement +24largest prior motion. No labels/scores used. Ranking is review priority, not an error label. Original hands preserved numerically. Review sample does not certify all added wrist anatomy.',ready_for_training=False)
    write(U/'RISK_INDEX.json',result)
    print(json.dumps({k:result[k] for k in ('all_additions','selected','pages','origin_counts')}),flush=True)
if __name__=='__main__':main(sys.argv[1])
