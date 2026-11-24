"""Render EVERY ROI-added hand-frame for qualitative review, no new GT fitting."""
from pathlib import Path
import json,cv2,numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
    adds=[r for r in read(U/'ADDITIONS.json') if r['roi_kept_joints']>0];f=read(U/'FROZEN.json');rs={r['clip_id']:r for r in f['rows']};ix={r['clip_id']:r for r in read(U/'INDEX.json')}
    (U/'addition_review').mkdir(exist_ok=True);tiles=[];index=[]
    for k,r in enumerate(adds):
        cid=r['clip_id'];i=r['frame_id'];s=r['side'];meta=rs[cid]
        with np.load(ix[cid]['arrays'],allow_pickle=False) as z:p=z['my_xy'][i,s];v=z['my_valid'][i,s]
        cap=cv2.VideoCapture(meta['rgb_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();cap.release();assert ok
        raw=cv2.resize(img,(768,432));ann=raw.copy()
        for j in np.flatnonzero(v):cv2.circle(ann,tuple((p[j]*.4).astype(int)),3,(0,180,255),-1)
        if v[0]:cv2.circle(ann,tuple((p[0]*.4).astype(int)),8,(0,0,255),2)
        tile=np.hstack([raw,ann]);tile=cv2.copyMakeBorder(tile,34,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
        text=f'A{k+1:03} {cid} f{i} '+('LEFT' if s==0 else 'RIGHT')
        cv2.putText(tile,text,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile);index.append({'review_id':f'A{k+1:03}',**r})
    for start in range(0,len(tiles),4):cv2.imwrite(str(U/'addition_review'/f'page_{start//4+1:02}.jpg'),np.vstack(tiles[start:start+4]))
    (U/'ADDITION_REVIEW_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
    print('ALL_ROI_ADDITION_FRAMES',len(adds),'PAGES',(len(tiles)+3)//4)
if __name__=='__main__':main()
