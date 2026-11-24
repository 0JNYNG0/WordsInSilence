"""Review queue selected by geometry/provenance only, before new classification scores."""
from common import *
from collections import Counter,defaultdict
import cv2
from PIL import Image,ImageDraw,ImageFont
def main():
    iv=read(U/'INPUTS_VERIFIED.json');q=iv['queue'];dest=U/'review_visual';dest.mkdir()
    selected={};cache={}
    for i,a in enumerate(q):
        if a['stage']=='yolo':selected.setdefault(i,[]).append('every_yolo_addition')
        if a['clip_id'] not in cache:cache[a['clip_id']]=load(U/'outputs'/(a['clip_id']+'.npz'))
    # Prioritise overlapping hands, independent of recognition labels/scores.
    risks=[]
    for i,a in enumerate(q):
        p=cache[a['clip_id']];k=1 if a['stage']=='mp' else 2;t=a['frame'];hands=p['xy_px'][k,t,33:].reshape(2,21,2);v=p['usable'][k,t,33:].reshape(2,21)
        if v[:,0].all():
            sh=p['xy_px'][0,t,[11,12]];scale=np.linalg.norm(sh[0]-sh[1])
            if np.isfinite(scale) and scale>1:risks.append((float(np.linalg.norm(hands[0,0]-hands[1,0])/scale),i))
    for _,i in sorted(risks)[:24]:selected.setdefault(i,[]).append('smallest_interhand_wrist_gap')
    # Reproducible 12 MP additions per recorded performer ID, no score selection.
    groups=defaultdict(list)
    for i,a in enumerate(q):
        if a['stage']=='mp':groups[a['clip_id'].split('_REAL')[1].split('_')[0]].append(i)
    for s,ids in groups.items():
        for j in np.linspace(0,len(ids)-1,min(12,len(ids)),dtype=int):selected.setdefault(ids[j],[]).append('uniform_queue_sample_ID'+s)
    seq=[dict(queue_index=i,selection_reasons=selected[i],**q[i]) for i in sorted(selected)]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    edges=[(ch[i],ch[i+1]) for ch in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)) for i in range(4)]
    pages=[]
    for start in range(0,len(seq),8):
        board=Image.new('RGB',(1440,1440),'#edf1f6');draw=ImageDraw.Draw(board)
        for ii,a in enumerate(seq[start:start+8]):
            z=cache[a['clip_id']];i=a['frame'];s=a['side'];k=1 if a['stage']=='mp' else 2
            img=cv2.imread(a['image']);assert img is not None
            hands=z['xy_px'][k,i,33:].reshape(2,21,2);v=z['usable'][k,i,33:].reshape(2,21);pose=z['xy_px'][0,i,:33]
            pts=np.vstack([hands[v],pose[[13,14,15,16]]]);pts=pts[np.isfinite(pts).all(-1)]
            lo=np.maximum(0,np.floor(pts.min(0)-65)).astype(int);hi=np.minimum([1920,1080],np.ceil(pts.max(0)+65)).astype(int)
            assert ((hi-lo)>40).all();overlay=img.copy()
            for side,color in ((1-s,(235,150,40)),(s,(30,150,250))):
                for x,y in edges:
                    if v[side,x] and v[side,y]:cv2.line(overlay,tuple(hands[side,x].astype(int)),tuple(hands[side,y].astype(int)),color,2)
                for xy,ok in zip(hands[side],v[side]):
                    if ok:cv2.circle(overlay,tuple(xy.astype(int)),4,color,-1)
            x=(ii%2)*720;y=(ii//2)*360
            for j,im in enumerate((img,overlay)):
                crop=Image.fromarray(cv2.cvtColor(im[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2RGB));crop.thumbnail((354,292));board.paste(crop,(x+j*360,y+40))
            draw.text((x+5,y+5),f"{start+ii:02d} {a['clip_id'][7:22]} f{i} {'L' if s==0 else 'R'}",font=font,fill='#132a44')
            draw.text((x+5,y+334),f"{a['stage']} pts={a['points']} angle proxy={a['comparable_angle_count']} q={a['queue_index']}",font=font,fill='#132a44')
        path=dest/f'page_{start//8:02d}.jpg';board.save(path,quality=95);pages.append(str(path))
    write(dest/'INDEX.json',dict(at=now(),pages=pages,queue=seq,all_queue_size=len(q),selected=len(seq),
      note='Before classifier scoring: every YOLO addition,24 smallest interhand gaps,12 uniform MP additions per ID. Not all added hands visually checked. Raw left and overlay right; orange recovered side, blue other side. Coarse side/location review only, no hidden-finger truth.'))
    print('SELECTED',len(seq),'OF',len(q),'PAGES',len(pages),flush=True)
if __name__=='__main__':main()
