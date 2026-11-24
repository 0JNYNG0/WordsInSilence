"""Public RGB contact sheets, full frame plus hand detail; no image inference."""
from study import U,read,load,write
from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
from collections import defaultdict
font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",16)
small=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",13)
rows={r["clip_id"]:r for r in read(U/"PREPARED_INDEX.json")}
audit=read(U/"INPUT_VERIFY.json")
out=U/"visual";out.mkdir()
items=[]
for s in audit["normal_samples"]:items.append(dict(s,kind="NORMAL"))
for s in audit["new_added_hand_frames"]:
    if s["fresh"]:items.append(dict(s,kind="ADDED"))
for c in audit["previous_controls"]:
    if "cross" in c["group"]:
        z=load(rows[c["clip_id"]]["prepared"])
        items.append(dict(c,kind="CROSS",frame_id=int(z["frame_id"][c["index"]])))
# Three most withheld-but-computable frames per word, distinct clips.
for word in sorted({r["word_code"] for r in rows.values()}):
    candidates=[]
    for cid,r in rows.items():
        if r["word_code"]!=word:continue
        z=load(r["prepared"]);meta=read(U/"queries"/(cid+".json"))
        hs=[("L","R").index(s) for s in meta["required"]]
        for i in np.flatnonzero(z["core"]):
            ss=z["angle_state"][i,hs]
            comp=int((ss>=2).sum());held=int(np.isin(ss,[2,3]).sum())
            candidates.append((held,comp,cid,int(i)))
    seen=set()
    for held,comp,cid,i in sorted(candidates,reverse=True):
        if cid in seen:continue
        seen.add(cid);z=load(rows[cid]["prepared"])
        items.append(dict(clip_id=cid,index=i,frame_id=int(z["frame_id"][i]),kind="WITHHELD"))
        if len(seen)==3:break

bycid=defaultdict(list)
for n,s in enumerate(items):
    s["item_id"]=n;bycid[s["clip_id"]].append(s)
panels={}
for cid,ss in bycid.items():
    r=rows[cid];z=load(r["prepared"]);cap=cv2.VideoCapture(r["video"])
    for s in ss:
        i=s["index"];cap.set(cv2.CAP_PROP_POS_FRAMES,int(z["frame_id"][i]))
        ok,bgr=cap.read();assert ok,(cid,i)
        rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        p=Image.new("RGB",(600,530),"#101925");d=ImageDraw.Draw(p)
        d.text((8,4),f'#{s["item_id"]} {s["kind"]} {r["label"]} / 사람 {r["signer_id"]} / f{z["frame_id"][i]}',font=font,fill="white")
        # First full raw image, no skeleton to bias the visual observation.
        full=rgb.copy();full.thumbnail((592,333));p.paste(full,((600-full.width)//2,30))
        for h,name in enumerate(("L","R")):
            xy=z["xy_px"][i,33+21*h:54+21*h];valid=z["usable"][i,33+21*h:54+21*h]&np.isfinite(xy).all(-1)
            pts=xy[valid]
            sx=8+296*h
            if len(pts):
                lo=pts.min(0);hi=pts.max(0);pad=max(25,float((hi-lo).max())*.2)
                box=(max(0,int(lo[0]-pad)),max(0,int(lo[1]-pad)),
                     min(rgb.width,int(hi[0]+pad)),min(rgb.height,int(hi[1]+pad)))
                detail=rgb.crop(box);detail.thumbnail((150,132));p.paste(detail,(sx,375))
            st=z["angle_state"][i,h]
            d.text((sx+155,377),f'{name} 15각도\n비교 {(st==4).sum()}\n미확인 {(st==2).sum()}\n상충 {(st==3).sum()}\n누락 {(st<2).sum()}',font=small,fill="#b9cee3")
        panels[s["item_id"]]=p
    cap.release()
groups=defaultdict(list)
for s in items:groups[s["kind"]].append(s)
pages=[]
for kind,ss in groups.items():
    for start in range(0,len(ss),9):
        chunk=ss[start:start+9];sheet=Image.new("RGB",(1800,530*((len(chunk)+2)//3)),"#101925")
        for n,s in enumerate(chunk):sheet.paste(panels[s["item_id"]],((n%3)*600,(n//3)*530))
        path=out/f"{kind}_{start//9+1:02}.jpg";sheet.save(path,quality=88)
        pages.append(dict(path=str(path),kind=kind,items=[s["item_id"] for s in chunk]))
write(U/"VISUAL_MANIFEST.json",dict(items=items,pages=pages))
print("VISUAL",len(items),"panels",len(pages),"pages", {k:len(v) for k,v in groups.items()})
