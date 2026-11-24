import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import study as d
U=d.U;f=d.read(U/"FROZEN.json");rows={r["clip_id"]:r for r in f["rows"]}
oldrows={r["clip_id"]:r for r in d.read(d.P/"FROZEN.json")["rows"]}
font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",16)
small=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",13)
selected=[];seen=set()

def add(cid,i,h,why):
    key=(cid,int(i),int(h),why)
    if key not in seen:
        seen.add(key);selected.append(dict(clip_id=cid,index=int(i),side=int(h),reason=why))

for cid,r in rows.items():
    changes=d.read(U/"rechecked"/(cid+".json"))["changes"]
    if changes:
        q=d.load(r["quality"])
        if "core" in q:core=q["core"]
        else:
            ts=d.load(r["raw"])["timestamp_us"]
            core=np.zeros(len(ts),bool)
            for a,b in oldrows[cid]["intervals"]:core|=(ts>=a*1e6)&(ts<=b*1e6)
        first=next((c for c in changes if core[c["index"]]),changes[0])
        add(cid,first["index"],first["side"],"first new core corroboration")
    # Longest run and known identity-control clip: inspect endpoints and middle.
    if len(changes)>=30 or cid=="NIA_SL_WORD0651_REAL05_F":
        for h in (0,1):
            ids=np.array(sorted({c["index"] for c in changes if c["side"]==h}))
            if not len(ids):continue
            for group in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                for i in (group[0],group[len(group)//2],group[-1]):add(cid,i,h,"long/identity-sensitive run")
union={}
for p in (U/"schedules").glob("*__NIA*.npz"):
    rid=p.stem.split("__")[1];z=d.load(p)["rest"]
    union[rid]=union.get(rid,np.zeros_like(z))|z
for cid,rest in union.items():
    for h in (0,1):
        ids=np.flatnonzero(rest[:,h])
        if not len(ids):continue
        for group in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
            for i in (group[0],group[len(group)//2],group[-1]):add(cid,i,h,"reference-supported lowering")
for c in d.read(d.A/"VERIFY.json")["controls"]:
    cid=c["clip_id"];i=c["index"];z=d.load(U/"rechecked"/(cid+".npz"))
    if c["group"]=="known_identity_error" or not np.array_equal(z["angle_state"][i],z["before_state"][i]):
        add(cid,i,-1,"previous control: "+c["group"])
(U/"visuals").mkdir();panels=[]
for n,x in enumerate(selected):
    r=rows[x["clip_id"]];cid=r["clip_id"];i=x["index"]
    if "video" in r:v=Path(r["video"])
    elif cid in oldrows:v=Path(oldrows[cid]["video"])
    else:
        # All broader100 videos were retained during the previous preparation.
        w=d.O/"word100_finger_preservation_20261020"
        wr=next(z for z in d.read(w/"FROZEN.json")["rows"] if z["clip_id"]==cid)
        v=Path(wr["retained_video"]) if wr["retained_video"] else w/"videos"/(cid+".mp4")
    assert v.is_file(),v
    raw=d.load(r["raw"]);cap=cv2.VideoCapture(str(v));cap.set(cv2.CAP_PROP_POS_FRAMES,int(raw["frame_id"][i]))
    ok,bgr=cap.read();cap.release();assert ok
    h,w=bgr.shape[:2]
    im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((480,270))
    canvas=Image.new("RGB",(480,334),"#101c2a");canvas.paste(im,(0,0));dr=ImageDraw.Draw(canvas)
    obs=d.load(r["observation"]);z=d.load(U/"rechecked"/(cid+".npz"))
    dr.line((0,r["box"][3]*270/h,480,r["box"][3]*270/h),fill="#ffcd44",width=2)
    for hand,color in [(0,"#43e0e0"),(1,"#ff8493")]:
        a=obs["xy_px"][i,33+21*hand:54+21*hand];valid=obs["usable"][i,33+21*hand:54+21*hand]
        for p in a[valid]:
            xx,yy=p[0]*480/w,p[1]*270/h;dr.ellipse((xx-2,yy-2,xx+2,yy+2),fill=color)
    dr.text((8,273),f"{n+1:02d} {r['label']} {cid.split('_')[-2]} f{raw['frame_id'][i]} · {'L' if x['side']==0 else 'R' if x['side']==1 else 'both'}",font=font,fill="white")
    dr.text((8,298),x["reason"],font=small,fill="#b6cde7")
    x.update(video=str(v),frame_id=int(raw["frame_id"][i]),before_eligible=int((z["before_state"][i]==4).sum()),
             after_eligible=int((z["angle_state"][i]==4).sum()))
    panels.append(canvas)
pages=[]
for k in range(0,len(panels),10):
    sheet=Image.new("RGB",(960,334*5),"#0b1421")
    for j,p in enumerate(panels[k:k+10]):sheet.paste(p,((j%2)*480,(j//2)*334))
    path=U/"visuals"/f"review_{k//10+1:02}.jpg";sheet.save(path,quality=92);pages.append(str(path))
d.write(U/"VISUAL_MANIFEST.json",dict(panels=selected,pages=pages,scope="Raw full-source scenes with small existing-coordinate overlay; no model training or user images."))
print("VISUALS",len(selected),len(pages))
