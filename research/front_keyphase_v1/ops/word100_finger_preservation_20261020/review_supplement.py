"""Read-only audit of normal controls and exact transition endpoints."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
U = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}
def fit(canvas, im, box):
    x,y,w,h=box
    k=min(w/im.width,h/im.height)
    im=im.resize((max(1,int(im.width*k)),max(1,int(im.height*k))))
    canvas.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",18)
small=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",16)
targets=read(U/"VISUAL_TARGETS.json")
f=read(U/"FROZEN.json")
normal=[]
cross=[]
for r in f["rows"]:
    out=U/"outputs"/r["clip_id"]
    z=load(out/"merged.npz");e=load(out/"evidence.npz");x=load(out/"inputs.npz")
    xy=z["xy_px"][:,33:].reshape(-1,2,21,2)
    v=z["usable"][:,33:].reshape(-1,2,21)
    proxy=(xy[:,0,0,0]<xy[:,1,0,0])&v[:,0,0]&v[:,1,0]&e["core"]
    for i in np.flatnonzero(proxy):
        cross.append(dict(clip_id=r["clip_id"],label=r["label"],index=int(i),
            local_points_held=int((x["merged"][i,237:279]-x["safe"][i,237:279]).sum()),
            positions_preserved=bool(np.array_equal(x["merged"][i,:102],x["safe"][i,:102]))))
for q in targets["normal_controls"]:
    out=U/"outputs"/q["clip_id"];x=load(out/"inputs.npz");z=load(out/"loss_audit.npz")
    i=q["index"]
    normal.append(dict(clip_id=q["clip_id"],label=q["label"],index=i,
        local_points_held=int((x["merged"][i,237:279]-x["safe"][i,237:279]).sum()),
        angles_before=z["angle_valid_before"][i].sum(-1).tolist(),
        angles_merged=z["angle_valid_merged"][i].sum(-1).tolist(),
        angles_safe=z["angle_valid_safe"][i].sum(-1).tolist()))

def panel(q,indices,title):
    out=U/"outputs"/q["clip_id"];d=read(out/"detections.json")
    z=load(out/"merged_native.npz");loss=load(out/"loss_audit.npz")
    cap=cv2.VideoCapture(d["video"]);assert cap.isOpened()
    tile=Image.new("RGB",(1200,360),"#f5f8fc");draw=ImageDraw.Draw(tile)
    draw.text((10,5),title,font=font,fill="#15293e")
    for n,i in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(z["frame_id"][i]));ok,im=cap.read();assert ok
        im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        fit(tile,im,(10+n*395,40,145,100))
        sides=[q["side"]] if "side" in q else [0,1]
        pts=[]
        for s in sides:
            p=z["xy_px"][i,33+21*s:54+21*s];v=z["observed"][i,33+21*s:54+21*s]
            pts.extend(p[v])
        if pts:
            pts=np.array(pts);lo=np.maximum(0,np.floor(pts.min(0)-35)).astype(int)
            hi=np.minimum(im.size,np.ceil(pts.max(0)+35)).astype(int)
            if min(hi-lo)>0:fit(tile,im.crop((*lo,*hi)),(10+n*395,145,375,205))
        old=loss["angle_valid_before"][i,sides].sum()
        merged=loss["angle_valid_merged"][i,sides].sum()
        safe=loss["angle_valid_safe"][i,sides].sum()
        draw.text((165+n*395,53),f"f{i}\n기존 {old}\n통합 {merged} / 보류 후 {safe}",font=small,fill="#23445b")
    cap.release()
    return tile
panels=[]
for n,q in enumerate(targets["motion_targets"],1):
    indices=[q["start_index"],q["end_index"],q["end_index"]+1]
    tile=panel(q,indices,f"{n:02d} {q['label']} · 큰 각도 변화의 양끝과 다음 프레임")
    path=U/"visual"/f"motion_exact_{n:02d}.jpg";tile.save(path,quality=94)
    panels.append(dict(**q,exact_indices=indices,exact_image=str(path)))
for n in range(0,len(panels),2):
    sheet=Image.new("RGB",(1200,720),"white")
    for j,q in enumerate(panels[n:n+2]):sheet.paste(Image.open(q["exact_image"]),(0,j*360))
    sheet.save(U/"visual"/f"motion_exact_sheet_{n//2+1:02d}.jpg",quality=94)
for n,q in enumerate(cross,1):
    i=q["index"];indices=[max(0,i-1),i,i+1]
    path=U/"visual"/f"cross_proxy_{n:02d}.jpg"
    panel(q,indices,f"{q['label']} · 손목 X순서 역전 표본, 정상 교차의 자동 정답은 아님").save(path,quality=94)
    q["indices"]=indices;q["image"]=str(path)
result=dict(normal_controls=normal,crossing_proxy=cross,motion_exact=panels,
    normal_controls_with_hold=sum(q["local_points_held"]>0 for q in normal))
with (U/"REVIEW_SUPPLEMENT.json").open("x",encoding="utf8") as fp:
    json.dump(result,fp,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!="motion_exact"},ensure_ascii=False))
