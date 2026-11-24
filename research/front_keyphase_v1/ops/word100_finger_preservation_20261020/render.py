"""Public raw panels and time-aligned feature-loss plots for review."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",20)
small=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",17)
f=read(U/"FROZEN.json");a=read(U/"ANALYSIS.json")
rows={r["clip_id"]:r for r in f["rows"]}
(U/"visual").mkdir()
def source(cid,index):
    out=U/"outputs"/cid
    d=read(out/"detections.json");z=load(out/"merged_native.npz")
    path=d["video"];fid=int(z["frame_id"][index])
    cap=cv2.VideoCapture(path);assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,frame=cap.read();cap.release();assert ok
    return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),z,fid
def hand_patch(im,z,i,s):
    p=z["xy_px"][i,33+s*21:54+s*21];v=z["observed"][i,33+s*21:54+s*21]
    pts=p[v]
    if not len(pts):
        wrist=z["xy_px"][i,15+s]
        if not np.isfinite(wrist).all():return None
        pts=np.array([wrist+[-60,-60],wrist+[60,60]])
    lo=np.maximum(0,np.floor(pts.min(0)-30).astype(int))
    hi=np.minimum(im.size,np.ceil(pts.max(0)+30).astype(int))
    if min(hi-lo)<=0:return None
    return im.crop((*lo,*hi))
def paste_fit(tile,im,box):
    if im is None:return
    x,y,w,h=box;ratio=min(w/im.width,h/im.height)
    size=(max(1,int(im.width*ratio)),max(1,int(im.height*ratio)))
    tile.paste(im.resize(size),(x+(w-size[0])//2,y+(h-size[1])//2))
def write(p,v):
    with p.open("x",encoding="utf8") as fp:json.dump(v,fp,ensure_ascii=False,indent=2)

controls=[]
for n,q in enumerate(a["normal_controls"],1):
    cid=q["clip_id"];i=q["index"];r=rows[cid]
    im,z,fid=source(cid,i)
    tile=Image.new("RGB",(1200,330),"#f5f8fc");d=ImageDraw.Draw(tile)
    d.text((12,6),f"{n:02d} {r['label']} · {cid} · f{fid}  |  표시 없이 원본부터 확인",font=font,fill="#172b42")
    paste_fit(tile,im,(10,55,390,220))
    d.text((445,39),"왼손 주변 원본",font=small,fill="#43576b")
    d.text((845,39),"오른손 주변 원본",font=small,fill="#43576b")
    for s in range(2):paste_fit(tile,hand_patch(im,z,i,s),(410+395*s,65,380,220))
    d.text((12,293),"100단어 중 일정 간격으로 선택한 20개 핵심 구간 중간 장면 · 보류 여부를 보고 고르지 않음",font=small,fill="#50667b")
    path=U/"visual"/f"normal_{n:02d}.jpg";tile.save(path,quality=94)
    controls.append(dict(**q,frame_id=fid,image=str(path)))
normal_sheets=[]
for start in range(0,len(controls),4):
    sheet=Image.new("RGB",(1200,330*min(4,len(controls)-start)),"white")
    for j,q in enumerate(controls[start:start+4]):sheet.paste(Image.open(q["image"]),(0,j*330))
    path=U/"visual"/f"normal_sheet_{start//4+1:02d}.jpg";sheet.save(path,quality=94);normal_sheets.append(str(path))

def loss_panel(q,n,group):
    cid=q["clip_id"];r=rows[cid];out=U/"outputs"/cid
    x=load(out/"inputs.npz");loss=load(out/"loss_audit.npz")
    s=q["side"];start=q["start_index"];end=q["end_index"];middle=(start+end)//2
    indices=[max(0,start-1),middle,min(len(x["frame_id"])-1,end+1)]
    tile=Image.new("RGB",(1200,595),"#f5f8fc");d=ImageDraw.Draw(tile)
    d.text((12,5),f"{n:02d} {r['label']} · {cid} · {'왼손' if s==0 else '오른손'}",font=font,fill="#172b42")
    d.text((12,34),f"{group}  |  주황 시간대: 원래 계산 가능하던 각도가 보류된 구간",font=small,fill="#96570b")
    for k,i in enumerate(indices):
        im,z,fid=source(cid,i)
        paste_fit(tile,im,(12+395*k,70,140,92))
        paste_fit(tile,hand_patch(im,z,i,s),(12+395*k,170,375,215))
        before=int(loss["angle_valid_merged"][i,s].sum());after=int(loss["angle_valid_safe"][i,s].sum())
        d.text((165+395*k,85),f"f{fid}\n각도 {before} → {after}",font=small,fill="#172b42")
    t=x["timestamp_us"]/1e6
    core=loss["core"];ov=loss["angle_valid_before"][:,s].sum(-1)
    av=loss["angle_valid_merged"][:,s].sum(-1);bv=loss["angle_valid_safe"][:,s].sum(-1)
    x0,y0,w,h=55,430,1090,110
    def xx(i):return x0+w*(t[i]-t[0])/max(1e-9,t[-1]-t[0])
    d.rectangle((x0,y0,x0+w,y0+h),fill="white",outline="#d1dbe7")
    core_ids=np.flatnonzero(core)
    if len(core_ids):d.rectangle((xx(core_ids[0]),y0,xx(core_ids[-1]),y0+h),fill="#edf4fe")
    d.rectangle((xx(start),y0,max(xx(end),xx(start)+2),y0+h),fill="#ffe3bc")
    for value,col in [(av,"#2366b3"),(bv,"#e28220"),(ov,"#a0aab5")]:
        points=[(xx(i),y0+h-h*int(value[i])/15) for i in range(len(t))]
        d.line(points,fill=col,width=3)
    d.text((8,y0),"15",font=small,fill="#53647b");d.text((25,y0+h-15),"0",font=small,fill="#53647b")
    d.text((55,393),"굽힘 각도 수 / 회색: 기존 · 파랑: 새 손 통합 · 주황: 보류 후 · 연파랑 배경: 주석 핵심 구간",font=small,fill="#53647b")
    d.text((55,552),f"시간 {t[0]:.2f}초 → {t[-1]:.2f}초 · 누락은 관측/모양 근거의 감소이며 사용자 동작 오류 판정이 아님",font=small,fill="#53647b")
    prefix="loss" if group.startswith("긴") else ("allshape" if group.startswith("한 손") else "motion")
    path=U/"visual"/f"{prefix}_{n:02d}.jpg";tile.save(path,quality=94)
    return dict(**q,image=str(path),indices=indices)

loss_targets=[loss_panel(q,n,"긴 핵심구간 정보 감소") for n,q in enumerate(a["top_loss_events"],1)]
motion_targets=[]
for n,r in enumerate(a["top_removed_variation"],1):
    q=r["largest_removed_transition"];i=q["index"]
    motion_targets.append(loss_panel(dict(clip_id=r["clip_id"],label=r["label"],side=q["side"],
        start_index=i-1,end_index=i,angle_index=q["angle_index"],delta_deg=q["delta_deg"]),n,"관측 각도 변화가 크게 줄어든 순간"))
allshape_targets=[]
for r in f["rows"]:
    z=load(U/"outputs"/r["clip_id"]/"loss_audit.npz")
    for i,s in np.argwhere(z["all_angles_lost"]&z["core"][:,None]):
        allshape_targets.append(loss_panel(dict(clip_id=r["clip_id"],label=r["label"],side=int(s),
            start_index=int(i),end_index=int(i)),len(allshape_targets)+1,"한 손의 계산 가능 굽힘 전체 보류"))
sheets={}
for kind,targets in (("loss",loss_targets),("motion",motion_targets),("allshape",allshape_targets)):
    pages=[]
    for start in range(0,len(targets),2):
        sheet=Image.new("RGB",(1200,595*min(2,len(targets)-start)),"white")
        for j,q in enumerate(targets[start:start+2]):sheet.paste(Image.open(q["image"]),(0,j*595))
        path=U/"visual"/f"{kind}_sheet_{start//2+1:02d}.jpg";sheet.save(path,quality=94);pages.append(str(path))
    sheets[kind]=pages
write(U/"VISUAL_TARGETS.json",dict(normal_controls=controls,normal_sheets=normal_sheets,
    loss_targets=loss_targets,motion_targets=motion_targets,allshape_targets=allshape_targets,sheets=sheets,
    note="Raw controls first, then loss plots; not independent manual ground-truth annotation."))
print("NORMAL",len(controls),"LOSS",len(loss_targets),"MOTION",len(motion_targets),"ALL_SHAPE",len(allshape_targets))
