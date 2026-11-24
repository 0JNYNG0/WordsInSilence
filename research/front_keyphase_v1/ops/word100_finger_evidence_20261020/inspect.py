"""Render public source frames and saved diagnostic overlays; no detector."""
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

U=Path(__file__).resolve().parent;O=U.parent
P=O/"word100_fullframe_hand_merge_20261020";Q=O/"word100_merge_scale_input_20261020"
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
rows={r["clip_id"]:r for r in read(P/"FROZEN.json")["rows"]}
evidence={cid:load(U/"evidence"/(cid+".npz")) for cid in rows}
font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",20)
small=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",17)
chains=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
names=["엄지","검지","중지","약지","새끼"]
targets=[];seen=set()
def add(q,group):
    cid=q["clip_id"];i=int(q["index"]);s=int(q.get("side",0))
    key=(cid,i,s)
    if key in seen:return False
    seen.add(key)
    targets.append(dict(clip_id=cid,index=i,side=s,group=group));return True
for q in read(Q/"ANGLE_REVIEW_QUEUE.json")[:12]:add(q,"이전 각도 불일치 상위 12")
used={x["clip_id"] for x in targets};n=0
for q in read(U/"REVIEW_QUEUE.json"):
    if q["clip_id"] not in used:
        used.add(q["clip_id"]);add(q,"추가 표시 · 영상별 최대 불일치");n+=1
        if n>=10:break
normals=read(O/"word100_image_arm_connectivity_20261018/ANNOTATED.json")
cross=read(O/"word100_image_arm_connectivity_20261018/CROSSING.json")
control_stats=[]
for group,source in [("기존 정상",normals),("기존 교차",cross)]:
    displayed=set()
    for q in source:
        cid=q["clip_id"]
        if q["view"]!="masked" or cid not in rows:continue
        i=q["index"];e=evidence[cid]
        counts=e["position_usable"][i,33:].reshape(2,21).sum(-1)
        s=int(np.argmax(counts))
        control_stats.append(dict(group=group,clip_id=cid,index=i,
            uncertain_fingers=int(e["finger_uncertain"][i].sum()),
            paired_sides=int(e["pair_available"][i].sum()),
            new_shape_hold=int(e["shape_hold_new"][i].sum())))
        if group=="기존 정상" or (cid not in displayed and len(displayed)<4):
            add({**q,"side":s},group);displayed.add(cid)
for group in ("추가 관측 · 비교 부족","추가 관측 · 충돌 미발견"):
    n=0
    for cid,e in evidence.items():
        added=e["source_added"].any(-1)
        selected=added&(~e["pair_available"] if group.endswith("부족") else
                        (e["pair_available"]&~e["finger_uncertain"].any(-1)))
        hits=np.argwhere(selected)
        if len(hits):
            i,s=map(int,hits[0])
            if add(dict(clip_id=cid,index=i,side=s),group):n+=1
        if n>=3:break

rendered=[]
for k,q in enumerate(targets):
    cid=q["clip_id"];r=rows[cid];e=evidence[cid];i=q["index"];s=q["side"]
    frame_id=int(e["frame_id"][i])
    cap=cv2.VideoCapture(r["video"]);assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,frame=cap.read();cap.release();assert ok
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
    a=e["position_xy_px"][i,33+21*s:54+21*s]
    va=e["position_usable"][i,33+21*s:54+21*s]
    b=e["pair_b"][i,s];vb=e["pair_valid"][i,s]
    pts=np.concatenate((a[va],b[vb]));assert len(pts)
    lo=np.maximum(0,np.floor(pts.min(0)-30).astype(int))
    hi=np.minimum([w,h],np.ceil(pts.max(0)+30).astype(int))
    patch=rgb[lo[1]:hi[1],lo[0]:hi[0]]
    factor=min(370/patch.shape[1],240/patch.shape[0]);sz=(max(1,int(patch.shape[1]*factor)),max(1,int(patch.shape[0]*factor)))
    tile=Image.new("RGB",(1200,370),"#f7f9fc");draw=ImageDraw.Draw(tile)
    draw.text((14,8),f"{k+1:02d} {r['label']} · {cid} · f{frame_id} · {'왼손' if s==0 else '오른손'}",font=font,fill="#142034")
    draw.text((14,39),q["group"],font=small,fill="#53647b")
    tile.paste(Image.fromarray(rgb).resize((400,225)),(10,75))
    draw.rectangle((10+lo[0]*400/w,75+lo[1]*225/h,10+hi[0]*400/w,75+hi[1]*225/h),outline="#ffbc00",width=2)
    im=Image.fromarray(patch).resize(sz)
    tile.paste(im,(420,75));tile.paste(im,(820,75))
    draw.text((420,47),"원본 손 주변",font=small,fill="#142034")
    draw.text((820,47),"초록=원래 좌표 / 주황=모양 보류",font=small,fill="#142034")
    pp=(a-lo)*factor+[820,75]
    for ch in chains:
        for j,l in zip(ch,ch[1:]):
            if va[j] and va[l]:draw.line([tuple(pp[j]),tuple(pp[l])],fill="#20ef90",width=2)
    for j in np.flatnonzero(va):
        x,y=pp[j];color="#ff8c19" if e["shape_hold_new"][i,s,j] or e["point_uncertain"][i,s,j] else "#20ef90"
        draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
    fs=[names[j] for j in np.flatnonzero(e["finger_uncertain"][i,s])]
    message=" · ".join(fs) if fs else ("두 관측 비교 불가" if not e["pair_available"][i,s] else "검출 간 충돌 미발견")
    draw.text((14,316),f"손가락 근거: {message}",font=font,fill="#9c4b09")
    draw.text((14,344),"주황은 사용자 동작 오류가 아님. 위치 좌표는 그대로 보존하며 정밀 모양의 확정만 보류.",font=small,fill="#53647b")
    target=U/"evidence"/f"review_{k+1:02d}.jpg";tile.save(target,quality=94)
    rendered.append(dict(**q,frame_id=frame_id,label=r["label"],image=str(target),
        paired=bool(e["pair_available"][i,s]),uncertain_fingers=fs))
pages=[]
for start in range(0,len(rendered),3):
    sheet=Image.new("RGB",(1200,370*min(3,len(rendered)-start)),"white")
    for j,q in enumerate(rendered[start:start+3]):sheet.paste(Image.open(q["image"]),(0,j*370))
    path=U/"evidence"/f"sheet_{start//3+1:02d}.jpg";sheet.save(path,quality=93);pages.append(str(path))
with (U/"VISUAL_TARGETS.json").open("x",encoding="utf8") as fp:
    json.dump(dict(selection="Prior12, largest ratio per other clip up to10, saved normal/cross controls, 3unassessed and3clear examples",
                   targets=rendered,sheets=pages,controls=control_stats),fp,ensure_ascii=False,indent=2)
print(json.dumps(dict(targets=len(rendered),sheets=len(pages),
    groups=dict(Counter(q["group"] for q in rendered)),
    controls_with_shape_flags=sum(q["uncertain_fingers"]>0 for q in control_stats),
    controls_with_new_hold=sum(q["new_shape_hold"]>0 for q in control_stats)),ensure_ascii=False))
