"""Diagnose disagreement with independent math and original public frames; no detector."""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

U=Path(__file__).resolve().parent
P=U.parent/"word100_fullframe_hand_merge_20261020"
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
f=read(P/"FROZEN.json"); rows={r["clip_id"]:r for r in f["rows"]}
queue=read(U/"ANGLE_REVIEW_QUEUE.json")
chains=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
triples=[(a,b,c) for ch in chains for a,b,c in zip(ch,ch[1:],ch[2:])]
font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",20)
small=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",17)
evidence=[]
for rank,q in enumerate(queue[:12]):
    cid=q["clip_id"];r=rows[cid]
    e=read(P/"records"/(cid+".json"))["frames"][q["index"]]
    d=next(x for x in e["decisions"] if x["accepted"] and x["side"]==q["side"])
    a=np.array(e["full"][d["index"]]["xy"]);b=np.array(e["crop"][d["crop_index"]]["xy"])
    def angles(p):
        v=p[[x[0] for x in triples]]-p[[x[1] for x in triples]]
        w=p[[x[2] for x in triples]]-p[[x[1] for x in triples]]
        return np.degrees(np.arctan2(abs(v[:,0]*w[:,1]-v[:,1]*w[:,0]),(v*w).sum(1)))
    da=angles(a);db=angles(b)
    with np.load(U/"evidence"/(cid+".npz"),allow_pickle=False) as z:
        allowed=np.isfinite(z["two_scale_angle_difference_deg"][q["index"],q["side"]])
    delta=np.where(allowed,abs(da-db),-1)
    k=int(np.argmax(delta));ids=list(triples[k])
    assert abs(float(delta[k])-q["max_angle_difference_deg"])<.1
    cap=cv2.VideoCapture(r["video"]);assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES,q["frame_id"])
    ok,bgr=cap.read();cap.release();assert ok
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
    points=np.concatenate([a,b])
    lo=np.maximum(0,np.floor(points.min(0)-35).astype(int))
    hi=np.minimum([w,h],np.ceil(points.max(0)+35).astype(int))
    patch=rgb[lo[1]:hi[1],lo[0]:hi[0]]
    factor=min(375/patch.shape[1],245/patch.shape[0])
    sz=(int(patch.shape[1]*factor),int(patch.shape[0]*factor))
    tile=Image.new("RGB",(1200,360),"#f6f8fc");draw=ImageDraw.Draw(tile)
    side="왼손" if q["side"]==0 else "오른손"
    draw.text((16,8),f"{rank+1:02d} {q['label']} · {cid} · f{q['frame_id']} · {side}",font=font,fill="#142034")
    im=Image.fromarray(rgb).resize((400,225))
    tile.paste(im,(10,70));draw.rectangle((10+lo[0]*400/w,70+lo[1]*225/h,10+hi[0]*400/w,70+hi[1]*225/h),outline="#ffcb00",width=2)
    raw=Image.fromarray(patch).resize(sz)
    tile.paste(raw,(415,70));tile.paste(raw,(810,70))
    draw.text((415,42),"원본 손 주변",font=small,fill="#142034")
    draw.text((810,42),"초록=전체 재검출 / 자홍=부분 검사",font=small,fill="#142034")
    for p,color in [(a,"#00ee65"),(b,"#ff35da")]:
        pp=(p-lo)*factor+[810,70]
        for chain in chains:
            draw.line([tuple(pp[i]) for i in chain],fill=color,width=2)
        for j in ids:
            x,y=pp[j];draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
    draw.text((16,316),f"각도 {ids}: 전체 {da[k]:.1f}° / 부분 {db[k]:.1f}° · 차이 {delta[k]:.1f}°",font=font,fill="#9b2948")
    draw.text((16,342),"서로 다른 사람의 차이가 아닌, 같은 순간을 두 방식으로 검사한 결과의 불일치",font=small,fill="#4f6176")
    tile.save(U/"evidence"/f"angle_{rank+1:02d}.jpg",quality=93)
    evidence.append(dict(**q,angle_index=k,joints=ids,full_angle_deg=float(da[k]),crop_angle_deg=float(db[k]),
                         independent_difference_deg=float(delta[k]),
                         joint_displacements_px=np.linalg.norm(a[ids]-b[ids],axis=1).tolist(),
                         full_bone_lengths_px=[float(np.linalg.norm(a[ids[x]]-a[ids[1]])) for x in (0,2)],
                         crop_bone_lengths_px=[float(np.linalg.norm(b[ids[x]]-b[ids[1]])) for x in (0,2)],
                         image=str(U/"evidence"/f"angle_{rank+1:02d}.jpg")))
for page in range(4):
    sheet=Image.new("RGB",(1200,1080),"white")
    for j in range(3):
        sheet.paste(Image.open(U/"evidence"/f"angle_{page*3+j+1:02d}.jpg"),(0,j*360))
    sheet.save(U/"evidence"/f"angle_sheet_{page+1}.jpg",quality=93)
with (U/"ANGLE_GEOMETRY_AUDIT.json").open("x",encoding="utf8") as fp:
    json.dump(dict(method="Independent atan2 geometry plus raw public source frames; no inference or automatic correctness label.",
                   targets=evidence,review_status="PENDING_VISUAL_REVIEW"),fp,ensure_ascii=False,indent=2)
print("RENDERED12, SHEETS4")
