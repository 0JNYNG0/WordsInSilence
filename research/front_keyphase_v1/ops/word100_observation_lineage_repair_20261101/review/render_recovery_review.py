"""Render fixed-source review pages AFTER root authorizes completed-output review.

No detector imports or inference. Uses only already-reviewed exact native JPEGs.
All executed fixed frames are retained, including no-addition/conflict frames.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;U=HERE.parent;OPS=U.parent
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
       (9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
small=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
crop=(350,100,1570,1080)
def paint_hand(im,xy,color,valid=None):
    d=ImageDraw.Draw(im);xy=np.asarray(xy);valid=np.isfinite(xy).all(1) if valid is None else np.asarray(valid)&np.isfinite(xy).all(1)
    for a,b in EDGES:
        if valid[a] and valid[b]:d.line([tuple(xy[a]),tuple(xy[b])],fill=color,width=4)
    for i,p in enumerate(xy):
        if valid[i]:
            x,y=p;d.ellipse((x-4,y-4,x+4,y+4),fill=color)

def main():
    target=HERE/"RECOVERY_RGB_MANIFEST.json"
    assert not target.exists(),"Preserve first rendered review output."
    rm=read(U/"RECOVERY_RESULT_MANIFEST.json");prep=read(U/"RECOVERY_PREPARED.json")
    original=read(OPS/"word100_action_feedback_casebook_20261101/review/RGB_MANIFEST.json")
    source={(f["alias"],f["frame_id"]):f for f in original["full_frames"]}
    selection={(p["alias"],p["frame_id"]) for p in prep["opportunities"]}
    assert len(selection)==303
    pages=[];pins={str(U/"RECOVERY_RESULT_MANIFEST.json"):sha(U/"RECOVERY_RESULT_MANIFEST.json"),
                  str(OPS/"word100_action_feedback_casebook_20261101/review/RGB_MANIFEST.json"):sha(OPS/"word100_action_feedback_casebook_20261101/review/RGB_MANIFEST.json")}
    for rec in rm["records"]:
        assert sha(rec["output"])==rec["output_sha256"] and sha(rec["ledger"])==rec["ledger_sha256"]
        ledger=read(rec["ledger"]);z=np.load(rec["output"]);base=np.load(ledger["source"]["files"]["native"])
        for fr in ledger["frames"]:
            if fr["status"]!="NEW_IMAGE_CANDIDATE_EXECUTED":continue
            key=(rec["alias"],fr["frame_id"]);assert key in selection and key in source
            sf=source[key];assert sf["timestamp_us"]==fr["timestamp_us"] and sha(sf["image"])==sf["image_sha256"]
            i=int(np.flatnonzero(z["frame_id"]==fr["frame_id"])[0]);j=fr["source_index"]
            im=Image.open(sf["image"]).convert("RGB");assert im.size==(1920,1080)
            panels=[im.copy(),im.copy(),im.copy()]
            # Middle panel: exact retained original + newly accepted native slots.
            for side,off in [(0,33),(1,54)]:
                old=np.asarray(base["observed"][j,off:off+21])
                paint_hand(panels[1],base["xy_px"][j,off:off+21],"#4fff52" if side==0 else "#41dfff",old)
                added=z["native_observed"][i,off:off+21]&~old
                paint_hand(panels[1],z["native_xy_px"][i,off:off+21],"#ffad32" if side==0 else "#ff42c7",added)
            # Right panel: all attempted full/crop candidates, including rejected.
            for d in fr["full"]:paint_hand(panels[2],d["xy"],"#b163ff")
            for d in fr["crop"]:paint_hand(panels[2],d["xy"],"#ff5b4a")
            for b in fr.get("boxes",{}).values():ImageDraw.Draw(panels[2]).rectangle(b,outline="#ffdc50",width=3)
            canvas=Image.new("RGB",(1800,670),"#20252b");draw=ImageDraw.Draw(canvas)
            draw.text((12,8),f'{key[0]} f{key[1]} | {fr["timestamp_us"]/1e6:.6f}s | native source, fixed review scope',font=font,fill="white")
            for col,title in enumerate(["Original RGB","Existing L green / R cyan; added L orange / R pink","All full purple / crop red candidates"]):
                thumb=panels[col].crop(crop);thumb.thumbnail((594,480));canvas.paste(thumb,(col*600+(600-thumb.width)//2,70))
                draw.text((col*600+6,42),title,font=small,fill="white")
            descriptions=[f'full#{d["index"]} side={d.get("side")} applied={d.get("applied",False)}: {d["reason"]}' for d in fr["decisions"]]
            draw.text((10,560)," | ".join(descriptions)[:210],font=small,fill="white")
            draw.text((10,590),f'additions={len(fr["additions"])}; overlay is stored estimate, NOT landmark or educational ground truth',font=small,fill="white")
            draw.text((10,620),"Original RGB and complete candidate/rejection ledger are preserved. New visual review has not yet occurred.",font=small,fill="white")
            dest=HERE/"recovery_rgb"/f"{key[0]}_f{key[1]:04}.jpg";dest.parent.mkdir(exist_ok=True)
            assert not dest.exists();canvas.save(dest,quality=95)
            pages.append({"alias":key[0],"frame_id":key[1],"timestamp_us":fr["timestamp_us"],"file":str(dest),"sha256":sha(dest),
                "original_rgb":sf,"ledger":rec["ledger"],"ledger_sha256":rec["ledger_sha256"],
                "additions":fr["additions"],"decisions":fr["decisions"],"display_crop":list(crop),
                "selection_reason":"ALL executed fixed frames, not selected by additions or apparent correctness."})
            pins[sf["image"]]=sf["image_sha256"]
    assert len(pages)==rm["executed_frames"]
    result={"status":"RENDERED_NOT_YET_VISUALLY_REVIEWED","pages":pages,"source_pins":pins,"script_sha256":sha(__file__),
       "scope":"All executed frames within prior303 only; original images already reviewed in casebook.",
       "actions":{"model_inference":0,"rgb_decode":0,"new_alignment":0,"new_image_scope":0}}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"pages":len(pages),"path":str(target),"sha256":sha(target)}))
if __name__=="__main__":main()
