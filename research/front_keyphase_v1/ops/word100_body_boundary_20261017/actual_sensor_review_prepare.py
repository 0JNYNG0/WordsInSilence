"""Nine frozen TRAIN sensor-gap scenes; saved observations + original RGB only."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

sys.dont_write_bytecode = True
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

U = Path(__file__).resolve().parent
CASES = [
    ("NIA_SL_WORD0920_REAL01_F", "left", 49, 60),
    ("NIA_SL_WORD1545_REAL01_F", "left", 56, 68),
    ("NIA_SL_WORD2348_REAL02_F", "right", 48, 82),
    ("NIA_SL_WORD0185_REAL02_F", "right", 62, 72),
    ("NIA_SL_WORD0185_REAL05_F", "right", 59, 71),
    ("NIA_SL_WORD0795_REAL02_F", "right", 102, 112),
    ("NIA_SL_WORD0866_REAL03_F", "left", 53, 77),
    ("NIA_SL_WORD1139_REAL06_F", "left", 46, 57),
    ("NIA_SL_WORD1998_REAL05_F", "left", 73, 88),
]
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
         (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(n):
    return ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", n)


def render(rgb, obs, i, bottom, title, width=480):
    im = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0,0,0,0)); d = ImageDraw.Draw(overlay)
    w,h = im.size
    d.rectangle((0,bottom,w,h),fill=(255,30,30,35))
    d.line((0,bottom,w,bottom),fill=(255,220,20,255),width=5)
    if obs is not None:
        xy = obs["xy_px"][i]
        good = obs["observed"][i] & obs["usable"][i] & np.isfinite(xy).all(-1)
        for name,base,body,color in (("L",33,(11,13,15),(65,255,115,255)),("R",54,(12,14,16),(10,210,255,255))):
            for a,b in zip(body,body[1:]):
                if good[a] and good[b]:d.line(tuple(xy[a])+tuple(xy[b]),fill=color,width=4)
            for a,b in EDGES:
                a+=base;b+=base
                if good[a] and good[b]:d.line(tuple(xy[a])+tuple(xy[b]),fill=color,width=3)
            for p in range(base,base+21):
                if good[p]:
                    x,y=xy[p];d.ellipse((x-5,y-5,x+5,y+5),fill=color)
            if good[base]:
                x,y=xy[base];d.text((x+10,y-40),name,font=font(35),fill=color,stroke_width=2,stroke_fill=(0,0,0,255))
    im=Image.alpha_composite(im,overlay).convert("RGB")
    im=im.resize((width,round(h*width/w)),Image.Resampling.LANCZOS)
    d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,30),fill=(13,20,30))
    d.text((5,4),title,font=font(16),fill="white")
    return im


def main():
    started=time.perf_counter()
    out=U/"actual_sensor_review_MONTAGES.json"
    assert not out.exists(), "Existing evidence preserved"
    context=read(U/"PREP_CONTEXT.json")
    manifest={r["clip_id"]:r for r in context["ordered_rows"] if r["role"]=="train_candidate"}
    triage={r["clip_id"]:r for r in read(U/"actual_input_triage_RESULT.json")["rows"]}
    pins={str(U/name):sha(U/name) for name in ("PREP_CONTEXT.json","actual_input_triage_RESULT.json","actual_sensor_review_prepare.py","actual_sensor_review_FIRST_ERROR.json")}
    cases=[]
    for cid,side,start,end in CASES:
        row=manifest[cid]
        rp=Path(context["output_root"])/"records"/(cid+".json");rec=read(rp)
        assert rec["role"]=="train_candidate" and 1<=rec["signer_id"]<=9
        assert sha(rp)==triage[cid]["record_sha256"]
        pins[str(rp)]=sha(rp)
        observations={}
        for mode in ("native","masked","sensor"):
            op=Path(rec["views"][mode]["observation"])
            assert sha(op)==rec["outputs"][str(op)]
            pins[str(op)]=sha(op)
            with np.load(op,allow_pickle=False) as z:
                observations[mode]={k:z[k] for k in ("xy_px","observed","usable","timestamp_us","frame_id")}
        ts=observations["native"]["timestamp_us"];fid=observations["native"]["frame_id"]
        for mode in ("masked","sensor"):
            assert np.array_equal(ts,observations[mode]["timestamp_us"])
            assert np.array_equal(fid,observations[mode]["frame_id"])
        assert np.array_equal(fid,np.arange(len(fid)))
        source=rec["source"]
        retained=Path(row["retained_video"]) if row.get("retained_video") else None
        if retained is not None and retained.is_file():
            data=retained.read_bytes();method="existing_retained_source"
        else:
            with zipfile.ZipFile(source["archive"]) as archive:
                entry=archive.getinfo(source["member"])
                assert entry.CRC==source["crc32"] and entry.file_size==source["bytes"]
                data=archive.read(entry)
            method="exact_zip_member_in_memory_no_video_written"
        assert len(data)==source["bytes"] and hashlib.sha256(data).hexdigest()==source["sha256"]
        frames_needed=sorted(set([max(0,start-2),start,round((3*start+end)/4),round((start+end)/2),
                                  round((start+3*end)/4),end,min(len(fid)-1,end+2)]))
        frames={};pts={};absolute_pts={};origin_pts=None;origin_us=None
        with av.open(io.BytesIO(data)) as video:
            for index,frame in enumerate(video.decode(video=0)):
                assert frame.pts is not None
                if origin_pts is None:
                    origin_pts=frame.pts
                    origin_us=float(frame.pts*frame.time_base)*1e6
                if index in frames_needed:
                    frames[index]=frame.to_ndarray(format="rgb24")
                    pts[index]=round(float((frame.pts-origin_pts)*frame.time_base)*1e6)
                    absolute_pts[index]=round(float(frame.pts*frame.time_base)*1e6)
                if index>=max(frames_needed):break
        assert set(frames)==set(frames_needed)
        assert all(abs(pts[i]-int(ts[i]))<=1 for i in frames_needed)
        bottom=int(rec["calibration"]["box_source_pixels"][3])
        tilew,tileh=480,270;header=84;rowh=308
        sheet=Image.new("RGB",(4*tilew,header+rowh*len(frames_needed)),(12,19,29));d=ImageDraw.Draw(sheet)
        d.text((12,7),f'{row["canonical_label"]} / {cid} / {side} f{start}~{end} / 하단{bottom}px',font=font(27),fill="white")
        d.text((12,45),'원본 전체 | 전체검출 native | 같은점+범위 masked | 실제RGB잘림 sensor / 초록L·하늘R / 노랑 하단선',font=font(22),fill=(205,217,230))
        frame_records=[]
        for y_index,index in enumerate(frames_needed):
            counts={}
            for mode,obs in observations.items():
                good=obs["observed"][index]&obs["usable"][index]&np.isfinite(obs["xy_px"][index]).all(-1)
                counts[mode]={s:int(good[b:b+21].sum()) for s,b in (("left",33),("right",54))}
            y=header+rowh*y_index
            for col,mode in enumerate((None,"native","masked","sensor")):
                title=f'f{index} {ts[index]/1e6:.3f}s / '+("원본RGB" if mode is None else f'{mode} L{counts[mode]["left"]} R{counts[mode]["right"]}')
                sheet.paste(render(frames[index],None if mode is None else observations[mode],index,bottom,title),(col*tilew,y))
            d.text((7,y+tileh+5),"지적 구간 안" if start<=index<=end else "구간 전후 비교",font=font(18),fill=(226,230,237))
            frame_records.append({"frame_id":index,"timestamp_us":int(ts[index]),"source_relative_pts_us":pts[index],"source_absolute_pts_us":absolute_pts[index],"counts":counts})
        image=U/f"actual_sensor_review_{cid}.jpg"
        if image.exists():
            assert sha(image)==read(U/"actual_sensor_review_FIRST_ERROR.json")["existing_images"][str(image.relative_to(Path.cwd()))]
        else:
            sheet.save(image,quality=95,subsampling=0)
        cases.append({"clip_id":cid,"label":row["canonical_label"],"role":"train_candidate","side":side,
                      "flag_frame_range":[start,end],"core_seconds":row["intervals"][0],"bottom":bottom,
                      "source":source,"source_read_method":method,"source_hash_verified":True,"source_pts_verified":True,"source_pts_origin_us":origin_us,"pts_convention":"same first-decoded-frame-relative PTS as frozen video_source.extract",
                      "record":str(rp),"image":str(image),"image_sha256":sha(image),"frames":frame_records,
                      "all_required_view_files_present":all(rec["views"][v]["channel_file"] is not None for v in ("native","masked","sensor")),
                      "feature_info":{v:rec["views"][v]["feature_info"] for v in ("native","masked","sensor")},
                      "prior_visual_lineage":triage[cid]["review_lineage"],
                      "status":"MONTAGE_CREATED_PENDING_ACTUAL_VIEW"})
        print(json.dumps({"clip_id":cid,"image":str(image)},ensure_ascii=False),flush=True)
    assert all(sha(p)==h for p,h in pins.items())
    result={"status":"NINE_TRAIN_SENSOR_SCENES_PREPARED_PENDING_VISUAL_REVIEW","created_utc":datetime.now(timezone.utc).isoformat(),
            "cases":cases,"source_pins":pins,"new_detector":0,"classifier":0,"training":0,"development_arrays":0,
            "elapsed_seconds":time.perf_counter()-started,
            "preparation_note":"RGB decoded from existing bytes only. No video file copied and no observations recomputed.",
            "first_read_only_error":"Initial record lookup omitted data/ in path; file not found. Corrected to frozen PREP_CONTEXT.output_root; no source or result changed."}
    with out.open("x",encoding="utf-8") as target:json.dump(result,target,ensure_ascii=False,indent=2)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
