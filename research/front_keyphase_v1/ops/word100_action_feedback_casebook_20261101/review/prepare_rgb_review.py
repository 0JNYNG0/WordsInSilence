"""Decode only acknowledged public RGB windows; no detector or model imports."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,traceback
import av
from PIL import Image,ImageDraw,ImageFont

R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def main():
 selection=read(R/"WINDOW_SELECTION.json");ack=read(R/"WINDOW_ACK.json")
 assert sha(R/"WINDOW_SELECTION.json")==ack["selection_sha256"]
 assert ack["status"]=="ROOT_ACK_FIXED10_BEFORE_RGB_REVIEW"
 wanted=defaultdict(dict);sources={};existing={}
 for w in selection["windows"]:
  for side in ("query_axis","reference_axis"):
   a=w[side];alias=a["alias"];sources[alias]=a
   for fid,pts in zip(a["context_frame_ids"],a["context_timestamp_us"]):
    assert fid not in wanted[alias] or wanted[alias][fid]==pts
    wanted[alias][fid]=pts
   for p in a["existing_full_rgb_samples"]:existing[alias,p["frame_id"]]=p
 frames={};pages=[]
 font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
 dest=R/"rgb_review";dest.mkdir(exist_ok=True)
 # Uniform body-view presentation; full decoded originals retained and linked.
 crop=(350,100,1570,1080)
 for alias in sorted(wanted):
  a=sources[alias];source=Path(a["source"]);assert sha(source)==a["source_sha256"]
  folder=dest/alias;folder.mkdir(exist_ok=True)
  needed=set(wanted[alias]);created=0;reused=0
  with av.open(str(source)) as con:
   for fid,frame in enumerate(con.decode(video=0)):
    if fid in needed:
     expected=wanted[alias][fid]
     actual=round(float(frame.pts*frame.time_base)*1e6)
     assert abs(actual-expected)<=1,(alias,fid,actual,expected)
     prior=existing.get((alias,fid))
     if prior:
      p=Path(prior["file"]);assert sha(p)==prior["sha256"];reused+=1;mode="EXACT_EXISTING_FULL_RGB_JPEG_REUSED"
     else:
      p=folder/f"f{fid:04d}.jpg"
      assert not p.exists()
      frame.to_image().save(p,quality=94);created+=1;mode="NEW_LOCAL_RGB_DECODE_NO_FEATURE_EXTRACTION"
     frames[alias,fid]=dict(alias=alias,frame_id=fid,timestamp_us=expected,decoded_timestamp_us=actual,
       image=str(p),image_sha256=sha(p),source=str(source),source_sha256=a["source_sha256"],mode=mode)
    if fid>=max(needed):break
  assert all((alias,fid) in frames for fid in needed)
  ids=sorted(needed)
  for start in range(0,len(ids),12):
   batch=ids[start:start+12];thumbs=[]
   for fid in batch:
    im=Image.open(frames[alias,fid]["image"]).convert("RGB")
    im=im.crop(crop)
    im.thumbnail((400,322))
    thumbs.append((fid,im))
   sheet=Image.new("RGB",(1600,3*356+50),(24,27,32));draw=ImageDraw.Draw(sheet)
   draw.text((12,8),f"{alias} original chronological RGB | frame {batch[0]}..{batch[-1]} | gaps retain original numbers",fill="white",font=font)
   for i,(fid,im) in enumerate(thumbs):
    x=(i%4)*400;y=50+(i//4)*356
    draw.text((x+5,y),f"f{fid}  {wanted[alias][fid]/1e6:.3f}s",fill="white",font=font)
    sheet.paste(im,(x+(400-im.width)//2,y+30))
   p=folder/f"chronological_{start//12+1:02}.jpg";assert not p.exists();sheet.save(p,quality=94)
   pages.append(dict(alias=alias,file=str(p),sha256=sha(p),frame_ids=batch,
       display_only_crop_pixels=list(crop),source_pixels_modified=False))
  print(alias,"frames",len(needed),"new",created,"reused",reused,flush=True)
 windows=[]
 for w in selection["windows"]:
  windows.append(dict(window_id=w["window_id"],axes={side:[frames[w[side]["alias"],fid] for fid in w[side]["context_frame_ids"]] for side in ("query_axis","reference_axis")}))
 report=dict(status="RGB_PREPARED_FOR_FIXED_WINDOW_REVIEW_NOT_YET_VISUALLY_REVIEWED",
  at_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(R/"WINDOW_SELECTION.json"),
  ack_sha256=sha(R/"WINDOW_ACK.json"),unique_frames=len(frames),window_frame_opportunities=sum(len(a) for w in windows for a in w["axes"].values()),
  pages=pages,windows=windows,full_frames=list(frames.values()),script_sha256=sha(__file__),
  interpretation="Contact sheets are chronological display derivatives only, with fixed body crop. Full RGB images are retained. No detector, model, rescore, label or alignment was run.",
  actions=dict(public_rgb_decoding_only=True,feature_extraction=0,model_inference=0,training=0,alignment=0,source_writes=0,private_access=0))
 with (R/"RGB_MANIFEST.json").open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print("manifest",len(frames),len(pages),sha(R/"RGB_MANIFEST.json"))

if __name__=="__main__":
 try:main()
 except Exception:
  p=R/"FIRST_RGB_PREPARATION_FAILURE.txt"
  if not p.exists():p.write_text(traceback.format_exc(),encoding="utf-8")
  raise
