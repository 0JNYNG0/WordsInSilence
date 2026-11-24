"""Display-only WebM proxies. Keep every source frame, full field of view and PTS."""
from pathlib import Path
from fractions import Fraction
import av,json,hashlib,time,numpy as np
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=json.loads((U/'AUTHORIZATION.json').read_text(encoding='utf-8-sig'))['rows']
 out=U/'media';out.mkdir(exist_ok=True);records=[]
 for r in rows:
  target=out/(r['alias']+'.webm');record=out/(r['alias']+'.json')
  if record.exists():
   old=json.loads(record.read_text(encoding='utf8'));assert old['output_sha256']==sha(target) and old['source_sha256']==sha(r['source']);records.append(old);continue
  assert not target.exists(),'Partial output preserved; inspect before recovery'
  assert sha(r['source'])==r['source_sha256']
  source_pts=[]
  with av.open(r['source']) as src,av.open(str(target),'w') as dst:
   st=src.streams.video[0];stream=dst.add_stream('libvpx-vp9',rate=30)
   stream.time_base=Fraction(1,90000);stream.codec_context.time_base=Fraction(1,90000)
   stream.width=960;stream.height=540;stream.pix_fmt='yuv420p';stream.options={'crf':'28','b':'0','deadline':'realtime','cpu-used':'8'}
   for f in src.decode(video=0):
    source_pts.append(float(f.pts*f.time_base));frame=f.reformat(width=960,height=540,format='yuv420p')
    frame.pts=round(float(f.pts*f.time_base)*90000);frame.time_base=Fraction(1,90000)
    for packet in stream.encode(frame):dst.mux(packet)
   for packet in stream.encode():dst.mux(packet)
  with av.open(str(target)) as check:
   target_pts=[float(f.pts*f.time_base) for f in check.decode(video=0)]
  assert len(source_pts)==len(target_pts)==r['frames']
  maxerr=max(abs(a-b) for a,b in zip(source_pts,target_pts));assert maxerr<=.0011
  data=dict(alias=r['alias'],source_sha256=r['source_sha256'],output_sha256=sha(target),frames=len(target_pts),source_pts_seconds=source_pts,display_pts_seconds=target_pts,max_pts_error_seconds=maxerr,full_field_of_view=True,width=960,height=540,cropped=False,interpolated=False,for_model_input=False)
  record.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');records.append(data)
  print(r['alias'],len(target_pts),'verified',flush=True)
 (U/'DISPLAY_MEDIA.json').write_text(json.dumps(dict(status='PASS',clips=records),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
