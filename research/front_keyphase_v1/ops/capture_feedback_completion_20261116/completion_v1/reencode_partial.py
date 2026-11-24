from train_shape import *
import av,shutil
from fractions import Fraction
out=V/'web';source=read(out/'SOURCES.json')['normal']['video'];target=out/'partial_h264.mp4'
with av.open(source)as video,av.open(str(target),'w')as dest:
 stream=dest.add_stream('libx264',rate=30);stream.width=video.streams.video[0].width;stream.height=video.streams.video[0].height;stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2;stream.options={'preset':'fast','crf':'19'}
 for i,frame in enumerate(video.decode(video=0)):
  if i>=55:break
  frame.pts=i;frame.time_base=Fraction(1,30)
  for packet in stream.encode(frame):dest.mux(packet)
 for packet in stream.encode():dest.mux(packet)
with av.open(str(target))as video:assert sum(1 for _ in video.decode(video=0))==55
old=out/'partial.mp4';backup=out/'partial_mpeg4_original.mp4'
if not backup.exists():shutil.copy2(old,backup)
target.replace(old)
write(out/'MEDIA_REPAIR.json',dict(cause='Browser does not decode OpenCV mp4v on this setup',fix='H264 yuv420p, same first55 native public frames and 30fps timestamps',frames=55,model_changed=False))
