"""New public video pixels, reversed and re-extracted. No private video access."""
from pathlib import Path
from fractions import Fraction
import json,sys,time,hashlib,traceback
import av,numpy as np
from motion_model import M,P,V,read
from label_motion import save,sha
def main():
 if not (M/'REVERSE_MANIFEST.json').exists():
  rows=[]
  for r in read(P/'INPUT_MANIFEST.json')['rows']:
   if r['role']!='development':continue
   dest=M/'reversed_videos'/f"{r['clip_id']}.mp4";dest.parent.mkdir(exist_ok=True)
   assert not dest.exists(),'Preserve partial video; inspect before recovery'
   with av.open(r['source']) as c:frames=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
   assert len(frames)==r['frames'];indices=list(range(len(frames)-1,-1,-1))
   with av.open(str(dest),'w') as out:
    stream=out.add_stream('libx264',rate=30);stream.width=frames[0].shape[1];stream.height=frames[0].shape[0];stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
    for i,j in enumerate(indices):
     f=av.VideoFrame.from_ndarray(frames[j],format='rgb24');f.pts=i;f.time_base=Fraction(1,30)
     for packet in stream.encode(f):out.mux(packet)
    for packet in stream.encode():out.mux(packet)
   rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],source=r['source'],source_sha256=sha(r['source']),
    path=str(dest),sha256=sha(dest),source_frame_indices=indices,role='development_not_training',label_scope='reversed physical direction, not automatically wrong KSL'))
   print('made',r['clip_id'],flush=True)
  save('REVERSE_MANIFEST.json',dict(rows=rows,originals_unchanged=True))
 sys.path.insert(0,str(V));import demo_adapter
 # All writes remain in the NEW unit; original code/results are unchanged.
 demo_adapter.U=M;demo_adapter.warmup();rows=[]
 for r in read(M/'REVERSE_MANIFEST.json')['rows']:
  d=M/'reverse_extracted'/r['clip_id'];p=d/'RESULT.json'
  if p.exists():out=read(p)
  else:out=demo_adapter.analyze(r['path'],d,r['word_id'])
  rows.append(dict(clip_id=r['clip_id'],result=str(p),seconds=out['latency']['total_analysis_seconds'],frames=out['frames']))
  (M/'REVERSE_PROGRESS.json').write_text(json.dumps(dict(done=len(rows),total=12,last=rows[-1]),ensure_ascii=False),encoding='utf8');print('extracted',rows[-1],flush=True)
 if not (M/'REVERSE_EXTRACTION.json').exists():save('REVERSE_EXTRACTION.json',dict(rows=rows,new_video_pixels_reextracted=True,user_inputs=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (M/'REVERSE_FIRST_FAILURE.txt').exists():(M/'REVERSE_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
