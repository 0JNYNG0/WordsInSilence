"""Actual saved public results only; no private/video inference or invented scores."""
from unit import *
import cv2,subprocess,imageio_ffmpeg
def main():
 assert not (U/'VIEW_MANIFEST.json').exists();entries=[r for r in read(U/'INPUTS.json')['rows'] if r['role']=='development' and r['kind'] in ['full_normal','omit_final_event','mask_final_event']]
 sources={r['clip_id']:r for r in read(D/'PREPARED.json')['rows']};reviews={r['clip_id']:r for r in read(S/'REVIEW_INPUTS.json')['rows']};annotations={r['clip_id']:r for r in read(S/'ANNOTATIONS.json')['rows']}
 labels=read(D/'FROZEN.json')['labels'];(U/'media').mkdir(exist_ok=True);data=[];media={};ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
 for cid in sorted({r['clip_id'] for r in entries}):
  meta=sources[cid];assert sha(meta['source'])==meta['source_sha256'];cap=cv2.VideoCapture(meta['source']);frames=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   frames.append(im)
  cap.release();roi=reviews[cid]['crop'];l,t,rr,b=roi;width=480;height=round((b-t)/(rr-l)*width/2)*2
  original=np.load(read(J/'records'/f'{cid}.json')['raw_path']);px=original['xy_px'];origin=(px[:,5]+px[:,6])/2;scale=float(np.median(np.linalg.norm(px[:,5]-px[:,6],axis=1)))
  for e in [r for r in entries if r['clip_id']==cid]:
   z=np.load(e['path']);ids=z['frame_id'];kind=e['kind'];vidkey=f'{cid}_{"full_normal" if kind=="mask_final_event" else kind}'
   if vidkey not in media:
    path=U/'media'/f'{vidkey}.mp4';proc=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{width}x{height}','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
    for frame in ids:proc.stdin.write(cv2.resize(frames[int(frame)][t:b,l:rr],(width,height)).tobytes())
    proc.stdin.close();error=proc.stderr.read().decode();code=proc.wait();assert code==0,error;media[vidkey]=dict(path=str(path),sha256=sha(path),frames=len(ids),width=width,height=height)
   decisions={name:read(U/'decisions'/name/f'{cid}_{kind}.json') for name in ['final_normal_only_42','final_attempt_supervised_42']}
   points=z['raw'][:,:102].reshape(-1,51,2)*scale+origin[ids,None,:];points=(points-np.array([l,t]))/np.array([rr-l,b-t]);valid=z['raw'][:,186:237]>.5;points[~valid]=-1
   pred=np.load(U/'predictions/final_attempt_supervised_42'/f'{cid}_{kind}.npz')['phase'];reference=[]
   for c in decisions['final_attempt_supervised_42']['cards']:
    fs=[f for group in c['groups'] for g in group for f in annotations[cid]['stages'].get(str(g),[])];reference.append([min(fs),max(fs)] if fs else None)
   data.append(dict(id=f'{cid}_{kind}',clip=cid,label=labels[e['word_id']],person=e['person'],kind=kind,video=f'media/{vidkey}.mp4',reference=f'media/{cid}_full_normal.mp4',source_indices=z['source_indices'].tolist(),points=np.round(points,5).tolist(),phase=pred.argmax(1).tolist(),decisions=decisions,reference_spans=reference,cut=e['cut']))
 write(U/'VIEW_DATA.json',dict(rows=data,media=media));js='window.VIEW_DATA='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';';(U/'view_data.js').write_text(js,encoding='utf-8')
 write(U/'VIEW_MANIFEST.json',dict(cases=len(data),clips=12,media=media,input_sha256=sha(U/'INPUTS.json'),data_sha256=sha(U/'VIEW_DATA.json'),render='Original public RGB; only the omission video is cut according to the fixed intervention. Mask test changes joint input only. No generated image or false visual measurement.',alignment='Known source frame correspondence used for reviewing the controlled video cut, not DTW or neural correctness evidence. No map is fed into the model.',web_model_changed=False))
 event('view_prepared',cases=len(data),videos=len(media))
if __name__=='__main__':main()
