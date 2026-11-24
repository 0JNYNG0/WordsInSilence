from pathlib import Path
import sys,json,hashlib,zipfile,time,traceback,os
from collections import Counter
import numpy as np
import av
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;OPS=U.parent
sys.path.insert(0,str(OPS/'word100_signrep_temporal_head_20261029'))
from prepare_inputs import compile_gpu,encode,read_observation,extract_features
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=U/n;p.parent.mkdir(exist_ok=True,parents=True)
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def event(**d):
 with (U/'EVENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(time=time.time(),**d))+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
def select():
 original=read(OPS/'word100_shared_feature_bundle_20261027/FROZEN.json')
 words=sorted({r['word_id'] for r in original['rows']})
 pool=read(OPS/'word100_body_boundary_20261017/PREP_CONTEXT.json')['ordered_rows']
 pool=[r for r in pool if r['role']=='train_candidate']
 novel=sorted({r['word_code'] for r in pool}-set(words))[:4]
 selected=[r for r in pool if (r['word_code'] in words and r['signer_id'] in [1,2,3,4,5,6]) or (r['word_code'] in novel and r['signer_id'] in [5,6])]
 selected=sorted(selected,key=lambda r:(r['word_code'],r['signer_id']))
 assert len(selected)==80 and len({r['clip_id'] for r in selected})==80
 rows=[]
 old_idx={r['clip_id']:r for r in read(OPS/'word100_signrep_temporal_head_20261029/INPUTS.json')['rows']}
 for i,r in enumerate(selected):
  p=OPS/'word100_body_boundary_20261017/data/records'/f"{r['clip_id']}.json";rec=read(p)
  role='fit' if r['signer_id'] in [1,2,3] else 'development' if r['signer_id']==4 else 'held_person' if r['word_code'] in words else 'held_expression_person'
  rows.append(dict(alias=f'P{i+1:03}',clip_id=r['clip_id'],word_id=r['word_code'],label=r['canonical_label'],person=r['signer_id'],role=role,intervals=r['intervals'],record=str(p),record_sha256=sha(p),archive_source=rec['source'],retained_video=r.get('retained_video'),box=rec['calibration']['box_source_pixels'],observation=rec['views']['native']['observation'],old_input=old_idx.get(r['clip_id'])))
 save('SELECTION.json',dict(status='FROZEN_BEFORE_NEW_IMAGE_LABELS_OR_SCORES',words=words,novel_words=novel,novel_selection='First four lexicographic public TRAIN word IDs outside original12, before image review; no outcome-based replacement',roles=dict(Counter(r['role'] for r in rows)),rows=rows,split_note='All videos previously in TCN TRAIN. IDs1/2/3 are fit for NEW observer; original36 now development-used fit, not independent validation. ID4 dev, IDs5/6 locked head test. History is not universally unseen.',scope='Use exact public source only; protected personal paths excluded'))
 return rows
def source_file(r):
 s=r['archive_source'];old=r['old_input']
 candidates=[old['source']] if old else []
 if r['retained_video']:candidates.append(r['retained_video'])
 for p in candidates:
  if Path(p).is_file() and sha(p)==s['sha256']:return Path(p),'existing_exact_cache'
 out=U/'videos'/f"{r['alias']}.mp4";out.parent.mkdir(exist_ok=True)
 if out.exists():assert sha(out)==s['sha256'];return out,'verified_new_cache'
 with zipfile.ZipFile(s['archive']) as z:
  zi=z.getinfo(s['member']);assert zi.CRC==s['crc32'] and zi.file_size==s['bytes']
  data=z.read(zi)
 assert hashlib.sha256(data).hexdigest()==s['sha256']
 out.write_bytes(data);return out,'extracted_from_existing_public_archive'
def images(r,source,obs):
 a,b=min(x[0] for x in r['intervals']),max(x[1] for x in r['intervals'])
 ts=obs['timestamp_us'];start=max(0,a-.12);end=min(ts[-1]/1e6,b+.12)
 wanted=np.linspace(start,end,13)*1e6;ids=np.unique([np.abs(ts-t).argmin() for t in wanted]);assert len(ids)==13
 selected={}
 with av.open(str(source)) as con:
  for i,f in enumerate(con.decode(video=0)):
   if i in ids:selected[i]=f.to_image()
   if i>=ids[-1]:break
 assert len(selected)==13
 xy=obs['xy_px'];mask=obs['usable']&np.isfinite(xy).all(-1)
 shoulder=xy[ids][:,[11,12]].reshape(-1,2);center=np.nanmedian(shoulder,axis=0)
 # Display context comes from body center and full image, never a fitted target hand.
 width,height=next(iter(selected.values())).size
 crop=(max(0,int(center[0]-620)),0,min(width,int(center[0]+620)),height)
 folder=U/'visuals'/r['alias'];folder.mkdir(parents=True,exist_ok=True)
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
 sheet=Image.new('RGB',(1600,1640),'#20252b');frames=[]
 for ordinal,fid in enumerate(ids):
  im=selected[int(fid)];impath=folder/f'f{fid:04}.jpg';im.save(impath,quality=94)
  cropim=im.crop(crop);cropim.thumbnail((392,370))
  x=(ordinal%4)*400;y=(ordinal//4)*410
  sheet.paste(cropim,(x+(400-cropim.width)//2,y+36))
  ImageDraw.Draw(sheet).text((x+8,y+7),f"{r['alias']} #{ordinal:02} f{fid} {ts[fid]/1e6:.3f}s",font=font,fill='white')
  frames.append(dict(ordinal=ordinal,frame_id=int(fid),seconds=float(ts[fid]/1e6),file=str(impath)))
 sheet.save(folder/'sequence.jpg',quality=94)
 detail=Image.new('RGB',(1500,1110),'#20252b')
 for row,ordinal in enumerate([3,6,9]):
  fid=int(ids[ordinal]);im=selected[fid];thumb=im.crop(crop);thumb.thumbnail((500,330));detail.paste(thumb,((500-thumb.width)//2,row*370+35))
  ImageDraw.Draw(detail).text((8,row*370+6),f"{r['alias']} #{ordinal} f{fid} {ts[fid]/1e6:.3f}s",font=font,fill='white')
  for si,(side,off) in enumerate([('R',54),('L',33)]):
   good=mask[fid,off:off+21];pts=xy[fid,off:off+21][good]
   if len(pts):
    lo=pts.min(0);hi=pts.max(0);cx,cy=(lo+hi)/2;rad=max(float(np.max(hi-lo))*.75,75)
   else:cx,cy=xy[fid,16 if side=='R' else 15];rad=130
   if not np.isfinite([cx,cy]).all():cx,cy=center;rad=250
   bb=(max(0,int(cx-rad)),max(0,int(cy-rad)),min(width,int(cx+rad)),min(height,int(cy+rad)))
   cut=im.crop(bb);cut.thumbnail((480,330));detail.paste(cut,(500+si*500+(500-cut.width)//2,row*370+35))
   ImageDraw.Draw(detail).text((508+si*500,row*370+6),side+' candidate detail; verify in body view',font=font,fill='white')
 detail.save(folder/'states.jpg',quality=94)
 return frames,str(folder/'sequence.jpg'),str(folder/'states.jpg')
def main():
 if (U/'PREP_RUNNING.json').exists():raise RuntimeError('Existing prep lock; inspect execution first')
 save('PREP_RUNNING.json',dict(pid=os.getpid(),started=time.time()))
 rows=read(U/'SELECTION.json')['rows'] if (U/'SELECTION.json').exists() else select()
 done=[];visual=[];comp=None
 for r in rows:
  rec=U/'prepared'/f"{r['alias']}.json"
  if rec.exists():
   d=read(rec)
   for f,h in d['pins'].items():assert sha(f)==h
   done.append(d['input']);visual.append(d['visual']);event(alias=r['alias'],reused_completed=True);continue
  source,method=source_file(r);assert sha(source)==r['archive_source']['sha256']
  obs=read_observation(r['observation']);assert sha(r['record'])==r['record_sha256']
  frames,sheet,detail=images(r,source,obs)
  b=r['box'];xy=obs['xy_px'];obs['usable'] &= (xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3])
  p=extract_features(obs);x=np.nan_to_num(p['x']).astype('float32');mask=p['mask']
  if r['old_input']:
   old=r['old_input'];assert sha(old['input'])==old['input_sha256'];z=dict(np.load(old['input'],allow_pickle=False));assert np.array_equal(x,z['x']) and np.array_equal(mask,z['mask']);secs=0
   rgb=z['rgb'];wf=z['window_features'];ws=z['window_starts'];ts=z['timestamp_us'];mid=z['window_center_us'];reused=True
  else:
   if comp is None:comp=compile_gpu()
   wf,ts,ws,secs=encode(source,b,len(x),comp);assert np.max(abs(ts-obs['timestamp_us']))<=1
   mid=(ts[np.array(ws)]+ts[np.array(ws)+15])/2;rgb=np.stack([np.interp(ts,mid,wf[:,j]) for j in range(768)],1).astype('float32');reused=False
  dest=U/'inputs'/f"{r['alias']}.npz";dest.parent.mkdir(exist_ok=True);assert not dest.exists()
  np.savez_compressed(dest,x=x,mask=mask,rgb=rgb,window_features=wf,window_starts=ws,window_center_us=mid,frame_id=obs['frame_id'],timestamp_us=ts)
  row={k:r[k] for k in ['alias','clip_id','word_id','label','person','role','intervals','box','observation','record','record_sha256']}
  row.update(source=str(source),source_sha256=sha(source),source_method=method,input=str(dest),input_sha256=sha(dest),frames=len(x),windows=len(ws),rgb_seconds=secs,reused_rgb=reused,feature_names=p['feature_names'])
  vis=dict(alias=r['alias'],sequence=sheet,states=detail,frames=frames,static_ordinals=[3,6,9],left_right='Anatomical; source non-mirrored. R usually screen left; crossing must follow arm chain.',note='ROI only affects later inputs; visual source includes original evidence, not target label')
  pins={str(path):sha(path) for path in [dest,Path(sheet),Path(detail),Path(r['record']),Path(r['observation']),source]}
  save('prepared/'+r['alias']+'.json',dict(input=row,visual=vis,pins=pins));done.append(row);visual.append(vis)
  event(alias=r['alias'],role=r['role'],frames=len(x),rgb_seconds=secs)
 save('INPUTS.json',dict(status='UNLABELED_PREPARED',rows=done,feature_names=done[0]['feature_names'],roles=dict(Counter(r['role'] for r in done))))
 save('VISUAL_INDEX.json',dict(rows=visual,labels_absent=True,scope='Same-Codex RGB observation labels; not expert linguistic truth'))
 lock=read(U/'PREP_RUNNING.json');save('PREP_COMPLETED.json',dict(**lock,completed=time.time(),clips=len(done)));(U/'PREP_RUNNING.json').unlink()
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'PREP_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
