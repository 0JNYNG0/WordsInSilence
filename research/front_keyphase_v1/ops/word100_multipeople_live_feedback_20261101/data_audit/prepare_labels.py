"""Bounded public RGB preparation only; no detector, encoder, training or scoring."""
from pathlib import Path
import json, hashlib, random, zipfile
import numpy as np
import av
from PIL import Image, ImageDraw, ImageFont

U=Path(__file__).resolve().parent; OPS=U.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,data):
 p=U/name
 assert not p.exists(),p
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def prepare_inputs():
 inv=read(U/'PUBLIC_PEOPLE_INVENTORY.json')
 ds=read(OPS/'word100_semantic_feature_cycle_20261101/DATASET.json')
 old={}
 for r in ds['rows']:
  old.setdefault(r['clip_id'],[]).append(r)
 rows=[]
 for r in inv['four_expression_inventory']:
  pid=r['recorded_person_id']
  if pid not in list(range(1,10))+[11,12,13]:continue
  rec=read(r['native_record']['path']); op=r['native_observation']['path']
  assert sha(op)==r['native_observation']['sha256']
  with np.load(op,allow_pickle=False) as z:
   ts=z['timestamp_us']; fid=z['frame_id']
  a=min(v[0] for v in r['word_interval_seconds']); b=max(v[1] for v in r['word_interval_seconds'])
  ids=[int(np.abs(ts-t).argmin()) for t in np.linspace(max(0,a-.12),min(ts[-1]/1e6,b+.12),13)*1e6]
  assert len(set(ids))==13
  centers=[ids[i] for i in [3,6,9]]
  assert all(6<=c<len(ts)-6 for c in centers)
  reused=sorted(old.get(r['clip_id'],[]),key=lambda x:x['center_frame'])
  if pid<=4:
   assert [x['center_frame'] for x in reused]==centers
  else:assert not reused
  source=r['source_video_caches'][0]['path'] if r['source_video_caches'] else None
  archive=rec['source'].copy()
  for e in r['archive_members']:
   if e['member']==archive['member']:archive['archive']=e['resolved_archive']
  rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],label=r['label'],person=pid,
    role='fit' if pid<=9 else 'development',source=source,source_sha256=rec['source']['sha256'],archive_source=archive,
    native=op,native_sha256=sha(op),record=r['native_record'],roi_box=rec['calibration']['box_source_pixels'],
    intervals=r['word_interval_seconds'],frames=len(ts),centers=centers,
    center_timestamp_us=[int(ts[c]) for c in centers],center_frame_ids=[int(fid[c]) for c in centers],
    context_frame_ids=[[int(x) for x in fid[c-6:c+7]] for c in centers],
    label_provenance='reuse_exact_previous_center' if pid<=4 else 'new_direct_RGB_required',
    old_labels=reused))
 assert len(rows)==48 and sum(r['person']<=4 for r in rows)==16
 fresh=[r for r in rows if r['person']>4];random.Random(2026110117).shuffle(fresh)
 for i,r in enumerate(fresh,1):r['visual_alias']=f'N{i:03}'
 save('ANNOTATION_INPUTS.json',dict(status='FIXED_BEFORE_NEW_RGB_REVIEW_AND_SCORES',rows=rows,targets=ds['targets'],
   selection='Original core +/-0.12s uniformly sampled at 13 nearest native PTS; ordinals3/6/9 only. Each exact center has 13-frame input context, not 13 labels.',
   counts={'fit_clips':36,'dev_clips':12,'fit_centers':108,'dev_centers':36,'old_centers_reused':48,'new_centers':96},
   new_protocol='New nine-person developmental protocol; old studies and scores unchanged; old04/05/06/07/08 exposure disclosed;11..13 project-TCN DEV, not fresh final test.',
   source_pins={str(U/'PUBLIC_PEOPLE_INVENTORY.json'):sha(U/'PUBLIC_PEOPLE_INVENTORY.json'),str(OPS/'word100_semantic_feature_cycle_20261101/DATASET.json'):sha(OPS/'word100_semantic_feature_cycle_20261101/DATASET.json'),str(Path(__file__)):sha(__file__)},
   new_inference=0,new_training=0))
 return rows

def prepare_sources(rows):
 for r in rows:
  src=Path(r['source']) if r['source'] else U/'videos'/f"{r['clip_id']}.mp4"
  if not src.exists():
   src.parent.mkdir(parents=True,exist_ok=True);s=r['archive_source']
   with zipfile.ZipFile(s['archive']) as z:
    zi=z.getinfo(s['member']);assert zi.CRC==s['crc32'] and zi.file_size==s['bytes'];data=z.read(zi)
   assert hashlib.sha256(data).hexdigest()==r['source_sha256']
   src.write_bytes(data)
  assert sha(src)==r['source_sha256'];r['source']=str(src)
 save('SOURCES.json',dict(status='48_EXACT_PUBLIC_SOURCES_READY',rows=rows,new_model_calls=0,new_labels=0))
 print('SOURCES_READY',flush=True)

def prepare_visuals(rows):
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
 visual=[]
 for r in sorted([r for r in rows if r['person']>4],key=lambda x:x['visual_alias']):
  alias=r['visual_alias'];src=Path(r['source'])
  if not src.exists():
   src.parent.mkdir(parents=True,exist_ok=True);s=r['archive_source']
   with zipfile.ZipFile(s['archive']) as z:
    zi=z.getinfo(s['member']);assert zi.CRC==s['crc32'] and zi.file_size==s['bytes'];data=z.read(zi)
   assert hashlib.sha256(data).hexdigest()==r['source_sha256']
   src.write_bytes(data)
  assert sha(src)==r['source_sha256']
  folder=U/'sheets'/alias;folder.mkdir(parents=True,exist_ok=True)
  with np.load(r['native'],allow_pickle=False) as z:
   xy=z['xy_px'];usable=z['usable'];ts=z['timestamp_us']
  selected={}
  with av.open(str(src)) as con:
   for i,f in enumerate(con.decode(video=0)):
    if i in r['centers']:
     assert abs(round(float(f.pts*f.time_base)*1e6)-int(ts[i]))<=1
     selected[i]=f.to_image()
    if i>=max(r['centers']):break
  assert len(selected)==3
  sheet=Image.new('RGB',(1500,1170),'#20252b');draw=ImageDraw.Draw(sheet);windows=[]
  center=np.nanmedian(xy[r['centers']][:,[11,12]].reshape(-1,2),axis=0)
  for k,c in enumerate(r['centers']):
   im=selected[c];W,H=im.size
   body=(max(0,int(center[0]-620)),0,min(W,int(center[0]+620)),H)
   b=im.crop(body);b.thumbnail((490,350));sheet.paste(b,((500-b.width)//2,k*390+35))
   draw.text((8,k*390+7),f'{alias} W{k} f{c} {ts[c]/1e6:.3f}s',font=font,fill='white')
   # Red line denotes the input ROI, solely for unknown/out-of-input visibility decisions.
   scale=b.height/(body[3]-body[1]);line_y=k*390+35+int(r['roi_box'][3]*scale)
   draw.line(((500-b.width)//2,line_y,(500+b.width)//2,line_y),fill='#f27272',width=2)
   boxes={}
   for si,(side,off,pose) in enumerate([('R',54,16),('L',33,15)]):
    good=usable[c,off:off+21]&np.isfinite(xy[c,off:off+21]).all(-1);pts=xy[c,off:off+21][good]
    if len(pts):
     lo=pts.min(0);hi=pts.max(0);cx,cy=(lo+hi)/2;rad=max(float(np.max(hi-lo))*.75,85)
    else:cx,cy=xy[c,pose];rad=140
    if not np.isfinite([cx,cy]).all():cx,cy=center;rad=270
    bb=(max(0,int(cx-rad)),max(0,int(cy-rad)),min(W,int(cx+rad)),min(H,int(cy+rad)));boxes[side]=list(bb)
    cut=im.crop(bb);factor=min(480/cut.width,345/cut.height);cut=cut.resize((int(cut.width*factor),int(cut.height*factor)),Image.Resampling.LANCZOS)
    sheet.paste(cut,(500+si*500+(500-cut.width)//2,k*390+35));draw.text((508+si*500,k*390+7),side+' crop proposal - verify body',font=font,fill='white')
   p=folder/f'f{c:04}.jpg';im.save(p,quality=95)
   windows.append(dict(sample_id=f'{alias}_W{k}',center_frame=c,timestamp_us=int(ts[c]),full_image=str(p),full_image_sha256=sha(p),detail_boxes=boxes,roi_box=r['roi_box']))
  dest=folder/'centers.jpg';sheet.save(dest,quality=96)
  visual.append(dict(alias=alias,sheet=str(dest),sheet_sha256=sha(dest),windows=windows,source=str(src),source_sha256=sha(src)))
  print(alias,'ready',flush=True)
 save('VISUAL_INDEX.json',dict(status='EXACT_STATIC_CENTERS_READY',rows=visual,new_centers=96,new_model_calls=0,new_labels=0))

if __name__=='__main__':
 rows=prepare_inputs();prepare_sources(rows);prepare_visuals(rows)
