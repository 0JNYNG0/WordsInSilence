from common import *
rows=read(U/'SOURCES.json')['rows'];missing=[r for r in rows if not r['raw_path']or not Path(r['raw_path']).exists()]
if missing:
 frozen=[{k:v for k,v in r.items()if k not in ['raw_path','channel_path']}|dict(sha256=r['views']['F']['sha256'])for r in missing]
 write(U/'FROZEN.json',dict(sources=frozen,model_pins=read(N/'FROZEN.json')['model_pins']))
 # Recover a fully written extraction if record serialization failed; verify decoded frames.
 for r in frozen:
  p=U/'raw'/(r['clip_id']+'.npz');rec=p.with_suffix('.json');cp=p.with_name(p.stem+'_channels.npz')
  if p.exists()and not rec.exists():
   import av
   z=np.load(p);hashes=[]
   with av.open(r['path'])as con:
    for f in con.decode(video=0):hashes.append(hashlib.sha256(f.to_ndarray(format='bgr24').tobytes()).hexdigest())
   assert list(z['frame_sha256'])==hashes and len(hashes)==r['views']['F']['frames'];assert cp.exists()
   write(rec,r|dict(raw_path=str(p),channel_path=str(cp),raw_sha256=sha(p),channel_sha256=sha(cp),frames=len(hashes),seconds=None,recovered='First extraction finished; metadata duplicate-key failure only. All decoded image hashes verified. Timing unavailable.'))
 if(U/'ENGINE.json').exists():(U/'ENGINE.json').rename(U/'before'/'ENGINE_first_attempt.json')
 s=importlib.util.spec_from_file_location('center_extract',O/'word100_public_g4_20261114/prepare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.U=U;m.main()
 ext={r['clip_id']:r for r in read(U/'EXTRACTED.json')['rows']}
 rows=[r|{k:ext[r['clip_id']][k]for k in ['raw_path','channel_path']}if r['clip_id']in ext else r for r in rows]
for r in rows:
 z=np.load(r['raw_path']);assert len(z['frame_id'])==r['views']['F']['frames'];r['raw_sha256']=sha(r['raw_path']);r['channel_sha256']=sha(r['channel_path'])
write(U/'READY_SOURCES.json',dict(rows=rows))
print('ready',len(rows),'new extraction',len(missing))
