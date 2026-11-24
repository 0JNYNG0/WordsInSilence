"""Build display-only reference axes from existing public FIT metadata."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
U=Path(__file__).resolve().parent; O=U.parent; P=O/'word100_learned_guide_feedback_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
rows=[r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['person']==1]
guides=read(P/'GUIDE_SUPPORT.json')['guides']; data={};pins={str(P/'INPUT_MANIFEST.json'):sha(P/'INPUT_MANIFEST.json'),str(P/'GUIDE_SUPPORT.json'):sha(P/'GUIDE_SUPPORT.json')}
for r in rows:
 assert r['role']=='fit'
 for key in ['source','input','observation']:assert sha(r[key])==r[key+'_sha256'];pins[r[key]]=r[key+'_sha256']
 a=dict(np.load(r['input']));ob=dict(np.load(r['observation']));side=read(Path(r['observation']).with_suffix('.sidecar.json'))
 mpath=O/'word100_multipeople_live_feedback_20261101'/'stage_mappings'/(r['clip_id']+'.json')
 mapping=read(mpath);pins[str(mpath)]=sha(mpath)
 assert mapping['path_axes']==['other_reference_native_index','canonical_native_index']
 path=np.asarray(mapping['path']);assert np.array_equal(path[:,0],path[:,1]),'ID01 canonical identity required'
 assert np.array_equal(a['frame_id'],ob['frame_id'])
 xy=ob['xy_px'].astype(float);xy[~ob['usable']]=np.nan
 data[r['word_id']]=dict(word_id=r['word_id'],label=r['label'],source_sha256=r['source_sha256'],person=1,
  canonical_frame_ids=a['frame_id'].tolist(),canonical_times=(a['timestamp_us']/1e6).tolist(),
  width=side.get('frame_width',side.get('width')),height=side.get('frame_height',side.get('height')),
  points=[[[float(v) if np.isfinite(v) else None for v in pt] for pt in f] for f in xy],
  guides=[g for g in guides if g['word_id']==r['word_id']],
  mapping='Learned canonical index is trained on ID01 native axis; example display only, no new pairwise DTW',
  timestamps=(ob['timestamp_us']/1e6).tolist())
write(U/'reference_data.json',data);write(U/'DISPLAY_SOURCE_PINS.json',pins)
print({k:(len(v['canonical_times']),v['width'],v['height']) for k,v in data.items()})
