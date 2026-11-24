"""Read only explicitly allowlisted public original videos and prior labels."""
from common import *
import re,zipfile,io,collections,traceback
import numpy as np
import av

def main():
 assert not (U/'SOURCE_AUDIT.json').exists(),'Completed audit is immutable'
 tcn=read(O/'word100_training_20261013/MANIFEST.json')
 fb=read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')
 fm={r['clip_id']:r for r in fb['rows']};rows=[];archives={};seen={};checks=[]
 oldcompare=read(O/'word100_rtmw_mediapipe_compare_20261108/FROZEN.json')
 cache={r['clip_id']:r for r in oldcompare['rows']}
 # Feedback words first so that its limited supervised study can run during the larger extraction.
 ordered=sorted(tcn,key=lambda r:(r['clip_id'] not in fm,r['role']!='train_candidate',r['clip_id']))
 for i,r in enumerate(ordered):
  cid=r['clip_id'];sid=r['signer_id'];assert re.fullmatch(r'NIA_SL_WORD\d{4}_REAL\d{2}_F',cid)
  assert sid in (range(1,10) if r['role']=='train_candidate' else [11,12,13]);assert f'REAL{sid:02d}_F' in cid
  e=r['video_entries'][0];zp=Path(e['archive'])
  if str(zp) not in archives:archives[str(zp)]=zipfile.ZipFile(zp)
  zi=archives[str(zp)].getinfo(e['member']);assert zi.CRC==e['crc32'] and zi.file_size==e['bytes']
  data=archives[str(zp)].read(e['member']);h=hashlib.sha256(data).hexdigest()
  if h in seen:raise ValueError('Duplicate public source: '+cid+' / '+seen[h])
  seen[h]=cid
  with av.open(io.BytesIO(data)) as con:
   s=con.streams.video[0];n=s.frames;w=s.width;hh=s.height;duration=float(s.duration*s.time_base)
  assert n>0 and w>0 and hh>0
  record=O/'word100_body_boundary_20261017/data/records'/f'{cid}.json'
  assert record.exists(),cid
  rr=read(record)
  out=dict(clip_id=cid,role='fit' if sid<10 else 'development',person=sid,word_id=r['word_code'],label=r['canonical_label'],label_index=r['label_index'],
           archive=str(zp),member=e['member'],bytes=len(data),crc32=zi.CRC,source_sha256=h,frames_header=n,width=w,height=hh,duration_header=duration,
           baseline_record=str(record),baseline_record_sha256=sha(record),rtmw_cache=None,feedback=False)
  if cid in cache:
   cr=cache[cid];cp=O/'word100_rtmw_mediapipe_compare_20261108/runs'/cid/'rtmw_gpu/rtmw.npz'
   if cr['source_sha256']==h and cp.exists():out['rtmw_cache']=str(cp);out['rtmw_cache_sha256']=sha(cp)
  if cid in fm:
   fr=fm[cid];assert h==fr['source_sha256']==sha(fr['source']),cid+' feedback source mismatch'
   assert sha(fr['input'])==fr['input_sha256'];a=np.load(fr['input'],allow_pickle=False)
   out['feedback']=True;out['feedback_frame_ids']=a['frame_id'].tolist();out['feedback_timestamp_us']=a['timestamp_us'].tolist()
   assert len(a['frame_id'])==fr['frames'] and a['frame_id'].max()<n
   for label in fr['labels']:
    ix=np.flatnonzero(a['frame_id']==label['center_frame']);assert len(ix)==1
    assert int(a['timestamp_us'][ix[0]])==label['center_timestamp_us']
   out['feedback_source']=str(Path(fr['source']))
  rows.append(out)
  if i%100==0:event('source_audit',done=i+1,total=len(ordered))
 for z in archives.values():z.close()
 assert len(rows)==1199 and sum(r['role']=='fit' for r in rows)==899 and sum(r['feedback'] for r in rows)==48
 # Pin all original model/web inputs already protected by the prior stream experiment.
 protected=read(O/'word100_rtmw_browser_capture_20261108/FROZEN.json')['protected']
 assert all(sha(p)==h for p,h in protected.items())
 pins={str(O/p):sha(O/p) for p in ['word100_training_20261013/MANIFEST.json','word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json',
 'word100_learned_guide_feedback_20261102/GUIDE_SUPPORT.json','word100_learned_guide_feedback_20261102/BLIND_STAGE_ANCHORS.json',
 'word100_learned_guide_feedback_20261102/TRAINING_REPORT.json','word100_rtmw_shared_input_20261108/adapter.py']}
 write(U/'SOURCES.json',rows)
 write(U/'SOURCE_AUDIT.json',dict(status='PASS',clips=len(rows),fit=899,development=300,feedback_fit=36,feedback_development=12,
  duplicate_source_hashes=0,verified_label_centers=sum(len(r['labels']) for r in fb['rows']),
  source_bytes=sum(r['bytes'] for r in rows),source_frames_header=sum(r['frames_header'] for r in rows),cache_reusable=sum(bool(r['rtmw_cache']) for r in rows),
  archive_stats={p:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in archives},protected=protected,pins=pins,
  limitations=['Previously exposed development, not a fresh test','Prior Codex visual physical labels, not independent expert correctness','DTW transferred intervals are weak training supervision only'],
  training=0,source_manifest_sha256=sha(U/'SOURCES.json')))
 event('source_audit_complete',clips=len(rows))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/f'AUDIT_FAILURE_{time.time_ns()}.txt';p.write_text(traceback.format_exc(),encoding='utf-8');raise
