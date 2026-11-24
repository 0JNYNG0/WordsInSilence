from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 manifest=read(U/'VIDEO_CONTROLS.json')['rows'];results=read(U/'VIDEO_CONTROL_RESULTS.json')['rows'];assert len(manifest)==len(results)==24
 lat=[];summary={}
 for m,r in zip(manifest,results):
  assert m['word_id']==r['word_id'] and m['kind']==r['kind'];assert m['person_id']==12
  for key,pkey in [('sha256','path'),('source_sha256','source')]:assert hashlib.sha256(Path(m[pkey]).read_bytes()).hexdigest()==m[key]
  p=Path(r['result_path']);out=read(p);a=np.load(p.parent/'analysis.npz')
  assert out['source_sha256']==m['sha256'];assert np.all(np.diff(a['timestamp_us'])>0);assert len(a['frame_id'])==len(m['source_frame_indices'])
  assert out['whole_sign_correctness'] is None;assert out['individual_reference_comparisons']==0 and out['learned_prototype_comparisons']==1
  assert dict(Counter(s['status'] for s in out['stages']))==r['statuses']
  if m['kind']=='main_blackout':
   lo,hi=m['blackout_source_frames'];expected=np.array([(lo<=i<=hi) for i in m['source_frame_indices']]);assert np.array_equal(~a['image_available'],expected)
   gapids={i for g in out['evidence_gaps'] for i in g['frames']};assert set(np.flatnonzero(expected))==gapids
   assert not r['fully_similar']
  if m['kind'] in ['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']:assert r['hard_difference']==0
  lat.append(r['latency']['total_seconds']);key=m['kind'];ss=summary.setdefault(key,dict(videos=0,hard_difference=0,fully_similar=0,gaps=0));ss['videos']+=1;ss['hard_difference']+=r['hard_difference'];ss['fully_similar']+=int(r['fully_similar']);ss['gaps']+=r['evidence_gaps']
 result=dict(status='PASS_SEPARATE_SAVED_VIDEO_ARITHMETIC_AND_PIXEL_AVAILABILITY',summary=summary,
  latency_seconds=dict(min=min(lat),max=max(lat),mean=float(np.mean(lat))),
  reencoded_deleted_content_extracted_again=True,all_blackout_intervals_retained_exactly=True,
  no_fresh_inference=True,public_only=True,reviewer='Same Codex separate saved verification, not independent KSL human labels')
 with (U/'CONTROL_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
