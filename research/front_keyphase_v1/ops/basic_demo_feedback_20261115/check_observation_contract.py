"""Disentangle low detector score from truly missing coordinates.

The existing RTMW adapter explicitly does not calibrate score as visibility.
Preserve the earlier diagnostic output and append this clarified check.
"""
from local_shape import *
sys.path.insert(0,str(O/'word100_rtmw_shared_input_20261108'));import adapter

def main():
 torch.set_num_threads(2);net=LocalShape('temporal');out=[]
 records=[r for r in read(U/'INPUTS.json')['rows']if r['role']=='development']+read(U/'source_corrected/CONFIRMATION_INPUTS.json')['rows']
 for r in records:
  z=dict(np.load(r['raw']));z['xy_px'][:,91:]=np.nan;z['score'][:,91:]=0
  ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']))
  s={h:net.sequence(ch['tcn280'],h)[0]for h in ['L','R']}
  assert all((q<0).all()for q in s.values())
  out.append(dict(clip_id=r['clip_id'],all_missing_unknown=True,frames=len(z['frame_id'])))
 write(V/'MISSING_CHECK.json',dict(rows=out,correction='CONTROLS.json kind=missing only set scores to0, leaving coordinates present. It is a low-score-only condition, NOT missing joints. This check uses NaN coordinates and score0; all outputs unknown.',adapter_changed=False))
 # Whole-hand candidates must not manufacture individual finger facts.
 write(V/'ADMISSION.json',dict(admitted=False,
  reason='Temporal model misclassifies edge-view normal P15 as flexed in34/43 active frames; source-view overlap remains unresolved. Global DEV gain alone is insufficient.',
  safe_use='Research observations only. Preserve existing individual-finger observer and full100 model; no web success producer from this candidate.',
  positive='No correct-to-wrong regression among existing known DEV predictions; 20 additional correct DEV answers; opposite hand invariant in all10 fist controls; truly missing hands allunknown.',
  limits='One added definite DEV error and persistent held-source edge ambiguity; artificial fists are not real human error videos.'))
 print('MISSING5PASS; LOCAL SHAPE NOT ADMITTED')
if __name__=='__main__':main()
