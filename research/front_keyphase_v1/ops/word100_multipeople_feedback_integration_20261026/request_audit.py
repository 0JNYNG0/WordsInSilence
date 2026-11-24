import run as R
import numpy as np
from collections import Counter
f=R.read(R.U/'FROZEN.json');R.check(f['pins']);prep=R.read(R.BP/'PREPARED.json');count=Counter();rows=[];pins={}
for row in prep['rows']:
 cid=row['clip_id'];z=R.load(row['output']);ep=row['source_evidence'];e=R.load(ep);vp=R.O/'word100_recovery_dev300_20261026'/'outputs'/(cid+'.npz');p=R.load(vp);rp=R.O/'word100_recovery_dev300_20261026'/'records'/(cid+'.json');rec=R.read(rp);records={r['i']:r for r in rec['frames']}
 assert np.array_equal(p['frame_id'],z['frame_id']) and len(records)==int(p['mp_called'].sum()) and set(records)==set(np.flatnonzero(p['mp_called']))
 core=z['core'][:,None,None];unknown=core&(z['angle_state']==2);uncalled=np.broadcast_to(~p['mp_called'][:,None,None],unknown.shape);no_pair=np.broadcast_to(~e['quality_pair'][0,:,:,None],unknown.shape);angle_unavailable=~e['quality_available'][0]
 a=unknown&uncalled;b=unknown&~uncalled&no_pair;c=unknown&~uncalled&~no_pair&angle_unavailable
 assert np.array_equal(a|b|c,unknown) and not(a&b).any() and not(b&c).any()
 counts=dict(unassessed=int(unknown.sum()),no_retry_requested=int(a.sum()),retry_without_pair=int(b.sum()),pair_without_angle=int(c.sum()),total_frames=len(z['core']),core_frames=int(z['core'].sum()),retry_frames=int(p['mp_called'].sum()),core_retry_frames=int((p['mp_called']&z['core']).sum()))
 count.update(counts);rows.append(dict(clip_id=cid,**counts));pins.update({str(x):R.sha(x) for x in [ep,vp,rp]})
assert count['unassessed']==R.read(R.U/'CAUSE_AUDIT.json')['base_core_states']['state_2'];R.check(f['protected'])
R.write(R.U/'REQUEST_AUDIT.json',dict(at=R.now(),status='PASS_EXCLUSIVE_SOURCE_REASON_ACCOUNTING',counts=dict(count),rows=rows,pins=pins,code=R.sha(__file__),explanation='Existing recovery intentionally triggers only on a wholly missing hand with visible arm wrist. Thus most otherwise computable original hand angles never requested a second detector. No retry is not a detector failure or gesture error. Do not call these low confidence by absence alone. This source audit changes no inputs or statuses.',training=0,inference=0));print('REQUEST_AUDIT',dict(count))
