from pathlib import Path
import sys,json,collections
import numpy as np,torch
from motion_core import endpoint_pause
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_guide_requirement_learning_20261111'))
import requirement_core as rc
read=rc.read;write=rc.write
assert not(U/'MOTION_FROZEN.json').exists()
rr=rc.public_rows();gg=[];rows=[];pins={}
for r in rr:
 raw=r['raw_x'];pins[r['path']]=rc.sha(r['path'])
 for pi,p in enumerate(r['pairs']):
  g,ok,info=rc.prior.geometry(raw,r['timestamp_us'],p['a'],p['b'],p['axis'],p.get('moving','R'),p.get('anchor','L'))
  for pause in ([0,.25,.5]if r['role']=='fit'else[0]):
   gg.append(endpoint_pause(g,pause)if ok else g);rows.append(dict(clip_id=r['clip_id'],word=r['word_id'],person=r['person'],role=r['role'],pair=pi,axis=p['axis'],target=p['target'],span=[p['a'],p['b']],source_frames=[p['frame_a'],p['frame_b']],moving=p.get('moving','R'),anchor=p.get('anchor','L'),eligible=bool(ok),pause_fraction=pause,observed_delta=info.get('signed_change'),original_label=p))
np.savez_compressed(U/'motion_arrays.npz',geometry=np.array(gg),targets=np.array([r['target']for r in rows]))
write(U/'MOTION_INPUTS.json',dict(rows=rows,unknown_retained=True,source_pins=pins,scope='Previously RGB-reviewed component intervals, not full semantic-completion truth',fit_augmentation='Repeat observed endpoint positions only; retains observed direction, not a newly correct sign video'))
cfg=dict(seeds=[42,43,44],steps=600,batch=64,lr=.0005,weight_decay=.001,init=str(rc.B/'runs/coupled_42/model.pt'),init_sha256=rc.sha(rc.B/'runs/coupled_42/model.pt'),confidence=.65,selection='Fixed final step and all three seeds; no tuning to DEV outcomes',design='Fine-tune existing movement conv/GRU/readout; shape and word context removed; enforce temporal-reversal equivariance by two shared passes',labels='Original FIT component targets; reverse swaps classes0/2 and preserves hold1',fit_only=True,normalization='Original FIT-only geometry normalization retained',public_original_and_same_handshape_and_reverse_and_combined=True,no_word_exception=True,automatic_windows_unchanged=True,
 promotion='No reduction in supported normal/short/hold cases or old correct error detections; no new opposite-direction warnings on normal public material; no all-unknown success',pins={str(U/'motion_arrays.npz'):rc.sha(U/'motion_arrays.npz'),str(U/'MOTION_INPUTS.json'):rc.sha(U/'MOTION_INPUTS.json'),str(U/'motion_core.py'):rc.sha(U/'motion_core.py')})
write(U/'MOTION_FROZEN.json',cfg)
print(json.dumps({split:dict(rows=sum(r['role']==split for r in rows),eligible=sum(r['role']==split and r['eligible']for r in rows),clips=len({r['clip_id']for r in rows if r['role']==split}),words=len({r['word']for r in rows if r['role']==split}))for split in['fit','development']}))
