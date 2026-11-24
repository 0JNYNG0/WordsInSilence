from projected_direction_core import *
from collections import Counter
import time,traceback
def main():
 V.mkdir(exist_ok=True);assert not(V/'INPUTS_COMPLETE.json').exists()
 old=read(M/'SCOPED_INPUTS.json');cfg=read(M/'FROZEN.json');src=np.load(old['path']);assert sha(old['path'])==old['sha256']
 write(V/'FROZEN.json',dict(at=stamp(),hypothesis='Avoid squared-short-forearm denominator; compare one predefined shoulder-normalized projection representation with the saved original.',measurement_change='axis2 projection denominator L**2 -> L; other three series, masks, coverage, .05 length gate unchanged',why='Public-normal direction diagnosis revealed ratio amplification for a foreshortened anchor. Original first failure retained; no DEV threshold selection.',training=dict(seeds=[42,43,44],steps=800,lr=.0003,weight_decay=.001,batch='128 original replay +128 reviewed, balanced axis and target',selected='final_step_only',normalization='FIT rows only'),runtime_scope='Axis2 only; all other direction/shape/candidate outputs identical to saved local_old.',gate='All three seeds: no normal/speed direction regression versus baseline; reversed and combined evidence retained; no missing-hand positive; no web promotion by partial gain.',pins={str(p):sha(p)for p in [Path(__file__),W/'projected_direction_core.py',M/'SCOPED_INPUTS.json',M/'VERIFICATION.json',L/'evaluation/COMPARISON.json',L/'evaluation/shadow_components/RESULTS.json']},private_access=False,final8=False,web_changed=False))
 ready={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'}
 crop={r['clip_id']:r for q in[R/'CROP_INPUTS_COMPLETE.json',D/'CROP_INPUTS_COMPLETE.json']for r in read(q)['rows']}
 sys.path.insert(0,str(O/'word100_observation_guard_20261108'));guard=load('_pr_guard',O/'word100_observation_guard_20261108/guard.py');ad=load('_pr_ad',O/'word100_rtmw_shared_input_20261108/adapter.py')
 cache={};geoms=[];rows=[];pins={};max_other=0.
 for i,r in enumerate(old['rows']):
  cid=r['clip_id'];view=r['view'];key=cid,view
  if key not in cache:
   rr=ready[cid];assert sha(rr['path'])==rr['sha256']and sha(rr['channel'])==rr['channel_sha256'];ch=np.load(rr['channel']);z=np.load(rr['path']);ts=z['timestamp_us'];pins[rr['path']]=rr['sha256'];pins[rr['channel']]=rr['channel_sha256']
   if view=='actual_crop':
    cr=crop[cid];assert sha(cr['path'])==cr['sha256'];a=np.load(cr['path']);q=guard.quarantine(a['xy_px'],a['score'],a['timestamp_us'],640,480);x=ad.convert(q['xy_px'],a['score'],a['timestamp_us'],a['frame_id'],640,480)['tcn280'];pins[cr['path']]=cr['sha256']
   else:x=ch['full']if view=='full'else ch['feedback']if'feedback'in ch.files else ch['upper_mask']
   cache[key]=(x,ts)
  x,ts=cache[key];g,ok,info=geometry(x,ts,r['a'],r['b'],r['axis'],r['moving'],r['anchor'],r['reverse']);assert ok
  if r['axis']!=2:assert np.array_equal(g,src['g'][i])
  rows.append({**r,'geometry':info});geoms.append(g)
 g=np.asarray(geoms);tr=np.array([r['split']=='train'for r in rows]);norm=dict(mean=g[tr].mean(0).tolist(),std=np.maximum(g[tr].std(0),.01).tolist())
 dest=V/'inputs.npz';np.savez_compressed(dest,context=src['context'],g=g,y=src['y']);assert np.isfinite(g).all()
 write(V/'INPUTS_COMPLETE.json',dict(at=stamp(),path=str(dest),sha256=sha(dest),rows=rows,norm=norm,pins=pins,source=str(M/'SCOPED_INPUTS.json'),source_sha256=sha(M/'SCOPED_INPUTS.json'),counts=dict(Counter(r['split']+'_'+r['origin']for r in rows)),contexts_unchanged=True,labels_unchanged=True,other_axes_unchanged=True))
 print('Projected geometry ready',len(rows),'rows; unique source/views',len(cache))
if __name__=='__main__':
 try:main()
 except BaseException:V.mkdir(exist_ok=True);(V/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
