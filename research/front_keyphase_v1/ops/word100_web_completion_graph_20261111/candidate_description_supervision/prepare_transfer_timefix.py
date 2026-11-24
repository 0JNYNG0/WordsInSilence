from transfer_local_timefix_core import *
import time,traceback

def main():
 assert not (TQ/'INPUTS_COMPLETE.json').exists();TQ.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
 source=read(N/'INPUTS_COMPLETE.json');a=np.load(source['arrays']);rows=source['rows'];need={r['clip_id'] for r in rows};raw={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit' and r['clip_id'] in need};weights=O/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt'
 cfg=dict(at=stamp(),hypothesis='Local-only small network removed outside-context bias but lost normal-source discrimination. Test learned common-100 observation features recomputed inside each candidate, without word/order hints.',source_diagnosis=str(Q/'FAILURE_DIAGNOSIS.json'),same_supervision_as_dense=True,source_clips=24,train_people=[2,4,5,6],probe_people=[8,9],upstream_probe_exposed=True,shared_weights_frozen=True,shared_encoder=str(weights),input_time_feature='actual inter-frame delta seconds; first zero; reverse intervals reversed',no_word_id_or_expected_order=True,frame_subsampling=False,reverse_augmentation='Each source window internally reversed for the same action-axis attempt; not a normal-sign target.',head='1160 ->96->5*3',seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,selection=dict(conditions=['baseline','local_from_scratch','local_pretrained']),candidate_rule='all same visible spans; fixed .5 presence; sum log three heads',semantic_completion=None,web_changed=False,pins={str(p):sha(p) for p in [weights,N/'INPUTS_COMPLETE.json',Path(__file__),W/'transfer_local_timefix_core.py',Q/'FAILURE_DIAGNOSIS.json']})
 write(TQ/'FROZEN.json',cfg);base=encoder();xx=np.zeros((len(rows),1160),np.float32);xr=np.zeros_like(xx);groups=sorted({(r['clip_id'],r['view']) for r in rows});pins={};tim=[]
 for cid,view in groups:
  ii=[i for i,r in enumerate(rows) if (r['clip_id'],r['view'])==(cid,view)];r=rows[ii[0]];assert sha(r['cache'])==r['cache_sha256'];src=raw[cid];assert sha(src['path'])==src['sha256'];z=np.load(r['cache'])['z'];q=z[:,292:392];pose=np.load(src['path'])['x_'+view];sp=np.array([rows[i]['span'] for i in ii]);t=time.perf_counter()
  for off in range(0,len(ii),128):
   inds=ii[off:off+128];ss=sp[off:off+128];xx[inds]=encode_windows(base,pose,q,ss);xr[inds]=encode_windows(base,pose,q,ss,True)
  pins[r['cache']]=r['cache_sha256'];pins[src['path']]=src['sha256'];tim.append(dict(clip_id=cid,view=view,windows=len(ii),seconds=time.perf_counter()-t));print('transfer inputs',cid,view,len(ii),flush=True)
 assert np.isfinite(xx).all() and np.isfinite(xr).all();p=TQ/'inputs.npz';np.savez_compressed(p,x=xx,x_reverse=xr,y=a['y'],mask=a['mask'],train=a['train']);write(TQ/'INPUTS_COMPLETE.json',dict(at=stamp(),arrays=str(p),arrays_sha256=sha(p),rows=rows,pins=pins,timing=tim,base_weights_unchanged=sha(weights)==cfg['pins'][str(weights)],no_private=True,no_final8=True));print('Prepared transfer-local',len(rows))
if __name__=='__main__':
 try:main()
 except BaseException:TQ.mkdir(exist_ok=True);(TQ/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
