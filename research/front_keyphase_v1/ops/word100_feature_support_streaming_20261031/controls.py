"""Prespecified measurement safeguards and development-only speed perturbations."""
from quality import *
from evidence_adapter import shape_support,interval_support,selftest
def main():
 torch.set_num_threads(4);rows=get_rows();rows=[r for r in rows if r['role']=='development'];stats=read(U/'NORMALIZATION.json');data,_=assemble(rows,stats);mx,my,mm,ml=data['motion'];sx,sy,sm,sv=data['shape'];results=[]
 for seed in [42,43,44]:
  heads=torch.load(U/'runs'/str(seed)/'heads.pt',weights_only=True);model=Motion();model.load_state_dict(heads['motion']);model.eval()
  with torch.no_grad():base=model(mx,ml).sigmoid().numpy()
  for factor in [.75,1.25]:
   altered=mx.clone()
   for j in [9,10]:
    observed=mx[:,:,11+j].bool();raw=mx[:,:,j]*stats['motion_std'][j]+stats['motion_mean'][j];value=(raw/factor-stats['motion_mean'][j])/stats['motion_std'][j];altered[:,:,j]=torch.where(observed,value,torch.zeros_like(value))
   with torch.no_grad():p=model(altered,ml).sigmoid().numpy()
   mask=mm.numpy();flips=int((((p>=.5)!=(base>=.5))&mask).sum());results.append(dict(seed=seed,time_factor=factor,known_cells=int(mask.sum()),label_flips=flips,max_probability_change=float(np.max(np.abs(p-base))),motion_balanced_accuracy=old.metric(p,my.numpy(),mask)['macro_balanced_accuracy']))
 guards=selftest();counts=[]
 for r,a in zip(rows,data['arrays']):
  for side in SIDES:
   ix=feature_ids(r,side,'shape');s=shape_support(a['mask'][:,ix]);assert len(s['observed_angle_count'])==len(a['x'])
   for lo,hi in intervals(r['alias']):
    ids=np.flatnonzero((a['frame_id']>=lo)&(a['frame_id']<=hi));names=r['feature_names'];w=a['mask'][ids][:,[names.index(side+'_hand_wrist_x'),names.index(side+'_hand_wrist_y')]];v=a['mask'][ids][:,[names.index(side+'_wrist_velocity_x'),names.index(side+'_wrist_velocity_y')]];support=interval_support(w,v);hidden=interval_support(np.zeros_like(w),np.zeros_like(v));assert support['expected_frames']==hidden['expected_frames'] and hidden['status']=='비교 근거 부족';counts.append(support)
 save('CONTROLS.json',dict(status='COMPLETE_NO_SELECTION',guards=guards,full_interval_missing_tests=len(counts),speed_rows=results,scope='Development12 only. Same frames/order, velocities adjusted for uniform timestamp stretch .75/1.25. Not a real fast-camera/blur test. No hypothesis or threshold altered after outputs.',unknown_intervals=sum(not c['comparison_supported'] for c in counts),educational_verdicts=0))
 print(json.dumps(results),flush=True)
if __name__=='__main__':main()
