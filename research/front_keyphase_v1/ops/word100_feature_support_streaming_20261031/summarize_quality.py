"""Describe saved predictions, without another inference or model selection."""
from quality import *
from evidence_adapter import shape_support,interval_support
def main():
 rows=get_rows(True);alllabels=[labels(r,True) for r in rows];conditions=['frozen_previous_pose','frozen_previous_rgb','shared_shape_interval_motion'];results=[]
 for cond in conditions:
  for seed in [42,43,44]:
   p=np.load(U/'predictions'/f'fresh_{cond}_{seed}.npy',allow_pickle=False);count=dict(shape_cross_known=0,shape_cross_correct=0,shape_known=0,shape_correct=0,shape_positive_known=0,shape_positive_correct=0,motion_known=0,motion_correct=0)
   for i,(r,y) in enumerate(zip(rows,alllabels)):
    n=len(y['y']);cross=(y['y'][:,ALL.index('hands_overlap')]==1)&y['target_mask'][:,ALL.index('hands_overlap')]
    for j,f in enumerate(TARGETS):
     k=ALL.index(f);m=y['target_mask'][:,k];correct=(p[i,:n,j]>=.5)==(y['y'][:,k]>=.5)
     if f in ['R_index_only','R_open_four','L_open_four']:
      for key,sel in [('shape',m),('shape_cross',m&cross),('shape_positive',m&(y['y'][:,k]==1))]:count[key+'_known']+=int(sel.sum());count[key+'_correct']+=int((sel&correct).sum())
     if '_motion_' in f:count['motion_known']+=int(m.sum());count['motion_correct']+=int((m&correct).sum())
   results.append(dict(condition=cond,seed=seed,**count))
 evidence=[]
 for r in rows:
  a=dict(np.load(r['input'],allow_pickle=False))
  for side in SIDES:
   support=shape_support(a['mask'][:,feature_ids(r,side,'shape')]);shape_known=int(support['comparison_supported'].sum())
   motion=[];names=r['feature_names']
   for lo,hi in intervals(r['alias'],True):
    ii=np.flatnonzero((a['frame_id']>=lo)&(a['frame_id']<=hi));w=a['mask'][ii][:,[names.index(side+'_hand_wrist_x'),names.index(side+'_hand_wrist_y')]];v=a['mask'][ii][:,[names.index(side+'_wrist_velocity_x'),names.index(side+'_wrist_velocity_y')]];motion.append(dict(start=lo,end=hi,**interval_support(w,v)))
   evidence.append(dict(alias=r['alias'],side=side,expected_shape_frames=len(a['x']),supported_shape_frames=shape_known,motion=motion))
 save('SAVED_OUTPUT_DIAGNOSTICS.json',dict(rows=results,evidence=evidence,scope='Descriptive subsets of saved fresh evaluation. No feature/seed selected from outcomes. Crossing is 2D silhouette overlap label, not all arm-crossing cases. Positive shape cells are physical normal examples, not verified linguistic correctness.'))
 print(json.dumps(results),flush=True)
if __name__=='__main__':main()
