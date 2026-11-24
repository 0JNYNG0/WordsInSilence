"""Engineered open-to-fist coordinate substitution, evaluation only.

This is not a real video of a learner making an error and is never training truth.
"""
import core
P=core.U;core.U=P/'source_corrected'
from core import *
def main():
 torch.set_num_threads(2);factory=Features()
 sys.path[:0]=[str(O/'word100_finger_repair_20261115'),str(O/'word100_clause_finger_diagnostics_20261114'),str(O/'word100_rtmw_shared_input_20261108')]
 from finger_runtime import FingerObserver
 import adapter
 finger=FingerObserver();source=read(O/'word100_rtmw_joint_learning_20261108/records/NIA_SL_WORD0739_REAL01_F.json');assert sha(source['raw_path'])==source['raw_sha256'];donor=np.load(source['raw_path'])['xy_px'][79,112:133].copy();donor-=donor[0];rows=[]
 inputs=[r for r in read(P/'INPUTS.json')['rows']if r['role']=='development']+read(U/'CONFIRMATION_INPUTS.json')['rows']
 for r in inputs:
  raw=dict(np.load(r['raw']));lo,hi=r['source_interval'];ids=np.flatnonzero((raw['timestamp_us']/1e6>=lo)&(raw['timestamp_us']/1e6<=hi))
  for kind in ['original','right_fist_control','left_fist_control']:
   d={k:v.copy()for k,v in raw.items()}
   if kind!='original':
    start=112 if kind.startswith('right') else 91
    for i in ids:
     p=d['xy_px'][i,start:start+21];u=p[9]-p[0];v=donor[9];theta=np.arctan2(u[1],u[0])-np.arctan2(v[1],v[0]);co,si=np.cos(theta),np.sin(theta);rot=np.array([[co,-si],[si,co]],np.float32);q=donor@rot.T*np.linalg.norm(u)/max(np.linalg.norm(v),1e-6)+p[0];d['xy_px'][i,start:start+21]=q
   ch=adapter.convert(d['xy_px'],d['score'],d['timestamp_us'],d['frame_id'],int(d['source_width']),int(d['source_height']));f,q,coarse=factory(ch)
   details={}
   for hand,col in [('R',1),('L',2)]:
    states=finger.infer(ch['tcn280'],hand)['states'];sel=states[ids]
    details[hand]=dict(coarse_open_mean=float(coarse[ids,col].mean()),coarse_open_min=float(coarse[ids,col].min()),all_four_open_frames=int((sel==1).all(1).sum()),all_four_closed_frames=int((sel==0).all(1).sum()),any_closed_frames=int((sel==0).any(1).sum()),frames=len(ids))
   rows.append(dict(clip_id=r['clip_id'],kind=kind,hands=details))
 write(U/'SHAPE_CONTROLS.json',dict(rows=rows,source_fist=dict(clip_id=source['clip_id'],frame=79,direct_source_label='right_four_closed from existing source-reviewed shared supervision'),scope='Coordinate transplantation diagnostic only; no physical-video error accuracy, no training',thresholds_changed=False))
 print('SHAPE CONTROLS COMPLETE',len(rows))
if __name__=='__main__':main()
