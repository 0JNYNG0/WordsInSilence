"""Public-only read-only diagnosis; no user footage, fitting, or runtime selection."""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent;J=O/'word100_rtmw_joint_learning_20261108';D=O/'word100_action_feedback_demo_20261108';G=O/'word100_learned_guide_feedback_20261102'
sys.path.insert(0,str(J));from runtime_candidate import Candidate,A
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ranks(logits,f,truth):
 logits=np.asarray(logits);idx=np.argsort(-logits,kind='stable');prob=np.exp(logits-logits.max());prob/=prob.sum();return dict(top3=[f['labels'][i] for i in idx[:3]],truth_rank=int(np.flatnonzero(idx==truth)[0]+1),truth_probability=float(prob[truth]),top1_correct=int(idx[0])==truth)
def main():
 assert not (U/'COUNTERFACTUAL.json').exists();U.mkdir(exist_ok=True);torch.set_num_threads(4);f=read(J/'FROZEN.json');model=Candidate();rows=[r for r in read(G/'INPUT_MANIFEST.json')['rows'] if r['role']=='development'];baseline={r['clip_id']:r for r in read(J/'tcn_runs/finetune_C_20261013/PREDICTIONS_full.json')};upper={r['clip_id']:r for r in read(J/'tcn_runs/finetune_C_20261013/PREDICTIONS_upper_mask.json')}
 conditions=['left_hand_wrist_absent','left_arm_hand_attached_to_right_all','left_arm_hand_attached_to_right_5frames']
 write(U/'FROZEN.json',dict(scope='Answer whether absent-side false coordinates can affect current TCN and identify missing safeguards',public_clips=[r['clip_id'] for r in rows],conditions=conditions,baseline='Read already completed saved predictions, never re-infer baseline',model='Current demo fixed finetune_C_20261013, not the preserved original C',interpretation='Controlled coordinate perturbations, not observed real-user errors or an accuracy improvement',training=False,private_used=False,production_changes=False,pins={str(p):sha(p) for p in [J/'tcn_runs/finetune_C_20261013/model.pt',D/'consumer.py',O/'word100_rtmw_shared_input_20261108/adapter.py',D/'app.js']}))
 out=[]
 for r in rows:
  cid=r['clip_id'];record=read(J/'records'/f'{cid}.json');assert sha(record['raw_path'])==record['raw_sha256'];z=np.load(record['raw_path']);truth=f['classes'].index(r['word_id']);row=dict(clip_id=cid,word=r['word_id'],label=r['label'],baseline=ranks(baseline[cid]['logits'],f,truth),saved_upper_mask=ranks(upper[cid]['logits'],f,truth),conditions={})
  for condition in conditions:
   xy=z['xy_px'].copy();score=z['score'].copy();ts=z['timestamp_us'];n=len(ts)
   if condition=='left_hand_wrist_absent':xy[:,[9]+list(range(91,112))]=np.nan
   else:
    if condition.endswith('5frames'):
     lo,hi=r['intervals'][0];centre=int(np.argmin(abs(ts-(lo+hi)/2*1e6)));idx=np.arange(max(0,centre-2),min(n,centre+3))
    else:idx=np.arange(n)
    xy[idx,7]=xy[idx,8];xy[idx,9]=xy[idx,10];xy[idx,91:112]=xy[idx,112:133]
   a=A.convert(xy,score,ts,z['frame_id'],int(z['source_width']),int(z['source_height']));x=a['tcn280']
   with torch.inference_mode():logit=model.tcn(torch.from_numpy(x[None]),torch.ones(1,len(x)))[0].numpy()
   row['conditions'][condition]=ranks(logit,f,truth)
  out.append(row)
 summary={}
 for w in f['feedback']['words']:
  rs=[r for r in out if r['word']==w];summary[w]=dict(label=rs[0]['label'],clips=len(rs),baseline_correct=sum(r['baseline']['top1_correct'] for r in rs),saved_upper_correct=sum(r['saved_upper_mask']['top1_correct'] for r in rs),conditions={k:dict(top1_correct=sum(r['conditions'][k]['top1_correct'] for r in rs),truth_ranks=[r['conditions'][k]['truth_rank'] for r in rs]) for k in conditions})
 write(U/'COUNTERFACTUAL.json',dict(rows=out,summary=summary,threshold_tuning=False,production_changed=False,private_used=False));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
