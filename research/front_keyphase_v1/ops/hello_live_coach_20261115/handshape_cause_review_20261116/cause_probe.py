from diagnose import *
def main():
 torch.set_num_threads(2);net=ShapeConsensus();rows=[]
 sources=json.loads((O/'basic_demo_feedback_20261115/local_shape/fresh/EXTRACTED.json').read_text(encoding='utf-8'))['rows']
 labels=json.loads((O/'basic_demo_feedback_20261115/local_shape/fresh/SOURCE_TRUTH.json').read_text(encoding='utf-8'))['rows']
 for src in sources:
  raw=load(src['raw_path']);x=channel(raw);arr={}
  for h in ['L','R']:
   q=inspect(net,x,h);arr.update({h+'_'+k:v for k,v in q.items()})
   for r in labels:
    if r['clip_id']==src['clip_id']and r['hand']==h:
     i=r['frame'];rows.append(dict(clip_id=r['clip_id'],frame=i,hand=h,expected=r['state'],state=int(q['states'][i]),before=int(q['before'][i]),expanded=int(q['expanded'][i]),fingers=q['fingers'][i].tolist(),available=q['available'][i].tolist(),old_prob=q['old_probs'][:,i].round(3).tolist(),new_prob=q['new_probs'][:,i].round(3).tolist()))
  np.savez_compressed(V/(src['clip_id']+'_predictions.npz'),**arr)
 write(V/'OTHER_PUBLIC_DIAGNOSIS.json',rows);print(json.dumps([r for r in rows if r['state']!=r['expected']],ensure_ascii=False))
 raw=load(U/'source/reference.npz');x=channel(raw);out=[]
 for mode in ['whole_hand_collapsed','index_chain_collapsed','palm_anchor_collapsed']:
  xx=x.copy();p=xx[:,:102].reshape(-1,51,2)
  if mode=='whole_hand_collapsed':p[:,30:51]=p[:,30:31]
  elif mode=='index_chain_collapsed':p[:,35:39]=p[:,35:36]
  else:p[:,39]=p[:,30]
  q=inspect(net,xx,'R');out.append(dict(case=mode,frame54_state=int(q['states'][54]),fingers=q['fingers'][54].tolist(),available=q['available'][54].tolist()))
 write(V/'DEGENERATE_BEFORE.json',out);print(out)
 # Previously source-reviewed wrong-keypoint vs own-chain classifier examples.
 lab=json.loads((O/'word100_finger_error_split_20261115/LABELS.json').read_text(encoding='utf-8'))['rows']
 rr=[]
 for cid,frame in [('NIA_SL_WORD2700_REAL10_F',66),('NIA_SL_WORD2718_REAL10_F',42)]:
  r=next(r for r in lab if r['clip_id']==cid and r['frame']==frame and r['hand']=='R');z=load(r['raw_path']);q=inspect(net,channel(z),'R');i=frame
  rr.append(dict(clip_id=cid,frame=i,image=r['image'],shape_truth=r['shape'],quality_truth=r['quality'],pred=q['fingers'][i].tolist(),coarse=int(q['states'][i])))
 write(V/'SOURCE_REVIEW_CASES.json',rr);print(rr)
if __name__=='__main__':main()
