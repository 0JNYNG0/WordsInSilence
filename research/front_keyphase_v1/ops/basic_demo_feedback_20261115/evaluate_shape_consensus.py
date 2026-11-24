from shape_consensus import *
from finger_runtime import classify,merge_observers

def main():
 torch.set_num_threads(2);net=ShapeConsensus();d=np.load(SOURCE/'INPUTS.npz');ps={}
 for kind,models in net.finger.models.items():
  pp=[]
  for m,mu,sd in models:
   x=(d['x']-mu)/sd if kind=='shared_finger_chain'else(d['x_old']-mu)/sd
   if kind!='shared_finger_chain':x[:,:,63:]=0
   with torch.inference_mode():p=m(torch.tensor(x)).sigmoid().numpy()
   if kind!='shared_finger_chain':p=p[:,1:]
   pp.append(classify(p,d['availability']))
  ps[kind]=np.array(pp)
 f,_=merge_observers(ps['expanded_whole_hand'],ps['shared_finger_chain'],ps['old_v2']);a,_=net.before.windows(d['x'],d['availability']);b,_=net.expanded.windows(d['x'],d['availability']);s,base=combine(f,a,b)
 target=np.load(V/'temporal/predictions.npz');dev=target['dev'];y=target['target'];metrics={}
 for k,q in [('existing',base),('supplemented',s)]:metrics[k]=dict(total=int(dev.sum()),correct=int((dev&(q==y)).sum()),wrong=int((dev&(q>=0)&(q!=y)).sum()),unknown=int((dev&(q<0)).sum()))
 metrics['old_correct_to_wrong']=int((dev&(base==y)&(s>=0)&(s!=y)).sum());metrics['old_correct_to_unknown']=int((dev&(base==y)&(s<0)).sum())
 np.savez_compressed(V/'CONSENSUS_DEVELOPMENT.npz',states=s,baseline=base,fingers=f,target=y,dev=dev)
 write(V/'CONSENSUS_DEVELOPMENT.json',metrics);print(metrics,flush=True)
 sys.path.insert(0,str(O/'word100_rtmw_shared_input_20261108'));import adapter
 source=read(O/'word100_rtmw_joint_learning_20261108/records/NIA_SL_WORD0739_REAL01_F.json');donor=np.load(source['raw_path'])['xy_px'][79,112:133].copy();donor-=donor[0];out=[]
 inputs=[r for r in read(U/'INPUTS.json')['rows']if r['role']=='development']+read(U/'source_corrected/CONFIRMATION_INPUTS.json')['rows']
 for r in inputs:
  raw=dict(np.load(r['raw']));lo,hi=r['source_interval'];ids=np.flatnonzero((raw['timestamp_us']/1e6>=lo)&(raw['timestamp_us']/1e6<=hi));original=None
  for kind in ['original','right_fist','left_fist','missing']:
   z={k:v.copy()for k,v in raw.items()}
   if kind=='missing':z['xy_px'][:,91:]=np.nan
   elif kind!='original':
    start=112 if kind=='right_fist'else 91
    for i in ids:
     p=z['xy_px'][i,start:start+21];u=p[9]-p[0];v=donor[9];theta=np.arctan2(u[1],u[0])-np.arctan2(v[1],v[0]);c,sn=np.cos(theta),np.sin(theta);rot=np.array([[c,-sn],[sn,c]],np.float32);z['xy_px'][i,start:start+21]=donor@rot.T*np.linalg.norm(u)/max(np.linalg.norm(v),1e-6)+p[0]
   ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));ss={h:net.sequence(ch['tcn280'],h)['states']for h in ['L','R']}
   if original is None:original=ss
   opposite=None
   if kind in ['right_fist','left_fist']:h='L'if kind=='right_fist'else'R';opposite=bool(np.array_equal(ss[h],original[h]));assert opposite
   out.append(dict(clip_id=r['clip_id'],kind=kind,opposite_unchanged=opposite,hands={h:dict(extended=int((q[ids]==0).sum()),flexed=int((q[ids]==1).sum()),mixed=int((q[ids]==2).sum()),unknown=int((q[ids]<0).sum()),frames=len(ids))for h,q in ss.items()}))
 write(V/'CONSENSUS_CONTROLS.json',dict(rows=out,scope='Previously-seen sources for diagnostic only; artificial coordinate substitution, not real error-video accuracy.'))
 print('CONSENSUS DIAGNOSTICS',len(out))
if __name__=='__main__':main()
