from local_shape import *
sys.path.insert(0,str(SOURCE));from finger_runtime import FingerObserver,classify,merge_observers

def main():
 torch.set_num_threads(2);d=np.load(SOURCE/'INPUTS.npz');rows=read(SOURCE/'INPUTS.json')['rows'];n=FingerObserver();ps={}
 for kind,models in n.models.items():
  pp=[]
  for m,mu,sd in models:
   x=(d['x']-mu)/sd if kind=='shared_finger_chain'else(d['x_old']-mu)/sd
   if kind!='shared_finger_chain':x[:,:,63:]=0
   with torch.inference_mode():p=m(torch.tensor(x)).sigmoid().numpy()
   if kind!='shared_finger_chain':p=p[:,1:]
   pp.append(classify(p,d['availability']))
  ps[kind]=np.array(pp)
 base,_=merge_observers(ps['expanded_whole_hand'],ps['shared_finger_chain'],ps['old_v2']);coarse=np.where((base==1).all(1),0,np.where((base==0).all(1),1,np.where((base>=0).all(1),2,-1)))
 predictions={k:dict(np.load(V/k/'predictions.npz'))for k in ['center','temporal']};t=predictions['temporal'];dev=t['dev'];y=t['target'];results={}
 for kind,r in predictions.items():
  s=r['states'];results[kind]=dict(old_correct_to_wrong=int(((coarse==y)&dev&(s>=0)&(s!=y)).sum()),old_correct_to_unknown=int(((coarse==y)&dev&(s<0)).sum()),added_correct=int(((coarse<0)&dev&(s==y)).sum()),added_wrong=int(((coarse<0)&dev&(s>=0)&(s!=y)).sum()),errors=[dict(rows[i],expected=int(y[i]),predicted=int(s[i]),old=int(coarse[i]))for i in np.flatnonzero(dev&(s>=0)&(s!=y))])
 write(V/'BASELINE_COMPARE.json',dict(baseline=dict(total=int(dev.sum()),correct=int((dev&(coarse==y)).sum()),wrong=int((dev&(coarse>=0)&(coarse!=y)).sum())),comparison=results))
 # Selection follows the frozen lower-error rule, not gratitude-only scores.
 selected=min(predictions,key=lambda k:(read(V/'DEVELOPMENT.json')['results'][k]['development']['wrong'],-read(V/'DEVELOPMENT.json')['results'][k]['development']['correct']))
 write(V/'CONTROL_FROZEN.json',dict(selected=selected,confirmation='The P14/15 shape predictions have already been seen in earlier baseline diagnostics; reuse is diagnostic, not an untouched test.',controls=['right_fist','left_fist','missing','unchanged_opposite_hand'],admission='No automatic whole-word pass; error findings must remain compatible with existing per-finger evidence'))
 sys.path.insert(0,str(O/'word100_rtmw_shared_input_20261108'));import adapter
 net=LocalShape(selected);source=read(O/'word100_rtmw_joint_learning_20261108/records/NIA_SL_WORD0739_REAL01_F.json');assert sha(source['raw_path'])==source['raw_sha256']
 donor=np.load(source['raw_path'])['xy_px'][79,112:133].copy();donor-=donor[0];out=[]
 inputs=[r for r in read(U/'INPUTS.json')['rows']if r['role']=='development']+read(U/'source_corrected/CONFIRMATION_INPUTS.json')['rows']
 for r in inputs:
  raw=dict(np.load(r['raw']));lo,hi=r['source_interval'];ids=np.flatnonzero((raw['timestamp_us']/1e6>=lo)&(raw['timestamp_us']/1e6<=hi));original=None
  for kind in ['original','right_fist','left_fist','missing']:
   z={k:v.copy()for k,v in raw.items()}
   if kind=='missing':z['score'][:,91:]=0
   elif kind!='original':
    start=112 if kind=='right_fist'else 91
    for i in ids:
     p=z['xy_px'][i,start:start+21];u=p[9]-p[0];v=donor[9];theta=np.arctan2(u[1],u[0])-np.arctan2(v[1],v[0]);c,s=np.cos(theta),np.sin(theta);rot=np.array([[c,-s],[s,c]],np.float32);z['xy_px'][i,start:start+21]=donor@rot.T*np.linalg.norm(u)/max(np.linalg.norm(v),1e-6)+p[0]
   ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));states={h:net.sequence(ch['tcn280'],h)[0]for h in ['L','R']}
   if original is None:original=states
   opposite_unchanged=None
   if kind in ['right_fist','left_fist']:
    hand='L'if kind=='right_fist'else'R';opposite_unchanged=bool(np.array_equal(states[hand],original[hand]))
   out.append(dict(clip_id=r['clip_id'],kind=kind,opposite_unchanged=opposite_unchanged,hands={h:dict(extended=int((s[ids]==0).sum()),flexed=int((s[ids]==1).sum()),mixed=int((s[ids]==2).sum()),unknown=int((s[ids]<0).sum()),frames=len(ids))for h,s in states.items()}))
 write(V/'CONTROLS.json',dict(rows=out,scope='Artificial keypoint substitution diagnostic, not real learner error videos, not training. Positive source frames not all individually labelled.'))
 print('BASELINE',results);print('CONTROLS',out)
if __name__=='__main__':main()
