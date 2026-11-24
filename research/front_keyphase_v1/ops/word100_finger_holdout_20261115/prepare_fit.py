from holdout_common import *
import torch,collections
sys.path[:0]=[str(P),str(D)]
from finger_core import local_sequence
from finger_model import windows,features
assert not(U/'TRAIN_FROZEN.json').exists()
I=[1,0,0,0];T=[1,1,1,0];F=[0,0,0,0]
spec={('WORD2700',3):{1:I,2:I,6:T,7:T,8:T},('WORD2700',6):{3:I,4:I,8:T,9:T,10:T},('WORD2700',9):{2:I,3:I,4:F,6:T,7:T,8:T,9:T},('WORD2718',3):{4:T,6:F,7:I,8:I,9:I},('WORD2718',6):{3:T,4:T,8:I,9:I,10:I},('WORD2718',9):{3:T,5:I,6:I,7:I,8:I}}
extra=[]
for r in read(U/'FIT_CROP_INPUTS.json')['rows']:
 y=spec[(r['word'],r['person'])].get(r['index'])
 if y is not None:extra.append(dict(**r,targets=[-1]+y,new=True,origin='Original RGB direct Codex visual review; no expert adjudication; no model labels'))
old=read(P/'INPUTS.json')['rows'];data=np.load(P/'INPUTS.npz');xx=[v for v in data['x_old']];yy=[v for v in data['y']];av=[v for v in data['availability']];keys={(r['clip_id'],r['frame'],r['side'])for r in old}
for r in extra:
 assert r['person']in[3,6,9] and r['role']=='fit' and (r['clip_id'],r['frame'],r['side'])not in keys
 assert sha(r['channels'])==r['channels_sha256'];z=np.load(r['channels']);x=z['feedback']if'feedback'in z.files else z['full'];fl,_=local_sequence(x,'R');_,valid=features(x,'R');t=int(np.flatnonzero(z['frame_id']==r['frame'])[0]);xx.append(windows(np.c_[fl,np.zeros((len(x),64),np.float32)],[t])[0]);yy.append(r['targets'][1:]);av.append(valid[t])
rows=old+extra;xx=np.array(xx);xx[:,:,63:]=0;yy=np.array(yy);av=np.array(av);assert av.shape==yy.shape,(av.shape,yy.shape)
np.savez_compressed(U/'TRAIN_INPUTS.npz',x=xx,y=yy,availability=av)
write(U/'FIT_LABELS.json',dict(rows=extra,unknown_frames_excluded=72-len(extra),long_fingers_only=True,existing_truth_unchanged=True))
write(U/'TRAIN_INPUTS.json',dict(rows=rows,fit_people=sorted({r['person']for r in rows if r['role']=='fit'}),new_anchors=len(extra)))
write(U/'TRAIN_FROZEN.json',dict(seeds=[42,43,44],steps=300,lr=.0001,weight_decay=.001,confidence=.85,init='Previous expanded_whole_hand same-seed weights; normalization unchanged',loss='FIT-only per-clip/per-finger state balanced BCE; no diagnostic17 or reserve18 gradient',candidates=['veto_only','require_agreement'],selection='Diagnostic17 and historical DEV only. No new confident states: existing output may only be retained or abstained. Reject if >10% previous correct cells lost on historical DEV or diagnostic left/front, or remaining diagnostic errors.',final_step_only=True,pins={str(U/n):sha(U/n)for n in ['FIT_LABELS.json','TRAIN_INPUTS.json','TRAIN_INPUTS.npz']}))
print('additional_RGB_labels',len(extra),'shapes',xx.shape,yy.shape)
