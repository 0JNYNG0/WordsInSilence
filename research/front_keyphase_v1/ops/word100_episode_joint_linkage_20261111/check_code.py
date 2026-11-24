from joint_core import *
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
torch.set_num_threads(4)
net=JointNet();q=np.ones((20,10,10),np.float32);bp=np.zeros((20,10,3),np.float32);bp[2:17,0,0]=.9;bp[2,0,1]=.9;bp[8,0,1]=.8;bp[16,0,2]=.9
cs=candidate_spans(bp,q,[]);ck('intermediate_start_does_not_erase_whole',any(c['a']==2 and c['b']==16 for c in cs));ck('retained_attempt_not_filtered_by_shape',any(c['source']=='retained' for c in candidate_spans(bp,q,[dict(a=1,b=19,axis=0,moving='R')])));ck('all_missing_no_new_candidates',not candidate_spans(bp,np.zeros_like(q),[]))
r=dict(frame_id=np.arange(20),annotations=[dict(axis=0,moving='R',events=[dict(onset=[2,3],offset=[7,8]),dict(onset=[9,10],offset=[15,16])],holds=[[17,19]],background=[[0,1]])]);ss,yy,ps,py=example_specs([r]);ck('fragment_repeat_and_mixed_labels',set(py)=={0,1} and set(yy)=={0,1,2,3});rs,ry,rps,rpy=example_specs([r],True);ck('reverse_labels_in_bounds',all(0<=a<=b<20 for i,a,b,k in rs))
h=torch.zeros((1,20,64));tq=torch.ones((1,20,10,10));fh=h.clone();f=net.segment(h,tq,fh,ss);ck('feature_dimension',f.shape==(len(ss),304));ck('quality_and_pair_shapes',net.quality(f).shape[1]==4 and net.pair(torch.zeros(1,609)).shape[1]==3)
write(U/'CODE_CHECK.json',dict(passed=True,checks=checks));write(U/'CODE_PINS.json',{str(p):sha(p) for p in [U/'joint_core.py',U/'train.py',U/'prepare_train.py',U/'check_code.py']});print('PASS',len(checks))
