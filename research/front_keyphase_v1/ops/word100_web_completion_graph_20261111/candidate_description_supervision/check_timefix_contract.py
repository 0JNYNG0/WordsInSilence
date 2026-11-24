from transfer_local_timefix_core import *
torch.set_num_threads(2);torch.set_num_interop_threads(1)
base=encoder();seen=[]
hook=base.input.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().numpy().copy()))
x=np.zeros((5,280),np.float32);x[:,279]=[0,.04,.08,.02,.05];q=np.zeros((5,100),np.float32)
encode_windows(base,x,q,np.array([[1,3]]));encode_windows(base,x,q,np.array([[1,3]]),True);hook.remove()
assert np.allclose(seen[0][0,:,279],[0,.08,.02]);assert np.allclose(seen[1][0,:,279],[0,.02,.08]);assert np.array_equal(x[:,279],np.array([0,.04,.08,.02,.05],np.float32))
pos=read(O/'word100_positive_direction_20261109/COMPARISON.json')['rows'];pm={(r['clip_id'],r['kind']):r for r in pos};paired=[]
for cid,kind in pm:
 if kind!='right_closed_hand_main':continue
 a=pm[cid,'crop_normal'];b=pm[cid,kind];aa=np.load(a['prediction']);bb=np.load(b['prediction']);assert np.array_equal(aa['x'][:,:18],bb['x'][:,:18]);assert np.array_equal(aa['x'][:,186:195],bb['x'][:,186:195]);assert np.array_equal(aa['x'][:,279],bb['x'][:,279]);paired.append(dict(clip_id=cid,normal=a['prediction'],changed=b['prediction'],body_and_dt_exactly_equal=True))
write(TQ/'TIME_UNIT_TEST.json',dict(at=stamp(),variable_frame_rate_forward=seen[0][0,:,279].tolist(),reverse=seen[1][0,:,279].tolist(),source_unchanged=True,synthetic_input_only=True,paired_crop_inputs=paired,note='Full-view normal and cropped edited controls are different detector input conditions. They are not an input-invariance pair. Use original crop_normal for shape-only invariance.',existing_results_rerun=False))
print('Delta-time and paired-control contracts verified')
