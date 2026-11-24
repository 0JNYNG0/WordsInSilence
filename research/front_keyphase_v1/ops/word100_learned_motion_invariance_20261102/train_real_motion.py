import time,traceback,torch,numpy as np
from real_motion import N,M,read,MotionNet,batch,predict,filter_input,records_and_sequences
from prepare_real_fit import exclusive,sha,write
def main():
 torch.set_num_threads(4);assert read(N/'INPUT_AUDIT.json')['status'].startswith('PASS');contract=read(N/'TRAINING_CONTRACT.json');seq,y,mask,records,groups=records_and_sequences()
 exclusive(N/'DATASET.json',dict(rows=records,y=y.tolist(),mask=mask.tolist(),pairs=groups,all_fit=True)) if not (N/'DATASET.json').exists() else None
 yy=torch.tensor(y);mm=torch.tensor(mask);pos=(yy*mm).sum(0);neg=((1-yy)*mm).sum(0);weight=(neg/pos.clamp_min(1)).clamp(.25,8);runs=[]
 for condition in contract['conditions']:
  data=[filter_input(x,condition) for x in seq]
  for seed in contract['seeds']:
   d=N/'runs'/f'{condition}_{seed}'
   if (d/'COMPLETE.json').exists():runs.append(read(d/'COMPLETE.json'));continue
   assert not d.exists(),'Preserve any partial run';d.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
   model=MotionNet('ordered_gru');checkpoint=M/'runs'/f'ordered_gru_{seed}/model.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
   initial={k:v.clone() for k,v in model.state_dict().items()};optimizer=torch.optim.AdamW(model.parameters(),lr=contract['lr'],weight_decay=.01);history=[];start=time.perf_counter()
   for epoch in range(contract['epochs']):
    model.train();losses=[]
    for ix in np.array_split(rng.permutation(len(data)),16):
     x,l=batch([data[i] for i in ix]);z=model(x,l);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,yy[ix],pos_weight=weight,reduction='none')*mm[ix]).sum()/mm[ix].sum()
     # Every pair is the same reviewed event in actual speed variants. No DEV.
     selected=rng.choice(len(groups),size=8,replace=False);pairs=[rng.choice(groups[i],size=2,replace=False) for i in selected];fi=np.array(pairs).ravel()
     px,pl=batch([data[i] for i in fi]);pp=model(px,pl).sigmoid().reshape(8,2,6);pm=mm[fi].reshape(8,2,6).all(1);consistency=(((pp[:,0]-pp[:,1])**2)*pm).sum()/pm.sum()
     loss=loss+.1*consistency;assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
    if epoch%30==0 or epoch==119:
     history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))));write(N/'TRAIN_PROGRESS.json',dict(condition=condition,seed=seed,epoch=epoch+1));print('training',condition,seed,epoch+1,flush=True)
   model.eval();p=predict(model,data);torch.save(model.state_dict(),d/'model.pt');np.savez_compressed(d/'fit_predictions.npz',p=p,y=y,mask=mask)
   update=float(sum(((v-initial[k])**2).sum().item() for k,v in model.state_dict().items())**.5)
   assert update>0
   rec=dict(condition=condition,seed=seed,epochs=120,fit_windows=len(data),history=history,training_seconds=time.perf_counter()-start,checkpoint_sha256=sha(d/'model.pt'),
    initialized_from_sha256=sha(checkpoint),weight_update_L2=update,new_backbone_update=False,development_not_read=True)
   exclusive(d/'COMPLETE.json',rec);runs.append(rec);print('trained',condition,seed,flush=True)
 exclusive(N/'TRAINING_RESULT.json',dict(status='SIX_ACTUAL_VARIANT_MOTION_FINETUNES_COMPLETE',runs=runs,labels_are_public_physical_episodes_not_semantic_error_gold=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (N/'TRAIN_FIRST_FAILURE.txt').exists():(N/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
