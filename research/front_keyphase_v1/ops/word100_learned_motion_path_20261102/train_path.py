import time,traceback,torch,numpy as np
from path_model import Q,N,PathNet,records_and_sequences,batch,predict,read
from prepare_real_fit import exclusive,sha,write

def main():
 torch.set_num_threads(4);assert read(Q/'PREFLIGHT.json')['status']=='PASS';contract=read(Q/'CONTRACT.json')
 assert sha(N/'DATASET.json')==contract['data_sha256'];seq,y,mask,records,groups=records_and_sequences();yy=torch.tensor(y);mm=torch.tensor(mask)
 pos=(yy*mm).sum(0);neg=((1-yy)*mm).sum(0);weight=(neg/pos.clamp_min(1)).clamp(.25,8);runs=[]
 for condition in contract['conditions']:
  for seed in contract['seeds']:
   d=Q/'runs'/f'{condition}_{seed}';assert not d.exists();d.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
   model=PathNet(condition);model.initialize(seed);initial={k:v.clone() for k,v in model.state_dict().items()};optimizer=torch.optim.AdamW(model.parameters(),lr=contract['lr'],weight_decay=.01)
   history=[];start=time.perf_counter()
   for epoch in range(contract['epochs']):
    model.train();losses=[]
    for ix in np.array_split(rng.permutation(len(seq)),16):
     x,l=batch([seq[i] for i in ix]);z=model(x,l);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,yy[ix],pos_weight=weight,reduction='none')*mm[ix]).sum()/mm[ix].sum()
     pairs=[rng.choice(groups[i],size=2,replace=False) for i in rng.choice(len(groups),size=8,replace=False)];fi=np.array(pairs).ravel();px,pl=batch([seq[i] for i in fi])
     pp=model(px,pl).sigmoid().reshape(8,2,6);pm=mm[fi].reshape(8,2,6).all(1)
     loss+=.1*((pp[:,0]-pp[:,1]).square()*pm).sum()/pm.sum();assert torch.isfinite(loss)
     optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
    if epoch%30==0 or epoch==119:
     history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))));write(Q/'TRAIN_PROGRESS.json',dict(condition=condition,seed=seed,epoch=epoch+1));print('training',condition,seed,epoch+1,flush=True)
   model.eval();p=predict(model,seq);torch.save(model.state_dict(),d/'model.pt');np.savez_compressed(d/'fit_predictions.npz',p=p,y=y,mask=mask)
   update=sum(float((v-initial[k]).square().sum()) for k,v in model.state_dict().items())**.5;assert update>0
   rec=dict(condition=condition,seed=seed,epochs=120,fit_windows=1044,history=history,training_seconds=time.perf_counter()-start,
    checkpoint_sha256=sha(d/'model.pt'),weight_update_L2=update,initial_motion_sha256=sha(N/'runs'/f'real_variants_motion_inputs_{seed}/model.pt'),
    fit_only=True,new_backbone_update=False,private_data_access=False)
   exclusive(d/'COMPLETE.json',rec);runs.append(rec);print('trained',condition,seed,flush=True)
 exclusive(Q/'TRAINING_RESULT.json',dict(status='SIX_PATH_FINETUNES_COMPLETE',runs=runs))

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (Q/'TRAIN_FIRST_FAILURE.txt').exists():(Q/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
