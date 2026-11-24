from pathlib import Path
import sys
M=Path(__file__).resolve().parent
sys.path.insert(0,str(M.parent))
from common import *

class NetDirection(torch.nn.Module):
    """Axis2 shoulder-normalized projected NET displacement only."""
    def __init__(self):
        super().__init__();self.raw_scale=torch.nn.Parameter(torch.tensor(2.)+.01*torch.randn(()));self.raw_band=torch.nn.Parameter(torch.tensor(-2.)+.01*torch.randn(()))
    def forward(self,d):
        s=torch.nn.functional.softplus(self.raw_scale);b=torch.nn.functional.softplus(self.raw_band)
        return torch.stack([-s*d,s*b*torch.ones_like(d),s*d],-1)

def main():
    assert not(M/'FROZEN.json').exists();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    meta=read(HERE/'INPUTS_COMPLETE.json');assert sha(meta['path'])==meta['sha256'];z=np.load(meta['path']);ix=[i for i,r in enumerate(meta['rows'])if r['axis']==2];rows=[meta['rows'][i]for i in ix];d=torch.tensor(z['projected_geometry'][ix,40]);y=torch.tensor(z['y'][ix],dtype=torch.long)
    pools={}
    for o in [True,False]:
        pools[o]={t:[i for i,r in enumerate(rows)if r['split']=='train'and(r['origin']=='original_motion_replay')==o and r['target']==t]for t in [0,1,2]};assert all(pools[o].values())
    assert not({r['source_sha256']for r in rows if r['split']=='train'}&{r['source_sha256']for r in rows if r['split']=='probe'})
    cfg=dict(at=stamp(),seeds=[42,43,44],steps=800,lr=.05,weight_decay=.001,architecture='2learned parameters: s=softplus(raw_scale),b=softplus(raw_band), logits[-s*d,s*b,s*d]',
        input='raw physical shoulder-normalized projected_geometry column40 delta; never mean-center. No handshape/temporal context/target word/desired direction input.',
        labels='same reviewed axis2 FIT labels; balance3classes within original_motion_replay vs reviewed pools. Probe not for fitting or selection.',
        scope='Axis2 net displacement only; cannot determine full path/repetitions/contact/completion. Existing eligibility and direction_verdict unchanged.',
        objective='256examples/step,128 each origin pool; class-balanced cross entropy; finalstep only',
        gate='Normal/speed support and reversed/compound positive opposing evidence without new shape-only wrong-direction warning; all other axes/shape/spans unchanged. No G2 adoption from this alone.',
        pins={str(p):sha(p)for p in[Path(__file__),HERE/'INPUTS_COMPLETE.json',Path(meta['path']),HERE/'axis2_sign_audit/RESULTS.json']},private_access=False,final8=False,web_changed=False)
    write(M/'FROZEN.json',cfg);runs=[]
    for seed in cfg['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=NetDirection();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);log=[]
        for step in range(cfg['steps']):
            ids=[int(rng.choice(pools[o][int(rng.integers(3))]))for o in [True,False]for _ in range(128)];loss=torch.nn.functional.cross_entropy(net(d[ids]),y[ids]);assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            if step%200==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())))
        net.eval();dest=M/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),dest/'model.pt')
        with torch.inference_mode():pred=net(d).numpy()
        np.savez_compressed(dest/'predictions.npz',logits=pred,d=d.numpy(),y=y.numpy(),original_indices=np.array(ix));s=float(torch.nn.functional.softplus(net.raw_scale).detach());b=float(torch.nn.functional.softplus(net.raw_band).detach());runs.append(dict(seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),scale=s,band=b,log=log));print(seed,s,b,flush=True)
    write(M/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=runs,rows=rows,shared_models_unchanged=True))

if __name__=='__main__':main()
