"""Fixed component ablation; no task-level interpretation of numeric labels."""
from prepare import *
import torch, time, os
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(63,32),nn.ReLU())
        self.ordinary=nn.Linear(32,3);self.numeric=nn.Linear(32,3);self.fingers=nn.Linear(32,5)
    def forward(self,x):
        h=self.shared(x);return self.ordinary(h),self.numeric(h),self.fingers(h)

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    cfg=read(U/'FROZEN.json');assert not (U/'RESULTS.json').exists()
    for p,h in dict(cfg['source_pins'],**cfg['protected']).items():assert sha(p)==h,p
    data=read(U/'INPUT_MANIFEST.json');rows=data['ordinary'];aux=data['auxiliary'];z=np.load(U/'INPUTS.npz')
    fit=np.array([r['role']=='fit' for r in rows]);valid=np.array([r['input_available'] for r in rows]);ix=np.where(fit&valid)[0];dev=np.where(~fit)[0]
    ai=np.array([i for i,r in enumerate(aux) if r['input_available']]);mu=z['ordinary'][ix].mean(0);sd=z['ordinary'][ix].std(0).clip(.05)
    x=torch.tensor((z['ordinary']-mu)/sd);ax=torch.tensor((z['auxiliary']-mu)/sd)
    truth=np.array([r['y'] for r in rows]);y=torch.tensor(truth.clip(0),dtype=torch.float32);ym=torch.tensor(truth>=0,dtype=torch.float32)
    ay0=np.array([r['finger_states'] for r in aux]);ay=torch.tensor(ay0.clip(0),dtype=torch.float32);am=torch.tensor(ay0>=0,dtype=torch.float32)
    ac=torch.tensor([r['shape_class'] for r in aux]);pos=(y[ix]*ym[ix]).sum(0);neg=((1-y[ix])*ym[ix]).sum(0);pw=(neg/pos.clamp_min(1)).clamp(.1,10)
    write(U/'NORMALIZATION.json',dict(mean=mu.tolist(),std=sd.tolist(),positive_weight=pw.tolist(),fit_only=True))
    write(U/'CURRENT_EXECUTION.json',dict(status='RUNNING',pid=os.getpid(),experiment='three paired conditions x three seeds'))
    runs=[]
    for seed in cfg['seeds']:
        for condition in cfg['conditions']:
            name=f'{condition}_{seed}';dest=U/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'RESULT.json').exists():
                runs.append(read(dest/'RESULT.json'));continue
            torch.manual_seed(seed);m=Net();opt=torch.optim.Adam(m.parameters(),lr=cfg['lr']);start=time.perf_counter()
            for step in range(cfg['steps']):
                opt.zero_grad();out,_,_=m(x[ix]);loss=(F.binary_cross_entropy_with_logits(out,y[ix],reduction='none',pos_weight=pw)*ym[ix]).sum()/ym[ix].sum()
                _,cl,fi=m(ax[ai])
                if condition=='plus_numeric_name':loss=loss+F.cross_entropy(cl,ac[ai])
                if condition=='plus_finger_attributes':loss=loss+(F.binary_cross_entropy_with_logits(fi,ay[ai],reduction='none')*am[ai]).sum()/am[ai].sum()
                loss.backward();opt.step()
            elapsed=time.perf_counter()-start;m.eval()
            with torch.inference_mode():p=m(x)[0].sigmoid().numpy()
            cells=[]
            for i in dev:
                for j,t in enumerate(truth[i]):
                    if t<0:continue
                    prob=float(p[i,j]);available=bool(valid[i]);pred=int(prob>=.5) if available else None
                    confidence=available and (prob>=cfg['confidence'] or prob<=1-cfg['confidence'])
                    cells.append(dict(clip_id=rows[i]['clip_id'],word=rows[i]['word'],person=rows[i]['person'],frame=rows[i]['frame'],side=rows[i]['side'],attribute=j,truth=int(t),prob=prob,pred=pred,available=available,confident=bool(confidence),correct=bool(pred==t)))
            def summary(cc):
                known=[r for r in cc if r['available']]; confident=[r for r in cc if r['confident']]
                # Positive/negative mean recalls avoids interpreting class imbalance as success.
                recalls=[sum(r['correct'] for r in cc if r['truth']==t)/sum(r['truth']==t for r in cc) for t in (0,1) if any(r['truth']==t for r in cc)]
                return dict(total=len(cc),correct=sum(r['correct'] for r in cc),unavailable=len(cc)-len(known),confident=len(confident),confident_wrong=sum(not r['correct'] for r in confident),balanced_accuracy=float(np.mean(recalls)) if recalls else None)
            t0=time.perf_counter()
            with torch.inference_mode():m(x[dev])
            ms=(time.perf_counter()-t0)*1000
            torch.save(dict(state_dict=m.state_dict(),mean=mu.tolist(),std=sd.tolist(),condition=condition,seed=seed),dest/'model.pt')
            run=dict(condition=condition,seed=seed,train_seconds=elapsed,parameters=sum(t.numel() for t in m.parameters()),dev_batch_ms=ms,summary=summary(cells),by_hand={s:summary([r for r in cells if r['side']==s]) for s in ['R','L']},by_attribute={str(j):summary([r for r in cells if r['attribute']==j]) for j in range(3)},by_word={w:summary([r for r in cells if r['word']==w]) for w in sorted({r['word'] for r in cells})},cells=cells,model_sha256=sha(dest/'model.pt'))
            write(dest/'RESULT.json',run);runs.append(run)
            print(name,json.dumps(run['summary']),flush=True)
    write(U/'RESULTS.json',dict(runs=runs,baseline='same hand-local frame network trained only on ordinary sparse labels; not existing production temporal model',main_model_replaced=False,limitations=cfg['scope_limits']))
    write(U/'CURRENT_EXECUTION.json',dict(status='FIXED_TRAINING_COMPLETE_PENDING_VERIFICATION',running=None))
if __name__=='__main__':main()
