from motion_model import *
import time

def main():
    torch.set_num_threads(2);out=D/'edge';out.mkdir(exist_ok=True)
    assert not(out/'FROZEN.json').exists()
    write(out/'FROZEN.json',dict(reason='Two development trend errors remain. Centre distance can differ from the gap between the observed hand edges.',
        truth_sha=sha(D/'MOTION_TRUTH.json'),source_model_sha=sha(D/'motion_model.py'),
        change='Append changes of hand-extent gap along the target palm axes; original labels and split unchanged.',
        epochs=400,seeds=[42,43,44],lr=.003,weight_decay=.001,confidence=.7,
        development_not_final=True,missing='All42 hand points must be available for new extent features; otherwise unknown.',
        output='Three-state motion only; no automatic repeat or word completion.'))
    rows=read(D/'MOTION_TRUTH.json')['rows'];sources={r['clip_id']:r for r in read(U/'INPUTS.json')['rows']}
    channels={k:dict(np.load(r['channels']))for k,r in sources.items()}
    fit=[r for r in rows if r['split']=='fit'];xx=[feature(channels[r['clip_id']],r['start'],r['end'],True)for r in fit]
    assert all(x is not None for x in xx)
    x=np.stack(xx);y=np.array([r['label']for r in fit]);mu=x.mean(0);sd=np.maximum(x.std(0),1e-4)
    xt=torch.tensor((x-mu)/sd);yt=torch.tensor(y);weights=torch.tensor(len(y)/(3*np.bincount(y)),dtype=torch.float32)
    runs=[]
    for seed in [42,43,44]:
        torch.manual_seed(seed);m=Net('mlp32',20);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001);t=time.perf_counter()
        for _ in range(400):
            opt.zero_grad();loss=nn.functional.cross_entropy(m(xt),yt,weight=weights);loss.backward();opt.step()
        p=out/f'mlp32_{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mu,std=sd),p);runs.append(dict(seed=seed,seconds=time.perf_counter()-t,sha256=sha(p)))
    ob=Observer('mlp32',out,True);result=[]
    for r in read(D/'MOTION_COMPARISON.json')['rows']:
        result.append(dict(**r,edge_mlp=ob.predict(feature(channels[r['clip_id']],r['start'],r['end'],True))))
    dev=[r for r in result if r['split']!='fit'];summary={k:metrics(dev,k)for k in ['static_state_proxy','mlp32','edge_mlp']}
    write(out/'RESULTS.json',dict(summary=summary,rows=result,runs=runs,web_admitted=False))
    print(json.dumps(summary))

if __name__=='__main__':main()
