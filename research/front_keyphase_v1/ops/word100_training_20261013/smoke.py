"""Throwaway training-only pipeline/timing check. No development inputs."""
import time
import numpy as np
import torch
from common import U,OUT,read,write,pin_check
from train import model,batch,seed

f=pin_check();torch.set_num_threads(4);seed(f['train']['seed'])
clips=[]
for r in read(U/'MANIFEST.json'):
    p=OUT/'records'/f"{r['clip_id']}.json"
    if r['role']!='train_candidate' or not p.exists():continue
    rec=read(p)
    if not rec['channel_file']:continue
    with np.load(rec['channel_file']) as z:
        clips.append(dict(id=r['clip_id'],y=r['label_index'],signer=r['signer_id'],whole=z['x_whole'],context=z['x_context'],parity=len(clips)%2))
    if len(clips)==6:break
assert len(clips)==6
cc=[clips[i%6] for i in range(32)] # timing only, no saved trained model
x,p,y=batch(cc,1,'whole_context');m=model(f)
m.eval()
with torch.no_grad():
    a=m(x,p);b=m(torch.nn.functional.pad(x,(0,0,0,17)),torch.nn.functional.pad(p,(0,17)))
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    assert a.shape==(32,100)
m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
times=[];losses=[]
for i in range(4):
    st=time.perf_counter();opt.zero_grad();l=torch.nn.functional.cross_entropy(m(x,p),y);l.backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(l) and torch.isfinite(norm)
    opt.step();times.append(time.perf_counter()-st);losses.append(l.item())
write(U/'SMOKE.json',dict(status='PASS',train_clips=[c['id'] for c in clips],development_reads=0,
    disposable_optimizer_steps=4,checkpoint_saved=False,logits_shape=[32,100],eval_padding_invariance=True,
    step_seconds=times,losses=losses,median_step_seconds=float(np.median(times[1:])),
    estimate_two80epoch_training_hours=float(np.median(times[1:])*29*80*2/3600),
    estimate_note='Timing check only, repeated first6 training clips; excludes full-data lengths/evaluation/checkpoint/report. Not a performance or final runtime guarantee.'))
print(read(U/'SMOKE.json'))
