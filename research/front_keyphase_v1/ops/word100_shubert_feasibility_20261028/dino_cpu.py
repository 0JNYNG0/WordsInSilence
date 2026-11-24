"""Pinned official DINOv2, official SHuBERT teachers, isolated CPU inference."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch

U=Path(__file__).resolve().parent
sys.path.insert(0,str(U/'vendor/dinov2'))
from dinov2.hub.backbones import dinov2_vits14_reg

def load_teacher(kind):
    path=U/'weights'/f'{kind}_dinov2_checkpoint.pth'
    if torch.serialization.get_unsafe_globals_in_checkpoint(path):
        raise RuntimeError('Unexpected globals in official DINO checkpoint')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    teacher=checkpoint['teacher']
    state={k.removeprefix('backbone.'):v for k,v in teacher.items() if 'dino_head' not in k}
    model=dinov2_vits14_reg(pretrained=False)
    model.pos_embed=torch.nn.Parameter(torch.zeros(1,257,384))
    expected=model.state_dict()
    audit={'kind':kind,'missing':sorted(set(expected)-set(state)),
           'unexpected':sorted(set(state)-set(expected)),
           'shape_mismatch':[k for k in set(state)&set(expected) if state[k].shape!=expected[k].shape],
           'excluded_training_head_keys':[k for k in teacher if 'dino_head' in k]}
    if audit['missing'] or audit['unexpected'] or audit['shape_mismatch']:
        raise RuntimeError(str(audit))
    model.load_state_dict(state,strict=True)
    model.eval();model.requires_grad_(False)
    audit['parameters']=sum(p.numel() for p in model.parameters())
    audit['load']='strict=True CPU float32; all inference weights accounted for'
    return model,audit

def preprocess(images):
    assert images.dtype==np.uint8 and images.shape[1:]==(224,224,3)
    batch=torch.from_numpy(np.array(images,copy=True)).permute(0,3,1,2).float().div_(255)
    return (batch-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]

def extract(model,images,batch_size=8):
    result=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            x=preprocess(images[start:start+batch_size])
            y=model(x).cpu().numpy()
            assert y.shape==(len(x),384) and np.isfinite(y).all()
            result.append(y)
    return np.concatenate(result)

if __name__=='__main__':
    torch.set_num_threads(4)
    rows=[]
    for kind in ('face','hands'):
        t=time.perf_counter();model,audit=load_teacher(kind);audit['load_seconds']=time.perf_counter()-t
        # Constant synthetic input checks tensor plumbing only, not sign accuracy or video latency.
        images=np.full((2,224,224,3),128,dtype=np.uint8)
        a=extract(model,images);b=extract(model,images)
        audit['synthetic_repeat_max_abs']=float(np.max(np.abs(a-b)))
        audit['synthetic_finite']=bool(np.isfinite(a).all())
        audit['scope']='Plumbing/strict load only; no public video performance result'
        rows.append(audit);print(json.dumps(audit),flush=True)
        del model
    (U/'DINO_LOAD_AUDIT.json').write_text(json.dumps({'rows':rows,'torch_threads':4},indent=2),encoding='utf-8')
