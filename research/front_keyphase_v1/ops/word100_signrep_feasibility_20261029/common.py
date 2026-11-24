from pathlib import Path
import os, json, hashlib, time, sys
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['INTEL_OPENVINO_TELEMETRY_CONSENT'] = '0'
os.environ['OPENVINO_TELEMETRY_DISABLED'] = '1'
# Use OpenVINO's supported import-error fallback to its no-op telemetry stub.
# No opt-out event, account-level setting change or network call is needed.
sys.modules['openvino_telemetry'] = None
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'/'SignRep'))
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name, data):
    p = ROOT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): raise FileExistsError(f'Preserve existing result: {p}')
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def event(stage, **kw):
    row=dict(time=time.time(), stage=stage, **kw)
    with (ROOT/'EVENTS.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
def config():
    dims={'full_body_angles':44,'full_right_angles':82,'full_left_angles':82,'body_pose':183,'right_pose':63,'left_pose':63,'body_dist':792,'right_dist':165,'left_dist':165}
    return dict(ckpt_dir=None,backbone_params=dict(drop_path_rate=0.1,num_cls_tokens=0),out_dim=768,mask_ratio=0.,init_weights=False,
      cls_head_name='models.head_models.dict_head_v6_FINAL_thoughtful',
      cls_head_params=dict(latents_params=dict(hidden_dim=512,out_dim=384),total_frames=16,projection_params={k:dict(hidden_dim=None,output_dim=v) for k,v in dims.items()}))
def load_model():
    import torch
    from models.final_models.FINAL_hiera_latent_model_head_v25_active import Model
    assert sha(ROOT/'assets/ckpt.pt') == read(ROOT/'FROZEN.json')['checkpoint_sha256']
    torch.set_num_threads(4)
    model=Model(**config()).eval()
    ckpt=torch.load(ROOT/'assets/ckpt.pt',map_location='cpu',weights_only=True)
    result=model.load_state_dict(ckpt['model'],strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.requires_grad_(False)
    return model
def feature_wrapper(model):
    import torch
    class Features(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=model.backbone; self.fc=model.fc
        def forward(self,x): return self.fc(self.backbone(x,eval=True))
    return Features().eval()
def starts(n):
    if n < 16: raise ValueError('Too few real frames; no padding as observations')
    seq=list(range(0,n-15,2))
    if seq[-1] != n-16: seq.append(n-16)
    return seq
def norm(rgb):
    import numpy as np
    # Same operations as official A.Normalize: subtraction then reciprocal multiply.
    mean=np.array([.485,.456,.406],np.float32)*255
    denom=1/(np.array([.229,.224,.225],np.float32)*255)
    x=(rgb.astype(np.float32)-mean)*denom
    return x.transpose(0,3,1,2).copy()
def batch(frames, ids, batchsize):
    import numpy as np
    x=np.stack([frames[s:s+16].transpose(1,0,2,3) for s in ids])
    valid=len(x)
    if valid<batchsize: x=np.concatenate([x,np.repeat(x[-1:],batchsize-valid,axis=0)])
    return np.ascontiguousarray(x),valid
def diff(a,b):
    import numpy as np
    a=np.asarray(a,dtype=np.float64).reshape(len(a),-1);b=np.asarray(b,dtype=np.float64).reshape(len(b),-1)
    rel=np.linalg.norm(a-b,axis=1)/np.maximum(np.linalg.norm(a,axis=1),1e-12)
    co=(a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
    return dict(max_abs=float(np.max(np.abs(a-b))),max_relative_l2=float(rel.max()),min_cosine=float(co.min()),finite=bool(np.isfinite(b).all()))
