"""Restricted checkpoint read followed by clean architecture/state-dict loading."""
from pathlib import Path
import os,json,importlib,hashlib
U=Path(__file__).resolve().parent
(U/'runtime_config').mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(U/'runtime_config')
os.environ['YOLO_AUTOINSTALL']='False'
os.environ['YOLO_OFFLINE']='True'
os.environ['WANDB_DISABLED']='true'
import torch,yaml
from ultralytics import YOLO,settings
settings.update({'sync':False})

ALLOWED_PREFIXES=('torch.nn.modules.','ultralytics.nn.modules.','ultralytics.nn.tasks.','ultralytics.utils.')
def load_model():
    p=U/'yolo_hand_best.pt'
    src=json.loads((U/'MODEL_SOURCE.json').read_text('utf8'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==src['files'][p.name]['sha256']
    safe=[]
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(p):
        assert name.startswith(ALLOWED_PREFIXES),f'Unapproved pickle global: {name}'
        mod,key=name.rsplit('.',1);safe.append(getattr(importlib.import_module(mod),key))
    with torch.serialization.safe_globals(safe):
        ckpt=torch.load(p,map_location='cpu',weights_only=True)
    old=ckpt['ema'] if ckpt.get('ema') is not None else ckpt['model']
    cfg=dict(old.yaml);assert cfg['kpt_shape']==[21,3] and cfg['nc']==1
    path=U/'yolov8n-hand.yaml'
    text=yaml.safe_dump(cfg)
    if path.exists():assert path.read_text('utf8')==text
    else:path.write_text(text,encoding='utf8')
    # Build current trusted implementation; do not run the deserialized object's forward.
    model=YOLO(str(path),task='pose',verbose=False)
    model.model.load_state_dict(old.float().state_dict(),strict=True)
    model.model.kpt_shape=[21,3]
    model.model.names={0:'hand'};model.model.eval()
    return model,{'torch':torch.__version__,'ultralytics':importlib.import_module('ultralytics').__version__,
                  'checkpoint_kpt_shape':cfg['kpt_shape'],'parameters':sum(p.numel() for p in model.model.parameters()),
                  'restricted_load':True,'strict_state_dict':True,'device':'cpu'}
if __name__=='__main__':
    model,meta=load_model();print(json.dumps(meta,indent=2))
    (U/'MODEL_LOAD_CHECK.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
