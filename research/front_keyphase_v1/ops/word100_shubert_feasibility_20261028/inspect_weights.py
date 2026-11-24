"""Inspect official weights without executing arbitrary checkpoint pickle globals."""
from pathlib import Path
import json
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

U=Path(__file__).resolve().parent
rows=[]
for path in sorted((U/'weights').glob('*.p*')):
    if path.suffix not in ('.pt','.pth'):continue
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(path)
    row={'file':path.name,'bytes':path.stat().st_size,'unsafe_globals':unsafe}
    if not unsafe:
        with FakeTensorMode():
            data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        row['top_keys']=list(data)
        if isinstance(data.get('cfg'),dict):row['cfg']=data['cfg']
        if isinstance(data.get('args'),dict):row['args']=data['args']
        keys={}
        for name in ('model','teacher','student'):
            if isinstance(data.get(name),dict):
                keys[name]={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in data[name].items() if isinstance(v,torch.Tensor)}
        row['tensor_manifest']=keys
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='tensor_manifest'},default=str),flush=True)
(U/'WEIGHT_INSPECTION.json').write_text(json.dumps({'torch':torch.__version__,'rows':rows},default=str,indent=2),encoding='utf-8')
