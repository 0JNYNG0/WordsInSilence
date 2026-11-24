"""Describe whether added increments affect distances or reference bounds."""
import json,hashlib
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
assert read(U/'VERIFY.json')['status']=='PASS'
assert not (U/'COMPONENT_DIAGNOSIS.json').exists()
r=read(U/'RESULT.json');groups={'angle':slice(0,15),'palm':slice(15,17),'elbow':slice(17,19),'wrist':slice(19,21)}
result={}
for cohort,kinds in [('identity',{'identity'}),('normal',{'identity','fast075','slow15','pace','pause'}),('short',{'skip3_25','skip3_50','skip3_75'})]:
    out={g:dict(comparable=0,distance_increased=0,bound_increased=0,both_increased=0,levels_only_bad=0,sequence_only_bad=0) for g in groups}
    cases=0
    for row in r['cross']:
        if row['kind'] not in kinds:continue
        z=load(U/'cross'/(row['name']+'.npz'));c=load(U/'calibration'/(row['query']+'__'+row['reference']+'.npz'));cases+=1
        for g,sl in groups.items():
            ok=z['comparable'][:,:,sl];d=z['sequence_distance'][:,:,sl]>z['levels_distance'][:,:,sl]+1e-6
            b=c['sequence_threshold'][:,:,sl]>c['levels_threshold'][:,:,sl]+1e-6
            vals={'comparable':ok,'distance_increased':ok&d,'bound_increased':ok&b,'both_increased':ok&d&b,
                'levels_only_bad':ok&z['levels_bad'][:,:,sl]&~z['sequence_bad'][:,:,sl],
                'sequence_only_bad':ok&z['sequence_bad'][:,:,sl]&~z['levels_bad'][:,:,sl]}
            for k,x in vals.items():out[g][k]+=int(x.sum())
    result[cohort]={'cases':cases,'groups':out}
result.update(interpretation='Components share observations. Same verdict need not mean transition distance absent. Joint max and its separately calibrated max bound may both change; no causal model improvement claim.',sha256={n:hashlib.sha256((U/n).read_bytes()).hexdigest() for n in ('RESULT.json','VERIFY.json')})
with (U/'COMPONENT_DIAGNOSIS.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result['identity'],ensure_ascii=False))
