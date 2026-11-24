"""Recount new frozen-head scores and reused pair without encoder inference."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=read(HERE/'result.json'); plan=read(HERE/'plan.json')
assert result['errors']==0 and result['new_encoder_forwards']==12 and result['reused_outputs']==2
assert sha(HERE/'plan.json')==result['plan_sha256']
assert sha(HERE/'public_scores.npz')==result['arrays_sha256']
assert sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256']
payload=torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=True)
classes=payload['classes']; weight=payload['state_dict']['head.weight'].numpy(); bias=payload['state_dict']['head.bias'].numpy()
previous=read(plan['reused_result'])
inputs={r['id']:r for r in plan['inputs']}
expected={(r['id'],c) for r in plan['inputs'] for c in ('whole_clip','provided_annotation')}
assert len(result['rows'])==14 and {(r['id'],r['condition']) for r in result['rows']}==expected
scalar_count=rank_count=0; max_error=0.
with np.load(HERE/'public_scores.npz',allow_pickle=False) as saved:
    assert saved['classes'].tolist()==classes
    for row in result['rows']:
        if row['status']=='REUSED':
            ref=next(r for r in previous['rows'] if r['condition']==row['prior_condition'])
            assert row['scopes']==ref['scopes']
            continue
        assert row['status']=='EVALUATED'
        prefix=row['id']+'__'+row['condition']+'__'
        feature=saved[prefix+'features']; scores=saved[prefix+'scores']
        scalar=np.array([math.fsum(float(w)*float(z) for w,z in zip(weights,feature))+float(b) for weights,b in zip(weight,bias)])
        error=float(np.max(np.abs(scalar-scores)))
        assert error<1e-10
        max_error=max(max_error,error); scalar_count+=len(scalar)
        y=classes.index(row['reference_label'])
        for scope, ids in result['scopes'].items():
            got=row['scopes'][scope]
            ranked=np.array(ids)[np.argsort(-scalar[ids],kind='stable')].tolist()
            assert got['prediction']==classes[ranked[0]]
            assert got['source_label_agreement']==(ranked[0]==y)
            assert got['source_label_rank']==ranked.index(y)+1
            margin=float(scalar[y]-max(scalar[i] for i in ids if i!=y))
            assert abs(got['margin']-margin)<1e-10
            rank_count+=1
        item=inputs[row['id']]
        with np.load(item['cache'],allow_pickle=False) as raw:
            ts=raw['timestamp_us']
            span=[0,len(ts)-1] if row['condition']=='whole_clip' else item['annotation_rows']
            assert row['source_rows']==span
            assert row['timestamps_us']==[int(ts[span[0]]),int(ts[span[1]])]
            assert row['input_shape']==[span[1]-span[0]+1,280]
aggregate={}
for scope in ('project18','full93'):
    pairs=[]
    for item in plan['inputs']:
        rr={r['condition']:r['scopes'][scope] for r in result['rows'] if r['id']==item['id']}
        pairs.append(rr)
    aggregate[scope]={'pairs':len(pairs), 'whole_correct':sum(r['whole_clip']['source_label_agreement'] for r in pairs),
                      'annotation_correct':sum(r['provided_annotation']['source_label_agreement'] for r in pairs),
                      'wrong_to_right':sum(not r['whole_clip']['source_label_agreement'] and r['provided_annotation']['source_label_agreement'] for r in pairs),
                      'right_to_wrong':sum(r['whole_clip']['source_label_agreement'] and not r['provided_annotation']['source_label_agreement'] for r in pairs),
                      'source_margin_increases':sum(r['provided_annotation']['margin']>r['whole_clip']['margin'] for r in pairs)}
audit={'status':'PASS','independent_scalar_scores':scalar_count,'independent_scope_rankings':rank_count,
       'reused_outputs_checked_against_original':2,'max_scalar_error':max_error,'aggregate':aggregate,
       'result_sha256':sha(HERE/'result.json'),'script_sha256':sha(__file__)}
with (HERE/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:
    json.dump(audit,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(audit,ensure_ascii=False))
