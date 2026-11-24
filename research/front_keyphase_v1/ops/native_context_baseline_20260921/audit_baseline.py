"""Independent scalar score and source-ID recount, no encoder or preprocessing repeat."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(HERE/'plan.json');result=read(HERE/'result.json')
assert result['errors']==0 and result['new_encoder_forwards']==74 and result['training']==0
assert sha(HERE/'features_and_scores.npz')==result['arrays_sha256']
assert sha(HERE/'plan.json')==result['plan_sha256']
assert sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256']
ck=torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=True)
classes=ck['classes'];w=ck['state_dict']['head.weight'].numpy();b=ck['state_dict']['head.bias'].numpy()
old=read(plan['old_prediction_results'])['arms']['treatment']['evaluations']
old_full={r['id']:r for r in old['full']['rows']};old_target={r['id']:r for r in old['target_nikl18']['rows']}
target_labels={r['reference_label'] for r in plan['inputs'] if r['annotation_prediction_project18'] is not None}
import sys
sys.path[:0]=[str(HERE.parents[1]/'src'),str(HERE.parents[1].parent/'sign_model_pilot_v1/src')]
from wis_lab.service.neural import PROJECT_18
target_ids=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())]
assert len(target_ids)==18 and len(target_labels)==6
assert len(result['rows'])==74 and [r['clip_id'] for r in result['rows']]==[r['clip_id'] for r in plan['inputs']]
max_error=0.;scores_count=rankings=0
with np.load(HERE/'features_and_scores.npz',allow_pickle=False) as arrays:
    assert arrays['classes'].tolist()==classes
    assert arrays['clip_ids'].tolist()==[r['clip_id'] for r in result['rows']]
    for i,(spec,row) in enumerate(zip(plan['inputs'],result['rows'])):
        assert row['status']=='EVALUATED' and row['signer_id'] in (11,12,13)
        assert row['annotation_reference_id']==spec['annotation_reference_id']
        gt=row['reference_label'];yi=classes.index(gt)
        assert old_full[row['annotation_reference_id']]['true']==gt
        feature=arrays['features'][i]
        scalar=np.array([math.fsum(float(v)*float(z) for v,z in zip(ww,feature))+float(bb) for ww,bb in zip(w,b)])
        error=float(np.max(np.abs(scalar-arrays['scores'][i])))
        assert error<1e-10;max_error=max(max_error,error);scores_count+=93
        for scope,got in row['comparisons'].items():
            ids=list(range(93)) if scope=='full93' else target_ids
            prior=old_full if scope=='full93' else old_target
            assert got['annotation_prediction']==prior[row['annotation_reference_id']]['pred']
            ordered=np.array(ids)[np.argsort(-scalar[ids],kind='stable')].tolist()
            assert got['whole_prediction']==classes[ordered[0]] and got['whole_true_rank']==ordered.index(yi)+1
            assert abs(got['whole_true_margin']-(scalar[yi]-max(scalar[j] for j in ids if j!=yi)))<1e-10
            assert got['annotation_correct']==(got['annotation_prediction']==gt)
            assert got['whole_correct']==(got['whole_prediction']==gt)
            label='correct_both' if got['annotation_correct'] and got['whole_correct'] else 'wrong_both' if not got['annotation_correct'] and not got['whole_correct'] else 'right_to_wrong' if got['annotation_correct'] else 'wrong_to_right'
            assert got['change']==label;rankings+=1
aggregates={}
for scope in ('full93','project18'):
    rows=[r for r in result['rows'] if scope in r['comparisons']]
    ag={'n':len(rows),'annotation_correct':sum(r['comparisons'][scope]['annotation_prediction']==r['reference_label'] for r in rows),
        'whole_correct':sum(r['comparisons'][scope]['whole_prediction']==r['reference_label'] for r in rows),
        'changes':dict(Counter(r['comparisons'][scope]['change'] for r in rows)),
        'by_signer':{str(s):{'n':sum(r['signer_id']==s for r in rows),'whole_correct':sum(r['comparisons'][scope]['whole_correct'] for r in rows if r['signer_id']==s)} for s in (11,12,13)}}
    for k in ('n','annotation_correct','whole_correct','changes'):assert ag[k]==result['metrics'][scope][k]
    aggregates[scope]=ag
audit={'status':'PASS','independent_scalar_scores':scores_count,'independent_rankings':rankings,'max_scalar_error':max_error,
       'aggregate':aggregates,'whole_prediction_counts':dict(Counter(r['comparisons']['full93']['whole_prediction'] for r in result['rows'])),
       'result_sha256':sha(HERE/'result.json'),'script_sha256':sha(__file__),'new_encoder_forwards':0,'training':0}
with (HERE/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(audit,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(audit,ensure_ascii=False))
