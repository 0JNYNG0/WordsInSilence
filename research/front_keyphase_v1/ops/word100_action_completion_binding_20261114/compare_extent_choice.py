"""Compare saved new/parent predictions once; no selection or retraining."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=read(U/'ABSENCE_BAGS.json');cfg=read(U/'TRAINING_FROZEN.json');trained=read(U/'TRAINING_COMPLETE.json')
assert trained['training_frozen_sha256']==sha(U/'TRAINING_FROZEN.json')
rows=[];arrays=np.load(meta['arrays'])
for run in trained['rows']:
    assert sha(Path(run['predictions']))==run['predictions_sha256'];parent=next(p for p in cfg['parents']if p['seed']==run['seed'])
    assert sha(Path(parent['predictions']))==parent['predictions_sha256']
    old=np.load(parent['predictions']);new=np.load(run['predictions'])
    for r in meta['rows']:
        j=[2,3,4,5,7].index(r['family']);ids=r['candidate_indices'];y=arrays[f"y_{r['index']}"]
        record={k:r[k]for k in ['index','clip_id','person','split','family','kind','view','reverse']};record['seed']=run['seed']
        for key,pred in [('parent',old),('new',new)]:
            scores=pred[f"logits_{r['original_bag_index']}"][ids,j];null=float(pred['null_logits'][j]);at=int(scores.argmax());selected=bool(scores[at]>null)
            record[key]=dict(null_selected=not selected,chosen_original_candidate=ids[at] if selected else None,
                full_source_correct=bool(selected and y[at]) if y.any() else None,
                no_full_extent_correct=not selected if not y.any() else None)
        record['protected_positive_lost']=record['parent']['full_source_correct']is True and record['new']['full_source_correct']is not True
        rows.append(record)
summary=[]
for seed in [42,43,44]:
    for split in ['train','probe']:
        for kind in dict.fromkeys(r['kind']for r in rows):
            rr=[r for r in rows if r['seed']==seed and r['split']==split and r['kind']==kind];metric='full_source_correct'if kind=='positive_source_extent'else'no_full_extent_correct'
            summary.append(dict(seed=seed,split=split,kind=kind,bags=len(rr),source_clips=len(set(r['clip_id']for r in rr)),
                parent_correct=sum(r['parent'][metric] is True for r in rr),new_correct=sum(r['new'][metric] is True for r in rr),
                protected_losses=sum(r['parent'][metric] is True and r['new'][metric] is not True for r in rr),
                recovered=sum(r['parent'][metric] is not True and r['new'][metric] is True for r in rr)))
protected=[r for r in rows if r['split']=='probe' and r['kind'] in ['positive_source_extent','reviewed_local_hold']]
losses=[r for r in protected if (r['parent']['full_source_correct']is True and r['new']['full_source_correct']is not True)or(r['parent']['no_full_extent_correct']is True and r['new']['no_full_extent_correct']is not True)]
out=dict(status='LOCAL_EXTENT_COMPARISON_COMPLETE',summary=summary,rows=rows,protected_probe_losses=losses,
    local_protection_gate_passed=not losses,model_adopted=False,educational_error_accuracy=None,
    warning='Derived windows are correlated; counts are not independent people or natural error videos. Full source extent is not semantic correctness.',
    pins={str(p):sha(p)for p in [U/'ABSENCE_BAGS.json',U/'TRAINING_FROZEN.json',U/'TRAINING_COMPLETE.json']})
with (U/'LOCAL_COMPARISON.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'probe':[r for r in summary if r['split']=='probe'],'protected_losses':len(losses),'gate':not losses},ensure_ascii=False))
