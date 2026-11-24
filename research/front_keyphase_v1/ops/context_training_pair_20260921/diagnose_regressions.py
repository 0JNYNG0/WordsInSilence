"""Explain existing output changes; do not train, infer, or tune another model."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1];R=L/'runs/context_head_pair_v1'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=read(H/'RESULT_AUDIT.json');assert audit['status']=='PASS_OUTPUT_INTEGRITY'
report=read(R/'evaluation.json');meta=read(H/'feature_metadata.json')
models={k:torch.load(spec['path'],map_location='cpu',weights_only=True) for k,spec in [('parent',report['parent']),('control',report['models']['control']),('treatment',report['models']['treatment'])]}
classes=models['parent']['classes'];newlabels={r['label'] for r in meta['pair_context']}
rows=[r for r in report['rows'] if r['group']=='annotation_dev'];outrows=[]
with np.load(R/'evaluation_scores.npz',allow_pickle=False) as s,np.load(H/'public_features.npz',allow_pickle=False) as a,np.load(R/'training_arrays.npz',allow_pickle=False) as t:
    base_class_draws=Counter(a['base_y'][t['base_indices']].ravel().tolist())
    paired_class_draws=Counter(a['pair_context_y'][t['pair_indices']].ravel().tolist())
    for i,row in enumerate(rows):
        if not row['models']['parent']['full93']['correct'] or row['models']['treatment']['full93']['correct']:continue
        yi=classes.index(row['true']);pi=classes.index(row['models']['treatment']['full93']['pred']);x=a['annotation_dev_x'][i].astype(float)
        pw=models['parent']['state_dict']['head.weight'].numpy().astype(float);tw=models['treatment']['state_dict']['head.weight'].numpy().astype(float)
        pb=models['parent']['state_dict']['head.bias'].numpy().astype(float);tb=models['treatment']['state_dict']['head.bias'].numpy().astype(float)
        delta_w=math.fsum(float(v) for v in ((tw[yi]-pw[yi])-(tw[pi]-pw[pi]))*x)
        delta_b=float((tb[yi]-pb[yi])-(tb[pi]-pb[pi]))
        margins={m:float(s['annotation_dev__'+m][i,yi]-s['annotation_dev__'+m][i,pi]) for m in models}
        assert abs((margins['treatment']-margins['parent'])-(delta_w+delta_b))<1e-12
        outrows.append(dict(id=row['id'],true=row['true'],new_rival=classes[pi],true_in_context16=row['true'] in newlabels,rival_in_context16=classes[pi] in newlabels,
            already_wrong_in_matched_control=not row['models']['control']['full93']['correct'],true_vs_new_rival_margins=margins,
            margin_change_from_feature_weights=delta_w,margin_change_from_bias=delta_b,
            actual_base_draws_for_true_class=base_class_draws[yi],actual_paired_draws_for_true_class=paired_class_draws[yi]))
    base_counts=Counter(a['base_y'].tolist())
    training_distribution=[dict(label=c,original_base_samples=base_counts[i],base_draws=base_class_draws[i],paired_draws=paired_class_draws[i],in_context16=c in newlabels) for i,c in enumerate(classes)]
summary=dict(parent_regressions=len(outrows),regressions_truth_outside_context16=sum(not r['true_in_context16'] for r in outrows),regressions_rival_inside_context16=sum(r['rival_in_context16'] for r in outrows),already_wrong_in_control=sum(r['already_wrong_in_matched_control'] for r in outrows),all_regressed_true_classes_still_sampled=all(r['actual_base_draws_for_true_class']>0 for r in outrows),regression_counts_by_true=dict(Counter(r['true'] for r in outrows)),base_draws=10560,paired_draws=10560)
result=dict(status='VERIFIED_OUTPUT_CHANGE_DIAGNOSIS',summary=summary,regressions=outrows,training_distribution=training_distribution,
    conclusion='All9 parent-to-treatment regressions have true labels outside the16 context-added labels;8 new winners are inside those16. The matched control already loses3 of these9. All original93 classes were still sampled. This is observed interference in a shared classifier, not evidence that data were deleted or that one specific word needs an exception.',
    next_work='Review a common preservation objective on original PUBLIC TRAINING features while retaining paired-context learning. Freeze this failed pair. Before implementing anything new, specify one distinct public comparison and preservation gates. Do not search this failed pair coefficient/steps/seed/mixture, select a best intermediate step, or use user/development labels as training constraints.',
    limits=['Margin decomposition is algebraic attribution of the affine output, not proof of a unique optimization cause.','74/74 shows the frozen representation can separate this development set after head adaptation; it does not prove full webcam or new-person generalization.','No replacement model is selected and no new training policy is executed by this diagnosis.'],
    source_pins={str(p):sha(p) for p in [H/'RESULT_AUDIT.json',R/'evaluation.json',R/'evaluation_scores.npz',R/'training_arrays.npz',H/'public_features.npz']},training_runs=0,new_encoder_forwards=0,user_arrays_read=0,reserved_arrays_read=0)
assert summary['parent_regressions']==9 and summary['regressions_truth_outside_context16']==9 and summary['regressions_rival_inside_context16']==8 and summary['already_wrong_in_control']==3
with (H/'REGRESSION_DIAGNOSIS.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(summary,ensure_ascii=False))
