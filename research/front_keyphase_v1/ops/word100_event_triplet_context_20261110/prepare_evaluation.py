from triplet_core import *
assert not (U/'EVALUATION_LABELS.json').exists()
spans={
 '0739':[(49,51,1,1),(53,67,1,2),(71,73,2,1),(75,79,2,2),(83,85,3,1),(89,103,3,2),(115,115,0,0)],
 '1133':[(40,42,0,0),(46,48,4,1),(54,60,4,2),(66,74,5,2),(88,92,0,0)],
 '1567':[(34,38,0,0),(44,46,6,1),(48,52,6,2),(58,62,7,2),(66,72,6,2),(80,88,7,2)],
 '2523':[(38,40,0,0),(42,44,8,1),(48,56,8,2),(58,60,9,1),(64,74,9,2),(86,88,0,0)]}
rows=[]
for w,ss in spans.items():
 p=B/'review'/f'NIA_SL_WORD{w}_REAL11_F.jpg'
 rows.append(dict(clip_id=f'NIA_SL_WORD{w}_REAL11_F',person=11,role='historical_development',sheet=str(p),sheet_sha256=sha(p),segments=[dict(a=a,b=b,context=c,state=s) for a,b,c,s in ss]))
write(U/'EVALUATION_LABELS.json',dict(rows=rows,reviewer='Same Codex visual review before candidate outputs, not independent human/expert',labels='Visible body-context and progressing/core appearance only. No label for hidden contact, finger correctness, actual completion, or exact boundary.',count=sum(len(r['segments']) for r in rows),new_independent_test=False))
write(U/'EVAL_PINS.json',{str(p):sha(p) for p in [U/'EVALUATION_LABELS.json',U/'evaluate.py',U/'connector.py',U/'triplet_core.py']})
print('Evaluation labels and code frozen')
