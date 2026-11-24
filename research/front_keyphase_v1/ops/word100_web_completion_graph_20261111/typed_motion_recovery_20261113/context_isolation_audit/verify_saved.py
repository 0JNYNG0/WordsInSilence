"""Arithmetic-only audit of the fixed-span hand-removal diagnostic."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (P/'VERIFICATION.json').exists()
f=rd(P/'FROZEN.json');r=rd(P/'RESULTS.json');z=np.load(r['saved_features']);checks=0
for p,h in f['pins'].items():assert sha(p)==h;checks+=1
assert sha(r['saved_features'])==r['saved_features_sha256'];checks+=1
changes=0;classes=0;maxdiff=0.;changed_context=0
for row in r['rows']:
    assert row['person'] in range(1,8) and row['geometry_exact'] and row['normal_context_matches'];checks+=1
    cid=row['clip_id'];d=float(np.max(np.abs(z[cid+'_normal_context']-z[cid+'_hand_absent_context'])))
    assert np.isclose(d,row['context_max_abs_difference']);checks+=1
    changed_context+=int(d>1e-6)
    for p in row['predictions']:
        probs=[]
        for mode in ['normal','hand_absent']:
            v=np.array(p[mode+'_logits']);q=np.exp(v-v.max());q/=q.sum();probs.append(q)
            assert np.allclose(q,p[mode+'_probabilities'],atol=1e-6);checks+=1
        class_change=int(probs[0].argmax())!=int(probs[1].argmax());delta=float(np.max(np.abs(probs[0]-probs[1])))
        assert class_change==p['class_changed'] and np.isclose(delta,p['probability_max_abs_difference'],atol=1e-7);checks+=1
        state_change=p['normal_state']!=p['hand_absent_state'];assert state_change==p['state_changed'];checks+=1
        changes+=int(state_change);classes+=int(class_change);maxdiff=max(maxdiff,delta)
s=r['summary'];assert s['clips']==len(r['rows'])==7 and s['seeds']==3 and s['paired_outputs']==21;checks+=1
assert s['context_changed_clips']==changed_context and s['verdict_changed']==changes and s['argmax_changed']==classes;checks+=1
assert np.isclose(s['max_probability_change'],maxdiff,atol=1e-7);checks+=1
out=dict(status='PASS',checks=checks,summary=s,no_model_inference=True,reviewer='Same delegated Codex separate arithmetic process, not independent human truth',result_sha256=sha(P/'RESULTS.json'),script_sha256=sha(__file__))
(P/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps(out,ensure_ascii=False))
