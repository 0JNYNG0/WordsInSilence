from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parent
p=W/'score_diagnostic.py';f=W/'EVALUATION_FROZEN.json'
assert not (W/'DIAGNOSTIC_SUMMARY.json').exists()
assert not (W/'SCORE_SERIALIZATION_AMENDMENT.json').exists()
(W/'score_diagnostic_initial_serialization_failure.py').write_bytes(p.read_bytes())
(W/'EVALUATION_FROZEN_INITIAL.json').write_bytes(f.read_bytes())
old=hashlib.sha256(p.read_bytes()).hexdigest();t=p.read_text()
t=t.replace("phase_changed_vs_A=sum(p[r['id']]!=predictions['A_cached_seed42'][r['id']].argmax() for r in phase)","phase_changed_vs_A=int(sum(p[r['id']]!=predictions['A_cached_seed42'][r['id']].argmax() for r in phase))")
p.write_text(t,encoding='utf-8');new=hashlib.sha256(p.read_bytes()).hexdigest();assert old!=new
d=json.loads(f.read_text());d['pins'][str(p.resolve())]=new
f.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
(W/'SCORE_SERIALIZATION_AMENDMENT.json').write_text(json.dumps(dict(first_error='TypeError: Object of type int64 is not JSON serializable at DIAGNOSTIC_SUMMARY write',repeated_arithmetic_failure=2,failed_shell_patch='PowerShell quoting parse error, no changes made',old_code=old,new_code=new,change='numpy integer counter converted to Python int for JSON only',metrics_or_model_outputs_changed=False,inference_repeated=False),indent=2),encoding='utf-8')
