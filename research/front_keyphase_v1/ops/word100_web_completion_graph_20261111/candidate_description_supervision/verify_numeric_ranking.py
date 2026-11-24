from framing_boundary_core import *
import math
V=R/'numeric_ranking';j=read(V/'COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in j['predictions']};checked=0
for r in j['rows']:
 saved=pm[r['seed'],r['clip_id'],r['kind']];assert sha(saved['path'])==saved['sha256'];z=np.load(saved['path'])
 for i,c in enumerate(r['conditions']['unclipped_probability']['cards']):
  p=z[f'crop_boundary_p_{i}'];sp=z[f'crop_boundary_spans_{i}'];ss=[sum(math.log(float(v)) for v in pp) if pp[0]>=.5 and (pp>0).all() else -math.inf for pp in p]
  if ss and math.isfinite(max(ss)):
   at=ss.index(max(ss));assert c['chosen']['span']==sp[at].tolist()
  else:assert c['chosen'] is None
  assert c['semantic_completion'] is None;checked+=1
protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items())
write(V/'VERIFICATION.json',dict(at=stamp(),pass_=True,python_math_log_argmax_recount=checked,no_model_forward=True,protected_unchanged=True,reviewer='same Codex saved arithmetic, not human ground truth'))
print('Numeric ranking independently recalculated',checked)
