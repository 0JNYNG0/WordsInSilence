import study as S
import numpy as np
def main():
 rows=S.read(S.BP/'PREPARED.json')['rows'];z=S.load(rows[0]['output']);q,c,ix=S.transformed(z,'identity')
 assert np.array_equal(q['features'],z['features'],equal_nan=True);assert np.array_equal(q['present'],z['present'])
 np.testing.assert_allclose(c['values'],z['values'],atol=2e-5,rtol=1e-6,equal_nan=True);assert np.array_equal(c['valid'],z['available'])
 results=[]
 for kind in ['identity','slow15','gapL','wholeR']:
  qq,cc,ii=S.transformed(z,kind);out=S.evaluate(qq,q,cc,c,ii,z['core'])
  assert not(out['sequence_old_ok']&~out['sequence_new_ok']).any();assert np.array_equal(out['domain'],np.broadcast_to(z['core'][:,None,None],out['domain'].shape))
  if kind=='wholeR':
   for a,b in [(0,17),(21,23)]:assert not out['point_ok'][:,1,a:b].any() and not out['sequence_new_ok'][:,1,a:b].any()
  results.append(dict(kind=kind,path_length=len(out['path']),lost_support=0))
 # No hidden timing input reaches phase(). Sequence fallback is availability-only.
 a=np.arange(15.);np.testing.assert_array_equal(S.P.phase(a),a)
 d=np.ones((9,2,23),bool);c=dict(values=np.zeros((9,2,23)),valid=d.copy(),quality=d.copy());c['valid'][4,0,1]=False;c['values'][4,0,1]=np.nan
 m=S.mapped(c,np.arange(9.),np.arange(9),np.arange(9));v,ok,quality=S.descriptor(m,d)
 assert not ok[2:7,0,1].any() and ok[2:7,0,2].all();assert np.isnan(v[2:7,0,1]).all()
 S.write(S.U/'PREFLIGHT.json',dict(at=S.now(),status='PASS',identity_features_exact=True,domain_independent_of_missing=True,local_angle_not_neighbor_suppressed=True,wholehand_missing_not_zero=True,original_support_preserved=True,fixtures=results,phase_tests=S.P.selftest(),model_change=False))
 print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
