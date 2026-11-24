import candidate as C
import numpy as np,json,datetime
from pathlib import Path
T=Path(__file__).resolve().parent
def make(n):
 v=np.arange(n,dtype=float)
 return dict(features=v[:,None],present=np.ones((n,1),bool),core=np.ones(n,bool),values=v[:,None,None]*np.ones((n,2,23)),available=np.ones((n,2,23),bool),quality_eligible=np.zeros((n,2,23),bool),quality_after=np.ones((n,2,23),bool))
def main():
 q=make(7);ix=np.array([0,0,1,2,2,2,3,4,4,5,6,6]);z={k:v[ix].copy() for k,v in q.items()};s,e,g=C.groups(z);assert len(s)==7 and len(g)==len(ix)
 compressed=C.compressed(z,s,e)
 for k in q:np.testing.assert_array_equal(q[k],compressed[k])
 for k in ('available','quality_after','quality_eligible'):
  x={k:v.copy() for k,v in z.items()};x[k][1,0,0]=not x[k][1,0,0];ss,_,_=C.groups(x);assert len(ss)==8
 x={k:v.copy() for k,v in z.items()};x['values'][1,0,0]+=.000001;assert len(C.groups(x)[0])==8
 x={k:v.copy() for k,v in z.items()};x['core'][1]=False;assert len(C.groups(x)[0])==8
 a=make(3);a={k:v[[0,1,0]].copy() for k,v in a.items()};assert len(C.groups(a)[0])==3
 phase=np.linspace(0,6,97);native=C.expand_phase(phase,s,e);np.testing.assert_allclose(np.interp(native,np.arange(len(ix)),ix),phase,atol=1e-12)
 (T/'PREFLIGHT.json').write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS_EXACT_REPEAT_LOSSLESS_GROUPS',tests=8,normal_distinct_motion_kept=True,missing_quality_core_changes_not_collapsed=True,aba_crossing_order_preserved=True,known_index_used_only_in_test=True,operating_changed=False)),encoding='utf8');print('EXACT_REPEAT_PREFLIGHT_PASS8')
if __name__=='__main__':main()
