import study as S
import numpy as np
from collections import Counter
def main():
 f=S.read(S.U/'FROZEN.json');r=S.read(S.U/'RESULT.json');S.check(f['pins']);S.check(r['pins']);S.check(f['protected']);assert len(r['rows'])==4500
 counts=Counter();groups={g:dict(count=0,old=0.,new=0.) for g in S.GROUPS};ts=dict(cells=0,old=0.,new=0.);losses=[]
 for row in r['rows']:
  z=S.load(S.B/'cases'/(row['name']+'.npz'));n=S.load(S.U/'cases'/(row['name']+'.npz'));qp=S.B/'quality_correction'/(row['name']+'.npz');q=S.load(qp) if qp.is_file() else z;oldq=q['sequence_quality'];newq=q['sequence_new_quality']
  fb=oldq&~newq;assert np.array_equal(fb,n['quality_fallback']);assert np.array_equal(n['sequence_quality'],oldq|newq);assert not(oldq&~n['sequence_quality']).any()
  assert np.array_equal(n['sequence_delta'][fb],z['sequence_old_delta'][fb],equal_nan=True);assert np.array_equal(n['sequence_delta'][~fb],z['sequence_new_delta'][~fb],equal_nan=True)
  good=np.isfinite(n['sequence_delta']).all(-1);assert np.array_equal(good,z['sequence_new_ok']);assert not(z['sequence_old_ok']&~good).any();assert not(n['sequence_quality']&~good).any()
  assert np.array_equal(n['source'],np.where(fb,2,z['phase_source']));expected=np.broadcast_to(z['old_error'][:,None,None],fb.shape)
  np.testing.assert_allclose(n['effective_error'][fb],expected[fb]);np.testing.assert_allclose(n['effective_error'][~fb],z['effective_error'][~fb],equal_nan=True)
  counts['cases']+=1;counts['quality_fallback']+=int(fb.sum());counts['quality_before']+=int(oldq.sum());counts['quality_after']+=int(n['sequence_quality'].sum());counts['coordinate_sequence_cells']+=int(good.sum())
  if row['kind'] in S.NORMAL:
   common=z['sequence_old_ok']&good;counts['normal_quality_fallback']+=int(fb.sum());counts['normal_quality_before']+=int(oldq.sum());counts['normal_quality_after']+=int(n['sequence_quality'].sum())
   ts['cells']+=int(common.sum());ts['old']+=float(expected[common].sum());ts['new']+=float(n['effective_error'][common].sum())
   for g,(a,b) in S.GROUPS.items():
    m=common[:,:,a:b];gg=groups[g];gg['count']+=int(m.sum())*5;gg['old']+=float(z['sequence_old_delta'][:,:,a:b][m].astype(float).sum());gg['new']+=float(n['sequence_delta'][:,:,a:b][m].astype(float).sum())
  if row['kind'].startswith('skip'):
   removed=np.setdiff1d(np.arange(len(z['core'])),z['source_index'].astype(int));region=np.arange(max(2,removed[0]-2),min(len(z['core'])-2,removed[-1]+3));common=z['sequence_old_ok'][region]&good[region];old=z['sequence_old_delta'][region];new=n['sequence_delta'][region]
   lost=common&(np.max(old,axis=-1)>1e-6)&(np.max(new,axis=-1)<=1e-6)
   if lost.any():losses.append(dict(name=row['name'],cells=int(lost.sum())))
 for g in groups.values():g.update(old_mean=g['old']/g['count'],new_mean=g['new']/g['count'])
 ts.update(old_mean_frames=ts['old']/ts['cells'],new_mean_frames=ts['new']/ts['cells'])
 gate=dict(original_coordinate_support_preserved=True,original_quality_support_preserved=True,normal_effective_time_lower=ts['new']<ts['old'],normal_each_group_lower=all(g['new']<g['old'] for g in groups.values()),no_deleted_signal_erased=not losses);gate['passed']=all(gate.values())
 S.check(f['pins']);S.check(r['pins']);S.check(f['protected']);S.write(S.U/'VERIFIED.json',dict(at=S.now(),status='PASS_SAVED_SELECTION_ARITHMETIC',counts=dict(counts),normal_time=ts,normal_groups=groups,deleted_positive_to_zero=losses,gate=gate,method='All4500 saved arrays: exact evidence-only selection, unchanged instantaneous parent hashes, no coordinate/quality support lost. SameCodex arithmetic, not independent human truth.',pins={str(S.U/n):S.sha(S.U/n) for n in ['FROZEN.json','RESULT.json','verify.py']},classifier_accuracy_improved=None,production_changed=False));print('VERIFIED_QUALITY',gate,dict(counts),ts,flush=True)
if __name__=='__main__':main()
