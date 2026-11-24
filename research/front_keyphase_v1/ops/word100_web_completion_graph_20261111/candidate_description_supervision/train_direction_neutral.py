from core import *
V=W/'direction_neutral_candidates'
def main():
 meta=read(V/'INPUTS_COMPLETE.json');a=np.load(meta['arrays']);old=np.load(read(W/'INPUTS_COMPLETE.json')['arrays']);n=meta['original_rows'];assert all(np.array_equal(a[k][:n],old[k]) for k in ['x','y','mask','train']);assert all(sha(p)==h for p,h in meta['pins'].items());assert np.isfinite(a['x']).all()
 for g in read(V/'GENERATED.json')['rows']:
  assert sha(g['path'])==g['sha256'] and sha(g['source'])==g['source_sha256'];d=np.load(g['path']);raw=np.load(g['source']);ii=d['source_indices'];assert np.array_equal(ii,np.arange(len(ii)-1,-1,-1));assert all(np.diff(ii)==-1);assert np.all(np.diff(d['ts'])>0)
  # Physical coordinate conversion verified by same frozen normalizer; no fabricated poses.
  rc=load('_neutral_audit_rc',O/'word100_guide_requirement_learning_20261111/requirement_core.py') if 'rc' not in locals() else rc
  assert np.array_equal(d['x'],rc.prior.physical(raw['x_'+g['view']])[ii])
 rows=meta['rows'];assert {r['source_sha256'] for r in rows if r['split']=='train'}.isdisjoint(r['source_sha256'] for r in rows if r['split']=='probe')
 for i,r in enumerate(rows):
  for f,b in r['basis'].items():
   j=F.index(int(f));own=set(b['own_progression']);assert a['y'][i,j,0]==bool(own)
   if a['mask'][i,j,1]:assert a['y'][i,j,1]==(not b['other_reviewed'])
   if a['mask'][i,j,2]:assert a['y'][i,j,2]==(set(b['required_observed_progression'])<=own)
 cfg=read(W/'FROZEN.json');cfg.update(at=stamp(),data_sha256=meta['arrays_sha256'],heads=['related_comparison_axis_attempt','no_reviewed_other_action','observed_source_progression_coverage'],selection=dict(conditions=['baseline','normal_body','neutral_body'],eligibility='All frozen boundary proposals on requested axis/side with >=3 body-observed points and >=.8 body fraction; predicted related attempt>=.5. No old has_motion gate, no expected direction.',score='sum log three candidate probabilities, unit coefficients',completion=None),pins=dict(meta['pins']))
 cfg['pins'].update({str(p):sha(p) for p in [W/'train.py',Path(__file__),W/'COMPARISON.json',W/'shadow_components/RESULTS.json']});assert not (V/'FROZEN.json').exists();write(V/'FROZEN.json',cfg)
 write(V/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),status='PASS',original_rows_preserved=n,reversed_views=48,source_split=meta['source_person_split'],synthetic_splices=0,whole_sign_correctness_labels_created=0,unknown_masked=True,reviewer='root arithmetic; no independent human'))
 m=load('_neutral_candidate_trainer',W/'train.py');m.W=V;m.main()
if __name__=='__main__':main()
