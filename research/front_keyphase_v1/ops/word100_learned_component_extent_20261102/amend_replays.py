"""Preserve v1; attach reviewed metadata/guards without model re-inference."""
from prepare_demo import *
def main():
 bridge=SourceBridge();rows=[]
 for case in read(X/'PUBLIC_REPLAY_VERIFIED.json')['cases']:
  base=read(case['source']);a=dict(np.load(case['input'],allow_pickle=False))
  if 'relation_probability' not in a:
   a['relation_probability']=np.load(V/'control_predictions'/f"rgb_geometry_42_{case['word']}_{case['kind']}.npz",allow_pickle=False)['probability']
  out=bridge.attach(base,a);dest=Path(case['output']).with_name('RESULT_v2.json')
  save(str(dest.relative_to(X)),out)
  assert all(out[k]==v for k,v in base.items())
  missing={k:v.copy() for k,v in a.items()};missing['image_available']=np.zeros(len(a['frame_id']),bool)
  held=bridge.attach(base,missing)
  assert all(not c.get('confirmed_frames') and c['status']!='RELATED_MOVEMENT_OBSERVED' for c in held['source_components'])
  no_joint={k:v.copy() for k,v in a.items()};no_joint['mask'][:]=False
  estimated=bridge.attach(base,no_joint)
  assert all(not c.get('confirmed_frames') for c in estimated['source_components'] if 'point_evidence' in c)
  unsupported=copy.deepcopy(base)
  for m in unsupported.get('motion_observations',[]):m['guide_time_binding']='UNCONFIRMED'
  linked=bridge.attach(unsupported,a)
  assert all(c['status']!='RELATED_MOVEMENT_OBSERVED' for c in linked['source_components'])
  rows.append(dict(word=case['word'],kind=case['kind'],v1_path=case['output'],v1_sha256=sha(case['output']),
    output=str(dest),sha256=sha(dest),old_outputs_exact=True,
    missing_image_held=True,jointless_positive_marked_image_estimate=True,unsupported_binding_not_promoted=True))
 save('REVIEW_FIXES_VERIFIED.json',dict(rows=rows,new_inference=0,new_training=0,
  fixes=['saved-result provenance across12GETresponses','image/time estimate differs from joint-supported point evidence','unconfirmed time-binding retained','busy UI source lock and response word check','restore captured file preview before analyzing upload','old analysis visible when source link has no validated component'],
  ui_checks_pending=True,pins={str(X/n):sha(X/n) for n in ['source_bridge.py','learner_adapter.py','run_demo.py','demo.js','demo.html']}))
 print('12v2publicreplays prepared;3guard probes passed for each;oldoutputs exact')
if __name__=='__main__':main()
