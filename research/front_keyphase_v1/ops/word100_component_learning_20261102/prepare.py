from common import *
def main():
 torch.set_num_threads(4);rows,meta,arrays=load()
 fit=[r for r in meta.values() if r['role']=='fit'];parts=[current_geometry(arrays[r['clip_id']]) for r in fit]
 x=np.concatenate([p[0] for p in parts]);mask=np.concatenate([p[1] for p in parts]);mean=[];std=[]
 for j in range(63):
  v=x[mask[:,j],j];mean.append(float(v.mean()) if len(v) else 0);std.append(max(float(v.std()) if len(v) else 1,.05))
 stats=dict(mean=mean,std=std,fit_clips=[r['clip_id'] for r in fit],current_frame_features=63,
  contextual_rgb_latent_excluded_from_temporal_controls=True)
 temporal=build_temporal(rows,arrays,stats);point=point_samples(rows,arrays)
 # Concrete invariants, including the context issue found in review.
 for i,s in enumerate(temporal):
  r=s['record'];base=temporal_sample(r,arrays[r['clip_id']],stats)
  if s['variant'].startswith('speed'):assert np.array_equal(s['x'],base['x'])
  if s['variant']=='reverse':assert np.array_equal(s['x'],base['x'][::-1])
  if s['variant']=='still':assert np.all(s['x']==s['x'][0])
  if s['variant']=='image_gap':assert not s['usable'] and not s['mask'] and np.count_nonzero(s['x'])==0
  assert s['native_extent']==base['native_extent']
 torch.manual_seed(42);a=TemporalNet('pooled').eval();torch.manual_seed(42);b=TemporalNet('ordered').eval()
 assert torch.equal(a.embed[0].weight,b.embed[0].weight) and torch.equal(a.out.weight,b.out.weight)
 xx,ll=batch(temporal[:2]);long=torch.cat([xx,torch.zeros(len(xx),7,126)],1)
 with torch.inference_mode():
  for model in [a,b]:assert torch.allclose(model(xx,ll),model(long,ll),atol=1e-6)
  rev=torch.flip(xx[:1,:ll[0]],dims=[1]);assert torch.allclose(a(xx[:1],ll[:1]),a(rev,ll[:1]),atol=1e-6)
 p=PointNet();old=torch.load(V/'runs/rgb_geometry_42/model.pt',map_location='cpu',weights_only=True)
 assert torch.equal(p.pose.weight,old['pose.weight'])
 save('NORMALIZATION.json',stats)
 save('INPUT_MANIFEST.json',dict(temporal=[metadata(s) for s in temporal],point=[metadata(s) for s in point],
  sources=meta,annotation_scope_not_automatic_runtime=True,geometry_control_not_reencoded_video=True))
 conditions=['pooled','ordered'];pins={str(p):sha(p) for p in [S/'DIRECT_TARGETS.json',S/'TEMPORAL_CONTROLS.json',
  S/'INDEPENDENT_PREPARATION_REVIEW.json',V/'NORMALIZATION.json',V/'runs/rgb_geometry_42/model.pt',
  U/'common.py',U/'train_evaluate.py',U/'NORMALIZATION.json',U/'INPUT_MANIFEST.json']}
 plan=dict(conditions=conditions,primary='ordered',seed=42,epochs=80,lr=.001,weight_decay=.01,
  point_shared=True,point_finetune='new2output + copied final32channel convolution; frozen earlier encoder',
  temporal='new3component ordered GRU vs order-agnostic pooled head, current-frame geometry+mask only',
  loss='one BCE for reviewed point or whole scoped interval/sequence; equal target/clip/class/representation weighting',
  train_variants=['native','reverse','still'],exclude_from_optimization=['unknown','speed_075','speed_125','image_gap'],
  scope='4expressions; recordedIDs01-09 fit,11-13 repeated development; no H14/15/private/mentor',
  source_times_as_model_feature=False,word_or_actor_id_as_model_input=False,
  existing16outputs_changed=False,thresholds=dict(classification=.5,confirmed_positive=.8,confirmed_negative=.2,temporal_critical_geometry_coverage=.8),
  gates=dict(point_oblique_recall_min=.8,point_oblique_specificity_min=.8,
   natural_temporal_recall_each_min=.8,natural_traversal_specificity_min=.8,
   constructed_reverse_direction_specificity_each_min=.8,constructed_alternation_reverse_recall_min=.8,
   constructed_still_specificity_each_min=.8,primary_no_lower_natural_correct_than_pool=True,
   primary_better_direction_reverse_correct_than_pool=True,gap_claims_zero=True,
   thumb_correction_allowed=False,whole_sign_correctness_allowed=False),
  no_auto_adoption='Passing annotation scopes still requires automatic learned-prototype proposal and full-demo tests; no natural learner-error precision claim',
  timing='head time only; video-to-feedback later',pins=pins)
 save('FROZEN.json',plan)
 save('PREFLIGHT.json',dict(passed=True,temporal_records=len(temporal),point_records=len(point),
  fit_clips=len(fit),all_clips=len(meta),native_time_preserved=True,order_control_context_coherent=True,
  pooled_reversal_invariant=True,padding_invariant=True,existing_checkpoints_unchanged=True,
  no_new_model_predictions_seen=True,new_training=0))
 print('Prepared',len(temporal),'temporal and',len(point),'point records; frozen80epochs2temporal + sharedpoint')
if __name__=='__main__':main()
