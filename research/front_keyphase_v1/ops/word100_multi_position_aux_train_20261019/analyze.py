from common import *
r=read(U/'COMPARISON_INDEPENDENT.json');ctx=read(U/'TRAIN_CONTEXT.json');d=read(U/'DIAGNOSTIC.json')
assert r['status'].startswith('PASS') and read(U/'DIAGNOSTIC_INDEPENDENT.json')['status']=='PASS'
events=[];errors=[];histories=[];seconds=0.
for seed in SEEDS:
 result=read(U/f'{ARM}_{seed}_RESULT.json');h=read(U/f'{ARM}_{seed}_history.json');seconds+=result['seconds']
 histories.append({'seed':seed,'selected_epoch':result['selected_epoch'],'completed_epochs':result['completed_epochs'],'first':h[0],'selected':h[result['selected_epoch']-1],'last':h[-1]})
 for v in VIEWS:
  for final,base,aux in zip(result['predictions'][v],result['base_predictions'][v],result['aux_predictions'][v]):
   assert final['clip_id']==base['clip_id']==aux['clip_id']
   y=final['label_index'];fp=final['prediction'];bp=base['prediction'];ap=aux['prediction']
   zz=np.asarray(final['logits']);bb=np.asarray(base['logits']);aa=np.asarray(aux['logits']);other=np.arange(100)!=y
   row={'seed':seed,'view':v,'clip_id':final['clip_id'],'target':ctx['labels'][y],'base_prediction':ctx['labels'][bp],
    'new_prediction':ctx['labels'][fp],'aux_prediction':ctx['labels'][ap],'base_correct':base['correct'],'new_correct':final['correct'],'aux_correct':aux['correct'],
    'target_rank':int(np.sum(zz>zz[y]))+1,'old_margin':float(bb[y]-bb[other].max()),'new_margin':float(zz[y]-zz[other].max()),
    'aux_target_minus_old_rival':float(aa[y]-aa[bp]) if bp!=y else None}
   if final['correct']!=base['correct']:events.append(row)
   if not final['correct']:errors.append(row)
summary={'status':'COMPLETE_VERIFIED_NOT_PROMOTED','at':now(),'model':ARM,'comparison_sha256':sha(U/'COMPARISON_INDEPENDENT.json'),
 'diagnostic_sha256':sha(U/'DIAGNOSTIC.json'),'summary':r['summary'],'changes_vs_T3F':events,'remaining_errors':errors,
 'histories':histories,'selected_epochs':r['selected_epochs'][ARM],'completed_epochs':r['completed_epochs'][ARM],
 'training_seconds_sum':seconds,'training_epoch_sum':sum(x['epochs'] for x in r['traces'].values()),
 'actual_batches':sum(x['actual_batches'] for x in r['traces'].values()),'actual_source_exposures':sum(x['actual_source_exposures'] for x in r['traces'].values()),
 'primary_gate':r['primary_review_gate_vs_T3F'],'existing_models_unchanged':True,'all_signs_proven':False,'user_data_used':False}
write(U/'ANALYSIS.json',summary,new=True)
write(U/'NEXT_PLAN.json',{'status':'RESULT_REVIEW_KEEP_C_T3F','automatic_training':False,'automatic_adoption':False,
 'decision':'M3 sensor mean improves0.11pp in only1/3seeds; masked new error violates signer/group preservation. Not promoted.',
 'next_single_action':'Review this comparison. Before another trained candidate, establish public observable-stage/ordering and normal-variation evaluation cases using the existing100 reference audit, with competent sign-language review for mandatory/omission labels. Do not automatically retune heads, epochs or weights.',
 'why':'Earlier-feature sensitivity and standalone auxiliary recognition improved, but reverse-order sensitivity did not consistently improve and overall gains were not repeatable. Word-label accuracy is not signing-completeness correctness.',
 'future_hypothesis':'A separate chronological-relation learning signal may be considered only after valid task labels/shortcut controls are designed and approved. No new order-loss coefficient is selected now.',
 'data_rules':'Existing C/T3-F and web/DTW preserved. User/mentor recordings never train/tune/select. ID14-16 remain untouched; no raw ID14-18 in this unit.',
 'generality':'Shared readout and loss, no word exceptions. Current evidence limited to100word DEV repeatedly used for development; no all-vocabulary or real-webcam guarantee.'},new=True)
print(json.dumps({'events':events,'remaining_sensor_errors':[x for x in errors if x['view']=='sensor'],'epochs':summary['training_epoch_sum'],'batches':summary['actual_batches'],'exposures':summary['actual_source_exposures'],'seconds':seconds},ensure_ascii=False,indent=2))
