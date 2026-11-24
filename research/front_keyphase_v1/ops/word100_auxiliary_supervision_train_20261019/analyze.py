"""Interpret already verified saved scores and histories; no training or inference."""
from common import *
import statistics
r=read(U/'COMPARISON_INDEPENDENT.json');assert r['status'].startswith('PASS')
ctx=read(U/'TRAIN_CONTEXT.json');check(ctx['pins']);rows=[];errors=[];gains=[]
for seed in SEEDS:
 result=read(U/f'{ARM}_{seed}_RESULT.json');hist=read(U/f'{ARM}_{seed}_history.json');selected=hist[result['selected_epoch']-1]
 traces=[b for h in hist for b in read(U/'traces'/f'{ARM}_{seed}'/f"{h['epoch']:03}.json")['batches']]
 first=traces[0];oldfirst=read(F/'traces'/f'FG16_{seed}'/'001.json')['batches'][0]
 assert first['clips']==oldfirst['clips'] and first['choices']==oldfirst['choices']
 rows.append({'seed':seed,'seconds':result['seconds'],'epochs':len(hist),'selected_epoch':result['selected_epoch'],
  'train_first_aux_ce':hist[0]['train_aux_ce'],'train_last_aux_ce':hist[-1]['train_aux_ce'],
  'first_batch_gradient_norm_old':oldfirst['gradient_norm'],'first_batch_gradient_norm_new':first['gradient_norm'],
  'all_batch_median_gradient_norm':statistics.median(t['gradient_norm'] for t in traces),
  'first_memory_gradient_zero':all(first['gradient_norm_by_tensor'][k]==0 for k in ctx['trainable_parameter_names'] if k.startswith('memory.')),
  'later_memory_nonzero':all(any(t['gradient_norm_by_tensor'][k]>0 for t in traces[1:]) for k in ctx['trainable_parameter_names'] if k.startswith('memory.')),
  'update_norm_min_max':[min(t['update_norm'] for t in traces),max(t['update_norm'] for t in traces)],
  'online_train_aux_selected_percent':selected['train_aux_correct_online']/899*100,
  'sensor_aux_correct_first_selected_last':[hist[0]['aux_sensor_top1'],selected['aux_sensor_top1'],hist[-1]['aux_sensor_top1']],
  'sensor_final_correct_first_selected_last':[hist[0]['sensor_top1'],selected['sensor_top1'],hist[-1]['sensor_top1']],
  'sensor_final_ce_first_selected_last':[hist[0]['sensor_ce'],selected['sensor_ce'],hist[-1]['sensor_ce']],
  'sensor_aux_ce_first_selected_last':[hist[0]['aux_sensor_ce'],selected['aux_sensor_ce'],hist[-1]['aux_sensor_ce']],
  'selection_ce_first_selected_last':[hist[0]['selection_ce'],selected['selection_ce'],hist[-1]['selection_ce']],
  'observed_sensor_top1_range':[min(h['sensor_top1'] for h in hist),max(h['sensor_top1'] for h in hist)],
  'late61_80_available':any(h['epoch']>=61 for h in hist)})
 for v in VIEWS:
  teacher={p['clip_id']:p for p in read(T/f'T3F_{seed}_RESULT.json')['predictions'][v]}
  aux={p['clip_id']:p for p in result['aux_predictions'][v]}
  for p in result['predictions'][v]:
   old=teacher[p['clip_id']]
   if p['status']!='OK':continue
   y=p['label_index'];a=aux[p['clip_id']];z=np.asarray(p['logits']);b=np.asarray(old['logits']);am=np.asarray(a['logits']);other=np.arange(100)!=y
   item={'seed':seed,'view':v,'clip_id':p['clip_id'],'label':ctx['labels'][y],'signer':p['signer_id'],
    'prediction':ctx['labels'][p['prediction']],'aux_prediction':ctx['labels'][a['prediction']],
    'old_target_margin':float(b[y]-b[other].max()),'new_target_margin':float(z[y]-z[other].max()),
    'aux_target_margin':float(am[y]-am[other].max()),'old_correct':old['correct'],'new_correct':p['correct']}
   if not old['correct'] or not p['correct']:errors.append(item)
   if not old['correct'] and p['correct']:gains.append(item)
group_rows=[]
for view in VIEWS:
 for group,info in r['metrics'][ARM][view][0]['groups'].items():
  entry={'view':view,'group':group,'videos':info['total']}
  for model in ['T3F',ARM]:
   counts=[x['groups'][group]['correct'] for x in r['metrics'][model][view]]
   entry[model]={'counts':counts,'mean_percent':sum(counts)/(3*info['total'])*100}
  group_rows.append(entry)
out={'at':now(),'status':'COMPLETE_SAVED_ANALYSIS','comparison_sha256':sha(U/'COMPARISON_INDEPENDENT.json'),
 'rows':rows,'gains_vs_T3F':gains,'error_cases':errors,'groups':group_rows,'training_seconds_sum':sum(x['seconds'] for x in rows),
 'facts':['Auxiliary supervision passed stronger gradient to the branch and auxiliary classification learned; final primary accuracy did not improve.',
          'One masked-view 속다 video changed to correct in seed20261015; no new incorrect selected predictions versusT3F in either view.',
          'Selection epochs1/1/8 are determined by final combined DEV CE, not highest classification accuracy or auxiliary-only score.',
          'Later auxiliary scores improve in logs while combined validation CE worsens in the first two seeds; longer training alone is not justified.'],
 'interpretation':'Direct supervision addresses weak auxiliary learning signal, but stronger standalone classification does not automatically add complementary information to a strong frozen classifier. Data support this limitation; they do not prove one sole cause or semantic order understanding.',
 'generality':'Objective is label-independent, but learned100class head and this repeatedDEV100word evaluation are not all-sign-language coverage.',
 'user_data_used':False,'new_inference':0}
write(U/'ANALYSIS.json',out,new=True)
print(json.dumps({'rows':rows,'gains':gains,'training_seconds_sum':out['training_seconds_sum']},ensure_ascii=False))
