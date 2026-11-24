"""Verify auxiliary/combined decomposition and independent scores; no model inference."""
from common import *
import statistics
def audit_aux(ctx,stored,recount):
 out=[]
 for seed in SEEDS:
  r=stored[ARM,seed];row={'seed':seed,'views':{}}
  for v in VIEWS:
   aux=r['aux_predictions'][v];base=r['base_predictions'][v];final=r['predictions'][v]
   auxstats=recount(aux,ctx['missingness_groups'][v]);basestats=recount(base,ctx['missingness_groups'][v])
   assert auxstats['correct']==r['aux_metrics'][v]['top1_correct'] and abs(auxstats['mean_ce']-r['aux_metrics'][v]['mean_ce'])<1e-10
   tf={p['clip_id']:p for p in stored['T3F',seed]['predictions'][v]}
   changes=[];margins=[];oldmargins=[];max_base_error=0.
   for f,b,a in zip(final,base,aux):
    assert f['clip_id']==b['clip_id']==a['clip_id'];gold=tf[f['clip_id']]
    assert f['status']==b['status']==a['status']==gold['status']
    if f['status']!='OK':continue
    # Torch and numpy both add stored float32 logits; serialization retains exact values.
    zz=np.asarray(b['logits'],np.float32)+np.asarray(a['logits'],np.float32)
    assert np.array_equal(zz,np.asarray(f['logits'],np.float32))
    error=float(np.max(abs(np.asarray(b['logits'])-np.asarray(gold['logits']))));max_base_error=max(max_base_error,error)
    assert error<1e-5 and b['prediction']==gold['prediction'] and b['top5']==gold['top5']
    y=f['label_index'];before=np.asarray(b['logits']);after=np.asarray(f['logits'])
    other=np.arange(100)!=y
    oldmargins.append(float(before[y]-before[other].max()));margins.append(float(after[y]-after[other].max()))
    changes.extend(abs(z) for z in a['logits'])
   row['views'][v]={'auxiliary_only':auxstats,'frozen_base':basestats,'base_saved_logit_max_error':max_base_error,
    'mean_abs_aux_logit':statistics.mean(changes),'max_abs_aux_logit':max(changes),
    'mean_target_margin_before':statistics.mean(oldmargins),'mean_target_margin_after':statistics.mean(margins),
    'same_forward_component_sum_exact':True}
  hist=read(U/f'{ARM}_{seed}_history.json');selected=hist[r['selected_epoch']-1]
  for v in VIEWS:
   assert selected['aux_'+v+'_top1']==r['aux_metrics'][v]['top1_correct']
   assert abs(selected['aux_'+v+'_ce']-r['aux_metrics'][v]['mean_ce'])<1e-10
  row['selected_train_aux_online_accuracy_percent']=selected['train_aux_correct_online']/899*100
  row['online_training_metric_not_frozen_model_eval']=True
  out.append(row)
 return {'status':'PASS','per_seed':out,'extra_saved_predictions_verified':3600,'final_rescoring':0}
