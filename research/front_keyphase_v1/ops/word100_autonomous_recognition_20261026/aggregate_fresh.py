"""Fixed E3, previously frozen weights. Reads each stored fresh score only."""
from common import *
from compare_saved import metrics,pairs

F=U/'fresh199'

def main():
 ctx=read(F/'EVAL_CONTEXT.json'); verified=read(F/'INPUTS_VERIFIED.json')
 assert verified['status']=='PASS' and not (F/'EVAL_RUNNING.json').exists()
 frozen=read(U/'FROZEN.json')
 for p,h in frozen['pins'].items():assert sha(p)==h,p
 models={f'{a}{s}':read(F/f'{a}_{s}_SCORES.json') for a,ss in [('C',(20261013,)),('T3F',SEEDS)] for s in ss}
 for v in models.values():
  assert v['context_sha256']==sha(F/'EVAL_CONTEXT.json') and v['model_state_before']==v['model_state_after']
  assert v['training']==0 and not v['checkpoint_reselected']
 byid={r['clip_id']:r for r in ctx['rows']};assert len(byid)==199
 output={}; stored={}
 for view in VIEWS:
  rr={k:sorted(d['predictions'][view],key=lambda x:x['clip_id']) for k,d in models.items()}
  for a in rr.values():assert len(a)==199 and {r['clip_id'] for r in a}==set(byid)
  ensemble=[]
  for i,r in enumerate(rr['T3F20261013']):
   current=[rr[f'T3F{s}'][i] for s in SEEDS]
   assert all((q['clip_id'],q['label_index'],q['signer_id'])==(r['clip_id'],r['label_index'],r['signer_id']) for q in current)
   assert all(r[k]==byid[r['clip_id']][k] for k in ['label_index','signer_id'])
   ok=all(q['logits'] is not None for q in current)
   p=combine([q['logits'] for q in current]) if ok else None; order=ranking(p) if ok else []
   ensemble.append(dict(clip_id=r['clip_id'],label_index=r['label_index'],signer_id=r['signer_id'],probabilities=p.tolist() if ok else None,prediction=order[0] if ok else None,top5=order[:5],correct=bool(ok and order[0]==r['label_index']),top5_correct=r['label_index'] in order[:5],status='OK' if ok else 'INPUT_UNAVAILABLE'))
  output[view]={'C13':metrics(rr['C20261013']),'T3F13':metrics(rr['T3F20261013']),'E3':metrics(ensemble),'individual_T3F':{str(s):metrics(rr[f'T3F{s}']) for s in SEEDS},'E3_vs_T3F13':pairs(rr['T3F20261013'],ensemble),'E3_vs_C13':pairs(rr['C20261013'],ensemble)}
  stored[view]=ensemble
 write(F/'ENSEMBLE_ROWS.json',stored)
 check={v:{'top1_nondecreasing':output[v]['E3']['correct']>=output[v]['T3F13']['correct'],'top5_nondecreasing':output[v]['E3']['top5']>=output[v]['T3F13']['top5'],'each_recorded_id_nondecreasing':all(output[v]['E3']['by_signer'][s]['correct']>=output[v]['T3F13']['by_signer'][s]['correct'] for s in output[v]['E3']['by_signer'])} for v in VIEWS}
 write(F/'COMPARISON.json',dict(status='FRESH_EVALUATED_PENDING_SCALAR_VERIFICATION',at=now(),scores=output,checks=check,records=199,scope='100 fixed words; recorded IDs14/15; actual individual identity not verified; one unresolved source kept outside original eligible manifest',failures_remain_in_denominator=True,selection_correction=read(U/'FRESH_SELECTION_CORRECTION.json'),fresh_model_pins={k:d['checkpoint'] for k,d in models.items()},raw_score_pins={str(F/f'{a}_{s}_SCORES.json'):sha(F/f'{a}_{s}_SCORES.json') for a,ss in [('C',(20261013,)),('T3F',SEEDS)] for s in ss},new_inference=0,training=0,method_changed_after_fresh=False))
 print(json.dumps(output,ensure_ascii=False))

if __name__=='__main__':main()
