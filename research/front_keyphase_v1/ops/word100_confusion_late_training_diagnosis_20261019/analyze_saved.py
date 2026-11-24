from common import *
from collections import Counter,defaultdict
import math

def main():
 c=read(U/'CONTRACT.json');ctx=read(FACT/'TRAIN_CONTEXT.json');labels=ctx['labels'];arms=c['fixed_analysis']['arms'];seeds=c['fixed_analysis']['seeds']
 clips={};runs=[];pins={};pred_n=0
 for a in arms:
  for s in seeds:
   rp=resultpath(a,s);hp=historypath(a,s);pins[str(rp)]=sha(rp);pins[str(hp)]=sha(hp);r=read(rp);hist=read(hp)
   for row in r['predictions']['sensor']:
    pred_n+=1;cid=row['clip_id'];y=row['label_index'];logits=row['logits'];assert row['status']=='OK'
    order=sorted(range(100),key=lambda k:(-logits[k],k));assert row['correct']==(order[0]==y)
    margin=logits[y]-max(z for k,z in enumerate(logits) if k!=y)
    q=clips.setdefault(cid,{'label':labels[y],'label_index':y,'signer':row['signer_id'],'wrong':0,'predictions':[]})
    q['wrong']+=int(order[0]!=y);q['predictions'].append({'arm':a,'seed':s,'correct':row['correct'],'predicted':labels[order[0]],'rank':order.index(y)+1,'margin':margin,'ce':row['cross_entropy']})
   def norm(h):return {'epoch':h['epoch'],'train_ce':h['train_ce'],'ce':h.get('selection_ce',h.get('selected_rule_ce')),'sensor':h['sensor_top1'],'masked':h['masked_top1']}
   hs=[norm(x) for x in hist];late=hs[60:80];prev=hs[40:60];assert len(late)==20
   def summary(xs):
    return {'sensor_range':[min(x['sensor'] for x in xs),max(x['sensor'] for x in xs)],'sensor_mean':sum(x['sensor'] for x in xs)/len(xs),'ce_mean':sum(x['ce'] for x in xs)/len(xs),'train_ce_mean':sum(x['train_ce'] for x in xs)/len(xs)}
   diffs=[late[i]['sensor']-late[i-1]['sensor'] for i in range(1,20)];lossdownaccdown=[x['epoch'] for p,x in zip(late,late[1:]) if x['ce']<p['ce'] and x['sensor']<p['sensor']]
   phase={str(p):summary([x for x in late if (x['epoch']-1)%4==p-1]) for p in [1,2,3,4]}
   # Remove each four-epoch block mean to describe phase association without an overall late trend.
   phase_res={str(p):float(np.mean([late[i]['sensor']-np.mean([x['sensor'] for x in late[(i//4)*4:(i//4+1)*4]]) for i in range(20) if i%4==p-1])) for p in [1,2,3,4]}
   changes=[{'epoch':x['epoch'],'delta_correct':x['sensor']-p['sensor'],'ce_before':p['ce'],'ce_after':x['ce']} for p,x in zip(late,late[1:]) if abs(x['sensor']-p['sensor'])>=4]
   best=hs[0]
   for h in hs[1:]:
    if h['ce']<best['ce']-1e-6:best=h
   assert best['epoch']==r['selected_epoch']
   runs.append({'arm':a,'seed':s,'selected_epoch':r['selected_epoch'],'selected':best,'late':summary(late),'preceding':summary(prev),'late_series':late,
    'late_adjacent_max_change':max(map(abs,diffs)),'late_adjacent_mean_abs_change':float(np.mean(np.abs(diffs))),
    'ce_improves_while_top1_falls_epochs':lossdownaccdown,'large_adjacent_changes':changes,'phase_summaries':phase,'phase_demeaned_top1':phase_res,
    'selected_vs_late_max_count_gap':max(x['sensor'] for x in late)-best['sensor'],'posthoc_reselection':False})
 ordered=sorted(clips,key=lambda k:(-clips[k]['wrong'],k));focus=ordered[:5]
 for cid,q in clips.items():
  q['wrong_predictions']=dict(Counter(x['predicted'] for x in q['predictions'] if not x['correct']))
  q['wrong_rank_histogram']=dict(Counter(x['rank'] for x in q['predictions'] if not x['correct']))
 totalwrong=sum(x['wrong'] for x in clips.values());persistent=[k for k in ordered if clips[k]['wrong']>=6]
 out={'at':now(),'status':'STORED_ONLY','predictions':pred_n,'history_epochs':sum(len(read(historypath(a,s))) for a in arms for s in seeds),
 'total_wrong_repeated':totalwrong,'wrong_rank_le2':sum(not x['correct'] and x['rank']<=2 for q in clips.values() for x in q['predictions']),
 'wrong_rank_le5':sum(not x['correct'] and x['rank']<=5 for q in clips.values() for x in q['predictions']),
 'unique_error_clips':sum(x['wrong']>0 for x in clips.values()),'persistent_error_clips':persistent,'focus':focus,
 'focus_wrong_total':sum(clips[k]['wrong'] for k in focus),'clips':clips,'runs':runs,'pins':pins,'code_sha256':sha(__file__),
 'no_per_epoch_individual_predictions':'Only selected checkpoint predictions are saved for all runs. Individual frame/clip flips over epochs cannot be reconstructed without new inference; not performed.',
 'normalization_code_facts':{'layers':8,'momentum':0.1,'per_epoch_batches':29,'last_batch_clips':3,'regular_batch_clips':32,'last_batch_weight':0.1,'last5_batches_weight':1-0.9**5,'lr':0.001,'lr_decay':False,'padding_excluded':True,'causal_confirmed':False}}
 write(U/'SAVED_ANALYSIS.json',out)
 print(json.dumps({'wrong':totalwrong,'unique':out['unique_error_clips'],'focus':[{**{k:clips[x][k] for k in ['label','signer','wrong','wrong_predictions']},'clip':x} for x in focus],'late':[{k:r[k] for k in ['arm','seed','selected_epoch','late','preceding','late_adjacent_max_change','ce_improves_while_top1_falls_epochs','selected_vs_late_max_count_gap','phase_demeaned_top1']} for r in runs]},ensure_ascii=False))
if __name__=='__main__':main()
