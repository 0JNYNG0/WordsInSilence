"""Descriptive overlap-window sensitivity after an unchanged early stop; no re-selection."""
from common import *
import math
from collections import Counter

def arithmetic(h,view):
 if len(h)<2:return {'epochs':len(h),'first':h[0]['epoch'] if h else None,'last':h[-1]['epoch'] if h else None,**{k:None for k in ['accuracy_mean','sd_pp','adjacent_mean_pp','adjacent_max_pp','accuracy_min','accuracy_max','mean_ce']}}
 x=[r[view+'_top1']/3 for r in h];ce=[r[view+'_ce'] for r in h];n=len(x);mean=sum(x)/n
 changes=[abs(x[i]-x[i-1]) for i in range(1,n)]
 return {'epochs':n,'first':h[0]['epoch'],'last':h[-1]['epoch'],'accuracy_mean':mean,
  'sd_pp':math.sqrt(sum((v-mean)**2 for v in x)/n),'adjacent_mean_pp':sum(changes)/(n-1),'adjacent_max_pp':max(changes),'accuracy_min':min(x),'accuracy_max':max(x),'mean_ce':sum(ce)/n}

def main():
 r=read(U/'COMPARISON_INDEPENDENT.json');check(r['pins']);seeds=[20261013,20261014,20261015]
 out={'at':now(),'method':'Prespecified secondary descriptive overlap comparison if either paired history ends early by frozen patience12. For each paired seed restrict both histories to [61,min(completed_epochs)]. No outcome-dependent interval search, no new training or checkpoint selection. Primary frozen-window results preserved in COMPARISON_INDEPENDENT.json.','paired':[],'user_data_used':False,'new_inference':0,'new_training':0}
 for seed in seeds:
  a=read(BASE/f'training_C_{seed}_history.json');b=read(U/f'T3_{seed}_history.json');end=min(a[-1]['epoch'],b[-1]['epoch']);aa=[x for x in a if 61<=x['epoch']<=end];bb=[x for x in b if 61<=x['epoch']<=end]
  assert [x['epoch'] for x in aa]==[x['epoch'] for x in bb]
  out['paired'].append({'seed':seed,'window':[61,end],'C':{v:arithmetic(aa,v) for v in VIEWS},'T3':{v:arithmetic(bb,v) for v in VIEWS}})
 keys=['accuracy_mean','sd_pp','adjacent_mean_pp','adjacent_max_pp','mean_ce']
 out['three_seed_mean']={a:{v:{k:sum(row[a][v][k] for row in out['paired'])/3 if all(row[a][v][k] is not None for row in out['paired']) else None for k in keys} for v in VIEWS} for a in ['C','T3']}
 labels=read(U/'TRAIN_CONTEXT.json')['labels'];mistakes={}
 for arm in ['C','T3']:
  counts=Counter();byclip={}
  for seed in seeds:
   p=BASE/f'training_C_{seed}_RESULT.json' if arm=='C' else U/f'{arm}_{seed}_RESULT.json'
   for row in read(p)['predictions']['sensor']:
    if not row['correct']:
     counts[row['clip_id']]+=1;byclip[row['clip_id']]={'clip_id':row['clip_id'],'true_label':labels[row['label_index']],'signer_id':row['signer_id']}
  mistakes[arm]=[{**byclip[c],'wrong_seeds':n} for c,n in counts.most_common()]
 out['selected_sensor_errors']=mistakes
 out['pins']={str(U/'COMPARISON_INDEPENDENT.json'):sha(U/'COMPARISON_INDEPENDENT.json'),str(Path(__file__).resolve()):sha(__file__)}
 write(U/'LATE_MATCHED_WINDOW_AUDIT.json',out,new=True)
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
