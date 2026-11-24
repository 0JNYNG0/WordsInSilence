"""Saved-logit recount and actual batch/selection/checkpoint audit. No inference."""
from common import *
import math,argparse,torch
from collections import Counter
SEEDS=[20261013,20261014,20261015];VIEWS=['masked','sensor']

def statehash(state):
 h=hashlib.sha256()
 for k,t in sorted(state.items()):
  h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.cpu().numpy().tobytes())
 return h.hexdigest()

def recount(scores,groups):
 assert len(scores)==300 and len({r['clip_id'] for r in scores})==300
 assert Counter(r['signer_id'] for r in scores)=={11:100,12:100,13:100}
 assert Counter(r['label_index'] for r in scores)=={k:3 for k in range(100)}
 out=[]
 for r in scores:
  if r['status']=='OK':
   logits=r['logits'];assert len(logits)==100 and all(math.isfinite(x) for x in logits)
   ranking=sorted(range(100),key=lambda i:(-logits[i],i));correct=ranking[0]==r['label_index'];top5=r['label_index'] in ranking[:5]
   maximum=max(logits);ce=maximum+math.log(sum(math.exp(x-maximum) for x in logits))-logits[r['label_index']]
   assert r['prediction']==ranking[0] and r['top5']==ranking[:5]
  else:
   assert r['logits'] is None and r['prediction'] is None;correct=top5=False;ce=math.log(100)
  assert r['correct']==correct and r['top5_correct']==top5 and abs(r['cross_entropy']-ce)<1e-10
  out.append({'clip_id':r['clip_id'],'signer':r['signer_id'],'correct':correct,'top5':top5,'ce':ce,'failure':r['status']!='OK','group':groups[r['clip_id']]})
 return {'correct':sum(x['correct'] for x in out),'top5':sum(x['top5'] for x in out),'mean_ce':sum(x['ce'] for x in out)/300,'failures':sum(x['failure'] for x in out),'signers':{str(s):sum(x['correct'] for x in out if x['signer']==s) for s in [11,12,13]},'groups':{g:{'total':sum(x['group']==g for x in out),'correct':sum(x['correct'] for x in out if x['group']==g)} for g in sorted(set(groups.values()))}}

def trace_check(result,context):
 stem=f"{result['arm']}_{result['seed']}";hist=read(U/(stem+'_history.json'));last=torch.load(U/(stem+'_last.pt'),map_location='cpu',weights_only=True)
 checkpoint=torch.load(U/(stem+'_best.pt'),map_location='cpu',weights_only=True);assert sha(U/(stem+'_best.pt'))==result['checkpoint_sha256']
 assert result['context_sha256']==sha(U/'TRAIN_CONTEXT.json')==checkpoint['context_sha256']==last['context_sha256']
 assert result['initial_sha256']==read(BASE/f"training_C_{result['seed']}_RESULT.json")['initial_state_sha256']
 assert statehash(checkpoint['state_dict'])==statehash(last['best']['state_dict']) and result['predictions']==last['best']['predictions']
 assert hist==last['history'] and len(hist)==result['completed_epochs']==last['epoch']
 best=None;clips=context['ordered_train_ids'];batches=0
 for epoch,h in enumerate(hist,1):
  assert h['epoch']==epoch
  if best is None or h['selection_ce']<best['selection_ce']-1e-6:best=h
  assert h['best_epoch']==best['epoch']
  if epoch<len(hist):assert epoch-best['epoch']<12
  path=U/'traces'/stem/f'{epoch:03}.json';assert sha(path)==h['trace_sha256'];tr=read(path)
  assert tr['arm']==result['arm'] and tr['seed']==result['seed'] and tr['epoch']==epoch
  expected=[clips[i] for i in np.random.default_rng(result['seed']+epoch).permutation(899)]
  actual=[cid for b in tr['batches'] for cid in b['clips']];assert actual==expected and len(set(actual))==899
  assert len(tr['batches'])==29
  for step,b in enumerate(tr['batches'],1):
   assert b['step']==step and len(b['clips'])==(32 if step<29 else 3)
   for cid,ch in zip(b['clips'],b['choices']):
    parity=int(hashlib.sha256(cid.encode()).hexdigest(),16)%2
    sp=int(hashlib.sha256(('sensor_view:'+cid).encode()).hexdigest(),16)%2
    assert ch==['sensor' if ((epoch-1)//2+sp)%2 else 'masked','x_context' if (epoch+parity)%2 else 'x_whole']
   batches+=1
 assert best['epoch']==result['selected_epoch']==checkpoint['epoch'] and abs(best['selection_ce']-result['selected_ce'])<1e-12
 assert len(hist)==80 or len(hist)-best['epoch']>=12
 return {'epochs':len(hist),'batches':batches,'actual_source_exposures':899*len(hist),'selected_epoch':best['epoch']}

def gate(new,old):
 nv=new['sensor'];ov=old['sensor'];checks={
  'sensor_mean_higher':sum(x['correct'] for x in nv)>sum(x['correct'] for x in ov),
  'sensor_better_in_two_seeds':sum(x['correct']>y['correct'] for x,y in zip(nv,ov))>=2,
  'masked_mean_preserved':sum(x['correct'] for x in new['masked'])>=sum(x['correct'] for x in old['masked']),
  'sensor_top5_preserved':sum(x['top5'] for x in nv)>=sum(x['top5'] for x in ov)}
 for v in VIEWS:
  for s in ['11','12','13']:checks[v+'_signer_'+s]=sum(x['signers'][s] for x in new[v])>=sum(x['signers'][s] for x in old[v])
  for g in new[v][0]['groups']:checks[v+'_group_'+g]=sum(x['groups'][g]['correct'] for x in new[v])>=sum(x['groups'][g]['correct'] for x in old[v])
 return {'pass':all(checks.values()),'checks':checks,'statistical_proof':False,'automatic_adoption':False}

def main():
 assert not (U/'TRAIN_RUNNING.json').exists()
 context=read(U/'TRAIN_CONTEXT.json');check(context['pins']);check(read(U/'READY.json')['pins']);check(read(U/'PREP_CONTEXT.json')['pins'])
 metrics={a:{v:[] for v in VIEWS} for a in ['C','P','Q']};traces={};pairs={};pins={};stored={}
 for arm in metrics:
  for seed in SEEDS:
   path=BASE/f'training_C_{seed}_RESULT.json' if arm=='C' else U/f'{arm}_{seed}_RESULT.json';r=read(path);pins[str(path)]=sha(path);stored[arm,seed]=r
   for v in VIEWS:
    m=recount(r['predictions'][v],context['missingness_groups'][v]);old=r['metrics'][v]
    assert m['correct']==old['top1_correct'] and m['top5']==old['top5_correct'] and m['failures']==old['input_failures'] and abs(m['mean_ce']-old['mean_ce'])<1e-10
    metrics[arm][v].append(m)
   if arm!='C':traces[f'{arm}_{seed}']=trace_check(r,context)
 summary={a:{v:{'top1_counts':[x['correct'] for x in mm],'top5_counts':[x['top5'] for x in mm],'accuracy_mean':sum(x['correct'] for x in mm)/9,'accuracy_range':[min(x['correct'] for x in mm)/3,max(x['correct'] for x in mm)/3]} for v,mm in vv.items()} for a,vv in metrics.items()}
 for arm in ['P','Q']:
  pairs[arm]={}
  for seed in SEEDS:
   pairs[arm][str(seed)]={}
   for v in VIEWS:
    before={x['clip_id']:x['correct'] for x in stored['C',seed]['predictions'][v]};after={x['clip_id']:x['correct'] for x in stored[arm,seed]['predictions'][v]};assert set(before)==set(after)
    gain=[k for k in before if not before[k] and after[k]];loss=[k for k in before if before[k] and not after[k]]
    pairs[arm][str(seed)][v]={'improved':gain,'regressed':loss,'delta_pp':(len(gain)-len(loss))/3}
  for v in VIEWS:summary[arm][v]['delta_C_pp']=sum(x['correct']-y['correct'] for x,y in zip(metrics[arm][v],metrics['C'][v]))/9
 result={'status':'PASS_SAVED_SCORES_AND_TRACES','at':now(),'summary':summary,'metrics':metrics,'paired':pairs,'gates':{'P_vs_C':gate(metrics['P'],metrics['C']),'Q_vs_C':gate(metrics['Q'],metrics['C']),'Q_vs_P':gate(metrics['Q'],metrics['P'])},'traces':traces,'pins':pins,'verification_code_sha256':sha(__file__),'trained_model_inference':0,'user_data_used':False,'production_changed':False,'test_scope':'Repeatedly used public development set; not blind or actual webcam user accuracy.'}
 write(U/'COMPARISON_INDEPENDENT.json',result,new=True);print(json.dumps({'summary':summary,'gates':result['gates']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
