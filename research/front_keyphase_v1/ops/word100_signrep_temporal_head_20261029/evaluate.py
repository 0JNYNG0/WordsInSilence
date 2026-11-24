from train_compare import *
def binary(y,p):
 y=np.asarray(y,int);p=np.asarray(p)>=.5
 tp=int(((y==1)&p).sum());tn=int(((y==0)&~p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum())
 return dict(n=len(y),positive=int((y==1).sum()),negative=int((y==0).sum()),tp=tp,tn=tn,fp=fp,fn=fn,
  accuracy=(tp+tn)/len(y) if len(y) else None,
  balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
def main():
 info=read(U/'INPUTS.json');rows=info['rows'];targets=info['targets'];training=read(U/'TRAINING.json');labels=read(U/'LABELS.json')
 arrays=[dict(np.load(r['input'],allow_pickle=False)) for r in rows];byalias={r['alias']:i for i,r in enumerate(rows)}
 groups={role:[i for i,r in enumerate(rows) if r['role']==role] for role in sorted({r['role'] for r in rows})};groups['all_held']=[i for i,r in enumerate(rows) if r['role']!='fit_candidate']
 scores=[];event_details=[]
 for run in training['runs']:
  path=U/'runs'/run['name']/'probabilities.npy';assert sha(path)==run['probabilities_sha256'];p=np.load(path,allow_pickle=False)
  for group,ids in groups.items():
   per={}
   for j,k in enumerate(targets):
    yy=np.concatenate([a['y'][:,j][a['target_mask'][:,j]] for i,a in enumerate(arrays) if i in ids]);pp=np.concatenate([p[i,:len(a['y']),j][a['target_mask'][:,j]] for i,a in enumerate(arrays) if i in ids]);per[k]=binary(yy,pp)
   ba=[v['balanced_accuracy'] for v in per.values() if v['balanced_accuracy'] is not None]
   scores.append(dict(run=run['name'],condition=run['condition'],seed=run['seed'],group=group,clips=len(ids),per_target=per,macro_balanced_accuracy=float(np.mean(ba)),known_cells=sum(v['n'] for v in per.values())))
  for s in labels['sequences']:
   i=byalias[s['source_alias']];a=arrays[i];lookup={int(f):ii for ii,f in enumerate(a['frame_id'])}
   for feature in ['index_only','open_four']:
    k=s['side']+'_'+feature
    if k not in targets:continue
    j=targets.index(k);known=[f for f in s['frames'] if f['direct'][feature] is not None and a['target_mask'][lookup[f['frame_id']],j]]
    for left,right in zip(known,known[1:]):
     u,v=left['direct'][feature],right['direct'][feature];li=lookup[left['frame_id']];ri=lookup[right['frame_id']]
     # Non-adjacent equal labels do not prove no hidden transition in a gap.
     if u==v and right['frame_id']-left['frame_id']!=1:continue
     correct=bool((p[i,li,j]>=.5)==bool(u) and (p[i,ri,j]>=.5)==bool(v))
     event_details.append(dict(run=run['name'],condition=run['condition'],seed=run['seed'],case=s['case_id'],alias=s['source_alias'],role=s['role'],feature=k,left=left['frame_id'],right=right['frame_id'],true_left=u,true_right=v,change=u!=v,pred_left=float(p[i,li,j]),pred_right=float(p[i,ri,j]),both_endpoints_correct=correct,predicted_flip=bool((p[i,li,j]>=.5)!=(p[i,ri,j]>=.5))))
 summaries=[]
 for group in groups:
  for condition in ['pose_only','pose_plus_signrep']:
   ss=[r for r in scores if r['group']==group and r['condition']==condition];vals=[r['macro_balanced_accuracy'] for r in ss]
   summaries.append(dict(group=group,condition=condition,mean=float(np.mean(vals)),range=[min(vals),max(vals)],per_seed=vals,clips=ss[0]['clips'],known_cells=ss[0]['known_cells']))
 temporal=[]
 for condition in ['pose_only','pose_plus_signrep']:
  for scope in ['fit','held']:
   for seed in [42,43,44]:
    ev=[e for e in event_details if e['condition']==condition and e['seed']==seed and ((e['role']=='fit_candidate')==(scope=='fit'))];change=[e for e in ev if e['change']];stable=[e for e in ev if not e['change']]
    temporal.append(dict(condition=condition,scope=scope,seed=seed,change_intervals=len(change),correct_change_endpoints=sum(e['both_endpoints_correct'] for e in change),stable_adjacent_pairs=len(stable),false_flips=sum(e['predicted_flip'] for e in stable)))
 save('SCORES.json',dict(rows=scores,summary=summaries,temporal=temporal,note='Physical-label agreement on a small previously inspected development set, not word recognition or learner-correction accuracy. Adjacent frames correlated.'))
 save('TEMPORAL_DETAILS.json',event_details)
 print(json.dumps(dict(summary=summaries,temporal=temporal),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
