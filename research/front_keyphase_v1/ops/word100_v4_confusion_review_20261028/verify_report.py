"""Separate saved-array arithmetic, not new inference or independent human labels."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
U=Path(__file__).resolve().parent; S=U.parent/'word100_cpu_model_selection_20261028'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read(U/'SUMMARY.json');m=read(U/'MATRICES.json');classes=m['classes'];checks=[]
def ck(name,ok):
    checks.append(dict(name=name,pass_=bool(ok)))
    assert ok,name
preds={};keys=None
for model,runs in m['runs'].items():
    for r in runs:
        z=np.load(S/f'models/{model}_{r["seed"]}_dev.npz',allow_pickle=False)
        predictions=z['logits'].argmax(1).tolist();targets=z['targets'].tolist();ids=z['ids'].tolist()
        counts=Counter(zip(targets,predictions))
        ck(f'{model}_{r["seed"]}_45x45',all(r['cm'][a][b]==counts[a,b] for a in range(45) for b in range(45)))
        ck(f'{model}_{r["seed"]}_totals',sum(counts.values())==488 and sum(a==b for a,b in zip(targets,predictions))==r['correct'])
        if keys is None:keys=ids
        ck(f'{model}_{r["seed"]}_same_ids',ids==keys)
        preds[model,r['seed']]=predictions
v4=[[preds['v4hybrid',seed][i] for seed in [28101,28102,28103]] for i in range(488)]
ck('all_correct',sum(all(p==y for p in pp) for pp,y in zip(v4,targets))==s['right_all_three'])
ck('all_wrong',sum(all(p!=y for p in pp) for pp,y in zip(v4,targets))==s['wrong_all_three'])
ck('partition488',s['right_all_three']+s['wrong_all_three']+s['mixed']==488)
allpairs=Counter(tuple(sorted((y,p))) for pp,y in zip(v4,targets) for p in pp if p!=y)
ck('pairs',sum(allpairs.values())==s['v4_wrong_votes'] and all(allpairs[p['a'],p['b']]==p['count'] for p in s['top_pairs']))
ck('top4share',abs(sum(v for _,v in allpairs.most_common(4))/sum(allpairs.values())-s['top4_error_share'])<1e-12)
summary=read(S/'SUMMARY.json')
ck('original_v4_mean',abs(s['v4_accuracy']-next(x['accuracy_mean'] for x in summary['models'] if x['name']=='v4hybrid'))<1e-12)
manifest=read(S/'TRAINING_MANIFEST.json');rows={r['id']:r for r in manifest['rows']}
selection=read(U/'EXAMPLE_SELECTION.json')
ck('only_fit_dev_examples',all(rows[i]['role'] in ['fit','validation'] for i in selection['images_read']))
for pair in selection['examples']:
    a,b=pair['a'],pair['b'];eligible=[ids[i] for i,y in enumerate(targets) if (y==a and b in v4[i]) or (y==b and a in v4[i])]
    ck('pair_examples_'+str(a)+'_'+str(b),[r['id'] for r in pair['examples']]==sorted(eligible)[:3])
    for example in pair['examples']:
        idx=ids.index(example['id'])
        ck('display_prediction_'+example['id'][:10],example['predictions']==[classes[p] for p in v4[idx]])
for p,h in read(U/'SOURCE_PINS.json').items():ck('source_'+Path(p).name,sha(Path(p))==h)
for p,h in read(S/'PROTECTED.json').items():ck('operating_'+Path(p).name,sha(Path(p))==h)
for name in ['v4_overview.png','v4_confusion_full.png','v4_confusion_full.svg']:ck(name,(U/'figures'/name).exists())
result=dict(status='PASS',checks=checks,check_count=len(checks),verification='same Codex separate Counter-based arithmetic; not independent human relabeling',new_inference=False,training=False)
(U/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',checks=len(checks))))
