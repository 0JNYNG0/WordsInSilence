"""Paired arithmetic between new and original saved LOPO outputs; no models."""
from pathlib import Path
import json,hashlib,datetime
from collections import defaultdict
import numpy as np

U=Path(__file__).resolve().parent
OLD=U.parent/'hand_transition_learning'
E=U/'saved_evaluation'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def paired_summary(rows):
    known=[r for r in rows if r['eligible']]
    return dict(total=len(rows),eligible=len(known),
        original_correct=sum(r['old_correct'] for r in known),new_correct=sum(r['new_correct'] for r in known),
        repaired=sum(not r['old_correct'] and r['new_correct'] for r in known),
        regressed=sum(r['old_correct'] and not r['new_correct'] for r in known),
        unchanged_correct=sum(r['old_correct'] and r['new_correct'] for r in known),
        unchanged_wrong=sum(not r['old_correct'] and not r['new_correct'] for r in known))

def groups(rows,fields):
    g=defaultdict(list)
    for r in rows:g[tuple(r[f]for f in fields)].append(r)
    return [dict(**dict(zip(fields,k)),counts=paired_summary(v))for k,v in sorted(g.items())]

def compare():
    assert (E/'COMPLETE.json').exists() and (OLD/'saved_evaluation/COMPLETE.json').exists()
    files=[E/'ATTRIBUTE_ROWS.json',OLD/'saved_evaluation/ATTRIBUTE_ROWS.json',E/'WINDOW_DECISIONS.npz',OLD/'saved_evaluation/WINDOW_DECISIONS.npz',E/'CANONICAL_SOURCE_ROWS.json',OLD/'saved_evaluation/CANONICAL_SOURCE_ROWS.json',U/'EVALUATION_ITEMS.json',OLD/'EVALUATION_ITEMS.json']
    write(E/'PAIRED_COMPARISON_FROZEN.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rule='Exact seed/person/clip/frame/side/attribute or seed/window identity. Compare all stored cases, no threshold or candidate selection.',
        limitation='Same already examined developmental people and labels; rotation design was informed by original failure. Not a fresh generalization or educational benchmark.',
        pins={str(p):sha(p)for p in files+[Path(__file__)]}))
    ni=read(U/'EVALUATION_ITEMS.json');oi=read(OLD/'EVALUATION_ITEMS.json')
    assert ni==oi,'Fixed cases/labels must match exactly'
    key=lambda r:(r['seed'],r['clip_id'],r['frame'],r['side'],r['attribute'])
    a=read(E/'ATTRIBUTE_ROWS.json');b={key(r):r for r in read(OLD/'saved_evaluation/ATTRIBUTE_ROWS.json')};ar=[]
    assert len(a)==len(b)
    for r in a:
        o=b[key(r)];assert r['target']==o['target'] and r['eligible']==o['eligible']
        ar.append(dict(seed=r['seed'],person=r['person'],attribute=r['attribute_name'],side=r['side'],eligible=r['eligible'],
            old_correct=o['prediction']==r['target'],new_correct=r['prediction']==r['target']))
    a=np.load(E/'WINDOW_DECISIONS.npz');b=np.load(OLD/'saved_evaluation/WINDOW_DECISIONS.npz');wr=[]
    ix={(int(s),int(w)):i for i,(s,w)in enumerate(zip(b['seed'],b['window_id']))}
    assert len(a['seed'])==len(ix)
    for i,(seed,wid)in enumerate(zip(a['seed'],a['window_id'])):
        seed=int(seed);wid=int(wid);j=ix[seed,wid];w=ni['windows'][wid]
        assert a['target'][i]==b['target'][j] and a['eligible'][i]==b['eligible'][j]
        kind=w['kind']+('_closed_arm_raise'if w.get('body_raise_stable')else'')
        wr.append(dict(seed=seed,person=w['person'],kind=kind,reverse=w['reverse'],eligible=bool(a['eligible'][i]),
            old_correct=bool(b['prediction'][j]==b['target'][j]),new_correct=bool(a['prediction'][i]==a['target'][i])))
    a=read(E/'CANONICAL_SOURCE_ROWS.json');b={(r['seed'],r['clip_id'],r['kind']):r for r in read(OLD/'saved_evaluation/CANONICAL_SOURCE_ROWS.json')};cr=[]
    for r in a:
        o=b[r['seed'],r['clip_id'],r['kind']];assert r['source_frames']==o['source_frames'] and r['target']==o['target']
        cr.append(dict(seed=r['seed'],person=r['person'],clip_id=r['clip_id'],kind=r['kind'],frames=r['source_frames'],
            target=r['target'],old_prediction=o.get('prediction'),new_prediction=r.get('prediction'),
            old_confidence=o.get('confidence'),new_confidence=r.get('confidence'),old_pass=o['passed'],new_pass=r['passed']))
    out=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='PAIRED_SAVED_COMPARISON_COMPLETE_NO_OLD_OR_NEW_MODEL_EXECUTION',
        attributes_by_seed=groups(ar,['seed']),attributes_by_seed_attribute=groups(ar,['seed','attribute']),
        attributes_by_seed_person_attribute=groups(ar,['seed','person','attribute']),
        windows_by_seed=groups(wr,['seed']),windows_by_seed_kind=groups(wr,['seed','kind','reverse']),
        windows_by_seed_person_kind=groups(wr,['seed','person','kind','reverse']),canonical=cr,
        original_canonical_pass=sum(r['old_pass']for r in cr),new_canonical_pass=sum(r['new_pass']for r in cr),
        canonical_total=len(cr),source_people=6,transition_expressions=1,dependent_windows=True,
        old_models_rerun=False,thresholds_changed=False,labels_changed=False,private_access=False,final8=False)
    write(E/'PAIRED_COMPARISON.json',out)
    print('Paired canonical:',out['original_canonical_pass'],'->',out['new_canonical_pass'],'/',out['canonical_total'],flush=True)

if __name__=='__main__':compare()
