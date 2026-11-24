"""Apply the predeclared evidence rules; no model training or ground-truth claims."""
from pathlib import Path
import json, hashlib, copy
from collections import Counter
from temporal_support import support, selfcheck

U=Path(__file__).resolve().parent
FEATURES=('index_only','open_four')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def coherent(states,times,target,mode,identity_concern=False):
    raw={f:support(states[f],times,target,mode) for f in FEATURES}
    usable=copy.deepcopy(raw)
    if all(usable[f]['value']==1 for f in FEATURES):
        for f in FEATURES:
            if usable[f]['kind']=='temporal_estimate':
                usable[f].update(value=None,kind='unknown',reason='feature_conflict')
    if identity_concern:
        for f in FEATURES:
            if usable[f]['kind']=='temporal_estimate':
                usable[f].update(value=None,kind='unknown',reason='identity_uncertain')
    return raw,usable

def count_masks(rows):
    out={}
    for mode in ('A','B','C'):
        accepted=[r for r in rows if r['outputs'][mode]['value'] is not None]
        matches=sum(r['outputs'][mode]['value']==r['hidden_draft'] for r in accepted)
        out[mode]={'total':len(rows),'estimated':len(accepted),'draft_match':matches,
                   'draft_conflict':len(accepted)-matches,'withheld':len(rows)-len(accepted)}
    return out

def run():
    destination=U/'COMPARISON.json'
    if destination.exists(): raise RuntimeError('Saved comparison exists; do not overwrite/relabel to improve it.')
    frozen=read(U/'FROZEN.json'); idx=read(U/'visual_inputs/INDEX.json')
    draft=read(U/'review/IMAGE_LABELS.json'); independent=read(U/'independent/CENTER_LABELS.json')
    # Independent visual continuity review is attached later, without changing raw estimates.
    assert len(idx['rows'])==18 and len(draft['rows'])==234
    lut={(r['case_id'],r['image_key']):r for r in draft['rows']}
    assert len(lut)==234
    indep={r['case_id']:r for r in independent['rows']}
    assert len(indep)==18
    assert draft['rubric_sha256']==independent['rubric_sha256']==sha(frozen['rubric_source'])
    for r in draft['rows']:
        assert all(r[f] in (0,1,None) for f in FEATURES)
        assert not all(r[f]==1 for f in FEATURES)
    old=read(U.parent/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json')['rows']
    oldmap={(r['alias'],r['frame_id']):r for r in old}
    sequences=[]; masks=[]; centers=[]; transitions=[]; preserve=Counter()
    for case in idx['rows']:
        frames=sorted(case['frames'],key=lambda r:r['actual_pts_us'])
        times=[r['actual_pts_us'] for r in frames]
        assert len(frames)==13 and all(b['frame_id']==a['frame_id']+1 for a,b in zip(frames,frames[1:]))
        center=next(i for i,r in enumerate(frames) if r['offset_frames']==0)
        states={f:[lut[(case['case_id'],r['image_key'])][f] for r in frames] for f in FEATURES}
        seq={'case_id':case['case_id'],'label':case['label'],'group':case['group'],
             'person_id':case['recorded_person_id'],'inherited_role':case['inherited_role'],
             'side':case['side'],'center_index':center,'frames':[]}
        for i,fr in enumerate(frames):
            outputs={m:coherent(states,times,i,m) for m in ('A','B','C')}
            seq['frames'].append({'image_key':fr['image_key'],'frame_id':fr['frame_id'],
                'pts_us':times[i],'offset_frames':fr['offset_frames'],
                'direct':{f:states[f][i] for f in FEATURES},
                'raw_outputs':{m:outputs[m][0] for m in outputs},
                'outputs':{m:outputs[m][1] for m in outputs}})
            for f in FEATURES:
                if states[f][i] is not None:
                    for m in outputs:
                        assert outputs[m][1][f]['value']==states[f][i]
                        preserve[m]+=1
                # Every known interior state is masked independently. Endpoints excluded only.
                if states[f][i] is not None and 0<i<len(frames)-1:
                    masked=copy.deepcopy(states); masked[f][i]=None
                    raw={}; usable={}
                    for m in ('A','B','C'):
                        r,u=coherent(masked,times,i,m); raw[m]=r[f]; usable[m]=u[f]
                    masks.append({'case_id':case['case_id'],'group':case['group'],
                      'person_id':case['recorded_person_id'],'feature':f,'index':i,
                      'frame_id':fr['frame_id'],'pts_us':times[i],
                      'hidden_draft':states[f][i],'raw_outputs':raw,'outputs':usable})
                if i==center:
                    previous=oldmap.get((case['source_alias'],fr['frame_id']))
                    centers.append({'case_id':case['case_id'],'group':case['group'],
                       'person_id':case['recorded_person_id'],'feature':f,
                       'direct':states[f][i],'independent_direct':indep[case['case_id']][f],
                       'previous_sparse_available':previous is not None,
                       'previous_sparse':previous['labels'][case['side']+'_'+f] if previous else None,
                       'outputs':{m:outputs[m][1][f] for m in outputs}})
        for f in FEATURES:
            known=[i for i,x in enumerate(states[f]) if x is not None]
            for a,b in zip(known,known[1:]):
                if states[f][a]!=states[f][b]:
                    transitions.append({'case_id':case['case_id'],'feature':f,'from_index':a,
                        'to_index':b,'from':states[f][a],'to':states[f][b],
                        'elapsed_us':times[b]-times[a], 'unknown_frames_between':b-a-1,
                        'meaning':'Change of image-review label; not measured finger movement duration.'})
        sequences.append(seq)
    actual_summary={}
    for m in ('A','B','C'):
        unknown=[r for r in centers if r['direct'] is None]
        actual_summary[m]={'centers':36,'direct_known':36-len(unknown),
            'original_unknown':len(unknown),
            'new_estimates':sum(r['outputs'][m]['value'] is not None for r in unknown),
            'remaining_unknown':sum(r['outputs'][m]['value'] is None for r in unknown)}
    all_unknown={}
    for m in ('A','B','C'):
        cells=[x for s in sequences for x in s['frames']]
        originally_unknown=[(x,f) for x in cells for f in FEATURES if x['direct'][f] is None]
        all_unknown[m]={'original_unknown':len(originally_unknown),
            'new_estimates':sum(x['outputs'][m][f]['value'] is not None for x,f in originally_unknown)}
    agreement=Counter()
    for r in centers:
        a,b=r['direct'],r['independent_direct']
        k='both_unknown' if a is None and b is None else 'one_unknown' if a is None or b is None else 'known_agree' if a==b else 'known_conflict'
        agreement[k]+=1
    pins={str(p):sha(p) for p in [U/'FROZEN.json',U/'COHERENCE_CONTRACT.json',
       U/'temporal_support.py',Path(__file__),U/'visual_inputs/INDEX.json',
       U/'review/IMAGE_LABELS.json',U/'independent/CENTER_LABELS.json']}
    result={'status':'COMPUTED_NOT_HUMAN_VALIDATED','pins':pins,'selfcheck':selfcheck(),
       'evaluation_scope':'18 selected public windows; same-AI image drafts. No human truth, TCN or DTW accuracy evaluation.',
       'center_summary':actual_summary,'all_frames_unknown_summary':all_unknown,
       'center_review_agreement':dict(agreement),'known_states_preserved':dict(preserve),
       'masked_summary':count_masks(masks),
       'masked_by_group':{g:count_masks([r for r in masks if r['group']==g]) for g in sorted({r['group'] for r in masks})},
       'masked_by_person':{str(p):count_masks([r for r in masks if r['person_id']==p]) for p in (1,2,3)},
       'masked_by_feature':{f:count_masks([r for r in masks if r['feature']==f]) for f in FEATURES},
       'centers':centers,'sequences':sequences,'masked_trials':masks,'observed_label_transitions':transitions,
       'limitations':['Unknown real states have no hidden ground truth.',
          'Neighboring frames and masked trials are correlated, not independent subjects.',
          'No learned RGB recognizer; manual AI-observation inputs to a common temporal rule.',
          'C uses future frames and is unsuitable for zero-delay feedback.',
          'A motion hidden entirely between observations cannot be excluded from equal endpoints.',
          'Existing TCN training sources, not a fresh recognition test.'],
       'training_ready':False}
    dump(destination,result)
    print(json.dumps({k:result[k] for k in ('center_summary','all_frames_unknown_summary','center_review_agreement','known_states_preserved','masked_summary','observed_label_transitions')},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
