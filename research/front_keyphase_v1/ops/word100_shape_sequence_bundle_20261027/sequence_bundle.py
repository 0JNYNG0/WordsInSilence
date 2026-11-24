"""Lossless observed-feature sequences. No inference, imputation or correctness grading."""
from collections import Counter

FEATURES=('index_only','open_four')

def build_feature(frames,feature):
    assert frames and feature in FEATURES
    times=[f['pts_us'] for f in frames]
    ids=[f['frame_id'] for f in frames]
    assert all(a<b for a,b in zip(times,times[1:]))
    assert all(b==a+1 for a,b in zip(ids,ids[1:]))
    values=[f['direct'][feature] for f in frames]
    assert all(v is None or type(v) is int and v in (0,1) for v in values)
    runs=[]
    for i,v in enumerate(values):
        if not runs or runs[-1]['value']!=v:
            runs.append({'value':v,'start_index':i,'end_index':i,'indices':[i]})
        else:
            runs[-1]['end_index']=i;runs[-1]['indices'].append(i)
    for r in runs:
        a,b=r['start_index'],r['end_index']
        r.update(start_pts_us=times[a],end_pts_us=times[b],start_frame_id=ids[a],end_frame_id=ids[b],
                 count=len(r['indices']),kind='unobserved_run' if r['value'] is None else 'observed_equal_feature_run',
                 time_scope='recorded samples only; not proof of continuous state between frames')
    known=[i for i,v in enumerate(values) if v is not None]
    links=[]
    for a,b in zip(known,known[1:]):
        if values[a]!=values[b]: kind='bounded_feature_change'
        elif b>a+1: kind='same_endpoints_unresolved_gap'
        else: continue
        links.append({'kind':kind,'from_index':a,'to_index':b,'from_value':values[a],'to_value':values[b],
                      'from_pts_us':times[a],'to_pts_us':times[b], 'unobserved_indices':list(range(a+1,b)),
                      'exact_transition_time':None,'intermediate_shape':None,'linguistic_importance':None})
    unknown_runs=[r for r in runs if r['value'] is None]
    gaps=[]
    for r in unknown_runs:
        a,b=r['start_index'],r['end_index']
        left=a-1 if a>0 else None;right=b+1 if b+1<len(frames) else None
        gaps.append({'from_index':a,'to_index':b,'indices':r['indices'],
                     'left_anchor':left,'right_anchor':right,
                     'kind':'entire_window_unobserved' if left is None and right is None else 'leading_unobserved' if left is None else 'trailing_unobserved' if right is None else 'interior_unobserved',
                     'value':None,'gap_not_imputed':True})
    coverage={'known_samples':len(known),'unobserved_samples':len(frames)-len(known),'expected_samples':len(frames),
              'observed_fraction':len(known)/len(frames),'expected_from_pts_us':times[0],'expected_to_pts_us':times[-1],
              'meaning':'sample observability, not similarity or correctness; fixed full window denominator'}
    return {'feature':feature,'values':values,'runs':runs,'links':links,'gaps':gaps,'coverage':coverage,
            'semantic_complete':False,'no_change_certified':False,'whole_sign_scope':False,
            'feedback_verdict':None,'handshape_name':None}

def summarize(sequences):
    c=Counter()
    for s in sequences:
        for f in s['features'].values():
            c['feature_samples']+=len(f['values']);c['known_samples']+=f['coverage']['known_samples']
            c['unobserved_samples']+=f['coverage']['unobserved_samples']
            c['feature_runs']+=len(f['runs'])
            c.update(r['kind'] for r in f['runs']);c.update(e['kind'] for e in f['links'])
            c.update(e['kind'] for e in f['gaps'])
    return dict(c)
