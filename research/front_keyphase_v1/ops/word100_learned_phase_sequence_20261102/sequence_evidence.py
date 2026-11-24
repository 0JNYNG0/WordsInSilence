"""Physical state observations precede DTW's monotonic matching.

Learn the supported precedence inventory from FIT observations, not word-name
exceptions. This supplies evidence, not a complete linguistic error verdict.
"""
import numpy as np
from phase_model import read,S,exclusive

def learn_graph():
    t=read(S/'PHYSICAL_STATE_TARGETS.json');graphs={}
    for word in sorted({g['word_id'] for g in t['groups']}):
        gs=[g['index'] for g in t['groups'] if g['word_id']==word];rows=[r for r in t['rows'] if r['role']=='fit' and r['word_id']==word];edges=[];audit=[]
        for a in gs:
            for b in gs:
                if a>=b:continue
                before=[];after=[];mixed=[]
                for r in rows:
                    aa=[p['frame'] for p in r['points'] if p['mask'] and p['group']==a];bb=[p['frame'] for p in r['points'] if p['mask'] and p['group']==b]
                    if not aa or not bb:continue
                    if max(aa)<min(bb):before.append(r['person'])
                    elif max(bb)<min(aa):after.append(r['person'])
                    else:mixed.append(r['person'])
                audit.append(dict(pair=[a,b],a_before_b=before,b_before_a=after,interleaved=mixed))
                if len(before)>=3 and not after and not mixed:edges.append([a,b])
                elif len(after)>=3 and not before and not mixed:edges.append([b,a])
        graphs[word]=dict(groups=gs,precedences=edges,audit=audit,no_fixed_repetition_count=True)
    return dict(graphs=graphs,source='FIT sparse visible physical configurations; required educational validity is not claimed',runtime='no reference videos read; no clock position supplied to classifier')

def observe(prob,available,scope,graph,times):
    available=np.asarray(available,bool);scope=np.asarray(scope,int);n=len(prob)
    assert len(available)==n and scope.min()>=0 and scope.max()<n
    states=[];top=prob.argmax(1);conf=prob.max(1)
    inside=np.zeros(n,bool);inside[scope]=True
    for g in graph['groups']:
        chosen=(top==g)&(conf>=.8)&available&inside
        # Existing two-frame evidence requirement; never bridge an unavailable frame.
        keep=chosen&np.r_[False,chosen[:-1]] | chosen&np.r_[chosen[1:],False]
        ix=np.flatnonzero(keep);states.append(dict(group=g,observed=len(ix)>=2,frames=ix.tolist(),times=[float(times[k]) for k in ix],
            quartiles=np.quantile(ix,[.25,.5,.75]).tolist() if len(ix)>=2 else None))
    by={s['group']:s for s in states};edges=[]
    for a,b in graph['precedences']:
        aa=by[a]['quartiles'];bb=by[b]['quartiles']
        if aa is None or bb is None:status='ORDER_UNCONFIRMED'
        elif aa[2]<bb[0]:status='SUPPORTED_ORDER_OBSERVED'
        elif aa[0]>bb[2]:status='PHYSICAL_ORDER_CONFLICT'
        else:status='ORDER_OVERLAPS_OR_UNCERTAIN'
        edges.append(dict(before=a,after=b,status=status))
    gaps=scope[~available[scope]].tolist();missing=[s['group'] for s in states if not s['observed']]
    if gaps:status='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    elif any(e['status']=='PHYSICAL_ORDER_CONFLICT' for e in edges):status='PHYSICAL_ORDER_CONFLICT'
    elif missing or any(e['status']!='SUPPORTED_ORDER_OBSERVED' for e in edges):status='IMPORTANT_STATE_OR_ORDER_UNCONFIRMED'
    elif not edges:status='CONFIGURATIONS_OBSERVED_ORDER_RULE_UNSUPPORTED'
    else:status='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
    return dict(status=status,states=states,precedences=edges,missing_groups=missing,image_gap_frames=gaps,whole_sign_correctness=None,
      scope='native-order physical evidence in preserved coarse main window; not whole KSL correctness',no_reference_video_comparisons=True)
