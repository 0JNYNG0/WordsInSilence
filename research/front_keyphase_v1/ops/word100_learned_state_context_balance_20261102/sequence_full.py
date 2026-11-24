"""Never shrink the comparison axis to visible or DTW-selected frames."""
import numpy as np

def observe(prob,available,graph,times):
    available=np.asarray(available,bool);n=len(prob);assert len(times)==n and len(available)==n
    top=prob.argmax(1);conf=prob.max(1);states=[]
    for g in graph['groups']:
        chosen=(top==g)&(conf>=.8)&available
        keep=(chosen&np.r_[False,chosen[:-1]])|(chosen&np.r_[chosen[1:],False])
        ix=np.flatnonzero(keep);runs=[]
        for i in ix:
            if not runs or i!=runs[-1][-1]+1:runs.append([int(i)])
            else:runs[-1].append(int(i))
        states.append(dict(group=g,observed=len(ix)>=2,frames=ix.tolist(),runs=runs,times=[float(times[i]) for i in ix],
          isolated_candidate_frames=np.flatnonzero(chosen&~keep).tolist()))
    by={r['group']:r for r in states};edges=[]
    for a,b in graph['precedences']:
        aa=by[a]['frames'];bb=by[b]['frames']
        if not aa or not bb:status='ORDER_UNCONFIRMED'
        elif max(aa)<min(bb):
            # A single confident contrary instant is not silently erased by the
            # two-frame confirmation rule. Keep it as uncertain short evidence.
            sa=by[a]['isolated_candidate_frames'];sb=by[b]['isolated_candidate_frames']
            status='SHORT_EVENT_ORDER_UNCONFIRMED' if any(i>min(bb) for i in sa) or any(i<max(aa) for i in sb) else 'SUPPORTED_ORDER_OBSERVED'
        elif min(aa)>max(bb):status='PHYSICAL_ORDER_CONFLICT'
        else:status='ORDER_OVERLAPS_OR_UNCERTAIN'
        edges.append(dict(before=a,after=b,status=status))
    gaps=np.flatnonzero(~available).tolist();missing=[r['group'] for r in states if not r['observed']]
    if gaps:status='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    elif any(e['status']=='PHYSICAL_ORDER_CONFLICT' for e in edges):status='PHYSICAL_ORDER_CONFLICT'
    elif missing or any(e['status']!='SUPPORTED_ORDER_OBSERVED' for e in edges):status='IMPORTANT_STATE_OR_ORDER_UNCONFIRMED'
    elif not edges:status='CONFIGURATIONS_OBSERVED_ORDER_RULE_UNSUPPORTED'
    else:status='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
    return dict(status=status,states=states,precedences=edges,missing_groups=missing,image_gap_frames=gaps,
      comparison_frame_range=[0,n-1],comparison_frames=n,rest_observed_frames=np.flatnonzero((top==9)&(conf>=.8)&available).tolist(),
      uncertain_frames=np.flatnonzero((conf<.8)|~available).tolist(),whole_sign_correctness=None,
      scope='Whole native timeline, physical evidence only. Rest does not erase time. Repeated/interleaved states remain uncertain where no supported rule exists.')
