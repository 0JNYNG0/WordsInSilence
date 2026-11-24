"""Pure fixed observation qualification. No model imports or label access."""
import numpy as np

HIGH=.8
LOW=.2
MAX_INTERVAL_CANDIDATES=8

def _axis(point_probabilities,image_available,fragment_ids=None):
    point=np.asarray(point_probabilities,float);av=np.asarray(image_available,bool)
    assert point.shape==(len(av),2) and np.isfinite(point).all()
    assert ((point>=0)&(point<=1)).all()
    if fragment_ids is None:fragment_ids=np.zeros(len(av),dtype=int)
    frag=np.asarray(fragment_ids)
    assert frag.shape==av.shape
    # Any unavailable image breaks recurrence of a qualifying evidence chain.
    groups=[];current=[]
    for i in range(len(av)):
        if not av[i]:
            if current:groups.append(current);current=[]
        else:
            if current and frag[i]!=frag[current[-1]]:groups.append(current);current=[]
            current.append(i)
    if current:groups.append(current)
    return point,av,groups

def propose_motion_intervals(point_probabilities,image_available,fragment_ids=None):
    point,av,groups=_axis(point_probabilities,image_available,fragment_ids)
    candidates=[]
    for group in groups:
        # One-frame predicates are retained. No dwell/pause/minimum mass is required.
        setup=[i for k,i in enumerate(group) if point[i,0]>=HIGH and (k==0 or point[group[k-1],0]<HIGH)]
        terminal_runs=[]
        for i in group:
            if point[i,1]>=HIGH:
                if not terminal_runs or terminal_runs[-1][-1]!=i-1:terminal_runs.append([])
                terminal_runs[-1].append(i)
        for s in setup:
            for run in terminal_runs:
                after=[t for t in run if t>s]
                if not after:continue
                t=after[0]
                candidates.append(dict(start=s,end=t,output_indices=list(range(s,t+1)),
                    setup_index=s,terminal_index=t,setup_probability=float(point[s,0]),
                    terminal_probability=float(point[t,1])))
    candidates.sort(key=lambda r:(r["end"],r["start"]))
    return dict(candidates=candidates[:MAX_INTERVAL_CANDIDATES],uncapped_candidates=len(candidates),
        capped=len(candidates)>MAX_INTERVAL_CANDIDATES,max_candidates=MAX_INTERVAL_CANDIDATES,
        selection="Chronological terminal then setup; no score/duration optimization; one start per confident setup run and first later point of each terminal run")

def qualify_process(raw_process_probability,point_probabilities,image_available,
                    interval_probabilities,fragment_ids=None,fragment_provenance="availability_detector_only"):
    p=float(raw_process_probability);assert np.isfinite(p) and 0<=p<=1
    spec=propose_motion_intervals(point_probabilities,image_available,fragment_ids)
    scores=np.asarray(interval_probabilities,float)
    assert scores.shape==(len(spec["candidates"]),) and np.isfinite(scores).all()
    assert ((scores>=0)&(scores<=1)).all()
    candidates=[dict(r,motion_probability=float(q)) for r,q in zip(spec["candidates"],scores)]
    chain=[r for r in candidates if r["motion_probability"]>=HIGH]
    av=np.asarray(image_available,bool)
    if p>=HIGH and chain:status="OBSERVED_PROCESS_EVIDENCE";display=1
    elif not av.any():status="INSUFFICIENT_OBSERVATION";display=None
    elif p<=LOW and chain:status="PROCESS_AND_LOCAL_EVIDENCE_DISAGREE";display=None
    elif p<=LOW:status="PROCESS_NOT_CONFIRMED";display=None
    else:status="PROCESS_EVIDENCE_UNCONFIRMED";display=None
    return dict(status=status,qualified_display=display,raw_probability=p,
        raw_display=1 if p>=HIGH else 0 if p<=LOW else None,
        witnessed_chain=chain[0] if display==1 else None,interval_candidates=candidates,
        candidate_count=len(candidates),uncapped_candidates=spec["uncapped_candidates"],candidate_cap_reached=spec["capped"],
        unavailable_indices=np.flatnonzero(~av).tolist(),fragment_provenance=fragment_provenance,
        source_seam_oracle_diagnostic=fragment_provenance=="known_source_edit_metadata",
        observation_only=True,observed_process_absence_certified=False,
        whole_word_correctness=None,learner_error=None,required_repetitions=None,
        complete_action_verified=False,raw_low_is_not_word_error=True)

def qualify_with_scorer(raw_process_probability,point_probabilities,image_available,
                       score_intervals,fragment_ids=None,fragment_provenance="availability_detector_only"):
    """score_intervals receives one list of index lists, returns same-order probabilities.
    Caller makes ONE batched interval forward using the already prepared887 inputs;
    the pure functions never run a model or use annotations to select intervals.
    """
    spec=propose_motion_intervals(point_probabilities,image_available,fragment_ids)
    indices=[r["output_indices"] for r in spec["candidates"]]
    scores=score_intervals(indices) if indices else []
    return qualify_process(raw_process_probability,point_probabilities,image_available,scores,
                           fragment_ids,fragment_provenance)

def known_seam_diagnostic(runtime_result,seam_output_indices):
    """Filter the SAME scored <=8 candidates; no new proposal/forward or oracle score.
    This privileged edit-lineage diagnostic cannot replace real-runtime results.
    """
    import copy
    out=copy.deepcopy(runtime_result);seams=set(int(s) for s in seam_output_indices)
    for c in out['interval_candidates']:
        c['crosses_known_source_seam']=any(c['start']<s<=c['end'] for s in seams)
    chain=[c for c in out['interval_candidates'] if c['motion_probability']>=HIGH and not c['crosses_known_source_seam']]
    p=out['raw_probability'];out['qualified_display']=None;out['witnessed_chain']=None
    if p>=HIGH and chain:
        out['status']='OBSERVED_PROCESS_EVIDENCE';out['qualified_display']=1;out['witnessed_chain']=chain[0]
    elif p<=LOW and chain:out['status']='PROCESS_AND_LOCAL_EVIDENCE_DISAGREE'
    elif out['status']=='INSUFFICIENT_OBSERVATION':pass
    elif p<=LOW:out['status']='PROCESS_NOT_CONFIRMED'
    else:out['status']='PROCESS_EVIDENCE_UNCONFIRMED'
    out.update(fragment_provenance='known_source_edit_metadata',source_seam_oracle_diagnostic=True,
        source_seam_indices=sorted(seams),same_runtime_candidates_and_scores_reused=True,
        independent_real_runtime_generalization=False)
    return out

