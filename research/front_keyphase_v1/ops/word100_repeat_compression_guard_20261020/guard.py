"""Offline safeguards for alternative traversal evidence and collapsed motion samples."""
import itertools
import numpy as np
SIDES=('L','R');AXES=('handshape','location','movement')
PARTS=10;MIN_PEOPLE=5

def groups(required):
    return [0,1] if required==['L'] else [2,3] if required==['R'] else [0,1,2,3]

def motion_parts(ref,required):
    """Ten equal observed arm-travel portions; not linguistic/essential stages."""
    ids=groups(required);core=ref['reference_core'];v=ref['valid'][:,ids]
    joint=v[1:]&v[:-1];delta=np.nan_to_num(ref['xy'][1:,ids]-ref['xy'][:-1,ids])
    length=np.linalg.norm(delta,axis=2)
    step=np.where(joint,length,0).sum(1)/np.maximum(joint.sum(1),1)
    step*=core[1:]&core[:-1]&(joint.mean(1)>=.5)
    cum=np.r_[0,np.cumsum(step)];total=cum[-1]
    if total<=1e-12:return []
    # Boundaries are determined only from the TRAIN reference.
    cuts=np.searchsorted(cum,np.linspace(0,total,PARTS+1),side='left')
    cuts[0]=max(0,int(np.flatnonzero(step>0)[0]))
    out=[]
    for i,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        if b>a and cum[b]>cum[a]:
            out.append(dict(part=i,start=int(a),end=int(b),travel=float(cum[b]-cum[a]),
                            moving_edges=[int(e) for e in range(a,b) if step[e]>0],
                            source_fraction=float((b-a)/(len(core)-1))))
    return out

def all_return_witnesses(q,r,entries,required):
    n=len(q['ts']);m=len(r['ts']);ci=np.flatnonzero(r['reference_core']);anchor=np.zeros(n,bool)
    if len(ci) and ci[0]>0 and ci[-1]<m-1:
        ids=groups(required);joint=q['valid'][:,None,ids]&r['valid'][None,:,ids]
        dist=np.linalg.norm(np.nan_to_num(q['xy'][:,None,ids]-r['xy'][None,:,ids]),axis=3)
        cost=np.where(joint,dist,1.5).mean(2);support=joint.mean(2)
        ranges=[np.arange(ci[0]),np.arange(ci[-1]+1,m),ci]
        near=[rg[np.argmin(cost[:,rg],axis=1)] for rg in ranges];ii=np.arange(n)
        anchor=(support[ii,near[0]]>=.5)&(support[ii,near[1]]>=.5)
        anchor&=(cost[ii,near[0]]<cost[ii,near[2]])&(cost[ii,near[1]]<cost[ii,near[2]])
    out={}
    for e in entries:
        k=e['cycles']
        if k==1:continue
        bounds=[]
        p=np.asarray(e['path'],int);good=np.asarray(e['good'],bool);qi,ri=p.T
        for c in range(k-1):
            left=qi[good&(ri//m==c)&r['reference_core'][ri%m]]
            right=qi[good&(ri//m==c+1)&r['reference_core'][ri%m]]
            mask=np.zeros(n,bool)
            if e['available'] and len(left) and len(right):
                mask=anchor&(np.arange(n)>left.max())&(np.arange(n)<right.min())
            bounds.append(np.flatnonzero(mask).tolist())
        out[str(k)]=dict(available=e['available'],boundaries=bounds)
    return out

def joint_return(witnesses,people,k,n):
    eligible=[i for i,w in enumerate(witnesses) if w[str(k)]['available']]
    for ix in itertools.combinations(eligible,MIN_PEOPLE):
        shared=[]
        for c in range(k-1):
            common=set(range(n))
            for i in ix:common.intersection_update(witnesses[i][str(k)]['boundaries'][c])
            shared.append(sorted(common))
        if all(shared) and all(max(a)<min(b) for a,b in zip(shared,shared[1:])):
            return dict(supported=True,people=[people[i] for i in ix],boundaries=shared)
    return dict(supported=False,people=[],boundaries=[])

def compression_trace(ref,entry,parts,n):
    """One query sample cannot establish the order of observed reference motion."""
    mask=np.zeros(n,bool);events=[];ratios=[];unobserved=0;eligible=0
    if not entry['available']:
        return dict(available=False,mask=mask,events=[],eligible=0,unobserved=0,min_positive_ratio=None)
    m=len(ref['ts']);p=np.asarray(entry['path'],int);good=np.asarray(entry['good'],bool)
    qi,ri=p.T
    for cycle in range(entry['cycles']):
        for part in parts:
            start,end=part['start'],part['end']
            sel=good&(ri//m==cycle)&(ri%m>=start)&(ri%m<=end)&ref['reference_core'][ri%m]
            qs=np.unique(qi[sel]);rs=np.unique(ri[sel]%m)
            eligible+=1
            if len(qs)==0 or len(rs)<2:
                unobserved+=1;continue
            # At least one actual moving reference edge must have both endpoints supported.
            seen=set(rs.tolist())
            if not any(e in seen and e+1 in seen for e in part['moving_edges']):
                unobserved+=1;continue
            ratio=((qs[-1]-qs[0])/max(n-1,1))/part['source_fraction']
            if ratio>0:ratios.append(float(ratio))
            if len(qs)==1:
                mask[qs[0]]=True
                events.append(dict(cycle=cycle,part=part['part'],reference_start=start,reference_end=end,
                    supported_reference_samples=int(len(rs)),query_frame=int(qs[0]),query_samples=1,
                    source_fraction=part['source_fraction'],relative_duration=0.0))
    return dict(available=True,mask=mask,events=events,eligible=eligible,unobserved=unobserved,
                min_positive_ratio=min(ratios) if ratios else None)

def apply(q,refs,bank,people,required,projections,baseline,combine):
    assert len(set(people))==len(people)
    n=len(q['ts']);witnesses=[all_return_witnesses(q,r,b,required) for r,b in zip(refs,bank)]
    alternatives={str(k):joint_return(witnesses,people,k,n) for k in (2,3)}
    viable=[k for k,p in projections.items() if p['available']>=MIN_PEOPLE]
    previous=baseline['decision']['count']
    inspected=[previous] if previous is not None else viable
    traces=[];per_person=[]
    for person,ref,entries in zip(people,refs,bank):
        parts=motion_parts(ref,required)
        tt={str(k):compression_trace(ref,entries[k-1],parts,n) for k in inspected}
        # If count is unresolved, only flag collapse shared by every viable count.
        masks=[tt[str(k)]['mask'] for k in inspected]
        common=np.logical_and.reduce(masks) if masks else np.zeros(n,bool)
        per_person.append(common)
        traces.append(dict(person=person,parts=parts,counts=tt,common_mask=common))
    votes=np.sum(per_person,axis=0) if per_person else np.zeros(n,int)
    compression=votes>=MIN_PEOPLE
    conflicting=previous==1 and any(a['supported'] for a in alternatives.values())
    reasons=[]
    if conflicting:reasons.append('ALTERNATIVE_REPEAT_WITNESS')
    if compression.any():reasons.append('MOTION_COLLAPSED_TO_ONE_SAMPLE')
    chosen=None if reasons else previous
    p=combine(projections,dict(count=chosen),n)
    p['core']&=~compression
    p['unknown']=~(p['core']|p['outside'])
    for side in SIDES:
        p['rest'][side]&=~compression
        for axis in AXES:p['expected'][side][axis]&=~compression
    return dict(count=chosen,previous_count=previous,guard_reasons=reasons,
        alternative_evidence=alternatives,return_witnesses=witnesses,compression_traces=traces,
        compression_votes=votes,compression_mask=compression,projection=p,
        semantic_error=None,complete_performance=None)
