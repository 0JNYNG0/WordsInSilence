"""One fixed saved-path proposal; no model, DTW, labels or semantic verdict."""
import numpy as np

CONTEXT_FRAMES=12
MAX_PROPOSALS_PER_QUERY=1

def canonical_anchor(guides,word,canonical_frame_ids,prototype_length):
    ids=np.asarray(canonical_frame_ids)
    assert ids.ndim==1 and len(ids)==int(prototype_length)
    assert np.all(np.diff(ids)>0)
    rows=[g for g in guides if g["word_id"]==word]
    assert rows and all(g["boundary_provenance"]=="canonical01_saved_reviewed_span" for g in rows)
    lo=min(int(g["frames"][0]) for g in rows)
    hi=max(int(g["frames"][1]) for g in rows)
    # Convert source-native canonical frame IDs explicitly to prototype row indices.
    # Current saved engine uses these numbers directly: enforce its identity axis.
    selected=np.flatnonzero((ids>=lo)&(ids<=hi))
    assert len(selected) and ids[selected[0]]==lo and ids[selected[-1]]==hi
    assert np.array_equal(ids,np.arange(len(ids))),"Saved engine requires identity canonical native/index axis"
    return dict(source_native_extent=[lo,hi],prototype_index_extent=[int(selected[0]),int(selected[-1])],
        prototype_length=int(prototype_length),guide_ids=[g["canonical_stage_id"] for g in rows],
        scope="Hull of pre-existing canonical FIT01 guide context; not a component boundary or completion truth")

def make_proposal(path,frame_ids,timestamps_us,anchor):
    ids=np.asarray(frame_ids);ts=np.asarray(timestamps_us);p=np.asarray(path)
    n=len(ids);m=int(anchor["prototype_length"])
    assert n>=2 and ids.shape==ts.shape==(n,) and np.issubdtype(p.dtype,np.integer)
    assert np.all(np.diff(ids)>0) and np.all(np.diff(ts)>0)
    assert p.ndim==2 and p.shape[1]==2 and len(p)>=max(n,m)
    assert np.array_equal(p[0],[0,0]) and np.array_equal(p[-1],[n-1,m-1])
    delta=np.diff(p,axis=0)
    assert ((delta>=0)&(delta<=1)).all() and (delta.sum(1)>0).all()
    assert np.array_equal(np.unique(p[:,0]),np.arange(n))
    assert np.array_equal(np.unique(p[:,1]),np.arange(m))
    lo,hi=anchor["prototype_index_extent"]
    assert 0<=lo<=hi<m
    mapped=np.unique(p[(p[:,1]>=lo)&(p[:,1]<=hi),0])
    assert len(mapped)
    start=max(0,int(mapped[0])-CONTEXT_FRAMES);end=min(n-1,int(mapped[-1])+CONTEXT_FRAMES)
    return dict(status="PROPOSED_CONTEXT_ONLY",output_indices=list(range(start,end+1)),
        full_native_index_extent=[start,end],mapped_anchor_index_extent=[int(mapped[0]),int(mapped[-1])],
        source_native_frame_extent=[int(ids[start]),int(ids[end])],
        source_time_us_extent=[int(ts[start]),int(ts[end])],context_frames=CONTEXT_FRAMES,
        canonical_anchor=anchor,max_proposals_per_query=1,
        fallback="No full-clip or manually adjusted fallback. Invalid/missing path is unavailable and counts as a proposal miss.",
        exact_event_boundary=False,observed_process_truth=None,whole_word_correctness=None,
        additional_DTW=0,individual_reference_video_comparisons=0)

