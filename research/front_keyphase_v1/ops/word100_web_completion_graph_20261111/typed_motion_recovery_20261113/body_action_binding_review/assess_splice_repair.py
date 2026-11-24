"""Prospective FIT-only permutation provenance design; no arrays/labels/models generated."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import hashlib
import json

U = Path(__file__).resolve().parent
H, G = U.parent, U.parent.parent


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not (U/'SPLICE_REPAIR_DESIGN.json').exists()
    source_path=H/'palm_preserving_controls/FIXTURE_FROZEN.json'
    source=read(source_path)
    paths=[source_path,U/'BODY_POINT_CONTRACT.json',U/'EVIDENCE.json',
           G/'candidate_description_supervision/prepare_rank_learning.py',
           G/'candidate_description_supervision/prepare_local_rank.py',
           G/'candidate_description_supervision/train_local_rank.py',
           G/'candidate_description_supervision/source_rank_learning/INPUTS_COMPLETE.json',
           G/'candidate_description_supervision/local_rank_learning/FROZEN.json']
    pins={str(p):sha(p)for p in paths}
    old=read(paths[-2]);fit=[r for r in old['sequences']if r['split']=='train']
    rows=[]
    for r in source['sources']:
        if len(r['events'])!=2:
            continue
        a,b=sorted(r['events'],key=lambda e:e['source_frame_span'][0])
        wa,wb=a['direct_witness_frames'],b['direct_witness_frames']
        assert max(wa)<min(wb)
        cut=(max(wa)+min(wb))//2
        lo,hi=a['source_frame_span'][0],b['source_frame_span'][1]
        assert lo<=cut<hi
        blocks=[list(range(0,lo)),list(range(cut+1,hi+1)),list(range(lo,cut+1)),list(range(hi+1,r['frames']))]
        mapping=[f for block in blocks for f in block]
        assert sorted(mapping)==list(range(r['frames']))
        joins=[dict(query_right=i,source_left=mapping[i-1],source_right=mapping[i],
                    source_gap=mapping[i]-mapping[i-1])for i in range(1,len(mapping))if mapping[i]!=mapping[i-1]+1]
        assert set(wa)<=set(blocks[2]) and set(wb)<=set(blocks[1])
        rows.append(dict(clip_id=r['clip_id'],person=r['person'],source_sha256=r['source_sha256'],
            annotation_path=r['annotation_path'],event_ids=[a['event_id'],b['event_id']],
            original_source_region=[lo,hi],source_partition=[lo,cut,hi],
            blocks_in_query_order=[{'source_start':p[0],'source_end':p[-1]}for p in blocks if p],
            proposed_joins=joins,source_permutation_preserves_all_frames=True,
            each_event_witness_set_stays_in_one_contiguous_block=True,
            original_uncertain_bands_unchanged=True,
            provenance_use='Crossing a discontinuous source index join is negative ONLY for contiguous single-source occurrence selection, not for body-motion presence/direction/handshape/correctness.',
            caution='A split may pass through an old uncertainty band. Never declare whole occurrence complete from a block cut; original positives retained and inside-block targets mapped only if their entire original span remains contiguous.'))
    assert len(rows)==6
    decision=dict(status='DESIGN_ONLY_RECOMMEND_FIT_SPLICE_DATA_REPAIR_FIRST',
        recommendation='One isolated continuation of existing local_full candidate ranker architecture with FIT permutation negative bags and original/inside-block replay; no new body observer in this first test.',
        reason='All5 exposed failures are artificial joins with valid witness candidates already admitted. Current training code constructs original or whole-source-reversed windows only; it does not teach the explicit constructed noncontiguous case.',
        comparator='Saved immutable local_full baseline; do not rerun its completed training. Same frozen common100 encoder, hand/direction/presence modules and candidate enumeration.',
        candidate_scope='Only descriptor relation families2/3 read the new isolated rank output; families4/5/7 use old outputs byte-identically. No word-ID rule.',
        proposed_sources=rows,
        existing_training=dict(sequences=len(old['sequences']),fit_sequences=len(fit),fit_people=sorted({r['person']for r in fit}),
            source_transform_fields=sorted({k for r in fit for k in r if k in ('reverse','permutation','splice','source_map')}),
            code_review='prepare_rank_learning creates original/whole-source reverse and contiguous a:b windows; prepare_local_rank reuses those labels/spans; no synthetic source-join negative supervision.'),
        minimal_test_contract={
            'data':'Use only six FIT originals for new paired permutations. Original24FIT all-family bags stay available unchanged. All views and whole-query reversal follow existing preplanned recipe, not DEV score selection.',
            'mapping':'Preserve every source frame exactly once; swap two source partitions separated between the last first-event witness and first second-event witness. Preserve source frame IDs as provenance only; synthetic query has its own monotone PTS.',
            'negative':'For source-contiguous single-occurrence ranking, a candidate that crosses an explicit non-adjacent source join is a provenance-negative. This label says nothing about naturally filmed wrong movement or no motion. All mapped membership/visibility audits retained.',
            'positives':'Replay original normal/speed/reverse positives. A candidate wholly inside an unmodified block inherits the exact same source target only when all of its original frames are contiguous; complete-vs-partial target kept separate. No uncertainty band is cut into exact new gold.',
            'partial_presence':'Never feed join negatives to the progression-presence/hand/direction branches. Real partial and reversed attempted movement remain positive/unknown as originally annotated. A mixed window can contain motion while not being one source occurrence.',
            'features':'Recompute actual candidate-local input/relations and common100 encoding on the transformed coordinates and query dt; never splice old whole-video hidden states. No seam bit, source map, original ordinal or expected order enters the model/runtime.',
            'loss':'Keep original listwise rank objective; add one equally source/family-weighted paired synthetic bag per original bag. New seam-negative candidates contribute only to the single-occurrence rank denominator; unchanged inside-block positives retain numerator membership. Explicitly log bags with no intact full positive and do not force a positive there.',
            'duration_control':'Use the existing >=4-frame candidate enumerator. Balance/stratify seam negatives against unchanged windows of the same duration; no jump/speed cutoff, no failing DEV span templates.',
            'selection':'One pre-frozen recipe, final checkpoint, all seeds42/43/44; no DEV threshold/seed/epoch adjustment. Do not run a second body architecture in parallel.',
            'regression':'All4word frozen82 scenarios per seed only after FIT contract: normal/speed/cross/shape/direction preserved, full/partial witness recall preserved, static/hold absence unchanged; fewer seam selections with an intact single-source alternative. Report per family/length/person, no aggregate-only pass.',
            'scope_limit':'A pass repairs an artificial source-order stress test; it is not independent learner-error accuracy or full semantic recognition.',
            'stop':'If normal or partial attempts regress, or no good FIT positives survive source mapping, preserve failure and stop; do not tune on the82 exposed scenarios. Reconsider dense body observer only for residual non-seam association errors.'},
        alternative={
            'new_dense_body_observer':'Potential later diagnostic, not chosen first.',
            'cost':'Would introduce new sequence representation and ambiguous primary-role supervision from one expression/six people at the same time as known missing seam negatives.',
            'point_contract':'Body-wrist vertical motion is not the hand/palm descent primitive; forearm upright motion need not raise the palm. Do not use exclusive physical-axis negatives or coordinate-sign relabeling.',
            'why_deferred':'No evidence that the5 current failures need a new body observer once explicit FIT source-join negatives are supplied; isolate the data hypothesis first.'},
        no_training=True,no_model_inference=True,no_new_training_arrays=True,no_label_rewrite=True,
        synthetic_maps_are_proposals_not_executed_inputs=True,private_access=False,final8=False,pins=pins)
    for p,h in pins.items():
        assert sha(p)==h
    (U/'SPLICE_REPAIR_DESIGN.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n','utf-8')
    print('Recommended data repair first:',len(rows),'prospective FIT maps,',sum(len(r['proposed_joins'])for r in rows),'known joins; no arrays/train/inference')


if __name__=='__main__':
    main()
