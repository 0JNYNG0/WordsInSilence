"""Preserve a read-only review of existing block-permutation fixtures and choices."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
S=Path(__file__).resolve().parent;G=S.parent.parent;O=G.parent
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dest=S/'FIXTURE_SPLICE_REVIEW.json';assert not dest.exists()
    inp=G/'dynamic_blocks/INPUTS_COMPLETE.json';comparison=S/'evaluation/COMPARISON.json';basep=O/'word100_candidate_role_learning_20261111/EVALUATION_INPUTS.json'
    data=rd(inp)['rows'];comp=rd(comparison)['rows'];base={r['clip_id']:r for r in rd(basep)['rows'] if r['kind']=='full_normal'}
    pins={str(p):sha(p) for p in [inp,comparison,basep,G/'run_dynamic_block_diagnostic.py',G/'reviews/sequence_target/TARGETS.json',G/'reviews/sequence_target/MOTION_EXTENTS.json']}
    source_checks=[];cards=[]
    for r in data:
        before=base[r['clip_id']];assert sha(r['path'])==r['sha256'];assert sha(before['path'])==before['sha256']
        pins[r['path']]=r['sha256'];pins[before['path']]=before['sha256'];z=np.load(r['path']);b=np.load(before['path']);ids=z['source_indices'];ts=z['ts'];t=r['target']
        assert np.array_equal(np.sort(ids),np.arange(len(ids))) and np.array_equal(z['x'],b['x'][ids]) and np.array_equal(ts,b['ts'])
        assert (np.diff(ts)>0).all() and np.array_equal(ids,t['query_to_source_frame_ids'])
        cuts=(np.flatnonzero(np.diff(ids)!=1)+1).tolist();assert cuts==[j['query_right'] for j in t['splice_joins'] if j['synthetic_discontinuity']]
        joins=[dict(query_frames=[j-1,j],source_frames=ids[j-1:j+1].tolist(),elapsed_us=int(ts[j]-ts[j-1])) for j in cuts]
        source_checks.append(dict(clip_id=r['clip_id'],frames=len(ids),source_permutation_exact=True,coordinates_permutation_exact=True,timestamps_preserved=True,joins=joins,semantic_contract=t['semantic_direction_order']))
        for row in comp:
            if row['clip_id']!=r['clip_id'] or row['kind']!='dynamic_complete_blocks':continue
            for c in row['conditions']['progression_separate']['cards']:
                if 'truth' not in c:continue
                span=c['chosen']['span'] if c['chosen'] else None;a,b=span if span else [-1,-1]
                cs=[j for j in cuts if a<j<=b]
                cards.append(dict(seed=row['seed'],clip_id=r['clip_id'],card=c['card'],selected_span=span,source_frames=ids[a:b+1].tolist() if span else [],crossed_splices=cs,witness_preserved=c['truth']['all_witness'],wrong_points=c['truth']['wrong_points'],selection_is_current_frozen_extent=True,semantic_completion=None))
    failures=[r for r in cards if not r['witness_preserved'] or r['wrong_points']]
    assert len(source_checks)==4 and len(cards)==18 and len(failures)==5 and all(r['crossed_splices'] for r in failures)
    result=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='SAVED_FIXTURE_AND_CHOICE_AUDIT',source_checks=source_checks,cards=cards,
        summary=dict(input_clips=4,limited_descriptor_cards=18,witness_preserved=13,witness_failed=5,failed_spans_crossing_synthetic_join=5),failures=failures,
        finding='All five current witness-failed spans cross source-discontinuous synthetic joins. This is association, not proof that joins are the only cause.',
        scope='Original block internals and source-coordinate samples remain unchanged; distinct original times become adjacent on the retained query timestamp axis. This is an edited-observation robustness test, not naturally performed order-error accuracy.',
        existing_contract_preserved='Retain all failures and denominators. Egypt is source-identity-only and excluded from semantic order negative gold. Do not inject known cuts into runtime or choose a seam radius using these DEV results.',
        proposed_fit_plan=[
            'Before training, audit fixed public FIT continuous original transitions, fast short movements/crossing/occlusion returns, and predetermined synthetic joins. Keep source-person split and unknown labels.',
            'Source-contiguous frames are not automatically error-free keypoints; noncontiguous synthetic frames are not automatically physically distinguishable. Record ambiguous negatives instead of forcing continuity labels.',
            'Use body-normalized local relative trajectories, actual dt, masks, velocity consistency and both-arm geometry without target word, expected order or source indices as model features.',
            'If suitable labels exist, a continuity/quality component may withhold unsupported connections; any reranking must preserve raw candidates and prove retained target witnesses. Never call all-abstention success.',
            'Keep normal fast movement, crossing, short valid actions and missing-observation regressions separate from synthetic-splice detection. Natural continuous wrong-order performance remains unverified without corresponding labeled public evidence.'
        ],
        no_new_inference=True,no_training=True,old_files_modified=False,private_access=False,final8=False,web_changed=False,pins=pins,script_sha256=sha(__file__),reviewer='Separate Codex saved-data/code review, not independent human semantic validation')
    for p,h in pins.items():assert sha(p)==h
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps(result['summary']))
if __name__=='__main__':main()
