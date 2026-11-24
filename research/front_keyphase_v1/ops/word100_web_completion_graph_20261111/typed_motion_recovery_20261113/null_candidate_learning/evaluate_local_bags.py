"""Saved-score arithmetic + frozen old-body gate; root executes once.

Never re-infers the new model. Partial-progress masks are fixed from source
annotations BEFORE saved train/probe prediction arrays are opened.
"""
from pathlib import Path
import sys
N = Path(__file__).resolve().parent
sys.path.insert(0, str(N.parent))
from common import *
from collections import Counter, defaultdict
import time, traceback


def duration_bucket(n):
    return ('4..6' if n <= 6 else '7..10' if n <= 10 else
            '11..15' if n <= 15 else '16..21' if n <= 21 else '22+')


def partial_candidates(source, row, spans, n):
    """Existing windows with >=2 but not all direct progression witnesses.

    Containment is within one same-family group's outer onset/offset bounds.
    Reverse windows are mapped to ORIGINAL frames before comparing witnesses.
    These are observed fragments, NOT complete signs or linguistically errors.
    """
    assert int(source['frames']) == n
    events = {e['event_id']: e for e in source['events']}
    groups = []
    for g in source['groups']:
        if g['family'] != row['family']:
            continue
        ww = sorted({int(f) for eid in g['event_ids']
                     for f in events[eid]['progression_witness_frames']})
        groups.append((g, ww))
    out = []
    for i, (aa, bb) in enumerate(spans):
        aa, bb = int(aa), int(bb)
        a, b = (n-1-bb, n-1-aa) if row['reverse'] else (aa, bb)
        matches = []
        for g, ww in groups:
            inside = [f for f in ww if a <= f <= b]
            if (int(g['onset'][0]) <= a <= b <= int(g['offset'][1])
                    and 2 <= len(inside) < len(ww)):
                matches.append(dict(group_id=g['group_id'],
                    witnessed_frames=inside, all_group_witnesses=ww,
                    omitted_witnesses=[f for f in ww if f not in inside],
                    outer_bounds=[int(g['onset'][0]),int(g['offset'][1])]))
        if matches:
            out.append(dict(candidate_index=i, transformed_span=[aa,bb],
                original_span=[a,b], frames=bb-aa+1,
                duration_bucket=duration_bucket(bb-aa+1), groups=matches))
    return out


def pick(scores, eligible, spans, labels, null=None):
    ix = np.flatnonzero(eligible)
    if not len(ix):
        return dict(state='no_body_gate_admission', index=None, span=None,
                    known_positive=False, null_selected=False)
    k = int(ix[np.argmax(scores[ix])])
    if null is not None and float(scores[k]) <= null:
        return dict(state='learned_null', index=None, span=None,
                    known_positive=False, null_selected=True,
                    best_candidate_index=k, best_score=float(scores[k]),
                    null_logit=null)
    return dict(state='candidate', index=k, span=list(map(int,spans[k])),
                known_positive=bool(labels[k]), null_selected=False,
                score=float(scores[k]))


def summarize(rows, fields, metrics):
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[f] for f in fields)].append(r)
    return [dict(**dict(zip(fields, key)), rows=len(rr),
                 **{name:sum(int(fn(r)) for r in rr) for name,fn in metrics.items()})
            for key,rr in sorted(buckets.items())]


def main():
    E = N/'local_bag_evaluation'
    E.mkdir(exist_ok=True)
    assert not (E/'COMPARISON.json').exists(), 'Completed arithmetic must not rerun'
    assert not (E/'RULES_FROZEN.json').exists(), 'Use explicit audited recovery after failure'
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    training_frozen_sha = sha(N/'FROZEN.json')
    idx = read(N/'TRAINING_INPUT_INDEX.json'); rows = idx['rows']
    train = read(N/'TRAINING_COMPLETE.json')
    old_meta = read(old.L/'INPUTS_COMPLETE.json')
    old_lookup = {str(Path(r['path']).resolve()):i for i,r in enumerate(old_meta['bags'])}
    old_runs = {r['seed']:r for r in read(old.L/'TRAINING_COMPLETE.json')['rows']
                if r['condition']=='local_full'}
    body_runs = {r['seed']:r for r in read(old.B/'TRAINING_COMPLETE.json')['rows']
                if r['condition']=='body_window'}
    source_paths = [GRAPH/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',
                    PRIOR/'additional_source_review/SOURCE_EPISODES_ADDED.json']
    sources = {r['clip_id']:r for p in source_paths for r in read(p)['rows']}
    seq_meta = read(old.K/'INPUTS_COMPLETE.json')
    seqs = {(r['clip_id'],r['view'],r['reverse']):r for r in seq_meta['sequences']}
    data = []; partial = []; pins = {}; originals = {}; seq_lengths = {}
    # No prediction arrays opened in this section. Labels cannot depend on scores.
    for i,r in enumerate(rows):
        assert sha(r['path'])==r['sha256']; pins[r['path']]=r['sha256']
        z=np.load(r['path']); spans=z['spans'].copy(); y=z['y'].astype(bool)
        assert np.isfinite(z['local_full']).all()
        assert r['family'] in [3,4,5,7]
        assert r['split']==('probe' if r['person'] in [8,9] else 'train')
        assert r['person'] in [2,3,4,5,6,7,8,9]
        if r['origin']=='positive_source_extent':
            assert y.any()
            assert sha(r['original_path'])==r['original_sha256']
            pins[r['original_path']]=r['original_sha256']
            raw=np.load(r['original_path'])
            assert np.array_equal(raw['spans'],spans) and np.array_equal(raw['y'],y)
            bf=raw['x'].copy()
            key=(r['clip_id'],r['view'],r['reverse'])
            if key not in seq_lengths:
                ss=seqs[key]; assert sha(ss['path'])==ss['sha256']; pins[ss['path']]=ss['sha256']
                seq_lengths[key]=len(np.load(ss['path'])['x'])
            src=sources[r['clip_id']]
            assert src['source_sha256']==r['source_sha256']
            pieces=partial_candidates(src,r,spans,seq_lengths[key])
            assert all(not y[p['candidate_index']] for p in pieces), 'Partial mask overlaps full-compatible truth'
            old_index=old_lookup[str(Path(r['path']).resolve())]
        else:
            assert r['origin']=='reviewed_local_hold' and not y.any()
            bf=z['body_features'].copy(); pieces=[]; old_index=None
        assert len(bf)==len(spans)==len(y) and np.isfinite(bf).all()
        data.append(dict(spans=spans,y=y,body_features=bf,old_index=old_index))
        partial.append(dict(bag_index=i,clip_id=r['clip_id'],family=r['family'],
            split=r['split'],view=r['view'],reverse=r['reverse'],
            source_length=seq_lengths.get((r['clip_id'],r['view'],r['reverse']))
                if r['origin']=='positive_source_extent' else None,
            candidates=pieces))
    assert not ({r['source_sha256'] for r in rows if r['split']=='train'} &
                {r['source_sha256'] for r in rows if r['split']=='probe'})
    frozen=dict(at=stamp(),source='New logits are read only from saved predictions; old body gate and hold baseline may be calculated.',
        conditions=['saved_original_local_full','saved_new_local_null'],
        runtime='Frozen same-seed body_window logit>=0; candidate must beat family null strictly (tie selects null), same as full-path evaluate.py.',
        partial_rule='All EXISTING candidate windows wholly inside same-family group outer onset/offset bounds, containing>=2 but fewer than all direct progression witnesses of that group; no word/order-specific rules. Whole-source reverse mapped by n-1-frame.',
        partial_scope='Candidate score/admission diagnostic, not a new cropped-video trial, not semantic completion or educational error.',
        duration_buckets_frames=['4..6','7..10','11..15','16..21','22+'],
        observation='Input generator enforces body visibility. Zero body-gate admission is reported separately from learned null, and is not a new absence label or proof of physical observation shortage.',
        probe='ID08/09 person-separated diagnostic with upstream exposure, not final evaluation.',
        family2='Excluded from new null supervision/this diagnostic; preserved by original model in full-path evaluation.',
        selection='No weights, rules, thresholds or labels changed using this diagnostic.',
        training_frozen_sha256=training_frozen_sha,
        pins={str(p):sha(p) for p in [Path(__file__),N/'TRAINING_INPUT_INDEX.json',N/'TRAINING_COMPLETE.json',N/'INPUTS_COMPLETE.json',old.L/'TRAINING_COMPLETE.json',old.B/'TRAINING_COMPLETE.json',old.K/'INPUTS_COMPLETE.json']+source_paths},
        input_pins=pins,partial_masks=partial,new_model_reinference=False)
    write(E/'RULES_FROZEN.json',frozen)
    # The first saved new probe-score read occurs only below this frozen rule file.
    output=[]; partial_output=[]; evidence_pins={}
    for run in train['rows']:
        seed=run['seed']; assert seed in [42,43,44]
        assert sha(run['predictions'])==run['predictions_sha256']
        pred=np.load(run['predictions']); null=pred['null_logits']
        evidence_pins[run['predictions']]=run['predictions_sha256']
        obr=old_runs[seed]; assert sha(obr['predictions'])==obr['predictions_sha256']
        old_pred=np.load(obr['predictions']); evidence_pins[obr['predictions']]=obr['predictions_sha256']
        br=body_runs[seed]; assert sha(br['model'])==br['model_sha256']
        st=torch.load(br['model'],weights_only=True,map_location='cpu')
        body=old.BodyWindow(st['mean'].numpy(),st['std'].numpy()); body.load_state_dict(st); body.eval()
        evidence_pins[br['model']]=br['model_sha256']
        assert sha(obr['model'])==obr['model_sha256']; evidence_pins[obr['model']]=obr['model_sha256']
        st=torch.load(obr['model'],weights_only=True,map_location='cpu')
        baseline=old.LocalRankHead(st['mean'].numpy(),st['std'].numpy()); baseline.load_state_dict(st); baseline.eval()
        for i,(r,d) in enumerate(zip(rows,data)):
            j=old.F.index(r['family']); new=pred[f'logits_{i}'][:,j]; null_j=float(null[j])
            assert len(new)==len(d['spans']) and np.isfinite(new).all() and np.isfinite(null_j)
            with torch.inference_mode():
                presence=body(torch.tensor(d['body_features'])).numpy()[:,j,0]
            if d['old_index'] is None:
                # Baseline-only inference; the learned null model is never instantiated.
                z=np.load(r['path'])
                with torch.inference_mode(): old_score=baseline(torch.tensor(z['local_full'])).numpy()[:,j]
            else:
                old_score=old_pred[f"logits_{d['old_index']}"][:,j]
            assert np.isfinite(presence).all() and np.isfinite(old_score).all()
            eligible=presence>=0; all_candidates=np.ones(len(new),bool)
            old_pick=pick(old_score,eligible,d['spans'],d['y'])
            new_pick=pick(new,eligible,d['spans'],d['y'],null_j)
            raw_pick=pick(new,all_candidates,d['spans'],d['y'],null_j)
            out=dict(seed=seed,bag_index=i,clip_id=r['clip_id'],person=r['person'],
                split=r['split'],view=r['view'],reverse=r['reverse'],family=r['family'],origin=r['origin'],
                candidates=len(new),body_gate_admitted=int(eligible.sum()),
                known_positive_candidates=int(d['y'].sum()),
                known_positive_body_admitted=int((d['y']&eligible).sum()),
                baseline=old_pick,new=new_pick,new_ungated=raw_pick,
                baseline_false_hold_selection=r['origin']=='reviewed_local_hold' and old_pick['state']=='candidate',
                new_false_hold_selection=r['origin']=='reviewed_local_hold' and new_pick['state']=='candidate',
                semantic_completion=None)
            output.append(out)
            for p in partial[i]['candidates']:
                k=p['candidate_index']; admitted=bool(eligible[k]); survives=bool(float(new[k])>null_j)
                state='no_body_gate_admission' if not admitted else 'candidate_above_null' if survives else 'learned_null_suppressed'
                partial_output.append(dict(seed=seed,bag_index=i,clip_id=r['clip_id'],person=r['person'],
                    split=r['split'],view=r['view'],reverse=r['reverse'],family=r['family'],**p,
                    old_body_gate_admitted=admitted,body_gate_logit=float(presence[k]),
                    baseline_rank_score=float(old_score[k]),new_score=float(new[k]),null_logit=null_j,
                    above_null_ungated=survives,state=state,semantic_completion=None))
        print('local saved-score arithmetic seed',seed,flush=True)
    metrics=dict(baseline_candidate=lambda r:r['baseline']['state']=='candidate',
        new_candidate=lambda r:r['new']['state']=='candidate',
        no_body_gate_admission=lambda r:r['new']['state']=='no_body_gate_admission',
        learned_null=lambda r:r['new']['state']=='learned_null',
        baseline_known_positive=lambda r:r['baseline']['known_positive'],
        new_known_positive=lambda r:r['new']['known_positive'],
        new_ungated_known_positive=lambda r:r['new_ungated']['known_positive'],
        lost_previously_positive=lambda r:r['baseline']['known_positive'] and not r['new']['known_positive'],
        baseline_false_hold_selection=lambda r:r['baseline_false_hold_selection'],
        new_false_hold_selection=lambda r:r['new_false_hold_selection'])
    pmetrics=dict(body_gate_admitted=lambda r:r['old_body_gate_admitted'],
        no_body_gate_admission=lambda r:r['state']=='no_body_gate_admission',
        learned_null_suppressed=lambda r:r['state']=='learned_null_suppressed',
        candidate_above_null=lambda r:r['state']=='candidate_above_null',
        below_or_equal_null_ungated=lambda r:not r['above_null_ungated'])
    report=dict(at=stamp(),rules_sha256=sha(E/'RULES_FROZEN.json'),
        rows=output,summary=summarize(output,['seed','split','origin','family'],metrics),
        summary_all_families=summarize(output,['seed','split','origin'],metrics),
        partial_rows=partial_output,
        partial_summary=summarize(partial_output,['seed','split','family','duration_bucket'],pmetrics),
        partial_exact_duration=summarize(partial_output,['seed','split','family','frames'],pmetrics),
        partial_bags_without_supported_candidate=[{k:p[k] for k in ['bag_index','clip_id','family','split','view','reverse']}
            for p,r in zip(partial,rows) if r['origin']=='positive_source_extent' and not p['candidates']],
        evidence_pins=evidence_pins,new_model_reinference=False,
        interpretation='Sparse direct witnesses support visible partial progression only. Candidate suppression is a warning diagnostic, not fragment correctness or actual model response to an independently cropped input. Full-path regression remains separate.',
        private_access=False,final8=False,web_changed=False)
    assert sha(N/'FROZEN.json')==training_frozen_sha
    write(E/'COMPARISON.json',report)
    for r in report['summary_all_families']: print(r,flush=True)
    print('Partial existing candidate diagnostics',len(partial_output),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException:
        dest=N/'local_bag_evaluation'; dest.mkdir(exist_ok=True)
        (dest/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
