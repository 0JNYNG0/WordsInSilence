"""Arithmetic-only audit of fixed LOPO outputs. No model import or inference.

Labels and representative intervals are frozen before any prediction is read.
An unfinished/failed evaluation is preserved and cannot silently be rerun.
"""
from pathlib import Path
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import time
import traceback
import numpy as np

U = Path(__file__).resolve().parent
T = U.parent / 'typed_shape_transition_review'
E = U / 'saved_evaluation'
ATTRS = ['four_open', 'four_closed', 'index_only', 'intermediate_bending']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), 'Never overwrite an existing evaluation artifact: ' + str(p)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def ratio(a, b):
    return float(a / b) if b else None

def binary(rows, key='prediction'):
    eligible = [r for r in rows if r['eligible']]
    tp = sum(r['target'] == 1 and r[key] == 1 for r in eligible)
    tn = sum(r['target'] == 0 and r[key] == 0 for r in eligible)
    fp = sum(r['target'] == 0 and r[key] == 1 for r in eligible)
    fn = sum(r['target'] == 1 and r[key] == 0 for r in eligible)
    return dict(known=len(rows), eligible=len(eligible), unavailable=len(rows)-len(eligible),
                positive=tp+fn, negative=tn+fp, tp=tp, tn=tn, fp=fp, fn=fn,
                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                specificity=ratio(tn, tn+fp), accuracy=ratio(tp+tn, len(eligible)),
                balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None)

def transition(rows):
    known = [r for r in rows if r['eligible']]
    matrix = [[0]*3 for _ in range(3)]
    for r in known:
        matrix[r['target']][r['prediction']] += 1
    emit = [r for r in known if r['confidence'] >= .65]
    correct = sum(r['prediction'] == r['target'] for r in known)
    emitted_correct = sum(r['prediction'] == r['target'] for r in emit)
    return dict(total=len(rows), eligible=len(known), observation_unknown=len(rows)-len(known),
                confusion=matrix, class_order=['opening', 'stable_handshape', 'closing'],
                argmax_correct=correct, argmax_accuracy=ratio(correct,len(known)),
                emitted=len(emit), coverage=ratio(len(emit),len(rows)),
                low_confidence=len(known)-len(emit), emitted_correct=emitted_correct,
                emitted_wrong=len(emit)-emitted_correct,
                emitted_accuracy=ratio(emitted_correct,len(emit)))

def grouped(rows, fields, metric):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[f] for f in fields)].append(r)
    return [dict(**dict(zip(fields,k)), metrics=metric(v)) for k,v in sorted(groups.items())]

def category(w):
    if w['kind'] == 'synthetic_repeated_pose':
        return 'synthetic_constant'
    if w['kind'] == 'direct_stable_handshape':
        return 'real_stable_closed_arm_raise' if w['body_raise_stable'] else 'real_stable_other'
    return ('full' if w['kind']=='full_state_change' else 'partial') + ('_reverse_opening' if w['reverse'] else '_closing')

def representatives(review, windows):
    """Source-label rule only. Never select a model's best-scoring interval."""
    lookup = defaultdict(list)
    for i,w in enumerate(windows):
        if w['kind'] != 'synthetic_repeated_pose':
            lookup[w['clip_id'],tuple(w['source_frames']),bool(w['reverse'])].append(i)
    result = []
    for source in review['rows']:
        cid = source['clip_id']
        rr = sorted(source['frame_records'], key=lambda r:r['frame'])
        raises = [r['frame'] for r in rr if r['body_motion']=='forearm_raising' and r['hand_state']=='closed']
        opens = [r['frame'] for r in rr if r['hand_state']=='open']
        choices = []
        if raises and opens:
            closed_before = [r['frame'] for r in rr if r['hand_state']=='closed' and r['frame']<min(raises)]
            if closed_before:
                span = [min(opens),max(closed_before)]
                choices += [('canonical_full_closing',span,False,2),('canonical_reversed_opening',span,True,0)]
            choices += [('canonical_closed_arm_raise',[min(raises),max(raises)],False,1)]
        for name in ['canonical_full_closing','canonical_reversed_opening','canonical_closed_arm_raise']:
            c = next((c for c in choices if c[0]==name),None)
            if c is None:
                result.append(dict(clip_id=cid,kind=name,available=False,reason='Source labels do not support representative range'))
                continue
            _,span,rev,target = c
            hits = lookup[cid,tuple(span),rev]
            assert len(hits)<=1
            row = dict(clip_id=cid,kind=name,source_frames=span,reverse=rev,target=target,
                       available=bool(hits),window_id=hits[0] if hits else None)
            if hits:
                assert windows[hits[0]]['target']==target
            else:
                row['reason']='Label-defined range is absent from prepared windows; no new inference allowed'
            result.append(row)
    return result

def main():
    assert not E.exists(), 'Evaluation already started; preserve first outcome and investigate explicitly'
    meta = read(U/'INPUTS_COMPLETE.json')
    items = read(U/'EVALUATION_ITEMS.json')
    cfg = read(U/'FROZEN.json')
    tc = read(U/'TRAINING_CONTRACT.json')
    done = read(U/'TRAINING_COMPLETE.json')
    review = read(T/'REVIEW_COMPLETE.json')
    baseline = read(T/'BASELINE_CACHE_REVIEW.json')
    assert len(done['runs']) == len(cfg['folds'])*len(cfg['seeds']) == 18
    assert {(r['heldperson'],r['seed']) for r in done['runs']} == {(p,s) for p in cfg['folds'] for s in cfg['seeds']}
    for p,h in tc['pins'].items():
        assert sha(p)==h,p
    assert items['labels']==meta['labels'] and items['windows']==meta['windows']+meta['controls']
    canonical = representatives(review,items['windows'])
    assert len(canonical)==18
    freeze = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='RULES_FROZEN_BEFORE_READING_ANY_PREDICTION_VALUES',
        attributes=dict(decision='logit >= 0',unknown='Unknown RGB cells excluded, never negative',
                        reports='Per attribute, person, seed; new13 intermediate positives separate'),
        transition=dict(decision='argmax, class0 opening/1 stable/2 closing',emission='max softmax >= .65 AND observed eligible endpoints',
                        scope='Observed state change, never semantic completion; reverse opening is a synthetic contrast'),
        canonical_rule='Per source: earliest directly open to last directly closed before first closed-hand forearm-raising; plus first-to-last directly closed forearm-raising. Exact prepared windows only.',
        canonical=canonical,
        source_gate='Every one of six sources and all three seeds must classify AND confidently emit original closing, reverse opening and stable closed-arm-raising; any unavailable representative is unresolved, not pass.',
        baseline='Only stored requirement train.npz base logits at exact matched full-view expected1 cells. No old head/encoder execution. Old baseline trained on these FIT people; not a fair LOPO generalization comparison.',
        duration_groups=['4..7 frames','8..15 frames','16..24 frames'],
        limits=['Overlapping windows, reverse views and static copies are dependent observations.',
                'Six source people, one expression and right hand support transitions; static attributes span eleven expressions.',
                'Upstream RTMW/FIT exposure; not final8 or an independent educational correctness benchmark.',
                'Unknown interior frames do not prove the exact hidden hand trajectory.',
                'No missing-hand model run in this saved evaluation; abstention is an input contract.'],
        pins={str(p):sha(p) for p in [Path(__file__),U/'INPUTS_COMPLETE.json',U/'EVALUATION_ITEMS.json',U/'FROZEN.json',U/'TRAINING_CONTRACT.json',U/'TRAINING_COMPLETE.json',T/'REVIEW_COMPLETE.json',T/'BASELINE_CACHE_REVIEW.json']})
    write(E/'FROZEN.json',freeze)
    seq = {}
    for r in meta['sequences']:
        assert sha(r['path'])==r['sha256']
        seq[r['clip_id'],r['side']] = np.load(r['path'],allow_pickle=False)
    labels = defaultdict(list)
    for r in items['labels']:
        labels[r['clip_id'],r['frame'],r['side']].append(r)
    dense_frames = {(r['clip_id'],x['frame']) for r in review['rows'] for x in r['frame_records']}
    intermediate = {(r['clip_id'],x['frame']) for r in review['rows'] for x in r['frame_records'] if x['hand_state']=='flexing'}
    unknown = {(r['clip_id'],x['frame']) for r in review['rows'] for x in r['frame_records'] if x['hand_state']=='unknown'}
    assert len(intermediate)==13 and len(unknown)==8
    # This map contains existing saved logits only; no new model is loaded.
    old = baseline['sparse_saved_logits']
    assert sha(old['npz'])==old['npz_sha256'] and sha(old['manifest'])==old['manifest_sha256']
    bz = np.load(old['npz'],allow_pickle=False)
    bm = read(old['manifest'])
    bmap = {}
    for r in old['covered_current_cells']:
        ix=r['sample'];d=bm[ix]
        assert d['sample']==ix and d['clip_id']==r['clip_id'] and d['frames'][0]==r['frame'] and d['expected']==1 and d['view']=='full'
        bmap[r['clip_id'],r['frame'],r['side'],r['attribute']] = (float(bz['base'][ix]),bool(bz['mask'][ix]),ix)
    assert len(bmap)==268
    attr_rows=[];window_rows=[];paired=[];canonical_rows=[];pins={}
    for run in sorted(done['runs'],key=lambda r:(r['heldperson'],r['seed'])):
        held,seed=run['heldperson'],run['seed']
        complete=Path(run['predictions']).parent/'COMPLETE.json'
        rec=read(complete)
        assert rec==run and sha(run['predictions'])==run['predictions_sha256'] and sha(run['model'])==run['model_sha256']
        assert held not in run['split']['trainpeople'] and not set(run['split']['train_source_hashes'])&set(run['split']['held_source_hashes'])
        pins[str(complete)]=sha(complete);pins[run['predictions']]=run['predictions_sha256']
        p=np.load(run['predictions'],allow_pickle=False)
        expected_f=[i for i,r in enumerate(items['frames']) if r['person']==held]
        expected_w=[i for i,r in enumerate(items['windows']) if r['person']==held]
        assert p['frame_ids'].tolist()==expected_f==run['split']['test_frame_ids']
        assert p['window_ids'].tolist()==expected_w==run['split']['test_window_ids']
        assert p['frame_logits'].shape==(len(expected_f),4) and p['window_logits'].shape==(len(expected_w),3)
        assert np.isfinite(p['frame_logits']).all() and np.isfinite(p['window_logits']).all()
        for ix,logits in zip(expected_f,p['frame_logits']):
            fr=items['frames'][ix];cid=fr['clip_id'];f=fr['source_frame'];side=fr['side']
            for a in labels[cid,f,side]:
                j=a['attribute'];logit=float(logits[j]);target=int(a['target'])
                r=dict(seed=seed,person=held,clip_id=cid,frame=f,side=side,attribute=j,attribute_name=ATTRS[j],
                       target=target,prediction=int(logit>=0),logit=logit,eligible=bool(a['available']),
                       dense_new=(cid,f) in dense_frames and side=='R',new_intermediate_positive=(cid,f) in intermediate and side=='R' and j==3)
                attr_rows.append(r)
                if (cid,f,side,j) in bmap:
                    bl,be,bi=bmap[cid,f,side,j]
                    paired.append(dict(r,baseline_logit=bl,baseline_prediction=int(bl>=0),baseline_sample=bi,
                                       eligible=bool(a['available']) and be))
        bywid={}
        for ix,logits in zip(expected_w,p['window_logits']):
            w=items['windows'][ix];source=seq[w['clip_id'],w['side']];a,b=w['span']
            eligible=bool(source['available'][a] and source['available'][b])
            if w['kind']=='synthetic_repeated_pose':eligible=bool(source['available'][w['anchor']])
            assert source['available'][a:b+1].all(), 'Prepared window availability changed'
            v=logits.astype(np.float64);p3=np.exp(v-v.max());p3/=p3.sum();pred=int(v.argmax());n=b-a+1
            r=dict(seed=seed,person=held,window_id=ix,clip_id=w['clip_id'],kind=category(w),
                   reverse=w['reverse'],target=int(w['target']),prediction=pred,confidence=float(p3.max()),
                   eligible=eligible,frames=n,duration_group='4..7 frames' if n<=7 else '8..15 frames' if n<=15 else '16..24 frames')
            window_rows.append(r);bywid[ix]=r
        for c in canonical:
            cid=c['clip_id']
            if int(cid.split('_REAL')[1][:2])!=held:continue
            r=dict(c,seed=seed,person=held)
            if c['available']:
                d=bywid[c['window_id']]
                r.update(prediction=d['prediction'],confidence=d['confidence'],eligible=d['eligible'],
                         emitted=d['eligible'] and d['confidence']>=.65,
                         passed=d['eligible'] and d['confidence']>=.65 and d['prediction']==c['target'])
            else:r['passed']=False
            canonical_rows.append(r)
    assert len(canonical_rows)==54
    # Keep decisions inspectable without rewriting any input or score file.
    write(E/'ATTRIBUTE_ROWS.json',attr_rows)
    write(E/'BASELINE_PAIRED_ROWS.json',paired)
    write(E/'CANONICAL_SOURCE_ROWS.json',canonical_rows)
    arrays={k:np.asarray([r[k] for r in window_rows]) for k in ['seed','person','window_id','target','prediction','confidence','eligible','frames']}
    np.savez_compressed(E/'WINDOW_DECISIONS.npz',**arrays)
    paired_metric=lambda rr:dict(new_lopo=binary(rr),old_fit_exposed=binary(rr,'baseline_prediction'),
        changed_correct_to_wrong=sum(r['eligible'] and r['baseline_prediction']==r['target'] and r['prediction']!=r['target'] for r in rr),
        changed_wrong_to_correct=sum(r['eligible'] and r['baseline_prediction']!=r['target'] and r['prediction']==r['target'] for r in rr))
    results=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='SAVED_ARITHMETIC_COMPLETE_NO_MODEL_EXECUTION',
        original_sources=dict(state_source_clips=meta['counts']['source_clips'],state_words=meta['counts']['expressions'],
            source_people=meta['counts']['people'],held_people=cfg['folds'],transition_clips=6,transition_words=1,
            transition_hand='right',dense_rgb_frames=206,unknown_frames_excluded=8,new_intermediate_positive_frames=13),
        attributes=dict(by_seed=grouped(attr_rows,['seed'],binary),by_seed_attribute=grouped(attr_rows,['seed','attribute_name'],binary),
            by_seed_person_attribute=grouped(attr_rows,['seed','person','attribute_name'],binary),
            dense_new_by_seed_attribute=grouped([r for r in attr_rows if r['dense_new']],['seed','attribute_name'],binary),
            intermediate_by_seed=grouped([r for r in attr_rows if r['attribute']==3],['seed'],binary),
            new13_positive_by_seed=grouped([r for r in attr_rows if r['new_intermediate_positive']],['seed'],binary)),
        windows=dict(dependent_observations=True,by_seed=grouped(window_rows,['seed'],transition),
            by_seed_kind=grouped(window_rows,['seed','kind'],transition),
            by_seed_person_kind=grouped(window_rows,['seed','person','kind'],transition),
            by_seed_kind_duration=grouped(window_rows,['seed','kind','duration_group'],transition)),
        canonical_gate=dict(passed=all(r['passed'] for r in canonical_rows),required=54,passed_cases=sum(r['passed'] for r in canonical_rows),
            unresolved=sum(not r['available'] for r in canonical_rows),
            per_seed=[dict(seed=s,required=18,passed=sum(r['passed'] for r in canonical_rows if r['seed']==s),
                           all_six_people_all_three_cases=all(r['passed'] for r in canonical_rows if r['seed']==s)) for s in cfg['seeds']],
            rows_file=str(E/'CANONICAL_SOURCE_ROWS.json'),scope='Narrow source-level diagnostic gate only; not overall adoption or educational correctness'),
        baseline=dict(available_saved_cells_total=268,paired_cells_per_seed=dict(Counter(str(r['seed']) for r in paired)),
            paired_by_seed=grouped(paired,['seed'],paired_metric),paired_by_seed_attribute=grouped(paired,['seed','attribute_name'],paired_metric),
            limitations=freeze['baseline'],missing_outputs=['intermediate_bending','opening/closing transition']),
        limitations=freeze['limits'],predictions_pins=pins,
        artifacts={str(p):sha(p) for p in [E/'FROZEN.json',E/'ATTRIBUTE_ROWS.json',E/'BASELINE_PAIRED_ROWS.json',E/'CANONICAL_SOURCE_ROWS.json',E/'WINDOW_DECISIONS.npz']})
    write(E/'RESULTS.json',results)
    write(E/'COMPLETE.json',dict(at=results['at'],results=str(E/'RESULTS.json'),sha256=sha(E/'RESULTS.json'),
        no_model_execution=True,no_new_labels=True,no_threshold_selection=True,old_models_changed=False))
    print(json.dumps(dict(canonical=results['canonical_gate'],
        intermediate=results['attributes']['intermediate_by_seed'],baseline_paired=results['baseline']['paired_cells_per_seed']),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        U.mkdir(parents=True,exist_ok=True)
        (U/f'SAVED_EVALUATION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
