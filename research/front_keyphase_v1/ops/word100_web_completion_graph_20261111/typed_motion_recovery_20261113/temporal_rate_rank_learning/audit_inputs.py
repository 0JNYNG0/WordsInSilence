"""Freeze and audit fixed-rate FIT mapping/raw observations/labels ONLY.

No torch, model loading, encoder, guard/adapter rerun, training or DEV inputs.
Saved raw packs and label audits are explicitly NOT model-ready features.
"""
from pathlib import Path
import collections, hashlib, json, sys, time, traceback
import numpy as np
from rate_core import rate_mapping, labels
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent
J=H/'splice_rank_learning';W=G/'candidate_description_supervision'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p

def freeze():
    assert not (U/'FROZEN.json').exists()
    parent=read(J/'FROZEN.json');ready=read(J/'INPUTS_COMPLETE.json');trained=read(J/'TRAINING_COMPLETE.json')
    assert len(parent['sequences'])==36 and len(ready['bags'])==72 and len(ready['replay'])==178
    pins={str(p):sha(p)for p in [U/'rate_core.py',U/'audit_inputs.py',U/'test_mapping.py',J/'FROZEN.json',
        J/'INPUTS_COMPLETE.json',J/'TRAINING_COMPLETE.json',W/'prepare_rank_learning.py',
        W/'local_rank_learning/FROZEN.json',H/'body_action_binding_review/SPLICE_REPAIR_DESIGN.json']}
    raw={}
    for cid,v in parent['inputs'].items():
        rp=O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json';r=read(rp);pins[str(rp)]=sha(rp)
        assert sha(r['raw_path'])==r['raw_sha256'];pins[r['raw_path']]=r['raw_sha256']
        crop=v['crop'];assert sha(crop['path'])==crop['sha256'];pins[crop['path']]=crop['sha256']
        raw[cid]=dict(full_path=r['raw_path'],full_sha256=r['raw_sha256'],crop_path=crop['path'],crop_sha256=crop['sha256'],source_sha256=r['source_sha256'])
    for row in parent['sequences']+ready['sequences']:
        assert sha(row['path'])==row['sha256'];pins[row['path']]=row['sha256']
    for row in ready['replay']+ready['bags']:
        if not row.get('skipped',False):pins[row['path']]=row['sha256']
    for row in ready['bags']:pins[row['target_debug']]=row['target_debug_sha256']
    for row in trained['rows']:
        assert sha(row['model'])==row['model_sha256'];pins[row['model']]=row['model_sha256']
    cfg=dict(status='PURE_MAPPING_RAW_LABEL_AUDIT_ONLY_ENCODER_TRAINING_NOT_AUTHORIZED',at=time.time(),
        rates=[0.8,1.2],bases=['normal','splice'],family_scope=[2,3],family_order=[2,3,4,5,7],
        source_people=[2,3,4,5,6,7],source_count=6,sequences=parent['sequences'],splice_sequences=ready['sequences'],
        replay_original=ready['replay'],replay_splice=ready['bags'],raw=raw,parent_models=trained['rows'],
        initialization_if_later_authorized='Matching J seed42/43/44 weights and normalization, never retrain or overwrite J.',
        timestamp_rule='Uniform source-time grid including both endpoints. Number of intervals round(duration/(median_dt*rate)), at least1. Nearest real source sample, exact ties earlier. Query duration=original/rate; integer microsecond queryPTS evenly spaced. Mapping ignores labels.',
        raw_rule='Copy original raw xy/score by composed source index. Save original source frameID/PTS, intermediate base query index/PTS, and synthetic final queryPTS separately. No interpolated coordinate or duplicated sample counted as new observation.',
        full_positive='Old source band/witness/exclusion/valid contract AND all original observed positive witness IDs retained AND endpoints in old bands AND >=4 distinct observed source frames AND new mapped body qualification. No full-positive if any required witness disappears.',
        negative='Normal: inherited original extent disagreement with still-retained observed evidence. Splice: ONLY known source-block crossing. Sampling gaps within a block are never splice negatives. Lost witness alone is partial/unknown, never negative.',
        held_bag='No positive or no valid negative => hold entire rank bag. Count every case in denominator. Never add witness frames to resampling map, narrow bands, tune rate or choose seed to restore positives.',
        future_encoder='Must recompute canonical channels, guards/masks and normalization on the actual transformed observations, then frozen common100 inside EACH new candidate. No resampling/concatenating old hidden1160. Full/upper/crop view policy must match current route.',
        current_audit_limit='Eligibility inherits original source valid masks. Raw packs include upper view metadata but no canonical upper mask or resampled guard run. Counts are label/provenance viability, not final encoder-input readiness. Future canonical availability can reduce them.',
        runtime_protection='Original178+J72 replay preserved. Only family2/3 new head considered; other family runtime output stays old exactly. No wordID, seam/provenance, required order, completion or existence claim.',
        no_dev_inputs=True,no_new_expert_labels=True,no_encoder=True,no_training=True,pins=pins)
    check(pins);write(U/'FROZEN.json',cfg);print('FROZEN rates .8/1.2;144 sequence variants;288 family cases; no encoder')

def audit():
    assert not (U/'INPUT_VALIDITY_AUDIT.json').exists()and not(U/'raw_audit').exists()
    cfg=read(U/'FROZEN.json');check(cfg['pins'])
    (U/'raw_audit').mkdir();(U/'label_audit').mkdir()
    sm={(r['clip_id'],r['view'],r['reverse']):r for r in cfg['splice_sequences']}
    parent=read(J/'INPUTS_COMPLETE.json');jm={(r['clip_id'],r['view'],r['reverse'],r['family']):r for r in parent['bags']}
    om={(r['clip_id'],r['view'],r['reverse'],r['family']):r for r in parent['replay']}
    records=[];sequences=[];checks=collections.Counter()
    def ck(ok,k):assert bool(ok),k;checks[k]+=1
    for s in cfg['sequences']:
        cid,view,reverse=s['clip_id'],s['view'],s['reverse'];rawpath=cfg['raw'][cid]['crop_path'if view=='actual_crop'else'full_path']
        raw=np.load(rawpath);source=np.load(s['path']);n=len(raw['timestamp_us'])
        for kind in cfg['bases']:
            if kind=='normal':
                src=np.arange(n-1,-1,-1,dtype=np.int32)if reverse else np.arange(n,dtype=np.int32)
                baseline=np.arange(n,dtype=np.int32);block=np.zeros(n,np.int32)
                pts=raw['timestamp_us'][-1]-raw['timestamp_us'][::-1]if reverse else raw['timestamp_us']-raw['timestamp_us'][0]
            else:
                z=np.load(sm[cid,view,reverse]['path']);src=z['source_index'];baseline=z['baseline_index'];pts=z['timestamp_us']
                block=np.searchsorted(z['join_right'],np.arange(n),side='right').astype(np.int32)
            ck(np.array_equal(np.sort(src),np.arange(n)),'base_source_permutation')
            # Identity algebra must reproduce every original/J full-positive span.
            for f in cfg['family_scope']:
                identity=labels(source,baseline,block,cfg['family_order'].index(f),kind)
                prior=(om if kind=='normal'else jm)[cid,view,reverse,f];bag=np.load(prior['path'])
                ck({tuple(v)for v in identity['spans'][identity['positive']]}=={tuple(v)for v in bag['spans'][bag['y']]},'identity_full_positive_exact')
                if kind=='splice':
                    debug=np.load(prior['target_debug'])
                    ck(np.array_equal(identity['negative'],debug['negative']),'identity_splice_negative_exact')
            for rate in cfg['rates']:
                rm=rate_mapping(pts,rate);idx=rm['base_query_index'];ids=src[idx];m=baseline[idx];bi=block[idx]
                name=f'{cid}_{view}_{int(reverse)}_{kind}_{str(rate).replace(".","p")}'
                pack=dict(xy_px=raw['xy_px'][ids],score=raw['score'][ids],timestamp_us=rm['timestamp_us'],
                    frame_id=np.arange(len(idx),dtype=np.int64),source_frame_id=raw['frame_id'][ids],source_timestamp_us=raw['timestamp_us'][ids],
                    source_index=ids,baseline_index=m,source_block_id=bi,base_query_index=idx,
                    base_query_pts=rm['base_query_pts'],requested_source_time_us=rm['source_time_requested_us'],
                    source_width=raw['source_width'],source_height=raw['source_height'])
                path=U/'raw_audit'/f'{name}.npz';np.savez_compressed(path,**pack)
                ck(np.array_equal(pack['xy_px'],raw['xy_px'][ids])and np.array_equal(pack['score'],raw['score'][ids]),'raw_coordinates_scores_exact')
                ck(np.all(np.diff(pack['timestamp_us'])>0),'query_pts_monotone')
                ck(np.array_equal(pack['source_timestamp_us'],raw['timestamp_us'][ids]),'source_pts_preserved')
                ck(np.all(np.diff(idx)>=0),'rate_mapping_monotone')
                sr=dict(clip_id=cid,person=s['person'],view=view,reverse=reverse,base_kind=kind,rate=rate,
                    path=str(path),sha256=sha(path),source_frames=n,query_frames=len(idx),unique_source_frames=len(np.unique(ids)),
                    duplicated_samples=len(ids)-len(np.unique(ids)),omitted_source_frames=n-len(np.unique(ids)),
                    known_splice_joins_retained=int(np.count_nonzero(np.diff(bi))),raw_source=rawpath,raw_source_sha256=sha(rawpath),
                    source_sequence=s['path'],source_sequence_sha256=s['sha256'],canonical_features_ready=False)
                sequences.append(sr)
                for f in cfg['family_scope']:
                    d=labels(source,m,bi,cfg['family_order'].index(f),kind)
                    lp=U/'label_audit'/f'{name}_{f}.npz';np.savez_compressed(lp,**d)
                    positives=int(d['positive'].sum());negatives=int(d['negative'].sum());reasons=[]
                    if len(d['missing_witnesses']):reasons.append('required_witness_removed')
                    if not len(d['onset_retained']):reasons.append('onset_band_removed')
                    if not len(d['offset_retained']):reasons.append('offset_band_removed')
                    if not positives and not reasons:reasons.append('no_full_span_after_eligibility_or_distinct_observation_contract')
                    if not negatives:reasons.append('no_rank_negative')
                    ck(not len(d['missing_witnesses'])or positives==0,'lost_witness_never_full_positive')
                    ck(not np.any(d['positive']&d['crossing']),'positive_never_splice_crossing')
                    ck(np.all(d['distinct_observed'][d['positive']]>=4),'duplicates_not_extra_evidence')
                    if kind=='normal':ck(not d['crossing'].any(),'sampling_gap_not_splice_join')
                    records.append(dict(clip_id=cid,person=s['person'],view=view,reverse=reverse,base_kind=kind,rate=rate,family=f,
                        raw_path=str(path),label_path=str(lp),label_sha256=sha(lp),positive=positives,negative=negatives,
                        eligible=int(d['eligible'].sum()),partial_unknown_with_witness=int(d['partial'].sum()),
                        unknown=int((~(d['positive']|d['negative'])).sum()),held=not positives or not negatives,hold_reasons=reasons,
                        required_witnesses=d['required_witnesses'].tolist(),missing_witnesses=d['missing_witnesses'].tolist(),
                        onset_retained=d['onset_retained'].tolist(),offset_retained=d['offset_retained'].tolist()))
        print('rate audit',cid,view,int(reverse),'cases',len(records),flush=True)
    summary=[]
    for rate in cfg['rates']:
        for kind in cfg['bases']:
            for f in cfg['family_scope']:
                rr=[r for r in records if(r['rate'],r['base_kind'],r['family'])==(rate,kind,f)]
                summary.append(dict(rate=rate,base_kind=kind,family=f,total_cases=len(rr),viable_cases=sum(not r['held']for r in rr),
                    held_cases=sum(r['held']for r in rr),positive_candidates=sum(r['positive']for r in rr),
                    missing_witness_cases=sum(bool(r['missing_witnesses'])for r in rr),reasons=dict(collections.Counter(q for r in rr for q in r['hold_reasons']))))
    ck(len(sequences)==144 and len(records)==288,'all_fixed_denominators')
    check(cfg['pins'])
    write(U/'INPUT_VALIDITY_AUDIT.json',dict(status='PURE_RATE_RAW_LABEL_AUDIT_COMPLETE_NOT_ENCODER_READY',at=time.time(),
        frozen_sha256=sha(U/'FROZEN.json'),summary=summary,sequences=sequences,cases=records,checks=dict(checks),total_checks=sum(checks.values()),
        original178_and_J72_replay_untouched=True,matching_J_parent_models_pinned=True,
        no_encoder=True,no_training=True,no_model_inference=True,no_dev_arrays=True,
        interpretation='Viability under inherited source availability only. Lost witness cases remain in denominator and held. Not a speed-generalization result or complete canonical input check.'))
    print('RATE AUDIT COMPLETE',summary,flush=True)

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in ['freeze','audit']
        freeze()if sys.argv[1]=='freeze'else audit()
    except BaseException:
        (U/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
