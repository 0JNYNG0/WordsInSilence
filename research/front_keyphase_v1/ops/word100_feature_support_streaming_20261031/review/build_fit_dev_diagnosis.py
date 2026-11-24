"""Read-only fit/development support diagnosis; never load held target arrays."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import json,hashlib
from collections import Counter,defaultdict
import numpy as np

REVIEW=Path(__file__).resolve().parent
U=REVIEW.parent
OLD=U.parent/'word100_stage_feedback_cycle_20261031'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    manifest=read(OLD/'LEARNING_INPUTS.json')
    rows=[r for r in manifest['rows'] if r['role'] in ('fit','development')]
    allowed={r['alias'] for r in rows}
    assert len(rows)==48 and Counter(r['role'] for r in rows)=={'fit':36,'development':12}
    assert all(r['person'] in (1,2,3) if r['role']=='fit' else r['person']==4 for r in rows)
    audits={r['alias']:r for r in read(OLD/'LABEL_AUDIT.json')['rows'] if r['alias'] in allowed}
    primary={r['alias']:r for name in ('RGB_LABELS_A.json','RGB_LABELS_B.json') for r in read(OLD/'review'/name)['rows'] if r['alias'] in allowed}
    features=manifest['all_targets']
    static=read(OLD/'RGB_RUBRIC.json')['features']
    names=['R_index_only','L_index_only','R_open_four','L_open_four','R_motion_up','L_motion_up','R_motion_down','L_motion_down']
    arrays={};pins={}
    for r in rows:
        assert sha(r['input'])==r['input_sha256'] and sha(r['targets_file'])==r['targets_sha256']
        arrays[r['alias']]={'input':load(r['input']),'target':load(r['targets_file']),'raw':load(r['observation'])}
        pins[r['alias']]={k:sha(r[k]) for k in ('input','targets_file','observation')}
    support={};diagnostic={};temporal={}
    for role in ('fit','development'):
        group=[r for r in rows if r['role']==role]
        support[role]={};diagnostic[role]={}
        for feature in names:
            col=features.index(feature);counts=Counter();primary_counts=Counter();current_counts=Counter();positive=[];positive_current=set();direction_counts=Counter()
            final_positive=set();final_negative=set();roi_total=0
            for r in group:
                a=r['alias'];z=arrays[a]['target'];ids=arrays[a]['input']['frame_id']
                for val in (0,1):
                    ix=np.flatnonzero((z['y'][:,col]==val)&z['target_mask'][:,col]);counts[str(val)]+=len(ix)
                    target_set=final_positive if val else final_negative
                    target_set.update((a,int(ids[j])) for j in ix)
                    if val and len(ix):positive.append(dict(alias=a,word=r['word_id'],person=r['person'],cells=len(ix)))
                if feature in static:
                    for f in primary[a]['static']:
                        val=f['values'][static.index(feature)]
                        primary_counts['null' if val is None else str(val)]+=1
                else:
                    side,d=feature.split('_motion_')
                    for f in primary[a]['motion']:
                        direction=f[side];direction_counts[str(direction)]+=1
                        val=None if direction in (None,'mixed') else int(d in direction)
                        primary_counts['null' if val is None else str(val)]+=1
                for e in audits[a]['observations']:
                    if e['feature']==feature:
                        current_counts['null' if e['value'] is None else str(e['value'])]+=1
                        if e['value']==1:positive_current.add((a,e['frame_id']))
                roi_total+=sum(e['feature']==feature for e in audits[a]['roi_exclusions'])
            support[role][feature]=dict(final_counts=dict(counts),positive_words=sorted({r['word'] for r in positive}),positive_people=sorted({r['person'] for r in positive}),positive_clips=len(positive),positive_cells_by_clip=positive,
                current_primary_counts=dict(primary_counts),current_reconciled_counts=dict(current_counts),current_positive_retained=len(positive_current&final_positive),
                final_positive_not_from_current_three_static_states=len(final_positive-positive_current) if feature in static else 0,
                roi_excluded_candidate_cells=roi_total,raw_motion_categories=dict(direction_counts),
                original_global_cell_class_weights={str(v):(sum(counts.values())/(2*counts[str(v)]) if counts[str(v)] else None) for v in (0,1)})
        for feature in ('L_open_four','R_motion_up'):
            counters=Counter();removed=[];unknown_inside=[]
            for r in group:
                a=r['alias'];raw=arrays[a]['raw'];data=arrays[a]['input'];z=arrays[a]['target'];lookup={int(f):i for i,f in enumerate(data['frame_id'])};col=features.index(feature)
                for e in audits[a]['observations']:
                    if e['feature']!=feature:continue
                    fid=e['frame_id'];lo,hi=e.get('interval_frames',[fid,fid]);ids=[lookup[f] for f in range(lo,hi+1)]
                    off=33 if feature.startswith('L') else 54
                    p=raw['xy_px'][ids,off:off+21];good=raw['observed'][ids,off:off+21]&np.isfinite(p).all(-1);b=r['box']
                    inside=(p[:,:,0]>=b[0])&(p[:,:,0]<b[2])&(p[:,:,1]>=b[1])&(p[:,:,1]<b[3])
                    missing=~good.any(-1);outside=(good&~inside).any(-1)
                    reason='missing_and_outside' if missing.any() and outside.any() else 'missing_hand_only' if missing.any() else 'observed_outside_only' if outside.any() else 'all_observed_inside'
                    val='null' if e['value'] is None else str(e['value']);counters[val+'|'+reason]+=1
                    retained=bool(z['target_mask'][lookup[fid],col]);counters[val+'|'+('retained' if retained else 'not_retained')]+=1
                    if e['value']==1 and not retained:
                        removed.append(dict(alias=a,frame_id=fid,interval_frames=[lo,hi],reason=reason,missing_original_frames=[ids[k] for k in np.flatnonzero(missing)],outside_original_frames=[ids[k] for k in np.flatnonzero(outside)]))
                    if e['value'] is None and reason=='all_observed_inside':unknown_inside.append(dict(alias=a,frame_id=fid))
            diagnostic[role][feature]=dict(current_evidence_cells=dict(counters),positive_exclusions=removed,unknown_with_detector_points_inside_roi=unknown_inside,
                caution='Detector availability/ROI categories explain gate behavior only. They do not make unknown RGB labels positive or negative.')
        spans=[];pose_coverage=[];rgb_coverage=[];rgb_width=[];rgb_extra=[];retained_spans=[]
        for r in group:
            a=r['alias'];data=arrays[a]['input'];target=arrays[a]['target'];lookup={int(f):i for i,f in enumerate(data['frame_id'])}
            for e in audits[a]['observations']:
                if e['feature']!='R_motion_up':continue
                lo,hi=e['interval_frames'];l,h=lookup[lo],lookup[hi];t=lookup[e['frame_id']];ids=set(range(l,h+1))
                seconds=float(data['timestamp_us'][h]-data['timestamp_us'][l])/1e6;spans.append([len(ids),seconds])
                head_ids=set(range(max(0,t-2),min(len(data['frame_id']),t+3)))
                pose_coverage.append(len(ids&head_ids)/len(ids))
                wins=set();mid=data['window_center_us'];ws=data['window_starts']
                for ti in head_ids:
                    k=int(np.searchsorted(mid,data['timestamp_us'][ti]))
                    wins.update([0] if k==0 else [len(mid)-1] if k==len(mid) else [k-1,k])
                pixels=set(f for w in wins for f in range(int(ws[w]),int(ws[w])+16))
                rgb_coverage.append(len(ids&pixels)/len(ids));rgb_width.append(len(pixels));rgb_extra.append(len(pixels-ids))
                if target['target_mask'][t,features.index('R_motion_up')]:retained_spans.append([len(ids),seconds])
        temporal[role]=dict(intervals=len(spans),span_quantile_levels=[0,.25,.5,.75,1],span_frame_and_second_quantiles=np.quantile(spans,[0,.25,.5,.75,1],axis=0).tolist(),
            retained_R_up_binary_interval_span_quantiles=np.quantile(retained_spans,[0,.25,.5,.75,1],axis=0).tolist(),
            pose5frame_fraction_of_interval_median=float(np.median(pose_coverage)),rgb_source_interval_coverage_median=float(np.median(rgb_coverage)),
            rgb_source_support_width_frame_quantiles=np.quantile(rgb_width,[0,.5,1]).tolist(),rgb_source_frames_outside_label_interval_median=float(np.median(rgb_extra)),
            note='Two k3 stride1 convolutions yield5 feature-center frames. Each RGB feature center already interpolates16-source-frame window embeddings; RGB support is therefore wider than5 raw frames. Pixel support derived from saved window starts/times without model inference.')
    pooled={}
    for feat in ('index_only','open_four'):
        pos=[];neg=[]
        for r in rows:
            if r['role']!='fit':continue
            z=arrays[r['alias']]['target']
            for side in ('R','L'):
                j=features.index(side+'_'+feat)
                for value,dest in ((0,neg),(1,pos)):
                    for t in np.flatnonzero((z['y'][:,j]==value)&z['target_mask'][:,j]):dest.append((r['alias'],r['word_id'],r['person'],int(t),side))
        pooled[feat]=dict(positive_hand_state_cells=len(pos),negative_hand_state_cells=len(neg),positive_words=len({p[1] for p in pos}),positive_clips=len({p[0] for p in pos}),positive_people=sorted({p[2] for p in pos}),negative_words=len({p[1] for p in neg}),
                          caveat='Pooling shares supervision across anatomical hands; paired hands and adjacent frames remain correlated, and reflection does not create independent observations.')
    visual_files=[OLD/'visuals'/a/'states.jpg' for a in ('P051','P057','P069')]
    files=[OLD/'LEARNING_INPUTS.json',OLD/'LABEL_AUDIT.json',OLD/'build_labels.py',OLD/'learning.py',OLD/'RGB_RUBRIC.json',Path(__file__)]+visual_files
    report=dict(status='FIT_DEV_DIAGNOSIS_COMPLETE_NO_LEARNING',reviewer='Same Codex subagent rgb_labels_a; separate arithmetic/code review and3 fit RGB sheets, not independent human truth',
        authorized_unit=str(U),scope=dict(fit=36,development=12,people_fit=[1,2,3],people_development=[4],aliases=[r['alias'] for r in rows],
            prior_held32_target_arrays_read=0,prior_held32_images_reviewed=0,prior_held32_scores_read_for_this_diagnosis=0,
            contextual_disclosure='Mandatory project context/handoff contains historical aggregate held results; no held score file, prediction, target array, image, or held label was used for this diagnosis or candidate selection.',
            mixed_manifests='Role metadata used to filter to48 allowed aliases before any target/raw array load or label-row analysis.'),
        no_training=True,no_new_model_inference=True,old_outputs_modified=False,
        pins={str(p):sha(p) for p in files},fit_dev_array_pins=pins,support=support,gate_diagnosis=diagnostic,temporal_contract=temporal,shared_shape_support=pooled,
        conclusions=[
            'L_open_four scarcity is not caused by dropping current positive states at ROI: all7 fit and1 dev current positives survive. Fit30 positives include23 additional prior fit annotation cells. Current108 fit states contain59 nulls;36 have observed points outside ROI,2 no observed hand,21 observed points inside but RGB uncertainty remains.',
            'Current left positive support is intrinsically sparse and view-dependent in these sampled clips. Three reviewed fit sheets show edge-on/overlapping left digits; availability inside ROI cannot establish all4 digit extension. No unknown labels were upgraded.',
            'R_motion_up has21 reconciled fit-positive intervals, of which12 are gated out:9 for at least one missing-hand frame only,3 for both missing and observed outside-ROI frames. Development9 positives reduce to4, with5 missing-only exclusions. Full-interval gate is conservative but should not be weakened by body-wrist fallback.',
            'The interval direction describes trajectory over the whole annotated span, while its target was placed on one nearest midpoint frame. The pose head sees5 local feature frames, substantially less than median13/14-frame fit/dev intervals. A plateau or turn near midpoint can therefore disagree with a whole-interval target.',
            'RGB baseline is not restricted to5 raw frames: its16-frame window representation plus interpolation and5-center head can cover most/all intervals and include adjacent phases. This contextual mismatch and smearing remain even when its source interval coverage is complete.',
            'Class-balanced loss already upweights rare classes; it cannot supply missing shape viewpoints or reconstruct excluded motion evidence. Support counts are correlated cells, not30 or9 independent videos.'
        ],
        visual_spot_checks=[
            dict(alias='P051',ordinals=[3,6,9],finding='Left hand is edge-on with right hand overlapping/contacting its near edge; individual non-thumb extension is not fully separated. Retain uncertainty; no relabel.'),
            dict(alias='P057',ordinals=[3,6,9],finding='Left palm is horizontal under the moving right hand; side-view digits overlap. Inside-ROI geometry does not prove all4 extension. Retain uncertainty; no relabel.'),
            dict(alias='P069',ordinals=[3,6,9],finding='Hands form a joined outline then flatten in projection; fingers overlap and later blur. Do not infer hidden finger extension from neighboring states. No relabel.')
        ],
        recommended_one_bounded_method=dict(name='Shared physical hand-shape observation with a separate whole-interval direction branch',
            reason='One compact candidate fixes the observation/supervision unit while sharing existing known shape evidence. No word-specific exception, new backbone, or reflected RGB extraction is required.',
            shape='One weight-shared hand shape function for anatomical R and L, using non-thumb12 projected finger internal angles plus their12 masks.14 angles may be retained as inputs but thumb invariance then is not guaranteed; targets still ignore thumb. No word or side ID. Keep per-hand unknown masks and existing ROI rules.',
            invariance='2D internal finger angles are invariant to rigid image reflection/rotation. Absolute palm-axis x/y, body-relative wrist x, and image direction are not invariant; exclude them from the shared invariant shape input or explicitly transform them, rather than call them invariant.',
            interval='One direction prediction per [0,3],[3,6],[6,9],[9,12] evidence interval from its original full-frame torso-relative hand wrist trajectory, native consecutive-frame velocity, availability masks, and actual elapsed time. Preserve mixed/null as excluded/unknown; do not turn them into still or four negatives.',
            resample16='Choose16 deterministic temporal bins or nearest original timestamps while preserving missing masks. Never interpolate through missing observations or label synthetic/duplicated samples as new evidence. Compute velocity on original consecutive frames before resampling. If bins pool multiple source frames, retain ordered/time and availability information so all original frames contribute; audit reversal cases.',
            aggregation='Do not reduce an interval to only mean position, endpoints, or unsigned displacement: these erase reversals/loops. Preserve temporal order and use one loss unit per original labeled interval; repeated16 slots do not increase sample counts.',
            gate='Keep no-hand fallback excluded. Only a separate direct RGB audit of missing-only allowed fit/dev frames could establish ROI coverage; it cannot fabricate missing geometry or restore an interval containing an actually outside-ROI hand.',
            fair_evaluation='Freeze candidate, label/ROI policy, feature columns and interval output timing before new ID7/8 test. Compare common physical tasks and report shape vs direction separately. State changed supervision/architecture/input context; do not attribute all gains solely to sharing. Never tune on consumed ID5/6 held32.',
            runtime='Interval endpoint must be reached before its full-interval direction is emitted; report that latency instead of presenting the midpoint label as causal online output.'),
        alternatives=[
            dict(method='Horizontal RGB reflection with physical R/L label exchange',decision='Not first choice',reason='Existing768D SignRep vectors have no valid simple reflection transform; pixel reflection requires new frozen-backbone extraction. Reflection does not add independent samples or reveal hidden left digits, and preserves no claim of lexical sign identity.'),
            dict(method='Increase class weights or duplicate rare cells',decision='Insufficient as primary fix',reason='The existing loss already uses inverse class counts. Duplication changes weighting, not independent word/person/view support or interval semantics.'),
            dict(method='Targeted full-source RGB review of missing-only fit/dev motion frames',decision='Useful optional evidence supplement',reason='9 fit-positive R-up intervals and5 dev-positive intervals fail only missing detector frames, but recovery needs every missing frame visually verified inside ROI. No automatic recovery or geometry interpolation.'),
            dict(method='Loosen ROI/missing gate or copy neighboring labels',decision='Reject',reason='Would reintroduce unsupported input evidence and change unknown into apparent truth.')],
        limitations=['No causal attribution of previous quality scores is made. This is support/contract diagnosis, not a performance experiment.','Sparse current static sampling can miss brief shapes; stable neighboring labels cannot fill unobserved states.','2D angle geometry remains sensitive to foreshortening, wrong hand identity, and occlusion; shared weights do not solve those observation errors.'])
    output=REVIEW/'FIT_DEV_DIAGNOSIS.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status=report['status'],fit=36,development=12,pooled=pooled,temporal=temporal,output=str(output)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
