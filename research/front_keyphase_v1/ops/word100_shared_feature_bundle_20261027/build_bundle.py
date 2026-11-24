"""Reconcile image-only drafts without turning agreement into ground truth."""
import json, hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

U = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, obj): (U/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def reconcile(x,y):
    assert x in (0,1,None) and y in (0,1,None)
    if x is not None and y == x: return x, 'both_visible_agree'
    if x is None and y is None: return None, 'both_unknown'
    if x is None or y is None: return None, 'one_unknown'
    return None, 'visible_disagreement'

def counts(rows, field):
    out={}
    for value, name in [(1,'present'),(0,'absent'),(None,'unknown')]:
        sel=[r for r in rows if r['labels'][field] == value]
        out[name]={'cells':len(sel), 'clips':len({r['clip_id'] for r in sel}),
                   'expressions':sorted({r['word_id'] for r in sel}),
                   'recorded_people':sorted({r['recorded_person_id'] for r in sel})}
    return out

def main():
    frozen=read(U/'FROZEN.json'); rubric=read(U/'RUBRIC.json')
    root=read(U/'ROOT_VISUAL_LABELS_RAW.json')['rows_by_alias']
    index={r['alias']:r for r in read(U/'visual_inputs/INDEX.json')['rows']}
    features=list(rubric['features']); ords=rubric['static_sample_ordinals']
    other_files=[U/'prior_labels/IMAGE_LABELS.json',U/'review/IMAGE_LABELS.json']
    secondary={}; secondary_seq={}
    for p in other_files:
        obj=read(p); assert obj['rubric_sha256']==sha(U/'RUBRIC.json')
        for r in obj['rows']:
            key=(r['alias'],r['ordinal']); assert key not in secondary
            secondary[key]=r
        for r in obj['sequences']:
            assert r['alias'] not in secondary_seq
            secondary_seq[r['alias']]=r
    expected={(a,n) for a in index for n in ords}
    assert set(secondary)==expected and set(root)==set(index)==set(secondary_seq)
    assert len(index)==36 and len(expected)==108
    rows=[]; seq=[]; disagreements=[]
    static_status=Counter(); sequence_status=Counter()
    for a in sorted(index):
        source=index[a]; frames={f['ordinal']:f for f in source['frames']}
        assert len(frames)==16
        meta={k:source[k] for k in ['alias','clip_id','word_id','label','recorded_person_id','role']}
        for i,n in enumerate(ords):
            assert len(root[a]['values'][i])==len(features)
            labels={}; status={}; pvals=dict(zip(features,root[a]['values'][i])); svals={f:secondary[a,n][f] for f in features}
            for f in features:
                labels[f], status[f]=reconcile(pvals[f],svals[f]); static_status[status[f]]+=1
                if status[f] in ('one_unknown','visible_disagreement'):
                    disagreements.append({'alias':a,'ordinal':n,'feature':f,'primary':pvals[f],'secondary':svals[f],'status':status[f]})
            for vals in (pvals,svals,labels):
                for side in ('R','L'):
                    assert not(vals[side+'_index_only']==1 and vals[side+'_open_four']==1),(a,n,side)
            rows.append(dict(**meta,ordinal=n,frame_id=frames[n]['frame_id'],pts_us=frames[n]['pts_us'],image=frames[n]['file'],
                             labels=labels,status=status,primary=pvals,secondary=svals,primary_note=root[a]['note'],secondary_note=secondary[a,n].get('note','')))
        labels={}; status={}
        for i,s in enumerate(('R','L')):
            labels[s],status[s]=reconcile(root[a]['seq'][i],secondary_seq[a][s]);sequence_status[status[s]]+=1
            if status[s] in ('one_unknown','visible_disagreement'):
                disagreements.append({'alias':a,'ordinal':'sequence','feature':s+'_open_grouped_change',
                                      'primary':root[a]['seq'][i],'secondary':secondary_seq[a][s],'status':status[s]})
        seq.append(dict(**meta,labels=labels,status=status,primary=root[a]['seq'],secondary=secondary_seq[a],
                        primary_evidence_ordinals=list(range(1,17)),scope='16 sampled states only; includes transition/rest context, not necessarily lexical contrast'))
    roles=['fit_candidate','held_person','held_expression_seen_person','held_expression_new_person']
    coverage=[]
    for f in features:
        by_role={role:counts([r for r in rows if r['role']==role],f) for role in roles}
        fit=by_role['fit_candidate']; held=counts([r for r in rows if r['role'].startswith('held_expression')],f)
        gate={
            'fit_present_at_least2_expressions':len(fit['present']['expressions'])>=2,
            'fit_absent_at_least2_expressions':len(fit['absent']['expressions'])>=2,
            'held_expression_present':held['present']['cells']>0,
            'held_expression_absent':held['absent']['cells']>0}
        coverage.append(dict(feature=f,all=counts(rows,f),by_role=by_role,composition_checks=gate,
                             composition_pass=all(gate.values()),truth_validated=False))
    seq_cov={s:{'all':counts(seq,s),'by_role':{role:counts([r for r in seq if r['role']==role],s) for role in roles}} for s in ('R','L')}
    save('DRAFT_LABELS.json',dict(status='TWO_CODEX_IMAGE_DRAFTS_NOT_HUMAN_TRUTH',rubric_sha256=sha(U/'RUBRIC.json'),rows=rows,sequences=seq,
                                disagreements=disagreements,policy='Any unknown or disagreement remains unknown; neither source text nor estimated joints fills it.'))
    manifest={
        'status':'PREPARED_NOT_TRAINING_READY',
        'why':'Composition is a necessary small-sample check, not validation of visual labels or educational correctness.',
        'frozen_roles':frozen['counts'],
        'static':[{k:r[k] for k in ['alias','clip_id','word_id','recorded_person_id','role','ordinal','pts_us','image','labels']} for r in rows],
        'sequence':seq,
        'excluded_model_inputs':['target word ID','Gloss','dictionary explanation','teacher correct answer','user footage','mentor footage'],
        'allowed_model_inputs_future':['RGB at recorded PTS','public observed landmarks plus observation mask','time/order'],
        'unknown_handling':'exclude only that feature-cell from supervised loss; retain availability denominator and other visible features',
        'data_leakage_warning':'All clips already belong to prior TCN TRAIN. This split is for a NEW feature learner, not a fresh evaluation of C/T3-F or their embeddings.',
        'training_authorized_by_this_file':False,'human_verified_labels':0,
        'next_before_training':['Review physical visible labels independently; focus disagreements and each polarity/feature.','Freeze independent truth subset and permissible fit labels.','Report missing polarity coverage instead of swapping held expressions.'],
        'future_metrics':['per-feature precision/recall and unknown rate','per-person and held-expression reporting','normal crossing/rest not treated as errors','false correction rate requires separate expert reviewed correctness evidence'],
        'not_targets':['exact 3D contact','hidden fingers','sign correctness','mandatory stages','allowable height tolerance','facial emotion']}
    save('LEARNING_MANIFEST.json',manifest)
    protection=read(U.parent/'word100_face_relation_alignment_20261027/FROZEN.json')['protected']
    assert isinstance(protection,dict)
    for p,h in protection.items(): assert sha(p)==h, p
    for p,h in frozen['source_pins'].items(): assert sha(p)==h,p
    for r in frozen['rows']: assert sha(r['record'])==r['record_sha256'],r['record']
    pins={str(p):sha(p) for p in [U/'FROZEN.json',U/'RUBRIC.json',U/'ROOT_VISUAL_LABELS_RAW.json',*other_files,U/'visual_inputs/INDEX.json',U/'visual_inputs/VERIFY_SAVED.json']}
    result={
        'status':'PUBLIC_SHARED_FEATURE_DRAFT_BUNDLE_COMPLETE_TRAINING_NOT_RUN', 'at':datetime.now(timezone.utc).isoformat(),
        'counts':{'expressions':12,'recorded_people':3,'clips':36,'old_reused_clips':12,'new_clips':24,'rgb_sample_states':576,
                  'fixed_static_states':108,'static_feature_cells':756,'sequence_hand_cells':72,
                  'known_static_consensus':static_status['both_visible_agree'],'unknown_static_consensus':756-static_status['both_visible_agree'],
                  'known_sequence_consensus':sequence_status['both_visible_agree'],'unknown_sequence_consensus':72-sequence_status['both_visible_agree'],
                  'composition_pass_features':sum(c['composition_pass'] for c in coverage),'features':len(features),
                  'training_runs':0,'human_verified_labels':0,'operating_changes':0,'user_access':0},
        'static_review_status':dict(static_status),'sequence_review_status':dict(sequence_status),
        'roles':dict(Counter(r['role'] for r in index.values())), 'coverage':coverage,'sequence_coverage':seq_cov,
        'protected_pins_unchanged':len(protection),'pins':pins,
        'limitations':['Same-AI consistency is not accuracy or independent human truth.','16 samples per video do not label unsampled movements or exact repetition.',
                       'Spatial projection overlap is not physical contact or an error.','Sequence changes may include preparation/rest; not mandatory lexical phases.',
                       'Only 3 recorded IDs and 12 expressions; no all-sign generalization claim.','No new TCN/DTW performance evaluation.'],
        'completion':'The approved next preparation unit is complete; no trained feature predictor or educational feedback model is claimed.'}
    save('RESULT.json',result)
    print(json.dumps({'counts':result['counts'],'static':dict(static_status),'sequence':dict(sequence_status),
                      'coverage':[(x['feature'],x['composition_pass'],x['composition_checks']) for x in coverage]},ensure_ascii=False))

if __name__=='__main__': main()
