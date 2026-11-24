"""Record reviewed single-run readiness after all fixed prerequisites exist.

No model import/forward, optimizer, extraction, DTW or evaluation is run here.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import ast,hashlib,json
import numpy as np

A=Path(__file__).resolve().parent
OUT=A/'PROCESS_FINAL_READINESS_REVIEW.json'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
PINS={};CHECKS=Counter()

def require(value,message,category='contract'):
    CHECKS[category]+=1
    if not value:raise AssertionError(message)

def pin(path,expected=None):
    p=Path(path).resolve();name=str(p)
    if name not in PINS:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        PINS[name]=h.hexdigest()
    require(expected is None or PINS[name]==expected,'Pinned file changed: '+name,'hash')
    return PINS[name]

def main():
    require(not OUT.exists(),'Preserve completed readiness review')
    require(not (A/'PROCESS_TRAIN_RESULT.json').exists(),'No post-training readiness')
    require(not (A/'PROCESS_EVALUATION.json').exists(),'No post-evaluation readiness')
    require(not Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/process_run/seed42').exists(),
            'No existing/interrupted training output')
    required=['process_head.py','train_process_head.py','evaluate_process_head.py','verify_process_evaluation_saved.py',
      'preflight_process_training.py','freeze_process_comparison.py','process_auto_proposal.py','process_qualification.py',
      'prepare_process_auto_native.py','evaluate_process_auto.py','check_process_auto_contract.py',
      'PROCESS_AUTO_NATIVE_MANIFEST.json','PROCESS_AUTO_PROPOSAL_PLAN.json','PROCESS_AUTO_CONTRACT_VERIFIED.json',
      'PROCESS_VIEW_TARGETS.json','DEV_PROCESS_VIEW_TARGETS.json','PROCESS_TARGETS.json','DEV_PROCESS_TARGETS.json',
      'PROCESS_INPUT_MANIFEST.json','DEV_PROCESS_INPUT_MANIFEST.json',
      'PROCESS_TRAINING_DESIGN_REVIEW.json','PROCESS_HEAD_CODE_REVIEW.json','PROCESS_EVALUATION_CODE_REVIEW.json',
      'PROCESS_EVALUATOR_REVIEW_ADJUSTMENT.txt','PROCESS_ACTUAL_INPUT_PREFLIGHT.json','PROCESS_INPUTS_VERIFIED.json',
      'DEV_PROCESS_INPUTS_VERIFIED.json','PROCESS_VIEW_TARGETS_INDEPENDENT.json','DEV_PROCESS_VIEW_TARGETS_VERIFIED.json',
      'PROCESS_EXTRACTION_FROZEN.json','DEV_PROCESS_EXTRACTION_FROZEN.json','PROCESS_HEAD_SYNTHETIC_VERIFIED.json',
      'PROCESS_TRAIN_BINDING_STATIC_VERIFIED.json','PROCESS_LABEL_PROTOCOL.json','PHASE_LABEL_ADDENDUM.json',
      'write_process_final_readiness.py']
    for name in required:
        require((A/name).is_file(),'Required file not ready: '+name)
        pin(A/name)
        if name.endswith('.py'):ast.parse((A/name).read_text(encoding='utf8'))
    for name in ['PROCESS_ACTUAL_INPUT_PREFLIGHT.json','PROCESS_INPUTS_VERIFIED.json','DEV_PROCESS_INPUTS_VERIFIED.json',
                 'PROCESS_VIEW_TARGETS_INDEPENDENT.json','DEV_PROCESS_VIEW_TARGETS_VERIFIED.json','PROCESS_AUTO_CONTRACT_VERIFIED.json',
                 'PROCESS_HEAD_SYNTHETIC_VERIFIED.json','PROCESS_TRAIN_BINDING_STATIC_VERIFIED.json']:
        require(read(A/name)['status'].startswith('PASS'),'Passed verification required: '+name,'review')
    for name in ['PROCESS_EXTRACTION_FROZEN.json','DEV_PROCESS_EXTRACTION_FROZEN.json','PROCESS_AUTO_NATIVE_MANIFEST.json']:
        d=read(A/name)
        for path,h in d['pins'].items():pin(path,h)

    pre=read(A/'PROCESS_ACTUAL_INPUT_PREFLIGHT.json')
    require(pre['views']==217 and pre['inputs']==216 and pre['optimization_steps']==0 and pre['development_accessed'] is False,
            'FIT actual preflight scope','review')
    for path,h in pre['pins'].items():
        # Evaluator reporting-only addition is retained as a named amendment.
        if Path(path).name=='evaluate_process_head.py':continue
        pin(path,h)
    for name in ['PROCESS_HEAD_SYNTHETIC_VERIFIED.json','PROCESS_TRAIN_BINDING_STATIC_VERIFIED.json']:
        d=read(A/name)
        for path,h in d.get('code_pins',d.get('pins',{})).items():pin(path,h)
    for path,h in read(A/'PROCESS_AUTO_CONTRACT_VERIFIED.json')['pins'].items():pin(path,h)
    checked=read(A/'PROCESS_EVALUATION_CODE_REVIEW.json')
    for name,h in checked['pins'].items():pin(A/name,h)

    counts={};points=intervals=0;actual_rows=0
    for role,tn,mn,vn,expected,n_inputs,ids,label_counts in [
      ('fit','PROCESS_VIEW_TARGETS.json','PROCESS_INPUT_MANIFEST.json','PROCESS_INPUTS_VERIFIED.json',217,216,set(range(1,10)),{'1':79,'0':128,'None':10}),
      ('development','DEV_PROCESS_VIEW_TARGETS.json','DEV_PROCESS_INPUT_MANIFEST.json','DEV_PROCESS_INPUTS_VERIFIED.json',80,80,{11,12,13},{'1':30,'0':45,'None':5})]:
        target=read(A/tn)['rows'];im=read(A/mn)['rows'];verified=read(A/vn)
        require(len(target)==expected and len(im)==n_inputs,'Frozen input/target counts','source')
        require(Counter(str(r['observed_process_evidence']) for r in target)==label_counts,'Process label counts','target')
        require({r['recorded_person_id'] for r in target}==ids,'Registered split','source')
        require(verified['manifest_sha256']==pin(A/mn),'Input verification manifest lineage','hash')
        if 'target_sha256' in verified:require(verified['target_sha256']==pin(A/tn),'Target verification lineage','hash')
        by_key={r['pixel_view_key']:r for r in im};require(len(by_key)==n_inputs,'Unique inputs','source')
        axes={}
        for key,r in by_key.items():
            require(r['role']==role and r['person'] in ids,'Actual input split','source')
            pin(r['input'],r['input_sha256'])
            with np.load(r['input'],allow_pickle=False) as z:
                n=len(z['frame_id'])
                require(z['x'].shape==z['mask'].shape==(n,69) and z['rgb'].shape==(n,768),'Actual feature shapes','input')
                require(z['mask'].dtype==bool and z['image_available'].dtype==bool,'Mask dtypes','input')
                require(np.isfinite(z['x'][z['mask']]).all() and np.isfinite(z['rgb']).all(),'Actual finite inputs','input')
                axes[key]={k:z[k].copy() for k in ['frame_id','timestamp_us','image_available']}
                actual_rows+=n
        for r in target:
            inp=by_key[r['pixel_view_key']];a=axes[r['pixel_view_key']];fm=r['ordered_frame_map']
            require(r['role']==role and inp['clip_id']==r['clip_id'] and inp['person']==r['recorded_person_id'],
                    'Target/input clip-person binding','source')
            require(inp['source_sha256']==r['source_sha256'] and r['view_id'] in inp['aliases'],'Source SHA/alias','source')
            require(a['frame_id'].tolist()==[f['output_index'] for f in fm]==list(range(len(fm))),'Real output axis','input')
            require(a['timestamp_us'].tolist()==[f['view_timestamp_us'] for f in fm],'Real view time','input')
            require(a['image_available'].tolist()==[f['image_available'] for f in fm],'Real availability','input')
            require(inp['source_frame_indices']==[f['source_native_frame'] for f in fm] and
                    inp['source_timestamps_us']==[f['source_native_timestamp_us'] for f in fm],'Native source axis','input')
            require(all(r[k] is None for k in ['whole_word_correctness','learner_error','required_repetitions']),
                    'No invented educational truth','target')
            for v in r['sparse_relation_targets']:
                i=v['output_index'];require(fm[i]['image_available'] and fm[i]['source_native_frame']==v['source_frame'],'Sparse visible target','target')
            for v in r['fully_retained_motion_interval_targets']:
                ii=v['output_indices'];ff=v['source_frames']
                require(len(ii)>=2 and ii==list(range(ii[0],ii[-1]+1)) and ff==list(range(ff[0],ff[-1]+1))
                        and ff==[fm[i]['source_native_frame'] for i in ii] and all(fm[i]['image_available'] for i in ii),
                        'Whole observed interval target','target')
            points+=len(r['sparse_relation_targets']);intervals+=len(r['fully_retained_motion_interval_targets'])
        counts[role]={'views':len(target),'unique_inputs':len(im),'source_queries':len({r['query_id'] for r in target}),
          'source_clips':len({r['clip_id'] for r in target}),'label_counts':dict(label_counts),'registered_ids':sorted(ids)}

    auto=read(A/'PROCESS_AUTO_NATIVE_MANIFEST.json')
    require(len(auto['rows'])==60 and len(auto['unique_inputs'])==48,'Automatic native scope','auto')
    require(auto['manual_review_boundaries_used_for_proposals'] is False and auto['new_DTW']==0 and auto['no_model_forward'] is True,
            'No boundary/score input to proposal','auto')
    for row in auto['rows']:
        people=set(range(1,10)) if row['role']=='fit' else {11,12,13}
        require(row['person'] in people and row['actual_sensor_extent']==[0,0,1920,1080],'Automatic public full-frame scope','auto')
        proposal=row['proposal']
        require(proposal['max_proposals_per_query']==1 and proposal['observed_process_truth'] is None
                and proposal['whole_word_correctness'] is None and proposal['additional_DTW']==0,
                'Proposal is context, not truth','auto')
    report={'at':datetime.now(timezone.utc).isoformat(),'status':'READY_FOR_ONE_FIXED_TRAINING_RUN','training_allowed':True,
      'reviewer':'Codex /root/predicate_contract_review; separate code/input review, not independent human/KSL expert gold',
      'scope':'Only the one separately frozen seed42/final80epoch additive process-head experiment; no runtime replacement.',
      'pins':PINS,'counts':counts,'actual_unique_input_rows_checked':actual_rows,'sparse_targets_checked':points,
      'whole_interval_targets_checked':intervals,'automatic_query_proposals':60,'automatic_unique_native_inputs':48,
      'closed_blockers':['DEV15 native and65 partial/sentinel direct reviews compiled and separately reconstructed.',
        'FIT216 andDEV80 actual input shape/source/native-time/view-time/availability bindings verified.',
        'FIT-only actual untrained preflight and synthetic excluded-feature/unknown tests passed.',
        'Single saved-path canonical-guide context proposal and observation-only qualification fixed before predictions.',
        'Raw/qualified/manual/automatic evaluation scopes and source-query denominators kept separate.'],
      'scientific_limits':['Process1 is supported physical evidence in the observed scope, never whole-word completion or education correctness.',
        'Process0 can retain positive local motion; unavailable or ambiguous data remains unknown.',
        'DEV is repeatedly exposed and transformed views are dependent, not an independent population test.',
        'Native automatic truth is existential physical presence, not exact proposed boundaries or semantic completion; the model-selected local interval may still have unreviewed local truth.',
        'Short-input repeated encoder context is not independent per-frame RGB evidence.',
        'Retimed identical pixels and known edit seams cannot establish natural-speed or real-occlusion robustness.',
        'Input time/edit/person/word labels are excluded from learned features; implicit length/jump shortcuts still require fixed controls.',
        'No post-result epoch/threshold/seed/weight/window selection is authorized.'],
      'excluded_learned_inputs':['word/person/view/edit identifiers','source/output frame IDs and timestamps','percentage progress',
        'annotation boundaries and truth','seam flags and padding counts','geometry diagnostics and precomputed edit velocity'],
      'accepted_reporting_amendment':'PROCESS_ACTUAL_INPUT_PREFLIGHT retains original evaluator pin. Reporting-only change is documented; unchanged binding/model was not rerun.',
      'runtime_replacement_authorized':False,'old_models_rerun':False,'reviewer_new_model_forward':0,'reviewer_new_training':0,
      'new_accuracy_claim':False,'checks':dict(CHECKS),'checks_total':sum(CHECKS.values()),'whole_goal_complete':False}
    with OUT.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':report['status'],'pins':len(PINS),'counts':counts,'checks':report['checks_total']},ensure_ascii=False))

if __name__=='__main__':main()
