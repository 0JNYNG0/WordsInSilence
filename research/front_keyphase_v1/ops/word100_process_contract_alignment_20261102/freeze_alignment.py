"""Root releases one finite trial only after actual source/contract review."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from process_data import read,sha,build_fit_dataset,multiplicities
from train_process_head import TRAINING_PROFILE,OUT

U=Path(__file__).resolve().parent
AB=U.parent/'word100_action_completion_supervision_20261102'


def main():
    dest=U/'ALIGNMENT_FROZEN.json';assert not dest.exists() and not OUT.exists()
    pins={}
    def pin(p,expected=None):
        p=str(Path(p).resolve());h=sha(p)
        if expected is not None:assert h==expected,p
        if p in pins:assert pins[p]==h,p
        pins[p]=h
    def artifact(p,expected_status=None):
        p=Path(p);d=read(p);pin(p)
        if expected_status:assert d['status'].startswith(expected_status),(p,d['status'])
        for field in ('pins','freezable_pins','reviewed_code_pins','code_pins'):
            for f,h in d.get(field,{}).items():pin(f,h)
        return d
    v2=artifact(U/'SUPERVISION_V2_SAVED_VERIFIED.json','PASS')
    support=artifact(U/'RGB_SUPPORT_VERIFIED.json','PASS')
    nominal=artifact(U/'NOMINAL_TARGETS_VERIFIED.json','PASS')
    code=artifact(U/'EVALUATION_CODE_REVIEW.json','PASS')
    assert code['training_and_evaluation_allowed_after_root_freeze'] is True
    training_review=artifact(U/'TRAINING_CODE_REVIEW.json','PASS')
    assert training_review['training_allowed_after_root_freeze'] is True
    artifact(U/'RUNTIME_CONTRACT_CHECKED.json','PASS')
    artifact(AB/'PROCESS_RUNTIME_SUPERVISION_BINDING_VERIFIED.json','PASS')
    artifact(U/'RGB_SUPPORT_MANIFEST.json')
    assert support['manifest_sha256']==sha(U/'RGB_SUPPORT_MANIFEST.json')
    targets=artifact(U/'NOMINAL_PROCESS_TARGETS.json','DIRECT_NOMINAL_PROCESS_TARGETS_READY')
    assert nominal['target_sha256']==sha(U/'NOMINAL_PROCESS_TARGETS.json')
    for name in ['FIT_VIEW_TARGETS_V2.json','DEV_VIEW_TARGETS_V2.json','FIRST_RUNTIME_INTERVAL_REVIEW_MANIFEST.json',
        'TRAINING_IMPLEMENTATION_REVIEW.json','BASE_TRAINING_INPUT_PREFLIGHT.json','FOLLOWUP_PLAN.json','SCOPE.json']:
        artifact(U/name)
    baseline=artifact(AB/'PROCESS_COMPARISON_FROZEN.json')
    for name in ['PROCESS_TRAIN_RESULT.json','PROCESS_EVALUATION.json','PROCESS_AUTO_EVALUATION.json',
        'PROCESS_DECISION.json','PROCESS_EVALUATION_SAVED_VERIFIED.json','PROCESS_AUTO_EVALUATION_VERIFIED.json',
        'PROCESS_INPUT_MANIFEST.json','DEV_PROCESS_INPUT_MANIFEST.json','PROCESS_AUTO_NATIVE_MANIFEST.json',
        'PROCESS_RUNTIME_SUPERVISION_BINDING.json']:
        artifact(AB/name)
    for name in ['process_head.py','process_data.py','process_rgb_support.py','process_runtime_contract.py',
        'train_process_head.py','evaluate_contract_candidate.py','verify_evaluation_saved.py','decide_alignment.py',
        'build_nominal_process_targets.py','freeze_alignment.py']:
        pin(U/name)
    assert sha(U/'process_head.py')==sha(AB/'process_head.py')
    data=build_fit_dataset(U/'NOMINAL_PROCESS_TARGETS.json',require_spans=True)
    multiplicities(data['records'])
    for p,h in data['pins'].items():pin(p,h)
    oldtrain=read(AB/'PROCESS_TRAIN_RESULT.json');oldstats=read(oldtrain['normalization'])
    for key in ('mean','std'):
        assert np.array_equal(np.asarray(oldstats[key]),np.asarray(data['stats'][key])), 'Normalization changed'
    assert data['nominal_span_status']['positive']>0 and data['nominal_span_status']['negative']>0
    assert data['development_arrays_opened']==0 and data['model_forwards']==0
    for p,h in pins.items():assert sha(p)==h,p
    d=dict(status='READY_FOR_ONE_FIXED_TRAINING_RUN',at=datetime.now(timezone.utc).isoformat(),
        training_allowed=True,seed=42,epochs=80,candidates=1,training_profile=TRAINING_PROFILE,
        output_directory=str(OUT),pins=pins,
        nominal_span_manifest=dict(path=str((U/'NOMINAL_PROCESS_TARGETS.json').resolve()),
            sha256=sha(U/'NOMINAL_PROCESS_TARGETS.json'),required=True),
        fixed_gates=baseline['fixed_gates'],fit_counts=data['counts'],nominal_counts=data['nominal_span_status'],
        evaluation_budget=dict(fixed_global_contexts=357,reviewed_views=297,automatic_contexts=60,
            max_runtime_local_intervals_per_context=8,old_model_forward=0,
            same_selection_cached=True,annotated_netmotion_diagnostics=True,
            nominal_fit_diagnostics=True,development_used_for_optimization=False),
        fixed_semantics='Whole-context and directly reviewed nominal linked physical process; netmotion separate. RGB context disclosed, not exact event boundary truth.',
        combined_changes_not_separate_causal_effects=True,
        current_C_T3F_and_web_preserved=True,personal_or_mentor_material_used=False,
        new_downloads=False,whole_goal_complete=False,
        root_code_review='Root read data/train/runtime/eval implementations and source contracts; agent reviewed separate scope/arithmetic. Same Codex, not independent human or KSL expert.',
        next='Run once80epochs, evaluate once, verify saved results and apply unchanged gates. No threshold/epoch/seed search. Exact runtime timestamps and real demo still require validation.')
    with dest.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print('READY',len(pins),data['nominal_span_status'])

if __name__=='__main__':main()
