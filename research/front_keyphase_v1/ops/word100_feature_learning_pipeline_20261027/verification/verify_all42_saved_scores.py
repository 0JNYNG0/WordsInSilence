"""Audit completed 42-run saved outputs, without forward passes or fitting."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import numpy as np
from verify_finetune_independent import metrics, same, sha

U=Path(__file__).resolve().parents[1]
V=U/'verification'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    checks=[]
    def check(name,ok,detail=None):
        checks.append({'name':name,'passed':bool(ok),'detail':detail});assert ok,name
    c=read(U/'EXPERIMENT.json');rows=c['rows'];classes=c['classes'];seeds=c['seeds']
    roles={name:np.asarray([i for i,r in enumerate(rows) if r['role']==name]) for name in ['fit','validation','test']}
    ii=roles['test'];truth=np.asarray([rows[i]['target'] for i in ii]);valtruth=np.asarray([rows[i]['target'] for i in roles['validation']])
    original=read(U/'TRAINING_FROZEN_BEFORE_TEST.json');ft=read(U/'FINETUNE_COMPLETE.json');adapted=read(U/'ADAPTED_TRAINING_FROZEN.json')
    manifests=[original['checkpoints'],ft['checkpoints'],adapted['checkpoints']]
    allpins={k:v for d in manifests for k,v in d.items()}
    expected={f'probe_{name}_{s}' for name in c['encoder_candidates'] for s in seeds}
    expected|={f'observer_{condition}_{s}' for condition in c['modalities']+['MASK_T'] for s in seeds}
    expected|={f'ft_{name}_{s}' for name in c['encoder_candidates'] for s in seeds}
    expected|={f'adapted_{condition}_{s}' for condition in ['R_T','RJ_F','RJ_T'] for s in seeds}
    check('exact_42_frozen_final_checkpoints',len(allpins)==42 and [len(d) for d in manifests]==[24,9,9]
          and {Path(x).stem for x in allpins}==expected and not any('_resume' in x for x in allpins))
    check('all_final_checkpoint_bytes_preserved',all(sha(Path(p))==h for p,h in allpins.items()))
    check('all_42_outputs_present_and_no_extra_test_files',
          {p.name[:-len('_test.json')] for p in (U/'models').glob('*_test.json')}==expected
          and {p.name[:-len('_test.npz')] for p in (U/'models').glob('*_test.npz')}==expected)
    check('execution_complete',not (U/'ADAPTED_RUNNING.json').exists() and not (U/'TRAINING_RUNNING.json').exists()
          and (U/'COMPARISON.json').exists() and (U/'ADAPTED_COMPARISON.json').exists())
    inputpins={}
    for filename in ['TEST_INPUT_PINS.json','ADAPTED_TEST_PINS.json']:
        ip=read(U/filename)
        check('fixed_input_bytes_'+filename,all(sha(Path(p))==h for p,h in ip['pins'].items()))
        inputpins[filename]=sha(U/filename)
    check('fixed_source_and_training_contracts',original['contract_sha256']==sha(U/'EXPERIMENT.json')
          and ft['protocol_sha256']==sha(U/'FINETUNE_PROTOCOL.json')
          and adapted['protocol_sha256']==sha(U/'ADAPTED_FUSION_PROTOCOL.json'))
    # Inspect only this unit's current public observation cache, not operating files.
    j=np.load(U/'data/PUBLIC_JOINTS.npz',allow_pickle=False);index={str(v):i for i,v in enumerate(j['image_ids'])}
    ordered=np.asarray([index[r['id']] for r in rows]);raw=j['raw_xyz'][ordered].astype(np.float32)
    rel=raw-raw[:,:1];scale=np.linalg.norm(rel[:,[5,9,13,17]],axis=-1).max(axis=1)
    seen=j['mask'][ordered].all(axis=1)&(scale>1e-6)
    masks={'detector_missing':~seen[ii],'detector_present':seen[ii]}
    for sid in ['signer08','signer09']:masks[sid]=np.asarray([rows[i]['signer']==sid for i in ii])
    masks['near_label_change']=np.asarray([any(k>=0 and rows[k]['target']!=rows[i]['target'] for k in rows[i]['context_indices']) for i in ii])
    check('all_test_denominators_preserved',len(ii)==382 and len(set(truth))==28
          and int(masks['near_label_change'].sum())==19
          and sum(int(masks[k].sum()) for k in ['detector_missing','detector_present'])==382
          and [int(masks[k].sum()) for k in ['signer08','signer09']]==[253,129])
    val_by_stem={};computed={};predictions={};artifacts={}
    for stem in sorted(expected):
        cp=U/'models'/(stem+'.pt');vp=U/'models'/(stem+'_validation.npz');rp=cp.with_suffix('.json')
        numeric=U/'models'/(stem+'_test.npz');report=U/'models'/(stem+'_test.json')
        r=read(report);z=np.load(numeric,allow_pickle=False)
        pinname='ADAPTED_TEST_PINS.json' if stem.startswith(('ft_','adapted_')) else 'TEST_INPUT_PINS.json'
        check('test_numeric_identity_'+stem,np.array_equal(z['indices'],ii)
              and z['logits'].shape==(382,45) and np.isfinite(z['logits']).all()
              and str(z['checkpoint_sha256'])==sha(cp)
              and str(z['input_pins_sha256'])==inputpins[pinname]
              and r['checkpoint_sha256']==sha(cp) and r['stem']==stem)
        if pinname=='ADAPTED_TEST_PINS.json':
            check('adapted_json_hash_binding_'+stem,r['input_pins_sha256']==inputpins[pinname]
                  and r['numeric_sha256']==sha(numeric))
        else:
            check('original_json_freeze_binding_'+stem,r['freeze_sha256']==sha(U/'TRAINING_FROZEN_BEFORE_TEST.json'))
        calc=metrics(truth,z['logits'],classes)
        check('independent_score_arithmetic_'+stem,same(calc,r['test']))
        for key,mask in masks.items():
            check('independent_subgroup_'+stem+'_'+key,same(metrics(truth[mask],z['logits'][mask],classes),r['subgroups'][key]))
        vz=np.load(vp,allow_pickle=False);vr=read(rp)
        valcalc=metrics(valtruth,vz['logits'],classes)
        check('saved_validation_'+stem,np.array_equal(vz['indices'],roles['validation'])
              and vz['logits'].shape==(284,45) and np.isfinite(vz['logits']).all()
              and same(valcalc,vr['validation']))
        if stem.startswith('adapted_'):
            tr=vr['trace'];check('adapted_training_trace_'+stem,vr['protocol_sha256']==sha(U/'ADAPTED_FUSION_PROTOCOL.json')
                and vr['checkpoint_sha256']==sha(cp) and vr['test_seen'] is False
                and len(tr)==60 and all(x['epoch']==i+1 and x['seen_count']==2693 and math.isfinite(x['train_loss']) for i,x in enumerate(tr)))
        val_by_stem[stem]=valcalc;computed[stem]=calc;predictions[stem]=z['logits'].argmax(axis=1)
        artifacts[stem]={'checkpoint_sha256':sha(cp),'test_npz_sha256':sha(numeric),'test_json_sha256':sha(report),
                         'validation_npz_sha256':sha(vp)}
    for filename,subset in [('COMPARISON.json',{Path(p).stem for p in manifests[0]}),
                            ('ADAPTED_COMPARISON.json',{Path(p).stem for d in manifests[1:] for p in d})]:
        summary=read(U/filename)
        check('aggregate_exact_outputs_'+filename,len(summary['results'])==len(subset)
              and {r['stem'] for r in summary['results']}==subset and summary['test_tuning'] is False
              and all(same(r,read(U/'models'/(r['stem']+'_test.json'))) for r in summary['results']))
    ftchoice=read(U/'ADAPTED_ENCODER_SELECTION.json');ftgroups=ft['models']
    best=max(x['validation_macro_f1_mean'] for x in ftgroups)
    bestname=min([x for x in ftgroups if x['validation_macro_f1_mean']>=best-.01],key=lambda x:x['cpu_p95_mean_ms'])['name']
    check('adapted_encoder_choice_and_seed_preserved',ftchoice['selected']==bestname=='mobilenet_v3_small'
          and ftchoice['seed_for_common_features']==seeds[0] and ftchoice['test_seen'] is False)
    fp=read(U/'ADAPTED_FUSION_PROTOCOL.json');fm=read(U/'data/ADAPTED_RGB_META.json')
    features=np.load(U/'data/adapted_rgb_features.npz',allow_pickle=False)
    check('adapted_input_contract_links',fp['experiment_sha256']==sha(U/'EXPERIMENT.json')
          and fp['selection_sha256']==sha(U/'ADAPTED_ENCODER_SELECTION.json')
          and fp['feature_sha256']==fm['features_sha256']==sha(U/'data/adapted_rgb_features.npz')
          and fp['joint_sha256']==sha(U/'data/PUBLIC_JOINTS.npz')
          and fm['checkpoint_sha256']==sha(U/'models'/f'ft_{bestname}_{seeds[0]}.pt')
          and fm['seed']==seeds[0] and list(features['ids'])==[r['id'] for r in rows]
          and features['features'].shape==(3359,576) and np.isfinite(features['features']).all()
          and np.allclose(np.linalg.norm(features['features'],axis=1),1.,rtol=0.,atol=2e-6))
    obs=read(U/'OBSERVER_SELECTION.json');candidate_groups=[]
    for kind,conditions in [('observer',c['modalities']),('adapted',['R_T','RJ_F','RJ_T'])]:
        for condition in conditions:
            scores=[val_by_stem[f'{kind}_{condition}_{s}'] for s in seeds]
            candidate_groups.append({'kind':kind,'condition':condition,
                'mean_validation_macro_f1':float(np.mean([x['macro_f1_supported'] for x in scores])),
                'mean_validation_accuracy':float(np.mean([x['accuracy'] for x in scores]))})
    selected=max(candidate_groups,key=lambda x:x['mean_validation_macro_f1'])
    check('observation_candidate_selection_uses_development_only',same(candidate_groups,obs['candidates'])
          and same(selected,obs['selected']) and selected['kind']=='observer' and selected['condition']=='RJ_F'
          and obs['production_enabled'] is False)
    group_reports=[]
    for group in sorted({s.rsplit('_',1)[0] for s in expected}):
        arr=[computed[f'{group}_{s}'] for s in seeds]
        subgroup_report={}
        for key,mask in masks.items():
            acc=[float((predictions[f'{group}_{s}'][mask]==truth[mask]).mean()) for s in seeds]
            subgroup_report[key]={'n_per_seed':int(mask.sum()),'accuracy_mean':float(np.mean(acc)),
                                  'accuracy_range':[min(acc),max(acc)]}
        group_reports.append({'group':group,'seeds':seeds,'n_per_seed':382,
            'correct_per_seed':[x['correct'] for x in arr],
            'accuracy_mean':float(np.mean([x['accuracy'] for x in arr])),
            'accuracy_range':[min(x['accuracy'] for x in arr),max(x['accuracy'] for x in arr)],
            'macro_f1_supported_mean':float(np.mean([x['macro_f1_supported'] for x in arr])),
            'macro_f1_supported_range':[min(x['macro_f1_supported'] for x in arr),max(x['macro_f1_supported'] for x in arr)],
            'subgroups':subgroup_report})
    paired=[]
    for before,after in [('observer_J_T','observer_RJ_F'),('observer_RJ_F','observer_RJ_T'),
                         ('observer_R_T','adapted_R_T'),('observer_RJ_F','adapted_RJ_F'),('observer_RJ_T','adapted_RJ_T')]:
        cases=[]
        for seed in seeds:
            a=predictions[f'{before}_{seed}']==truth;z=predictions[f'{after}_{seed}']==truth
            fix=int((~a&z).sum());reg=int((a&~z).sum())
            cases.append({'seed':seed,'corrected':fix,'regressed':reg,'both_correct':int((a&z).sum()),
                          'both_wrong':int((~a&~z).sum()),'accuracy_delta_pp':100*(fix-reg)/382})
        paired.append({'before':before,'after':after,'per_seed':cases,
                       'mean_accuracy_delta_pp':float(np.mean([r['accuracy_delta_pp'] for r in cases]))})
    report={'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'checks':checks,'count':len(checks),
            'fixed_run_count':42,'read_final_predictions':True,'inference_or_training_performed':False,
            'producer_functions_imported':False,'private_or_operating_model_files_read':0,
            'test_denominator_per_run':382,'supported_test_classes':28,
            'subgroup_denominators':{k:int(m.sum()) for k,m in masks.items()},
            'group_summaries':group_reports,'paired_changes':paired,
            'development_selected_candidate':selected,'test_selection_or_tuning_performed':False,
            'artifacts':artifacts,'script_sha256':sha(Path(__file__)),
            'training_vs_test_implementation':{'adapted_training_protocol_code_sha256':fp['code_sha256'],
                 'current_test_code_sha256':sha(U/'adapted_comparison.py'),
                 'reason':'Final-only exact manifest selection and saved-prediction integrity hardening after code review; original learning contract retained.',
                 'training_contract_overwritten':False},
            'limitations':['Three seeds use the same 382 examples; pooling them as 1,146 independent people/examples is invalid.',
                           '19 near-change center crops correspond to 6 changed adjacent pairs in 5 source clips, not 19 independent transitions.',
                           'No independent KSL normal-variation or genuine-error labels; these scores cannot validate educational corrections.',
                           'Test subset contains 28/45 classes and only two recorded signers; sparse rare-class support limits generalization.',
                           'Saved score verification cannot prove every unlogged historical action; frozen artifacts, code inspection and execution records provide bounded evidence.',
                           'Current model choice is retained from validation even if another method looks better on final test.']}
    path=V/'ALL42_SCORES_INDEPENDENT.json';assert not path.exists(),'Preserve first audit'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','checks':len(checks),'runs':42,'groups':group_reports},ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');main()
