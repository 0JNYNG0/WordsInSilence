"""Local data/API binding only; candidate algorithms are Claude-authored modules."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

U=Path(__file__).resolve().parent
W=U.parent.parent
OLD=U.parent/'stgcn_execution_ready_20261005'
OLD_RUN=W/'runs/stgcn_full_path_v1'
RUN=W/'runs/stgcn_valid_window_v1'
sys.path.extend([str(U.parent/'stgcn_reference_alignment_20261005'),
                 str(U.parent/'stgcn_transfer_compare_20261005')])
import reference_adapter as ra
from stgcn_core import read_encoder_tensors
import generic_pose_io as gp
import generic_workflow as gw
import training_engine as engine
from window_bridge import bind_window_forward
from valid_window_forward import forward_valid_windows

PLAN_SHA='e4dc2ad90f70bd337de37890d92d614f260d45338dbf97818547fe37b8bc2389'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def setup():
    assert sha(U/'FROZEN_EXPERIMENT.json')==PLAN_SHA
    for mod in (gp,gw,engine):
        assert Path(mod.__file__).resolve().parent==U, mod.__file__
    plan=read(U/'FROZEN_EXPERIMENT.json')
    cfg=read(U/'CONFIG.json')
    assert cfg==plan['training_config']
    meta=read(OLD/'prepared/TRAINING_META.json')
    audit=read(U/'METADATA_AUDIT.json')
    assert audit['status']=='PASS'
    old_ctx=read(OLD/'EXECUTION_CONTEXT.json')
    old_frozen=gw.validate_frozen(str(OLD_RUN),old_ctx)
    assert old_frozen['final_model_sha256']==plan['comparison_source']['old_model_sha256']
    old_meta_pin=next(v for p,v in old_ctx['library_pins'].items() if p.replace('\\','/').endswith('/prepared/TRAINING_META.json'))
    assert audit['source_metadata_sha256']==old_meta_pin
    previous_audit=read(OLD/'EVALUATION_INPUT_INDEPENDENT_CHECK.json')
    assert previous_audit['status']=='PASS'
    assert audit['old_evaluation_diagnostics_sha256']==previous_audit['artifact_sha256']['EVALUATION_INPUT_DIAGNOSTICS.json']
    assert sha(OLD/'prepared/training.npz')==meta['cache_sha256']==plan['comparison_source']['training_cache_sha256']
    assert sha(OLD/'prepared/TRAINING_META.json')==audit['source_metadata_sha256']
    assert sha(U/'WINDOW_METADATA.npz')==audit['metadata_sha256']
    assert sha(OLD/'TRAIN_SPLIT.json')==plan['pins']['train_split']
    assert sha(U/'valid_window_forward.py')==plan['comparison_source']['source_component_sha256']
    artifact=U.parent/'openhands_artifact_20261005'
    assert sha(artifact/'TENSOR_METADATA.json')==plan['pins']['pretrained_metadata']
    tensors,_=read_encoder_tensors(artifact/'raw_dpc.zip',artifact/'TENSOR_METADATA.json',plan['pins']['pretrained_zip'],encoder_prefix='model.')
    encoder={k[len('conv_encoder.'):]:v for k,v in tensors.items() if k.startswith('conv_encoder.')}
    gru={k[len('agg.'):]:v for k,v in tensors.items() if k.startswith('agg.')}
    dpc,tf,_=ra.load_reference(U.parent/'stgcn_reference_alignment_20261005/reference',plan['pins']['source_pins'])
    def factory():
        model,_=ra.make_reference_model(dpc,encoder,gru,num_classes=93,seed=cfg['seed'])
        return bind_window_forward(model,forward_valid_windows)
    shear,rotate=tf.ShearTransform(.1),tf.RotatationTransform(.1)
    augment=lambda x:gp.augment_windows(x,lambda t:rotate(shear({'frames':t}))['frames'])
    return plan,cfg,meta,factory,augment,tf


def datasets(meta):
    with np.load(OLD/'prepared/training.npz',allow_pickle=False) as z:
        all_data={k:torch.from_numpy(z[k].copy()) for k in ('x','y','weight')}
        fit=np.flatnonzero(z['fit']); val=np.flatnonzero(z['internal_val'])
        primary=torch.from_numpy(z['primary'][val].copy())
    with np.load(U/'WINDOW_METADATA.npz',allow_pickle=False) as z:
        all_data['window_valid']=torch.from_numpy(z['train_valid'].copy())
    all_data.update(ids=meta['ids'],origins=meta['origins'])
    subset=lambda ix:{k:(v[ix].clone() if isinstance(v,torch.Tensor) else [v[int(i)] for i in ix]) for k,v in all_data.items()}
    train,validation=subset(fit),subset(val)
    validation['primary']=primary
    return train,validation,all_data


def context(plan,meta):
    names=['training_engine.py','generic_workflow.py','generic_pose_io.py','window_bridge.py',
           'valid_window_forward.py','CONFIG.json','FROZEN_EXPERIMENT.json','runtime_binding.py',
           'run_comparison.py','evaluate_frozen_candidate.py','prepare_metadata.py',
           'check_integration.py','check_training_trace.py','check_comparison.py','check_evaluation_inputs.py',
           'WINDOW_METADATA.npz','METADATA_AUDIT.json','INTEGRATION_CHECK.json','CODE_REVIEW.json',
           'EVALUATION_AUDIT_PINS.json','CLAUDE_TEST_RESULT.json']
    paths=[U/n for n in names]+[OLD/'TRAIN_SPLIT.json',OLD/'prepared/TRAINING_META.json',
          U.parent/'stgcn_reference_alignment_20261005/reference_adapter.py',
          U.parent/'stgcn_transfer_compare_20261005/stgcn_core.py']
    return {'input_policy':plan['input_policy'],'input':plan['input'],'source_pins':plan['pins'],
            'classes':meta['classes'],'inference_shape':[6,2,10,27],
            'metadata':'explicit window_valid bool[N,6] from original frame indices',
            'library_pins':{str(p):sha(p) for p in paths},
            'prepared_cache_sha256':meta['cache_sha256'],
            'window_metadata_sha256':sha(U/'WINDOW_METADATA.npz'),
            'cpu_threads':2,'deterministic_algorithms':True,
            'comparison':'padding_policy_BN_GRU_with_same_selection_policy_not_architecture_only'}
