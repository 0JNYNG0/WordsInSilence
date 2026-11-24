from model_core import *

def main():
    assert not (U/'FROZEN.json').exists()
    assert read(U/'PRETRAIN_CHECK.json')['status']=='PASS'
    assert read(U/'shape_recovery/CODE_REVIEW.json')['status']=='PASS'
    paths=[U/p for p in ['common.py','model_core.py','prepare_inputs.py','train.py','evaluate.py','pretrain_check.py','DATA_READY.json','ROOT_LABEL_REVIEW.json','shape_recovery/EXTRA_LABELS.json','shape_recovery/CODE_REVIEW.json']]
    paths += [PREV/'model_core.py', B/'core.py',J/'feedback_model.py']
    paths += [B/f'runs/shared100_{s}/model.pt' for s in [42,43,44]]
    paths += [PREV/f'scores/A_{s}.json' for s in [42,43,44]]
    paths += [B/f'scores/shared100_{s}.json' for s in [42,43,44]]
    paths += [B/'EXTRA_EVAL_LABELS.json',old.S/'FROZEN.json',old.V/'FEEDBACK_NORMALIZATION.json',J/'FROZEN.json']
    write(U/'FROZEN.json',dict(status='FROZEN_BEFORE_TRAINING',conditions=['frozen','finetune'],seeds=[42,43,44],epochs=12,warmup_epochs=2,steps_per_epoch=38,lr_base=3e-5,lr_head=.001,retention_weight=10.,extra_words=read(U/'DATA_READY.json')['extra_words'],
      pins={str(p):sha(p) for p in paths},protected=read(PREV/'FROZEN.json')['protected'],
      initialization='paired existing shared100 seed; same random finger head; never deployed TCN',
      architecture='identical to prior FineNet; frozen common network vs low-LR shared feature update; no extra word name loss',
      cause='Sparse B ordinary truth and hand imbalance, fine-tune common regressions. New FIT-only RGB truth plus equal-state/clip/hand objective; contrasts do not isolate each individual change versus historical A.',
      sampling='24 common clips without replacement per epoch covering all899 + 6 ordinary fine +3 B-containing +4 coarse, deduplicated within batch;4 extra fine. Matched conditions and seeds.',
      labels='E0 B1 F2 U-1 at exact reviewed native times; no temporal propagation. Thumb BF unsupported; left B long fingers except index single-expression limitation.',
      eval=dict(status='historical development only, no untouched-generalization claim',reference='stored prior A paired seed per-finger and shared100 coarse/stage/word',
        gate='Each condition separately: normal macroF1 +0.01 over storedA in all3seeds; every stateF1 and hand accuracy preserved; no higher normal abstention; all shared100 coarse/stage/order/word preserved. Speed/missing variants no known-state regression. No automatic web promotion.',
        confidence=.85,full_path='requires independent G2/G3 timing/direction/order/omission and meaningful feedback validation; this finger experiment cannot certify completion'),
      max_runs=6,private_used=False,production_changed=False,early_stop='no new candidates from development scores; fail means audit evidence not threshold fitting'))
    print('FROZEN',sha(U/'FROZEN.json'))
if __name__=='__main__':main()
