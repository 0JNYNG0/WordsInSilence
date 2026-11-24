"""Independent audit of the existing run records; never trains or changes them."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

U=Path(__file__).resolve().parent
R=U.parent.parent/'runs/stgcn_full_path_v1'
schedule=json.loads((U/'SCHEDULE_AUDIT.json').read_text())
plan=json.loads((U/'FROZEN_EXPERIMENT.json').read_text())
context=json.loads((U/'EXECUTION_CONTEXT.json').read_text(encoding='utf-8'))
for path,expected in context['library_pins'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
checks=[]
records={}
for stage,file in [('select','SELECT_RESULT.json'),('refit','REFIT_RESULT.json')]:
    if (R/file).is_file():
        record=json.loads((R/file).read_text())
        records[stage]=record
        history=record['history']
    else:
        progress=json.loads((R/'progress.json').read_text())
        if progress['phase']!=stage:
            continue
        history=progress['history']
    expected=schedule['stages'][stage]
    for j,h in enumerate(history):
        e=expected['epochs'][j]
        assert h['epoch']==j+1==e['epoch']
        assert h['sample_sha256']==e['sample_sha256']
        counts=np.asarray(h['sample_counts'],dtype='<i8')
        assert hashlib.sha256(counts.tobytes()).hexdigest()==e['row_counts_sha256']
        assert int(counts.sum())==expected['draws_per_epoch']==h['train_count']
        assert h['updates']==expected['draws_per_epoch']//plan['training_config']['batch_size']
        assert h['grad_missing']==0 and h['grad_finite_checks']==108*h['updates']
        assert math.isfinite(h['train_ce'])
        want=.5*plan['training_config']['base_lr']*(1+math.cos(math.pi*j/plan['training_config']['schedule_epochs']))
        assert h['lr']==want
        if stage=='select':
            assert h['val']['primary_count']==233 and h['val']['all_count']==257
        else:
            assert h['val'] is None
    checks.append({'stage':stage,'completed_epochs':len(history),'sampling_gradients_schedule':'PASS'})
if 'select' in records:
    selected=records['select']
    best=min(selected['history'],key=lambda h:(-h['val']['primary_correct'],h['val']['primary_ce'],h['epoch']))
    assert best['epoch']==selected['best_epoch']
    assert 20<=len(selected['history'])<=40
    if selected['stop_reason']=='patience':
        assert selected['history'][-1]['stale']>=10
    else:
        assert len(selected['history'])==40
    checks.append({'selection_independent_best_epoch':best['epoch'],'status':'PASS'})
if 'refit' in records:
    assert records['refit']['selected_epoch']==records['select']['best_epoch']==len(records['refit']['history'])
    i1=json.loads((R/'INIT_SELECT.json').read_text())
    i2=json.loads((R/'INIT_REFIT.json').read_text())
    assert i1==i2
    frozen=json.loads((R/'FROZEN_MODEL.json').read_text())
    assert frozen['selected_epoch']==records['refit']['selected_epoch']
    assert hashlib.sha256((R/'final_model.npz').read_bytes()).hexdigest()==frozen['final_model_sha256']==records['refit']['final_model_sha256']
    checks.append({'fresh_initial_digest_equal':True,'refit_epochs_equal_selected':True,'final_hash_exact':True})
out=dict(status='PASS',finalized=('refit' in records),checks=checks,read_only=True)
(U/'TRAINING_TRACE_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
