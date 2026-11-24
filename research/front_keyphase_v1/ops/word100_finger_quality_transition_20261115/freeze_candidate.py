from quality_common import *
assert not(U/'CANDIDATE_FROZEN.json').exists()
d=read(U/'diagnostic_RESULTS_V2.json')['summary'];h=read(U/'HISTORIC_RESULTS_V2.json')['summary']
assert d['quality_only']['wrong']<d['old']['wrong']
for s in [d,h]:assert s['quality_only']['old_correct_lost']/s['old']['correct']<=.05
pins={str(p):sha(p)for p in [U/'quality_model_v2.py',U/'quality_runtime_v2.py',U/'FIT_LABELS_V2.json',U/'DIAGNOSTIC_LABELS.json',U/'TRAIN_FROZEN_V2.json',U/'TRAINED_V2.json',U/'diagnostic_RESULTS_V2.json',U/'HISTORIC_RESULTS_V2.json']+list((U/'models_v2').glob('*.pt'))}
write(U/'CANDIDATE_FROZEN.json',dict(selected='quality_only',candidate='v2',selection='fewer wrong claims diagnostic6->4 with no diagnostic correct loss; historic910->870correct (40unknown,4.40% loss), one error remains. Phase combined fails historical5% loss gate (60/910), plus still confuses rotations; do not activate phase head.',scope='shadow until reserve and122regression pass; no confident-state imputation; quality_unknown retains previous decision',pins=pins,reserved_predictions_released=False))
print('quality_only candidate frozen before reserve15 review/predictions')
