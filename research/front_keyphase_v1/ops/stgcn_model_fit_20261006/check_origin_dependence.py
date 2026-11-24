"""Independent verification of frozen preprocessing/model; no candidate or training.

Calls pinned, previously verified Claude/reference APIs. All actual inputs are
existing public training references. The intervention changes coordinate origin,
not hand identity, observation availability, time, or true sign form.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch

U = Path(__file__).resolve().parent
OPS = U.parent
OLD = OPS / 'stgcn_execution_ready_20261005'
RUN = OPS.parent / 'runs/stgcn_full_path_v1'
OUT = U / 'ORIGIN_PROBE.json'
assert not OUT.exists(), 'Preserve completed probe; no repeat'
sys.path[:0] = [str(OLD), str(OPS/'stgcn_reference_alignment_20261005'),
               str(OPS/'stgcn_transfer_compare_20261005')]
import generic_pose_io as gp
import generic_workflow as gw
import reference_adapter as ra
from stgcn_core import read_encoder_tensors

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((U/'PROBE_PLAN.json').read_text(encoding='utf8'))
context = json.loads((OLD/'EXECUTION_CONTEXT.json').read_text(encoding='utf8'))
previous = json.loads((OLD/'FROZEN_EXPERIMENT.json').read_text(encoding='utf8'))
manifest_path = OPS/'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json'
manifest = json.loads(manifest_path.read_text(encoding='utf8'))
assert sha(manifest_path) == context['source_pins']['input_manifest']
protected = dict(context['library_pins'])
protected[str(RUN/'final_model.npz')] = plan['model_sha256']
protected[str(RUN/'FROZEN_MODEL.json')] = sha(RUN/'FROZEN_MODEL.json')
protected[str(manifest_path)] = context['source_pins']['input_manifest']
protected[str(U/'PROBE_PLAN.json')] = sha(U/'PROBE_PLAN.json')
for p, h in protected.items():
    assert sha(p) == h, p
frozen = gw.validate_frozen(str(RUN), context)
assert frozen['final_model_sha256'] == plan['model_sha256']

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
dpc, tf, _ = ra.load_reference(OPS/'stgcn_reference_alignment_20261005/reference',
                             context['source_pins']['source_pins'])
normalize = tf.CenterAndScaleNormalize(reference_points_preset=ra.NORMALIZE_PRESET,
                                      frame_level=False)
joints = list(ra.LAYOUTS['mp75'][1])
def prepare(xy, mask):
    value, diag = gp.prepare_windows(xy, mask, joints, [3,4], 60, 10,
                                    lambda t: normalize({'frames': t})['frames'])
    return value[0], diag

rows = sorted([r for r in manifest['rows'] if r['role']=='train'
               and r['layout']=='mp75' and r['group']=='base'
               and r['condition']=='intact'], key=lambda r:r['id'])
assert len(rows)==18 and all(r['mask_hand'] is None for r in rows)
spec = json.loads((OLD/'prepared/TRAINING_META.json').read_text(encoding='utf8'))['layout_specs']['mp75']
base, exact, raw_shift, records = [], [], [], []
for row in rows:
    p=Path(row['path'])
    assert '/runs/ses_' not in p.as_posix() and sha(p)==row['sha256']
    protected[str(p)] = row['sha256']
    with np.load(p, allow_pickle=False) as z:
        sl=slice(row['start'],row['end_exclusive'])
        xy=z[spec['xy_key']][sl].astype(np.float32,copy=True)
        mask=z[spec['mask_key']][sl].copy()
        times=z[spec['time_key']][sl].copy()
    assert mask.dtype==np.bool_ and mask[:,[11,12]].all()
    assert np.isfinite(xy[mask]).all() and np.all(np.diff(times)>0)
    a, diag = prepare(xy,mask)
    # Match float32 scalar arithmetic of the official torch normalizer.
    shoulders=torch.from_numpy(xy[:,[11,12],:])
    width=float(torch.norm(shoulders[:,0]-shoulders[:,1],dim=1).mean())
    shifted=xy.copy()
    shifted[mask] += np.array([width,0],dtype=np.float32)
    b, bd = prepare(shifted,mask)
    assert diag['indices']==bd['indices']
    selected=mask[:,joints]
    sampled=np.zeros((60,27),dtype=bool)
    padding=np.zeros((60,27),dtype=bool)
    for t,idx in enumerate(diag['indices']):
        if idx < 0: padding[t,:]=True
        else: sampled[t]=selected[idx]
    missing=(~sampled & ~padding).reshape(6,10,27)
    visible=sampled.reshape(6,10,27)
    pad=padding.reshape(6,10,27)
    assert missing.any()
    c=a.clone()
    c[:,0,:,:][missing] -= 1.0
    assert torch.equal(c[:,0,:,:][visible],a[:,0,:,:][visible])
    assert torch.equal(c[:,1,:,:],a[:,1,:,:])
    assert torch.equal(c[:,0,:,:][pad],a[:,0,:,:][pad])
    difference=float(torch.max(torch.abs(b-c)))
    observed_difference=float(torch.max(torch.abs((b-a).permute(0,2,3,1)[visible])))
    assert difference <= 1e-5, (row['id'],'raw_shift_vs_exact',difference)
    base.append(a);raw_shift.append(b);exact.append(c)
    records.append(dict(id=row['id'],source_sha256=row['sha256'],T=len(xy),
                        raw_missing_selected=int((~selected).sum()),
                        sampled_missing_selected=int(missing.sum()),
                        shoulder_width=width,raw_shift_vs_exact_max=difference,
                        observed_numerical_max=observed_difference,
                        exact_visible_and_padding_bit_identical=True,
                        original_input_sha256=gp.tensor_sha256(a),
                        isolated_input_sha256=gp.tensor_sha256(c)))

# Fully observed synthetic control: geometry only, not a filled real hand.
syn=np.empty((71,75,2),dtype=np.float32)
for t in range(71):
    syn[t,:,0]=500+np.arange(75)*3+10*np.sin(t/8)
    syn[t,:,1]=400+np.arange(75)*2+7*np.cos(t/7)
syn[:,11,:]=[700,400];syn[:,12,:]=[400,400]
sm=np.ones((71,75),dtype=bool)
sa,_=prepare(syn,sm)
sb,_=prepare(syn+np.array([300,0],dtype=np.float32),sm)
synthetic_max=float(torch.max(torch.abs(sa-sb)))
assert synthetic_max<=1e-5, synthetic_max

artifact=OPS/'openhands_artifact_20261005'
assert sha(artifact/'TENSOR_METADATA.json')==context['source_pins']['pretrained_metadata']
tensors,_=read_encoder_tensors(artifact/'raw_dpc.zip',artifact/'TENSOR_METADATA.json',
                              context['source_pins']['pretrained_zip'],encoder_prefix='model.')
encoder={k[len('conv_encoder.'):]:v for k,v in tensors.items() if k.startswith('conv_encoder.')}
gru={k[len('agg.'):]:v for k,v in tensors.items() if k.startswith('agg.')}
model,_=ra.make_reference_model(dpc,encoder,gru,num_classes=93,
                                seed=previous['training_config']['seed'])
model=gw.load_final_state(model,str(RUN),context)
before=gw.state_digest(model.state_dict())
started=time.perf_counter()
# batch_size=1 keeps numerical conditions identical for all controls.
inputs=torch.stack(base+exact+raw_shift+base+[sa,sb])
scores=gw.score_batches(model,inputs,1)
n=18
a,b,c,repeat=scores[:n],scores[n:2*n],scores[2*n:3*n],scores[3*n:4*n]
repeat_max=float(np.max(np.abs(a-repeat)))
synthetic_logit_max=float(np.max(np.abs(scores[-2]-scores[-1])))
assert repeat_max<=1e-5, repeat_max
assert synthetic_logit_max<=1e-4, synthetic_logit_max
assert gw.state_digest(model.state_dict())==before
for p,h in protected.items(): assert sha(p)==h,p
for i,r in enumerate(records):
    r.update(max_logit_delta=float(np.max(np.abs(a[i]-b[i]))),
             centered_logit_delta=float(np.max(np.abs((a[i]-a[i].mean())-(b[i]-b[i].mean())))),
             raw_vs_exact_logit_max=float(np.max(np.abs(c[i]-b[i]))),
             original_top1=int(np.argmax(a[i])),isolated_top1=int(np.argmax(b[i])),
             top1_changed=bool(np.argmax(a[i])!=np.argmax(b[i])))
result=dict(status='CONTROLS_PASS',plan_sha256=protected[str(U/'PROBE_PLAN.json')],
            model_sha256=plan['model_sha256'],rows=records,
            probe_inferences=len(inputs),new_training=0,final855_rescoring=0,
            new_candidate=0,user_data_used=False,source_state_hashes_unchanged=True,
            synthetic_input_max=synthetic_max,synthetic_logit_max=synthetic_logit_max,
            duplicate_logit_max=repeat_max,
            materially_sensitive=sum(r['max_logit_delta']>1e-4 for r in records),
            relative_class_sensitive=sum(r['centered_logit_delta']>1e-4 for r in records),
            top1_flips=sum(r['top1_changed'] for r in records),
            seconds=time.perf_counter()-started,
            interpretation='Dependence diagnostic only; no accuracy/generalization/improvement claim')
np.savez(U/'ORIGIN_PROBE_SCORES.npz',original=a,isolated=b,raw_shift=c,repeat=repeat,
         synthetic=scores[-2:])
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
