"""Read-only diagnostic of two declared windows; annotations are oracle inputs."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys
import traceback
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path[:0] = [str(LAB), str(LAB/'src'), str(LAB.parent/'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as union
from pilot.data import collate
from wis_lab.core.observation import Observation
from wis_lab.service import neural

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = read(HERE/'plan.json')
assert not (HERE/'result.json').exists() and not (HERE/'public_scores.npz').exists()
for path, digest in plan['source_pins'].items():
    assert sha(path) == digest, path
assert sha(plan['checkpoint']['path']) == plan['checkpoint']['sha256']
payload = torch.load(plan['checkpoint']['path'], map_location='cpu', weights_only=True)
model, classes = union.rebuild_from_payload(payload)
assert len(classes) == 93
model.eval().requires_grad_(False)
torch.set_num_threads(2)
weight = payload['state_dict']['head.weight'].numpy().astype(np.float64)
bias = payload['state_dict']['head.bias'].numpy().astype(np.float64)
scopes = {'project18': [classes.index(c) for c in neural.PROJECT_18.values()], 'full93': list(range(93))}
previous = read(plan['reused_result'])
assert previous['checkpoint']['sha256'] == plan['checkpoint']['sha256']
previous_by_condition = {r['condition']: r for r in previous['rows']}
captured = []
hook = model.head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().numpy().copy()[0]))
rows, arrays = [], {'classes': np.array(classes)}
max_affine_error = 0.
try:
    with torch.inference_mode():
        for item in plan['inputs']:
            assert sha(item['cache']) == item['cache_sha256']
            obs = None if item['reuse_existing_result'] else Observation.load(item['cache'])
            for condition in ('whole_clip', 'provided_annotation'):
                row = {'id':item['id'], 'condition':condition, 'reference_label':item['reference_label'], 'status':'ERROR'}
                try:
                    if item['reuse_existing_result']:
                        prior_condition = 'web_entire118' if condition == 'whole_clip' else 'web_annotation_start_aligned36'
                        row.update({'status':'REUSED', 'scopes':previous_by_condition[prior_condition]['scopes'],
                                    'prior_condition':prior_condition, 'prior_result':plan['reused_result']})
                    else:
                        lo, hi = (0, obs.n_frames-1) if condition == 'whole_clip' else item['annotation_rows']
                        win = slice(lo,hi+1)
                        fields = {key:getattr(obs,key)[win].copy() for key in
                                  ('xy_px','timestamp_us','frame_id','observed','usable','conf','hand_score')}
                        cut = Observation(**fields,sidecar=dict(obs.sidecar))
                        clip, why, support = neural._channels(cut)
                        assert clip is not None, why
                        x = clip['x']
                        assert x.shape[1] == 280 and np.isfinite(x).all()
                        bx, pad, _ = collate([{'x':x, 'frames':len(x), 'label':0}])
                        actual = model(torch.from_numpy(bx),torch.from_numpy(pad))[0].numpy()
                        z = captured[-1].astype(np.float64)
                        scores = weight @ z + bias
                        error = float(np.max(np.abs(scores-actual)))
                        max_affine_error = max(max_affine_error,error)
                        assert error < 1e-4
                        prefix = item['id']+'__'+condition+'__'
                        arrays[prefix+'features'] = z
                        arrays[prefix+'scores'] = scores
                        yi = classes.index(item['reference_label'])
                        evaluations = {}
                        for scope, indices in scopes.items():
                            rank = sorted(indices,key=lambda i:-float(scores[i]))
                            other = next(i for i in rank if i != yi)
                            evaluations[scope] = {'prediction':classes[rank[0]], 'source_label_agreement':rank[0]==yi,
                                                  'source_label_rank':rank.index(yi)+1,
                                                  'margin':float(scores[yi]-scores[other])}
                        row.update({'status':'EVALUATED', 'scopes':evaluations, 'source_rows':[lo,hi],
                                    'timestamps_us':[int(cut.timestamp_us[0]),int(cut.timestamp_us[-1])],
                                    'input_shape':list(x.shape), 'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                                    'support':support})
                except Exception as exc:
                    row['error'] = type(exc).__name__+': '+str(exc)
                    row['traceback'] = traceback.format_exc()
                rows.append(row)
                print(json.dumps({k:row[k] for k in ('id','condition','status','scopes') if k in row},ensure_ascii=False),flush=True)
finally:
    hook.remove()
for path, digest in plan['source_pins'].items():
    assert sha(path) == digest, path
assert sha(plan['checkpoint']['path']) == plan['checkpoint']['sha256']
with (HERE/'public_scores.npz').open('xb') as f:
    np.savez_compressed(f,**arrays)
result = {'status':'COMPLETED_PENDING_RECOUNT', 'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
          'plan_sha256':sha(HERE/'plan.json'), 'script_sha256':sha(__file__), 'arrays_sha256':sha(HERE/'public_scores.npz'),
          'checkpoint':plan['checkpoint'], 'scopes':scopes, 'rows':rows,
          'new_encoder_forwards':len(captured), 'reused_outputs':sum(r['status']=='REUSED' for r in rows),
          'errors':sum(r['status']=='ERROR' for r in rows), 'max_forward_affine_error':max_affine_error,
          'training':0, 'new_extraction':0, 'user_inputs_read':0, 'reserved_inputs_read':0,
          'runtime_changed':False, 'candidate_reselected':False,
          'limits':[plan['reference_use'],plan['population'],
                    'Window length/content and normalization all change; not a detector-only or pure duration intervention.',
                    'Annotation-video frame-origin and exact action boundary remain unverified.']}
with (HERE/'result.json').open('x',encoding='utf-8',newline='\n') as f:
    json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    f.write('\n')
assert len(captured) == plan['planned_new_encoder_forwards'] and result['errors']==0
