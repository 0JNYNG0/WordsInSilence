"""Synthetic loss/derivative review only: no project model, data, optimizer or training."""
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

U = Path(__file__).resolve().parent
assert not (U / 'MATH_REVIEW.json').exists()
torch.set_num_threads(1)
dtype = torch.float64
lam = 0.1
target = torch.tensor([0])
# Hypothetical confident base. These are invented logits, not saved user/model scores.
b = torch.tensor([[math.log(0.99997)] + [math.log(0.00003 / 99)] * 99], dtype=dtype, requires_grad=True)
a = torch.zeros((1, 100), dtype=dtype, requires_grad=True)
final_loss = F.cross_entropy(b.detach() + a, target)
aux_loss = F.cross_entropy(a, target)
loss = final_loss + lam * aux_loss
g = torch.autograd.grad(loss, a)[0].numpy()[0]
def softmax(z):
    e = np.exp(z - np.max(z)); return e / e.sum()
def ce(z):
    return float(np.log(np.exp(z-z.max()).sum()) + z.max() - z[0])
bn = b.detach().numpy()[0]
one = np.eye(100)[0]
analytic = softmax(bn) - one + lam * (np.ones(100) / 100 - one)
assert np.max(np.abs(g - analytic)) < 1e-13
assert abs(float(aux_loss.detach()) - math.log(100)) < 1e-13
assert b.grad is None
finite = []
for k in [0, 1, 50, 99]:
    e = np.zeros(100); e[k] = 1e-5
    fd = ((ce(bn+e)+lam*ce(e))-(ce(bn-e)+lam*ce(-e))) / 2e-5
    finite.append(abs(fd-g[k]))
assert max(finite) < 1e-8
# Zero output head blocks the feature gradient initially even with auxiliary CE.
h = torch.linspace(-0.7, 0.8, 16, dtype=dtype)[None, :].requires_grad_()
w = torch.zeros((100, 16), dtype=dtype, requires_grad=True)
z = F.linear(h, w)
toy = F.cross_entropy(b.detach()+z, target)+lam*F.cross_entropy(z, target)
gw, gh = torch.autograd.grad(toy, (w, h))
assert gw.norm() > 0 and torch.count_nonzero(gh) == 0
# Separately prescribed nonzero weights: NOT an optimizer update.
w2 = (torch.arange(1600, dtype=dtype).reshape(100,16).sin() * 0.001).requires_grad_()
z2 = F.linear(h, w2)
toy2 = F.cross_entropy(b.detach()+z2,target)+lam*F.cross_entropy(z2,target)
gh2 = torch.autograd.grad(toy2,h)[0]
assert gh2.norm() > 0
out = {
    'status':'PASS_SYNTHETIC_MATH_ONLY_NOT_MODEL_READINESS', 'lambda_aux':lam,
    'aux_ce_at_zero_logits':float(aux_loss.detach()), 'weighted_aux_ce_at_zero_logits':float(aux_loss.detach())*lam,
    'hypothetical_base_correct_probability':0.99997,
    'hypothetical_final_ce':float(final_loss.detach()),
    'hypothetical_target_logit_gradient_final_only':-0.00003,
    'hypothetical_target_logit_gradient_aux_weighted':-0.099,
    'numpy_gradient_max_error':float(np.max(np.abs(g-analytic))),
    'finite_difference_max_error':max(finite),
    'zero_head_gradient_norm':float(gw.norm()), 'zero_head_feature_gradient_norm':float(gh.norm()),
    'fixed_nonzero_head_feature_gradient_norm':float(gh2.norm()),
    'limitation':'Logit-level synthetic illustration only. Not a model gradient magnitude, AdamW update ratio or accuracy forecast.',
    'project_models_loaded':0, 'raw_recordings_read':0, 'optimizer_steps':0, 'training_runs':0,
    'verification':'NumPy analytic derivative and finite differences versus PyTorch autograd, same Codex; no independent agent.'
}
(U/'MATH_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False))
