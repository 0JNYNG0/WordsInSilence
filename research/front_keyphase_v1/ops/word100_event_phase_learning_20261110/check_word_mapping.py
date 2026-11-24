"""Regression check of word-namespace correction; no public benchmark replay."""
from phase_core import *
import ast
torch.set_num_threads(4)
torch.manual_seed(721)
old_ids=list(read(O/'word100_action_feedback_demo_20261108/FROZEN.json')['labels'])
right_ids=shared.WORDS
net=prior.Network('coupled',read(P/'NORMALIZATION.json')).eval()
x=torch.randn(1,8,280);length=torch.tensor([8]);checks=[]
with torch.inference_mode():
 for word in right_ids:
  wrong=torch.tensor([old_ids.index(word)]);right=torch.tensor([right_ids.index(word)])
  a=net.shared(x,length,wrong);b=net.shared(x,length,right)
  checks.append(dict(word=word,old_id=int(wrong),correct_id=int(right),latent_identical=bool(torch.equal(a['latent'],b['latent'])),attributes_identical=bool(torch.equal(a['attributes'],b['attributes'])),stage_changed=not bool(torch.equal(a['stage'],b['stage']))))
assert all(c['latent_identical'] and c['attributes_identical'] and c['stage_changed'] for c in checks)
ast.parse((U/'partial_evidence.py').read_text('utf-8'))
write(U/'WORD_MAPPING_AUDIT.json',dict(checks=checks,passed=True,issue='Followup initially passed100wordclassindex instead of4guidewordindex. Only target-free latent was used; wrong stage posterior was discarded. Executed v0 source preserved; helper corrected.',effect='No saved localization or direction output changed; no benchmark rerun.',method='Code dataflow plus deterministic random-input equivalence, not human or benchmark validation.',executed_source_sha256=sha(U/'partial_evidence_executed_v0.py.txt'),corrected_source_sha256=sha(U/'partial_evidence.py')))
print('PASS: all4 target-free latent/attribute outputs identical; guide-stage namespace corrected.')
