from graph import *
W=U/'occurrence_supervision'
d=read(W/'DIRECT_ROLE_AUDIT.json')
assert not (W/'ROOT_ADMISSION.json').exists()
assert read(W/'STRONG_CONFLICT_REVIEW_VERIFY.json')
write(W/'ROOT_ADMISSION.json',dict(
 at=stamp(),approved_for_isolated_fit=True,
 audit_sha256=hashlib.sha256((W/'DIRECT_ROLE_AUDIT.json').read_bytes()).hexdigest(),
 accepted_conflicts=[f"{r['clip_id']}:{r['sequence_index']}" for r in d['rows'] if r['category']=='conflict'],
 root_visual_review=['STRONG_CONFLICTS.png: all7 frames','obedience01 dense1','obedience04 dense0','obedience07 dense1','egypt FIT01/04/07 hand sequences'],
 review_limit='Separate Codex visual/source review, not independent human/KSL instructor truth. Other admitted samples retain delegated RGB/image/hash provenance.',
 decisions=['Entry/return refine old nonmain grouping; not seven independent wrong-action annotations.',
            'Egypt01 frame37 restores a visible central pose rather than treating an uncertain spread-onset lower bound as certain spread.',
            'No absolute end timestamp, whole-sign correctness, hidden fingers, or thumb-gather class inferred.',
            'Only eligible sampled frames. Uncertain rows do not create new labels. Original frozen labels remain unchanged.'],
 scope='Isolated same-schedule controlled continuation of common action-role head; no web promotion.'))
print('Admitted reviewed label edition for isolated controlled training.')
