from probe_common import *
cfg=read(U/'EVALUATION_CONTRACT.json');check(cfg['pins']);check(cfg['protected_pins'])
assert len(cfg['protected_pins'])==477
write(U/'ROOT_REVIEW.json',dict(status='APPROVED_FOR_SINGLE_EVALUATION',at=stamp(),
    reviewer='root Codex source-code and contract review; separate saved arithmetic review follows',
    contract_sha256=sha(U/'EVALUATION_CONTRACT.json'),pins={str(U/n):sha(U/n) for n in ['evaluate_probe.py','prepare_probe.py','probe_common.py','RGB_REVIEW.json','INPUTS_COMPLETE.json','LABEL_AUDIT.json']},
    decision='Run all fixed three checkpoints once on all48 probe sequences, include failures. No choice of seed/threshold/model or retraining using these results.',
    label_scope='Body family movement vs hold only; same target/mask contract as BP, no completion.',
    no_new_user_permission_needed='Existing user authorization covers public learning and comparison; this is an internal code-review record.'))
