from pathlib import Path
import json,ast,hashlib
U=Path(__file__).resolve().parent;O=U.parent;W=O/'web_tutorial_live_integration_20261113'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for name in ['ARM_SAVED_VERIFIED.json','LOCAL_DIRECTION_COMPARISON_V2.json','DIRECTION_SAVED_VERIFIED.json','PUBLIC_RUNTIME_VERIFIED.json','SERVED_VERIFIED.json','UI_CONTRACT_VERIFIED.json','SPEED_TRUTH_TRANSPORT.json']:
    assert read(U/name)['passed'];checks.append(name)
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'))
for p,h in read(W/'PROTECTED.json').items():assert sha(Path(p))==h,p
comparison=read(U/'LOCAL_DIRECTION_COMPARISON_V2.json');assert comparison['candidate_sha256']==sha(U/'localized_direction.py')
assert not read(U/'ARM_COMPARISON.json')['adoption_gates_passed']
assert 'runtime_adapter' in (W/'server.py').read_text('utf-8')
out=dict(passed=True,checks=checks,protected_model_files=len(read(W/'PROTECTED.json')),
    source_hashes={str(p):sha(p) for p in [U/'localized_direction.py',U/'runtime_adapter.py',W/'server.py']},
    direction_component_web_applied=True,arm_learning_candidate_adopted=False,
    whole_project_complete=False,G2_pass=False,final8_opened=False,camera_opened=False,
    first_audit_correction_preserved='BINDING_CAUSE_AUDIT_V2 separates missing direct truth from binding failure; V1 preserved.',
    first_direction_verifier_correction='V2 matches comparator by card ID, rather than a self-equality typo. Both contracts happen to share direction, so first counts unchanged. V1 preserved.',
    verification_actor='root Codex implementation plus separate stored arithmetic/contracts; not independent human/KSL validation',server=read(W/'SERVER.json'))
with (U/'FINAL_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(passed=True,server=out['server'],whole_project_complete=False),ensure_ascii=False))
