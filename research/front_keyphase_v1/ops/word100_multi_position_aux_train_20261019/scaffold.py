"""Adapt audited prior trainer/verifier only in this new operation directory."""
from pathlib import Path
D=Path(__file__).resolve().parent;OLD=D.parent/'word100_auxiliary_supervision_train_20261019'
assert not (D/'train.py').exists()
for name in ['common.py','objective.py','verify_aux.py']:
 s=(OLD/name).read_text(encoding='utf8')
 if name=='common.py':s=s.replace("ARM='FG16AUX01'","ARM='FG16M3'")+"\nA=O/'word100_auxiliary_supervision_train_20261019';D=O/'word100_auxiliary_order_diagnosis_20261019'\n"
 (D/name).write_text(s,encoding='utf8')
s=(OLD/'train.py').read_text(encoding='utf8')
s=s.replace('from frozen_order_aux import FrozenOrderAux','from multi_position_aux import MultiPositionAux')
s=s.replace('m=FrozenOrderAux(', 'm=MultiPositionAux(')
s=s.replace(" assert preflight_hash(m)==read(P/'RESULT.json')['initial_state_hashes'][str(seed)]\n",'')
s=s.replace('4000','7200').replace('51236','54436').replace('original_C_280_frozen_T3F_GRU16','original_C_280_frozen_T3F_GRU16_elapsed3')
(D/'train.py').write_text(s,encoding='utf8')
s=(OLD/'verify.py').read_text(encoding='utf8')
s=s.replace('4000','7200')
s=s.replace("models=['C','T3F','FG16',ARM]","models=['C','T3F','FG16','FG16AUX01',ARM]")
s=s.replace("else F if model=='FG16' else U", "else F if model=='FG16' else A if model=='FG16AUX01' else U")
s=s.replace("for baseline in ['C','T3F','FG16']:","for baseline in ['C','T3F','FG16','FG16AUX01']:")
s=s.replace("'saved_predictions_verified':7200","'saved_predictions_verified':9000")
s=s.replace("'secondary_gate_vs_C':", "'primary_review_gate_vs_AUX01':gate(metrics[ARM],metrics['FG16AUX01']),'secondary_gate_vs_C':")
s=s.replace('Versus FG16 only auxiliary objective changes; semantic order-only causality not established.', 'Versus AUX01 only multi-position readout including3200extra head parameters changes; semantic stage/order-only causality not established.')
(D/'verify.py').write_text(s,encoding='utf8')
print('Created isolated trainer and saved-score verifier; prior files unchanged.')
