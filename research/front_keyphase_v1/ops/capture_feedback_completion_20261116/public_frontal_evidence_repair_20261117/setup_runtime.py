from pathlib import Path
import json
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;N=G/'frontal_scope_recovery_20261117'
for name in ['apple_runtime','web','site']:(D/name).mkdir(exist_ok=True)
src=(N/'apple_runtime/worker.py').read_text(encoding='utf-8')
src=src.replace("from apple_candidate import RoleEvidenceCandidate", "from evidence_candidate import EvidenceLesson")
src=src.replace('net=RoleEvidenceCandidate()', 'net=EvidenceLesson()')
assert 'RoleEvidenceCandidate' not in src
(D/'apple_runtime/worker.py').write_text(src,encoding='utf-8')
js=(O/'web_tutorial_live_integration_20261113/site/live.js').read_text(encoding='utf-8')
marker="$('capture-start').disabled=requestBusy||s.state!=='preview'||!target();"
assert js.count(marker)==1
js=js.replace(marker,"$('capture-start').disabled=requestBusy||s.state!=='preview'||!target()||s.auto_end_ready===false;")
marker="preview:'손을 내린 편한 자세로 준비해요'";assert js.count(marker)==1
js=js.replace(marker,"preview:s.preparation_message||'손을 내린 편한 자세로 준비해요'")
(D/'site/live.js').write_text(js,encoding='utf-8')
print('isolated worker / readiness UI prepared')
