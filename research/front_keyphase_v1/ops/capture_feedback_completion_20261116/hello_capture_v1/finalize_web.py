from batch_observer import *
import hashlib,ast
p=O/'web_tutorial_live_integration_20261113/site/live.js'
s=p.read_text('utf-8-sig');needle="$('ranks').hidden=false;"
replacement="if(!(data.tcn_top3||[]).length){const note=document.createElement('span');note.textContent=data.recognition_scope||'단어 인식 결과 없음';$('ranks').append(note)}$('ranks').hidden=false;"
assert s.count(needle)==1 and 'note.textContent=data.recognition_scope'not in s
p.write_text(s.replace(needle,replacement),encoding='utf-8')
for p in [U/'worker.py',U/'hello_web_adapter.py',U/'hello_routes.py',O/'web_tutorial_live_integration_20261113/server.py']:ast.parse(p.read_text('utf-8-sig'))
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[U/'feedback_lesson.py',U/'sparse_observer.py',U/'batch_observer.py',U/'worker.py',U/'presentation.py',U/'hello_web_adapter.py',U/'hello_routes.py']}
write(U/'web/ADMISSION.json',dict(practice_only=True,quest_qualified=False,source_people=1,actual_learner_accuracy=None,training=False,pins=pins,known_normal_and_controls=read(U/'FEEDBACK_COMPARISON_v3.json')['passed'],control_scope='24 conditions from one public source; not 24 different people or full error-detection accuracy',normal_views=read(U/'VIEW_CHECK.json'),capture=read(U/'web/capture/CAPTURE_CHECK.json'),limitations=['Reversed sweep and closed sweeping hand can remain unknown rather than a specific correction.','TCN has no SEN0354 output; no placeholder recognition.','Exact 3D contact, thumb and independent signer accuracy not certified.']))
print('Practice-only scope and syntax checks saved')
