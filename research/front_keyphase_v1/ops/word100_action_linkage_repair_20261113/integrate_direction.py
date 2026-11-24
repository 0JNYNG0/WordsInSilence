from pathlib import Path
import json,hashlib,ast,shutil
U=Path(__file__).resolve().parent;W=U.parent/'web_tutorial_live_integration_20261113'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
comparison=read(U/'LOCAL_DIRECTION_COMPARISON_V2.json')
assert comparison['passed'] and comparison['candidate_sha256']==sha(U/'localized_direction.py')
assert read(U/'PUBLIC_RUNTIME_VERIFIED.json')['passed']
for p,h in read(W/'PROTECTED.json').items():assert sha(Path(p))==h,p
server=W/'server.py';before=server.read_text('utf-8')
old='from coaching import attach\n'
new='sys.path.insert(0,str(O/"word100_action_linkage_repair_20261113"))\nfrom runtime_adapter import CaptureService, attach\n'
assert before.count(old)==1 and 'from runtime_adapter' not in before
after=before.replace(old,new);ast.parse(after)
backup=U/'before_web';backup.mkdir(exist_ok=True);shutil.copy2(server,backup/'server.py')
server.write_text(after,encoding='utf-8')
out=dict(applied=True,files={str(server):dict(before=sha(backup/'server.py'),after=sha(server))},
    changed='Research-web-only refinement dependency and localized cue formatter',
    model_weights_changed=False,old_server_code_changed=False,camera_started=False,requires_idle_restart=True)
with (U/'WEB_PATCH.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('Verified observer connected to research web; restart still required.')
