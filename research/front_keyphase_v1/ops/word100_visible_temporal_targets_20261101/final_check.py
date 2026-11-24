from pathlib import Path
import json,hashlib,urllib.request
U=Path(__file__).resolve().parent;root=U.parents[3]
def read(p):return json.loads(Path(p).read_text('utf8'))
s=read(U.parent/'STATE.json');assert s['active'] is None and s[U.name]['status']=='COMPLETE_LIMITED_FEATURE_STUDY_NOT_ADOPTED'
for p,h in read(U/'TRAIN_FROZEN.json')['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
for name in read(U/'COMPLETION.json')['review_files']:assert (U/name).is_file()
for n in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md']:assert '6회' in (root/n).read_text('utf8') or '6완료' in (root/n).read_text('utf8')
health=json.load(urllib.request.urlopen('http://127.0.0.1:8798/health',timeout=3));assert health['read_only'] and not health['camera'] and not health['inference']
assert hashlib.sha256((U/'index.html').read_bytes()).hexdigest()==read(U/'UI_VERIFICATION.json')['html_sha256']
report=dict(status='PASS_FINAL_DOCUMENT_AND_LINKS',report_health=health,training_frozen_pins_unchanged=True,state_and_handoff_synced=True,html_matches_ui_verified=True,no_active_training=not (U/'TRAIN_RUNNING.json').exists(),environment_retry='First restricted-shell localhost health request denied WinError10013. Same read-only check repeated in normal authorized server context; server not restarted.')
with (U/'FINAL_CHECK.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
