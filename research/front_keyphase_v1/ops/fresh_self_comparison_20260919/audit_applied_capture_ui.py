"""Independent boundary and isolated-browser event checks. No model inference."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json

ops = Path(__file__).resolve().parent
web = ops.parents[1] / 'web'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(ops/'protocol_LOCAL_ONLY.json')
flow = read(ops/'ui_flow_verification_v4.json')
events = read(ops/'browser_mock_events.json')
health = read(ops/'post_ui_health.json')
checks = []
def check(name, passed):
    checks.append({'name':name,'passed':bool(passed)})

for path, expected in plan['protected_sha256'].items():
    check('unchanged: '+str(Path(path).relative_to(ops.parents[3])), sha(Path(path)) == expected)
check('applied JS exactly the version with33 successful flow checks', sha(web/'frozen-capture.js') == flow['javascript_sha256'] and flow['passed'] == flow['total'] == 33)
posts = [x for x in events['events'] if x['method'] == 'POST']
opens = [x for x in posts if x['route'] == '/api/v1/sessions']
trials = [x for x in posts if x['route'].endswith('/trials')]
reviews = [x for x in posts if x['route'].endswith('/review')]
closes = [x for x in posts if x['route'].endswith('/close')]
check('browser loaded page before any session; explicit open created one synthetic session', len(opens) == 1 and opens[0]['body']['store_video'] is False)
check('two explicit trials, no auto capture on next or reload', len(trials) == len(events['trials']) == 2)
check('review retry preserved user selection', len(reviews) == 2 and reviews[0]['body'] == reviews[1]['body'])
check('cancelled attempt kept', events['trials']['synthetic_trial_02']['state'] == 'CANCELLED')
check('synthetic session closed once', len(closes) == 1 and events['session']['state'] == 'CLOSED' and events['session']['camera'] is False)
check('no hardware camera or human files used in mock', events['hardware_camera_calls'] == events['human_observation_files_written'] == 0)
check('real server camera remains closed without owning session', health['camera'] is None and health['camera_session_id'] is None)
report = {'at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(), 'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL', 'checks':checks, 'passed':sum(x['passed'] for x in checks), 'total':len(checks), 'applied_sha256':{name:sha(web/name) for name in ['frozen-capture.html','frozen-capture.js']}, 'new_human_trials':0, 'new_training':0, 'new_inference':0, 'scope':'UI flow and protected-file boundary only. No actual camera capture or new recognition accuracy validation.'}
(ops/'applied_ui_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
assert report['status'] == 'PASS'
