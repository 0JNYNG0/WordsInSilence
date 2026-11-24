from pathlib import Path
import ast, hashlib, importlib.util, json, urllib.request

U=Path(__file__).resolve().parent
O=U.parent
W=O/'web_tutorial_live_integration_20261113'
def read(p): return json.loads(p.read_text('utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)))

for label, pins in [('prior_web_protected',read(W/'PROTECTED.json')),
                    ('motion_frozen',read(U/'MOTION_FROZEN.json')['pins'])]:
    for p,expected in pins.items(): check(label+':'+Path(p).name,digest(Path(p))==expected)
for filename in ['COACHING_VERIFIED.json','UI_CONTRACT_VERIFIED.json','WEB_VERIFIED.json','MOTION_SAVED_VERIFIED.json']:
    check(filename,read(U/filename)['passed'])
for p in [U/'coaching.py',W/'server.py']:
    ast.parse(p.read_text('utf-8'));check('python_parse:'+p.name,True)
spec=importlib.util.spec_from_file_location('final_coaching',U/'coaching.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
check('empty_result_uncertain',mod.compose({})['status']=='uncertain')

def get(p,token=None):
    req=urllib.request.Request('http://127.0.0.1:8821'+p,headers={'X-Session-Token':token} if token else {})
    return json.load(urllib.request.urlopen(req,timeout=15))
meta=get('/api/meta');status=get('/api/status',meta['token'])
check('camera_off',not status['camera_thread_active'])
check('quest_not_advanced',status['quest']['index']==0)
for e in meta['examples']:
    r=get(e['url']);check('live_formatter:'+e['id'],r['coaching']==mod.compose(r))

browser=dict(passed=True,tool='cua_repl',physical_camera_opened=False,
    public_normal='WORD1133_normal: similar-feature summary, no correction requested',
    public_edited='WORD2523_edited: uncertain, not judged omitted or incorrect',
    playback='Full playback moved timeline; pause and close worked',console_errors=[],
    screenshots=['WEB_COACHING_PUBLIC.png','WEB_COACHING_UNCERTAIN.png'],
    limitation='Browser tested saved normal/uncertain examples. Nonzero correction and focused replay verified by saved-output/DOM contract tests, not physical webcam accuracy.')
for f in browser['screenshots']: check('screenshot:'+f,(U/f).stat().st_size>1000)
(U/'BROWSER_VERIFIED.json').write_text(json.dumps(browser,ensure_ascii=False,indent=2),encoding='utf-8')
changes=[U/'coaching.py',W/'server.py',*[W/'site'/f for f in ['live.js','feedback.js','live.css','index.html']]]
out=dict(passed=all(c['passed'] for c in checks),checks=checks,
    implementation_and_verifier='same root Codex; separate saved-output arithmetic/contracts; not independent human/expert KSL judgment',
    server=read(W/'SERVER.json'),changed_hashes={str(p):digest(p) for p in changes},
    models_changed=False,full_common100_retraining=False,private_data_used=False,
    motion_candidate_adopted=False,sealed_evaluation_opened=False,
    first_issue_preserved='First restart safety assertion rejected a relative command line; no process was stopped. PID/listener and exact runtime were subsequently verified before idle-only restart.',
    training_note='PyTorch float(loss) warning affected logging only; frozen experiment source was not rewritten; all 18 runs completed. No training rerun.')
(U/'FINAL_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),server=out['server']),ensure_ascii=False))
assert out['passed']
