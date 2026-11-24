"""Small generic implementation request; no private source or user records transmitted."""
import json,subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent
prompt='''Return ONLY JSON {"code":"Python insertion"}. No tools/prose. Author a minimal block, indent12spaces, for inside a loop in a Python in-memory session store method. General interfaces only: `self.sessions[sid]` is the expired session dict. Earlier loop code already marks state CLOSED, closed_reason LEASE_EXPIRED, cancels unfinished trials while preserving every trial_id, and releases the camera if this expired session owns it. Bug: disk session.json remains OPEN with the original empty trial_ids, because only explicit manual close persisted it. Your insertion goes after camera-release handling for EVERY expired session, including sessions with zero trials. Set that session's `camera` to False, then call the existing write_json helper with os.path.join(session['run_dir'],'session.json') and a dictionary snapshot of all session fields EXCEPT 'request_ids' (an internal dedup map). Preserve actual trial_ids/order, participant/client IDs, terminal reason, other existing fields. Do not invent fields or alter timestamps, labels, camera implementation, model code, live/non-expired sessions. Follow the existing explicit-close persistence behavior: do not swallow write errors. No new helper needed.''' 
(here/'generic_prompt.txt').write_text(prompt,encoding='utf-8')
argv=['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe','-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--effort','low','--output-format','json']
start=time.time()
with (here/'response.json').open('xb') as output,(here/'stderr.txt').open('xb') as error:
    child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=output,stderr=error,cwd=here)
    (here/'worker.json').write_text(json.dumps({'pid':child.pid,'private_source_or_data_sent':False}),encoding='utf-8')
    print(json.dumps({'pid':child.pid}),flush=True)
    child.communicate(prompt.encode())
done={'exit_code':child.returncode,'seconds':round(time.time()-start,3),'response_bytes':(here/'response.json').stat().st_size}
(here/'completion.json').write_text(json.dumps(done),encoding='utf-8');print(json.dumps(done),flush=True)
