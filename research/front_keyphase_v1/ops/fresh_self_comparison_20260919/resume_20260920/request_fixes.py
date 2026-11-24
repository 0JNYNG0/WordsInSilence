"""Minimal corrections to newly Claude-generated generic code only; no private source."""
from pathlib import Path
import ast,json,subprocess,time
here=Path(__file__).resolve().parent
instruction='''Repair the generic code you just generated. Return ONLY JSON {"edits":[{"file":"frozen_cohort.py|frozen_replay.py","old":"exact unique source substring","new":"replacement"}]}. Provide small exact literal replacements, not full-file rewrites. Preserve all other behavior. No tests.

Required corrections:
1. Driver validate_journal incorrectly requires status CLOSED and safe_id(request_id). The journal may be COMPLETED, STOPPED, INTERRUPTED or stale ACTIVE while the authoritative server session is CLOSED. Require a nonempty journal status string; server CLOSED remains mandatory. request_id is nullable and contains colons. Check only nullable/string there; build_cohort enforces exact stable request-ID equality. It is not a filename.
2. In cmd_prepare, a protected_sha256 file absent must raise, not accept and pin absence. These are expected existing protected implementation/model files; ordinary missing capture artifacts remain allowed.
3. In cmd_execute, snapshot existing registry with deepcopy (already import copy). Runtime errors of recognise, including ValueError, become per-arm ERROR, while normalize contract/hash violations remain fatal; put normalize outside the recognise exception handler. Set normalized elapsed_ms to the finite stopwatch duration dt on EVERY arm, including errors; otherwise summarize rejects elapsed_ms=None. Preserve raw timing separately in log if useful.
4. Observation.load failure must keep that eligible row, record an observation_load_failed log event, create ERROR/null predictions with elapsed_ms=0 for all three roles and continue the next observation. It must NOT increase inference_calls because recognise was not invoked. Do not condition predictions on labels or requested words. Include an observation_load_failures list in final payload, initialized before loop. No silent skip when an eligible row has missing artifact path: raise integrity error.
5. Manifest execute output directory must be exactly the manifest file's parent; a manifest must not redirect writes elsewhere. Require expected manifest schema and exactly three model roles. Reverify real session/trials roots and each artifact path are still confined to web_root/runs/session_id (including symlink changes) before inference. Existing file hash checks remain.
6. Core _check_plan must enforce input slot array is in ascending1..54 order and round==((slot-1)//18)+1; _check_journal also enforce ascending order and integer current_slot1..54. Current set-based round checks permit swapping rounds or list order. In _check_results require clip_id key and exact value, not optional. _check_events reject unknown trial_closed outcome instead of treating arbitrary strings as a trusted terminal.

Below are ONLY the functions from your newly generated generic module, provided for exact source matching. No private project source/data is included.
'''
selection={'frozen_replay.py':['validate_journal','cmd_prepare','reverify','cmd_execute'],
           'frozen_cohort.py':['_check_plan','_check_journal','_check_results','_check_events']}
for filename,names in selection.items():
    source=(here/'generated_v1'/filename).read_text(encoding='utf-8')
    lines=source.splitlines(keepends=True)
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            instruction+='\nFILE '+filename+' FUNCTION '+node.name+'\n'+''.join(lines[node.lineno-1:node.end_lineno])+'\n'
(here/'fixes_prompt.txt').write_text(instruction,encoding='utf-8')
argv=['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe','-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--effort','low','--output-format','json']
start=time.time()
with (here/'fixes_response.json').open('xb') as out,(here/'fixes_stderr.txt').open('xb') as err:
    child=subprocess.Popen(argv,cwd=here,stdin=subprocess.PIPE,stdout=out,stderr=err)
    (here/'fixes_worker.json').write_text(json.dumps({'pid':child.pid,'private_source_data_sent':False}),encoding='utf-8')
    print(json.dumps({'pid':child.pid,'status':'STARTED'}),flush=True)
    child.communicate(instruction.encode())
record={'exit_code':child.returncode,'seconds':round(time.time()-start,3)}
(here/'fixes_completion.json').write_text(json.dumps(record),encoding='utf-8')
print(json.dumps(record),flush=True)
