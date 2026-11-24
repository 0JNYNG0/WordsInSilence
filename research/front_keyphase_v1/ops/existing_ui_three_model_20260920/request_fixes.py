"""Return only the two corrections to newly Claude-generated generic source."""
import ast,json,subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent
src=(here/'session_replay.py').read_text(encoding='utf-8')
tree=ast.parse(src)
review=next(ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_collect_reviews')
runtime='''                        except (ValueError, RuntimeError):
                            raise
                        except Exception as exc:'''
assert src.count(runtime)==1
prompt='''Two independent synthetic tests fail in your NEW generic offline evaluation code. Return ONLY JSON {"replacements":[{"old":"exact literal","new":"exact replacement"},...]} with exactly two edits. No tools or prose.
1) _collect_reviews only detects conflicting duplicate revisions when they equal the CURRENT latest revision. A stream revision1 -> revision2 -> conflicting revision1 wrongly passes. Track every (trial_id,revision) content independently, reject disagreement even when superseded; still choose latest revision as truth. Preserve normal revision ordering behavior.
2) ValueError/RuntimeError RAISED BY neural.recognise are runtime errors just like other Exceptions and must become one arm's ERROR; otherwise a single failed model aborts the whole cohort. Remove the special re-raise branch. _normalize_raw is already in the try/except's ELSE suite, hence its contract/hash errors are outside the catch and remain fatal. Do not move normalization into the caught region.
New generic code you authored (no private project source/data):
'''+review+'\n\nLiteral runtime branch:\n'+runtime
(here/'fix_prompt.txt').write_text(prompt,encoding='utf-8')
argv=['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe','-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--effort','low','--output-format','json']
start=time.time()
with (here/'fix_response.json').open('xb') as output,(here/'fix_stderr.txt').open('xb') as error:
    child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=output,stderr=error,cwd=here)
    (here/'fix_worker.json').write_text(json.dumps({'pid':child.pid,'private_source_or_data_sent':False}),encoding='utf-8')
    print(json.dumps({'pid':child.pid}),flush=True)
    child.communicate(prompt.encode())
done={'exit_code':child.returncode,'seconds':round(time.time()-start,3),'response_bytes':(here/'fix_response.json').stat().st_size}
(here/'fix_completion.json').write_text(json.dumps(done),encoding='utf-8')
print(json.dumps(done),flush=True)
