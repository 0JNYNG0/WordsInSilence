from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;R=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
# Close the report-author handoff: only the explanatory text was completed late.
c=read(U/'COMPLETION.json')
changes=[p for p,h in c['evidence_pins'].items() if sha(p)!=h]
if changes:
 assert changes==[str(U/'COMPARATOR_RESULT_KO.txt')],changes
 before=U/'review/COMPLETION_BEFORE_FINAL_REPORT_TEXT.json'
 assert not before.exists()
 before.write_bytes((U/'COMPLETION.json').read_bytes())
 c['evidence_pins'][changes[0]]=sha(changes[0])
 c['report_finalization_note']='Consumer report author completed the explicit negative-negative/unknown-cause explanation after initial completion metadata. Only this prose pin refreshed; earlier metadata retained. Code, saved comparison, execution and scores unchanged.'
 c['evidence_pins'][str(U/'QUERY_COVERAGE_LEDGER.json')]=sha(U/'QUERY_COVERAGE_LEDGER.json')
 dump(U/'COMPLETION.json',c)
 state=read(U.parent/'STATE.json');state[U.name]=c|dict(active=None);dump(U.parent/'STATE.json',state)
checks=0
def check(b):
 global checks
 assert b
 checks+=1
for p,h in c['evidence_pins'].items():check(sha(p)==h)
for p,h in read(U/'COMPARATOR_EXECUTION.json')['pins'].items():check(sha(p)==h)
comparison=read(U/'COMPARISON_RESULTS.json')
lookup={}
for result in comparison['results']:
 for ref in result['references']:
  if ref['status']!='LOCAL_ESTIMATED_EVIDENCE':continue
  for method,data in ref['methods'].items():
   lookup[(result['query'],ref['reference_alias'],method)]={p[0] for s in data['stages'] for p in s['saved_path']}
ledger=read(U/'QUERY_COVERAGE_LEDGER.json');check(len(ledger['rows'])==72)
check(ledger['source_result_sha256']==sha(U/'COMPARISON_RESULTS.json'))
for r in ledger['rows']:
 used=lookup[(r['query'],r['reference'],r['method'])]
 outside=r['outside_projection_frames'];ids=[a['query_frame'] for a in outside]
 check(len(ids)==len(set(ids)))
 check(used.isdisjoint(ids) and used|set(ids)==set(range(r['original_query_frames'])))
 check(len(used)==r['locally_projected_distinct_query_frames'])
 check(all(x['status']=='NOT_COMPARED_NOT_REST_NOT_CORRECT' for x in outside))
 check(r['whole_sign_verdict'] is None and r['full_query_accounting_preserved'])
state=read(U.parent/'STATE.json');check(state['current_stage']==c['status'])
check(state[U.name]==c|dict(active=None))
ctx=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf8')
for heading in ['## 8. 현재 상태','## 9. 다음 단일 작업']:
 section=ctx[ctx.index(heading):];first=section.split('### ',2)[1];check('2026-11-01' in first)
handoff=(R/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
check(U.name in handoff and c['status'] in handoff and '실행중없음' in handoff)
history=(R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').read_text(encoding='utf8')
check('2026-11-01 이번 네 단계 실행 완료 추기' in history)
for p in [R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt',U/'RESULT_KO.txt',U/'COMPARATOR_RESULT_KO.txt',U/'PLAN_KO.txt']:
 check(p.exists() and p.stat().st_size>0)
check(not(U/'RESULT_KO.html').exists())
out=dict(status='PASS_FINAL_DOCUMENT_LINKS_SYNC_PINS_AND_FULL_QUERY_COVERAGE',checks=checks,
 finalization_note=c.get('report_finalization_note'),new_inference=0,new_training=0,
 current_status=c['status'],pinned_files={str(p):sha(p) for p in [U/'COMPLETION.json',R/'WordsInSilence_PROJECT_CONTEXT.md',U.parent/'STATE.json',R/'NEW_CHAT_HANDOFF.md',R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt',U/'QUERY_COVERAGE_LEDGER.json']})
dump(U/'review/FINAL_DOCUMENT_VERIFICATION.json',out)
print(json.dumps({'status':out['status'],'checks':checks}))
