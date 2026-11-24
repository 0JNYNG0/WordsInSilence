from pathlib import Path
import json,hashlib,copy,itertools
from ownership_evidence import inspect_ownership
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
f=read(U/'OWNERSHIP_FROZEN.json');inp=read(U/'FULL_GRAPH_COMPARISON.json');res=read(U/'OWNERSHIP_COMPARISON.json')
ck('model_input_not_modified',sha(U/'FULL_GRAPH_COMPARISON.json')==f['input_sha256'])
ck('code_unchanged',sha(U/'ownership_evidence.py')==f['code_sha256'])
ck('all246',len(inp['rows'])==len(res['rows'])==246)
for before,after in zip(inp['rows'],res['rows']):
 ck('case_identity',all(before[k]==after[k]for k in ['seed','clip_id','kind']))
 cards=before['cards'];expect=[]
 for a,b in itertools.combinations(cards,2):
  if a['family']==b['family']or not a.get('chosen')or not b.get('chosen'):continue
  # A different formulation checks discrete interior overlap without calling
  # the implementation helper. Endpoints may meet without crossing ownership.
  def hit(es,w):return any(e['span'][0]<w[1] and e['span'][1]>w[0]and e['span'][1]>e['span'][0] and w[1]>w[0]for e in es)
  if hit(a['role_confirmation']['matching'],b['chosen']['span'])and hit(b['role_confirmation']['matching'],a['chosen']['span']):expect.append([a['card'],b['card']])
 ck('independent_overlap_arithmetic',expect==[p['cards']for p in after['ownership']['conflicts']])
 ck('no_discard',len(after['ownership']['cards'])==len(cards)and all(c['action_retained']for c in after['ownership']['cards']))
 ck('no_full_completion',all(c['semantic_completion']is None for c in after['ownership']['cards']))
 snapshot=copy.deepcopy(cards);inspect_ownership(cards);ck('no_input_mutation',cards==snapshot)
 reversed_output=inspect_ownership(list(reversed(cards)))
 ck('expected_order_irrelevant',{frozenset(p['cards'])for p in reversed_output['conflicts']}=={frozenset(p['cards'])for p in after['ownership']['conflicts']})
ck('normal_speed_no_new_conflict',not res['normal_conflicts'])
ck('actual_pure_block_swaps_retained',all(not r['ownership']['conflicts']for r in res['rows']if r['kind']=='dynamic_complete_blocks'))
conf=[r for r in res['rows']if r['ownership']['conflicts']]
ck('six_missing_final_conflicts_flagged',len(conf)==6 and {r['kind']for r in conf}=={'omit_final_event','mask_final_event'})
write('OWNERSHIP_VERIFIED.json',dict(passed=all(c['passed']for c in checks),total=len(checks),checks=checks,
 failures=[c for c in checks if not c['passed']],flagged_cases=len(conf),
 source_count=len({r['clip_id']for r in conf}),normal_speed_conflicts=len(res['normal_conflicts']),
 scope='6 flags = one public recording x2 source manipulations x3 seeds; not six independent learner errors',
 same_Codex_separate_verification=True,not_semantic_accuracy=True))
print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),flagged=len(conf),failures=[c for c in checks if not c['passed']]),ensure_ascii=False))
