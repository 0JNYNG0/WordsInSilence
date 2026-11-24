from pathlib import Path
import json,hashlib,collections
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):
 with (U/n).open('x',encoding='utf-8')as f:json.dump(x,f,ensure_ascii=False,indent=2)
inp=read(U/'CORE_COMPARISON.json');truth_path=G/'reviews/sequence_truth_audit.json'
truth={r['clip_id']:r for r in read(truth_path)['clips']};checks=[];rows=[]
for r in inp['rows']:
 for c in r['candidates']:
  # Old chosen/compared extent remains the full observed candidate. The core
  # only adds role witnesses; it cannot delete offending shape/direction frames.
  assert c['role_confirmation']['candidates_retained']
  if r['kind']!='reversed_core_auto_order' or r['clip_id']not in truth:continue
  ta=truth[r['clip_id']]['reviewed_source_anchors'];same=[];different=[]
  for q in ta:
   if any(a['span'][0]<=q['query_index']<=a['span'][1]for a in c['role_confirmation']['matching']):
    (same if q['source_reviewed_family']==c['family']else different).append(q)
  rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],card=c['card'],family=c['family'],matching_source_anchors=same,
                  conflicting_source_anchors=different,source_not_semantic_label=True))
normal=[r for r in inp['rows']if r['kind']in ['full_normal','speed075','speed125']]
known=[]
for r in normal:
 for c in r['candidates']:
  t=c.get('truth',{})
  if not t.get('witness'):continue
  obs={i for e in c['role_confirmation']['matching']for i in range(e['span'][0],e['span'][1]+1)}
  known.append(dict(seed=r['seed'],clip_id=r['clip_id'],card=c['card'],witness=t['witness'],
    core_keeps_all=all(i in obs for i in t['witness']),original_candidate_keeps_all=t['all_witness'],
    lost_from_core=[i for i in t['witness']if i not in obs],original_candidate_unchanged=True))
summary=dict(edited_source_cards=len(rows),edited_source_cards_with_conflict=sum(bool(r['conflicting_source_anchors'])for r in rows),
 reviewed_normal_cards=len(known),normal_core_all_witness=sum(r['core_keeps_all']for r in known),
 original_normal_candidates_all_witness=sum(r['original_candidate_keeps_all']for r in known),
 normal_old_order_regressions=len(inp['normal_reference_regressions']))
write('CORE_SOURCE_AUDIT.json',dict(summary=summary,edited_source_rows=rows,normal_direct_rows=known,
 source_sha256=sha(truth_path),comparison_sha256=sha(U/'CORE_COMPARISON.json'),
 limitations=['Reverse source identity is not a complete semantic action label','Repeated DEV arithmetic is not new independent validation','Role core cannot replace whole candidate for handshape/direction inspection'],
 whole_G2_pass=False,reviewer='same Codex separate exact-source arithmetic audit; not independent KSL expert'))
print(json.dumps(summary,ensure_ascii=False))
