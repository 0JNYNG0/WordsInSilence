"""Link saved physical event proposals to source candidates without inventing semantics."""
from episode import *

def main():
 source=read(U/'review/SOURCE_USAGE_AND_RANGE.json');rs={r['alias']:r for r in rows()};results=[]
 for word in source['source_connections']:
  row=next(r for r in rs.values() if r['role']=='development' and r['word_id']==word['word_id']);a=arrays(row);pred=np.load(U/'runs/42/auto'/f"{row['alias']}.npz")['pred'];events=runs(pred,a['frame_id'],a['timestamp_us']);clauses=[]
  for clause in word['clauses']:
   state={'R_motion_up':'up','R_motion_down':'down'}.get(clause['feature'])
   candidates=[e for e in events if e['state']==state] if state else []
   clauses.append(dict(source_clause_id=clause['id'],source_span=clause['source_span'],source_url=clause['source_url'],physical_output=state,possible_intervals=candidates,status='physical_candidates_only' if candidates else 'not_supported_by_this_head',semantic_stage_identity_verified=False,reason='Same screen-motion direction can occur during preparation, core action or lowering. Do not assign a source clause from direction alone. Original source form/hand shape/arm relation must agree.',educational_verdict=None))
  results.append(dict(alias=row['alias'],label=row['label'],word_id=word['word_id'],source_form_verified=word['full_source_form_verified'],clauses=clauses,orders=word['orders'],source_order_verified_from_new_outputs=False,physical_events=events))
 write(U/'GUIDE_EPISODE_LINKS.json',dict(status='PHYSICAL_CANDIDATES_LINKED_SEMANTIC_STAGE_UNVERIFIED',rows=results,scope='Preserve source wording/form distinction and original event times. Do not call preparatory motion a verified lexical stage.',source_sha256=sha(U/'review/SOURCE_USAGE_AND_RANGE.json')))
if __name__=='__main__':main()
