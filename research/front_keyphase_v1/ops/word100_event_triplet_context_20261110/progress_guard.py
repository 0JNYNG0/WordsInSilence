"""Only withdraw an unsupported positive. Does not assert an error or omission."""
import copy
def assess(card,binding,predictions):
 byspan={(p['a'],p['b']):p for p in predictions};ev=binding['movement_links'];rows=[]
 for e in ev:
  if not e['eligible'] or e['confidence']<.65 or e['prediction']==1:continue
  p=byspan.get(tuple(e['span']))
  progress=bool(p and p['observed'] and p['context'] in binding['contexts'] and p['confidence']>=.65 and p['core_state']=='progress' and p['core_probabilities'][1]>=.65)
  rows.append(dict(span=e['span'],complete=e['complete'],direct_progress_evidence=progress,context=p['context'] if p else None,core_probabilities=p['core_probabilities'] if p else None))
 # A present complete boundary or an unclassified motion prevents this narrow veto.
 veto=bool(card['display_status']=='similar' and rows and not any(e['complete'] for e in ev) and all(not e['complete'] and e['direct_progress_evidence'] for e in rows))
 return dict(withdraw_positive=veto,checks=rows,kind='progress_observed_completion_unconfirmed',not_proof_of_omission=True)
def apply(out):
 out=copy.deepcopy(out);bind={b['card']:b for b in out['guide_link_bindings']};changes=[]
 for c in out['cards']:
  if c['id'] not in bind:continue
  audit=assess(c,bind[c['id']],out['triplet_segments']);c['triplet_completion_audit']=audit
  if audit['withdraw_positive']:
   changes.append(dict(card=c['id'],before=c['display_status'],checks=audit['checks']))
   c.update(display_status='uncertain',status='insufficient_completion_evidence',educational_wrong=False,message='움직이기 시작한 모습은 보이지만, 중요한 동작을 끝까지 이어갔는지는 확인하기 어려워요.')
   c['candidate_message']=c['message']
 out['triplet_completion_vetoes']=changes
 return out
