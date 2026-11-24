"""Show actual uncertain intervals, not a neighbouring successful interval."""
from form_presentation import present as previous_present
def present(q,reference_span):
 cards,coaching=previous_present(q,reference_span);by={c['id']:c for c in q['criteria']};ts=q['timestamp_us']
 for card in cards:
  c=by[card['id']]
  if c['status']!='unknown':continue
  spans=c.get('uncertain_spans')or c.get('unconfirmed_events')
  if not spans:continue
  a=min(s[0]for s in spans);b=max(s[1]for s in spans)
  card.update(query_spans=spans,query_indices=[a,b],query_seconds=[ts[a]/1e6,ts[b]/1e6],uncertainty_location='observed_unreliable_or_unconfirmed_interval')
 return cards,coaching
