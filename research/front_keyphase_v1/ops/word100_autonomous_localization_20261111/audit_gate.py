from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;G=U/'guide_warning_gate'
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compact(cards):
 display=[c['display_status'] for c in cards];return dict(display=display,warning='difference' in display,similar_no_warning=all(c['display_status']=='similar' for c in cards if c.get('required',True)) and 'difference' not in display)
rows=rd(G/'FULLPATH_COMPLETE.json')['rows'];checks=0
for r in rows:
 d=rd(r['result']);assert sha(r['result'])==r['sha256'];assert sha(d['baseline'])==d['baseline_sha256'];before=rd(d['baseline']);assert compact(before['cards'])==r['before'];r['after']=compact(d['cards']);checks+=3
 for a,b in zip(d['cards'],before['cards']):
  assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence');checks+=1
  if b['display_status']=='difference':assert a['display_status']=='difference';checks+=1
 for entry in d['decisions']:assert entry['all_observed_attempts_retained'];checks+=1
 if r['kind'] in ['full_normal','speed075','speed125']:assert not r['after']['warning'];checks+=1
 if r['kind'] in ['all_hand_missing','mask_final_event','omit_final_event']:assert not r['after']['similar_no_warning'];checks+=1
summary={}
for seed in [42,43,44]:
 for kind in sorted({r['kind'] for r in rows}):
  rs=[r for r in rows if r['seed']==seed and r['kind']==kind];summary[f'{seed}/{kind}']=dict(total=len(rs),before_warnings=sum(r['before']['warning'] for r in rs),after_warnings=sum(r['after']['warning'] for r in rs),before_similar=sum(r['before']['similar_no_warning'] for r in rs),after_similar=sum(r['after']['similar_no_warning'] for r in rs),changed=[r['clip_id'] for r in rs if r['before']!=r['after']])
(G/'VERIFIED_FULLPATH.json').write_text(json.dumps(dict(rows=rows),ensure_ascii=False,indent=2),'utf-8');(G/'VERIFIED_RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
(G/'VERIFICATION.json').write_text(json.dumps(dict(passed=True,checks=checks,reviewer='Separate same-Codex saved-data audit; not independent human',aggregation_correction='First gate summary counted warnings only on required cards, unlike frozen evaluator which includes optional child-card warnings. Actual cards did not lose warnings. Original summary preserved; VERIFIED_RESULT is canonical.',all_three_seed_normal_speed_preserved=True,added_final_feedback_gain=any(v['after_warnings']>v['before_warnings'] for k,v in summary.items() if not k.endswith(('full_normal','speed075','speed125')))),ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(dict(checks=checks,changed={k:v for k,v in summary.items() if v['changed']}),ensure_ascii=False))
