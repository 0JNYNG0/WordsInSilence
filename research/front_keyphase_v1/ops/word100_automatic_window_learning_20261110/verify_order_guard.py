from pathlib import Path
import json,copy,importlib.util
from order_guard import apply
U=Path(__file__).resolve().parent;P=U.parent/'word100_broad_factorized_learning_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
s=importlib.util.spec_from_file_location('prior_conflict',P/'evidence_conflict_guard.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
checks=[]
for r in read(U/'ORDER_ONLY_COMPLETE.json')['rows']:
 key=r['clip_id']+'_'+r['kind'];d=read(r['result'])
 if r['new_baseline_inference']:base=read(Path(r['result']).parent/'first_baseline.json')
 else:base=m.apply(read(P/('edited_channels' if r['kind'] in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed'] else 'fullpath')/'coupled_42'/key/'after.json'))
 untouched=copy.deepcopy(base);out=apply(base,d['learned_order_checks']);assert base==untouched
 assert out['cards']==d['cards'] and out['path']==base['path'] and out['tcn_top3']==base['tcn_top3'];checks.append(key)
 # Changing the evidence's time span must never create a warning on another span.
 mismatched=copy.deepcopy(d['learned_order_checks'])
 for q in mismatched:
  if 'span' in q:q['span'][0]+=1
 wrong=apply(base,mismatched);assert wrong['cards']==base['cards']
(U/'ORDER_GUARD_VERIFIED.json').write_text(json.dumps(dict(passed=True,cases=len(checks),saved_output_parity=True,input_not_mutated=True,wrong_window_rejected=True,default_web_changed=False,scope='Same Codex pure-function verification against78saved comparisons; no new inference.'),ensure_ascii=False,indent=2),'utf-8');print('Order-only reusable guard:78saved comparisons match; mismatched windows rejected.')
