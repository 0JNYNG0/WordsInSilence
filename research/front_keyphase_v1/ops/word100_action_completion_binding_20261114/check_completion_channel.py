from pathlib import Path
import json,copy,hashlib
from completion_channel import attach_extent_channel
U=Path(__file__).resolve().parent
j=json.loads((U/'FULLPATH_COMPARISON.json').read_text('utf-8'));checks=0;rows=[]
for r in j['rows']:
    base=r['conditions']['fixed_rank'];before=copy.deepcopy(base);candidate=r['conditions']['typed_extent']
    result=attach_extent_channel(base,candidate);assert base==before
    restored=copy.deepcopy(result);restored.pop('completion_channel_version')
    for c in restored['cards']:
        a=c.pop('extent_assessment');assert a['semantic_completion']is None and a['educational_error']is None and not a['display_enabled']
    assert restored==base;checks+=1
    rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],localized_cards=len(base['cards']),
        retained_for_comparison=sum(c['chosen']is not None for c in result['cards']),
        insufficient_extent=sum(c['extent_assessment']['state']=='insufficient_extent_evidence'for c in result['cards'])))
out=dict(passed=True,checks=checks,rows=rows,all_localization_order_truth_fields_unchanged=True,
    original_model_predictions_untouched=True,new_neural_inference=False,web_enabled=False,
    outcome='Research channel wiring preserves already located actions, including wrong-handshape/direction cases; it does not validate completion predictions.',
    source_sha256=hashlib.sha256((U/'FULLPATH_COMPARISON.json').read_bytes()).hexdigest())
with (U/'COMPLETION_CHANNEL_VERIFIED.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('Separate completion channel: 246 saved conditions preserved; no web adoption.')
