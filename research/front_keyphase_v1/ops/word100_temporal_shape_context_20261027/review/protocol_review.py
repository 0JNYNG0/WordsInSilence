import json, hashlib, collections
from pathlib import Path

U=Path(__file__).resolve().parents[1]
f=json.loads((U/'FROZEN.json').read_text(encoding='utf-8-sig'))
checks={
 '18_unique_cases': len(f['rows'])==18 and len({r['case_id'] for r in f['rows']})==18,
 '18_unique_public_clips':len({r['clip_id'] for r in f['rows']})==18,
 '13_frame_offsets':f['dense_offsets_frames']==list(range(-6,7)),
 'two_fixed_features':f['features']==['index_only','open_four'],
 'fixed_200ms_2_support':f['max_gap_us']==200000 and f['min_support_frames_each_direction']==2,
 'input_pins_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in f['pins'].items())
}
o={
 'status':'PASS_WITH_SCOPE_LIMITS' if all(checks.values()) else 'FAIL',
 'checks':checks,
 'protocol_sha256':hashlib.sha256((U/'FROZEN.json').read_bytes()).hexdigest(),
 'role_counts':dict(collections.Counter(r['inherited_role'] for r in f['rows'])),
 'expected_images':234,'expected_binary_field_values':468,
 'implementation_checks_required':[
  'Mask each known target independently from original direct labels. No recursive fill and no hidden target leaking into evidence.',
  'B reads past only; C requires two distinct valid frame indices on each side. Use actual PTS and the frozen 200000us bound.',
  'Directly visible labels stay unchanged. Unknown is never changed into absence. Predictions retain estimate provenance.',
  'Report fill coverage, disagreement among fills, rejection and all missing-denominator counts; include short-change/transient-state failures.',
  'Check index_only/open_four mutual exclusivity; jointly inconsistent fills must not be promoted into trusted state.',
  'Identity, occlusion and gap flags cannot silently disappear because neighboring binary states agree.'
 ],
 'limits':[
  'Same-Codex image reviews are consistency evidence, not independent expert or coordinate truth.',
  'Known-state masking is artificial missingness; it cannot validate accuracy at genuinely hidden centers.',
  '18 deliberately selected short windows cannot establish all-word or new-user recognition accuracy.',
  'C uses up to 200ms future context: offline analysis or an explicit live delay, not zero-latency causal tracking.',
  'This unit compares rules on image-reviewed labels; it does not build an automatic RGB shape recognizer or justify training on imputed guesses.'
 ],
 'images_seen_when_written':['T01/shuffled_contact.png'],
 'no_ordered_or_condition_result_read':True,
 'own_label_policy':'Read each shuffled K image directly. Familiar prior examples and frozen word metadata mean this is not fully blind. No source-description/sequence-driven completion.'
}
(U/'review'/'PROTOCOL_REVIEW.json').write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(o,ensure_ascii=False))
