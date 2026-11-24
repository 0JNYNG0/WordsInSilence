"""Known practice word routing from FIT public references; no query answer use."""
from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;OPS=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 src=OPS/'word100_core_feature_split_20261101';old=OPS/'word100_stage_feedback_cycle_20261031'
 inp=read(src/'INPUTS.json');selection=read(src/'SELECTION.json');rows=inp['rows'];original=read(old/'LEARNING_INPUTS.json')
 cards=read(OPS/'word100_guide_video_feedback_20261028/GUIDE_CARDS.json')['cards']
 profiles=read(old/'review/guide_stage_map.json')['expressions']
 phase=read(OPS/'word100_core_stage_evidence_20261031/PHASE_DATA.json')['clips']
 oldclip={r['clip_id']:r for r in original['rows']};card={r['word_id']:r for r in cards};prof={r['word_id']:r for r in profiles};phasealias={r['alias']:r for r in phase}
 def entry(r):
  assert sha(r['input'])==r['input_sha256']
  o=oldclip.get(r['clip_id']);legacy=o['alias'] if o else None
  return {k:r[k] for k in ['alias','clip_id','word_id','label','person','role','input','input_sha256','source','source_sha256','observation','observation_sha256','box']}|dict(legacy_alias=legacy,has_saved_full_feature_probabilities=legacy is not None,phase_reference_available=legacy in phasealias)
 known=sorted({r['word_id'] for r in rows if r['role']=='fit'});bank=[];queries=[]
 for word in known:
  refs=[r for r in rows if r['role']=='fit' and r['word_id']==word];dev=[r for r in rows if r['role']=='development' and r['word_id']==word]
  assert len(refs)==3 and {r['person'] for r in refs}=={1,2,3};assert len(dev)==1 and dev[0]['person']==4
  c=card.get(word);p=prof.get(word)
  bank.append(dict(word_id=word,label=refs[0]['label'],references=[entry(r) for r in refs],
   guide_candidates=[] if c is None else c['candidates'],guide_source_status='UNAVAILABLE' if c is None else 'SOURCE_EXPECTATION_ONLY',
   guide_form_match=None if p is None else p['form_match'],
   partial_feature_links=[] if p is None else p.get('supported_partial_guide_features',[]),
   guide_can_override_observation=False,mandatory_linguistic_rules_enabled=False,exact_required_repetition_count=None,
   unsupported_reason=None if all(oldclip.get(r['clip_id'],{}).get('alias') in phasealias for r in refs) else 'Reference/guide routing available; phase-linked automatic comparison not in reviewed16-video scope'))
  queries.append(dict(practice_word_id=word,query=entry(dev[0]),scope='Historical same-supported-word, recorded-person4 development replay; no actual learner error labels'))
 assert len(bank)==len(queries)==20 and sum(len(b['references']) for b in bank)==60
 pins=[src/'INPUTS.json',src/'SELECTION.json',old/'LEARNING_INPUTS.json',old/'review/guide_stage_map.json',OPS/'word100_guide_video_feedback_20261028/GUIDE_CARDS.json',OPS/'word100_core_stage_evidence_20261031/PHASE_DATA.json']
 write(U/'REFERENCE_BANK.json',dict(status='KNOWN_PRACTICE_WORD_FIT_REFERENCE_ROUTING',words=bank,development_queries=queries,
  inference_contract='Caller supplies practice_word_id independently of inferred TCN label or query truth. Query observations remain unchanged when practice word changes. Unknown/unsupported target never silently reroutes.',
  query_labels_allowed_for='Evaluation-only normal cohort construction and post-comparison audits; forbidden inside aligner/observer',
  primary_product_test='Same supported word on held-out people, normal variation and important deviations; unseen word supplementary diagnosis',
  source_pins={str(p):sha(p) for p in pins},protected=selection['protected'],training=0,new_inference=0))
 write(U/'AUTHORIZATION.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),request='User asks objective/progress chronology for future mentor reports, then reviewed concrete plan and execute all steps',
  scope='Document history; known-target public reference/guide routing; frozen observation+stored automatic alignment integration; finite normal/controlled-difference evaluation and independent verification',
  no_new_external_models_or_downloads=True,no_private_or_mentor_recordings=True,preserve_C_T3F_and_production=True,no_old_result_reinference_or_relabel=True,
  training_policy='Reuse existing frozen learned observation outputs first; training only if newly required verified targets and prospectively fixed valid experiment exist. No forced training to count completion.'))
 print(json.dumps(dict(supported_words=20,fit_references=60,dev_queries=20,phase_scoped_words=sum(x['unsupported_reason'] is None for x in bank),guide_candidates_available=sum(bool(x['guide_candidates']) for x in bank)),ensure_ascii=False))
if __name__=='__main__':main()
