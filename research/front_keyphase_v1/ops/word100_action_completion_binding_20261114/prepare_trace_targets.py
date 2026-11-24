"""Bind existing reviewed public events to clauses, without creating KSL truth."""
from pathlib import Path
import json,hashlib,collections
from source_trace_contract import evaluate_trace
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']
guides_path=O/'word100_action_feedback_demo_20261108/CARDS.json';guides=read(guides_path)
# Declarative linkage of existing source annotation families to saved guide cards.
# It is not used by the word-agnostic trace evaluator, nor as a word recognition rule.
bindings={2:('WORD0739','close_chest'),3:('WORD0739','raise_forearm'),4:('WORD1133','brush_arm'),5:('WORD1567','alternating'),7:('WORD2523','spread_hands')}
rows=[];cases=[];pins={str(p):sha(p) for p in sources+[guides_path,U/'PLAN.json',U/'source_trace_contract.py']};clips=set()
for path in sources:
    for src in read(path)['rows']:
        assert src['clip_id'] not in clips;clips.add(src['clip_id'])
        person=src['person'];assert person in range(2,10)
        # Old source annotation role says fit also for 08/09. The narrower,
        # already frozen head split takes precedence. Never move them to train.
        split='train' if person<=7 else 'exposed_probe'
        assert sha(Path(src['source']))==src['source_sha256']
        pins[src['source']]=src['source_sha256']
        for group in src['groups']:
            word,card_id=bindings[group['family']];assert word==src['word_id']
            card=next(c for c in guides['words'][word] if c['id']==card_id)
            related=[e for e in src['events'] if e['event_id'] in group['event_ids']]
            assert related and all(e['family']==group['family'] for e in related)
            assert all(e['linguistic_completion'] is None for e in related)
            assert group['onset'][0]<=group['onset'][1]<group['offset'][0]<=group['offset'][1]<src['frames']
            witnesses=sorted(set(k for e in related for k in e['progression_witness_frames']))
            assert len(witnesses)>=1 and all(group['onset'][0]<=k<=group['offset'][1] for k in witnesses)
            identity=dict(clip_id=src['clip_id'],person=person,word_id=word,split=split,group_id=group['group_id'],
                family=group['family'],card_id=card_id,guide_instruction=card['instruction'],hands=card['hands'],
                annotation_file=str(path),annotation_sha256=sha(path),source=src['source'],source_sha256=src['source_sha256'],
                onset_band=group['onset'],offset_band=group['offset'],events=related,
                progression_annotation_density='single_interior_witness' if len(witnesses)==1 else 'multiple_interior_witnesses',
                required_repetition_count=None,semantic_completion_target=None,
                training_admission='source identity/order/coverage only; no educational full/partial correctness',
                reviewer='prior RGB annotations, root Codex arithmetic review; not independent KSL expert')
            rows.append(identity)
            a,b=group['onset'][0],group['offset'][1];base=list(range(a,b+1));tail=group['offset'][0]
            # Distinct cut locations come from reviewed interior witnesses, not
            # a universal required duration or a new coordinate threshold.
            cut_candidates=[k for k in witnesses if a<k<tail]
            chosen=sorted(set(cut_candidates[i] for i in [0,len(cut_candidates)//2,len(cut_candidates)-1]))
            variants=[('full',base,[]),('with_context',list(range(max(0,a-6),min(src['frames'],b+7))),[]),
                ('reverse',base[::-1],[]),('mask_finish',base,list(range(tail,b+1))),
                ('endpoint_only',list(range(a,group['onset'][1]+1))+list(range(tail,b+1)),[])]
            variants += [('prefix_'+str(k),list(range(a,k+1)),[]) for k in chosen]
            variants += [('suffix_'+str(k),list(range(k,b+1)),[]) for k in chosen]
            # Known source-frame subsampling is an observation test; dropping a
            # direct witness is not labeled incomplete performance.
            variants += [('subsample_every2',base[::2],[]),('duplicate_each2',[k for k in base for _ in range(2)],[])]
            for kind,ids,masked in variants:
                result=evaluate_trace(group,src['events'],ids,masked,kind)
                cases.append(dict(case_id=group['group_id']+':'+kind,clip_id=src['clip_id'],person=person,split=split,
                    family=group['family'],group_id=group['group_id'],kind=kind,source_indices=ids,masked_source_frames=masked,
                    **result))
assert len(clips)==32 and len(rows)==40
counts=collections.Counter((r['split'],r['source_trace_state']) for r in cases)
out=dict(status='SOURCE_TRACE_TARGETS_PREPARED_NOT_A_NEW_TRAINED_MODEL',sources=rows,cases=cases,
    source_clips=32,train_clips=24,probe_clips=8,source_groups=40,train_groups=30,probe_groups=10,
    summary=[dict(split=k[0],state=k[1],cases=v) for k,v in sorted(counts.items())],
    pins=pins,semantic_completion_targets=0,
    limitations=['No new natural learner-error labels.', 'Repeated source submovements do not define a required repetition count.',
      'Speed labels here are sample-index transport controls, not new speed accuracy evaluation.',
      'The model must infer these states from observation; this provenance evaluator itself cannot infer states in a new webcam clip.'],
    no_private=True,no_final8=True,no_new_neural_training=True)
with (U/'SOURCE_TRACE_TARGETS.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:out[k] for k in ['source_clips','train_clips','probe_clips','source_groups','semantic_completion_targets','summary']},ensure_ascii=False))
