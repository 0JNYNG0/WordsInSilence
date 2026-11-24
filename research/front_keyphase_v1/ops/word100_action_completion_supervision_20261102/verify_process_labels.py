"""Validate provenance/time/schema and aggregate new direct FIT process labels.

This checks the annotation contract, not the correctness of a human-language
judgment. No model is loaded or executed and no interval target is densified.
"""
from pathlib import Path
import json,hashlib,collections,traceback
from datetime import datetime,timezone
U=Path(__file__).resolve().parent;checked={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    p=str(Path(p).resolve())
    if p not in checked:checked[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return checked[p]
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    assert not (U/'PROCESS_TARGETS.json').exists()
    man=read(U/'PROCESS_REVIEW_MANIFEST.json');intent=read(U/'PROCESS_REVIEW_INTENT.json')
    assert sha(U/'PROCESS_REVIEW_INTENT.json')==man['intent_sha256']
    for p,h in intent['source_pins'].items():assert sha(p)==h,p
    meta={r['id']:r for r in man['rows']};assert len(meta)==45
    unique_frames={}; sources={};labels=[];reviews=[];counter=collections.Counter();split=collections.defaultdict(collections.Counter)
    for q in meta.values():
        assert 1<=q['person']<=9 and sha(q['source'])==q['source_sha256']
        sources[q['clip_id']]=q['source_sha256']
        fs={x['frame']:x for x in q['frames']};lo,hi=q['display_bounds'];assert set(fs)==set(range(lo,hi+1))
        times=[fs[i]['timestamp_us'] for i in sorted(fs)];assert all(b>a for a,b in zip(times,times[1:]))
        covered=[]
        for page in q['pages']:
            assert sha(page['path'])==page['sha256'];covered.extend(page['frames'])
        assert covered==sorted(fs)
        for f in fs.values():
            assert sha(f['path'])==f['sha256'];unique_frames[(q['clip_id'],f['frame'])]=(f['timestamp_us'],f['sha256'])
    for word in ['0739','1133','1567','2523']:
        file=U/(f'REVIEW_{word}_COMPAT.json' if word=='1567' else f'REVIEW_{word}.json')
        a=read(file);reviews.append(dict(path=str(file),sha256=sha(file)))
        if word=='1567':
            compatibility=read(U/'REVIEW_1567_COMPAT_VERIFIED.json')
            assert sha(U/'REVIEW_1567.json')==compatibility['source_sha256_before']==compatibility['source_sha256_after']
            assert sha(file)==compatibility['output_sha256'] and compatibility['changed_original_values']==0
            reviews.append(dict(path=str(U/'REVIEW_1567.json'),sha256=sha(U/'REVIEW_1567.json'),role='unaltered_original_annotation'))
        assert len(a['rows'])==(18 if word=='1567' else 9)
        for r in a['rows']:
            q=meta[r['id']];assert q['word_id']==r['word_id']=='WORD'+word and q['person']==r['person']
            assert q['clip_id']==r['clip_id'] and q['source_sha256']==r['source_sha256']
            assert q['physical_query']==r['physical_query']
            pages={p['path']:p['sha256'] for p in r['reviewed_pages']};assert pages=={p['path']:p['sha256'] for p in q['pages']}
            assert r['predicate_definitions']['setup_relation'] and r['predicate_definitions']['terminal_relation']
            frames={f['frame']:f for f in q['frames']};seen=set();row=dict(r)
            relations=[]
            for rel in r['relations']:
                f=rel['frame'];assert isinstance(f,int) and f in frames and f not in seen;seen.add(f)
                z=dict(rel);z['timestamp_us']=frames[f]['timestamp_us'];z['input_image']=frames[f]
                if 'timestamp_us' in rel:assert rel['timestamp_us']==z['timestamp_us']
                for k in ['setup_relation','terminal_relation']:
                    assert rel[k] in [0,1,None];counter[k+':'+str(rel[k])]+=1;split[word][k+':'+str(rel[k])]+=1
                assert rel['reason'];relations.append(z)
            assert relations
            intervals=[]
            for m in r['motion_intervals']:
                lo,hi=m['start'],m['end'];assert isinstance(lo,int) and isinstance(hi,int) and lo<hi
                assert all(f in frames for f in range(lo,hi+1));assert m['label'] in [0,1,None] and m['reason']
                z=dict(m);z['native_timestamps_us']=[frames[lo]['timestamp_us'],frames[hi]['timestamp_us']]
                z['supervision_scope']='one whole reviewed motion interval; not a target for every frame'
                intervals.append(z);counter['motion:'+str(m['label'])]+=1;split[word]['motion:'+str(m['label'])]+=1
            assert intervals
            for seq in r['sequence_links']:
                assert seq['visible_chain'] in ['yes','partial','uncertain'];counter['chain:'+seq['visible_chain']]+=1
                for f in seq['setup_frames']+seq['terminal_frames']:assert f in frames
                lo,hi=seq['motion_interval'];assert lo<hi and all(f in frames for f in range(lo,hi+1))
                assert seq['reason']
            assert r['review_bounds_sufficient'] in [True,False]
            assert r.get('whole_word_correctness') is None and r.get('required_repetition_count') is None
            row.update(role='fit',relations=relations,motion_intervals=intervals,source=q['source'],
                review_bounds=q['display_bounds'],unlisted_is_unknown=True,linguistic_correctness=None,
                source_annotation=str(file),source_annotation_sha256=sha(file))
            labels.append(row)
    assert len(labels)==45 and len({r['id'] for r in labels})==45 and len(sources)==36 and len(unique_frames)==1354
    assert set(r['person'] for r in labels)==set(range(1,10))
    protocols={str(U/n):sha(U/n) for n in ['PROCESS_LABEL_PROTOCOL.json','PHASE_LABEL_ADDENDUM.json','PROCESS_REVIEW_INTENT.json','PROCESS_REVIEW_MANIFEST.json','PROCESS_DESIGN_REVIEW.json']}
    out=dict(at=datetime.now(timezone.utc).isoformat(),status='DIRECT_FIT_PHYSICAL_PROCESS_TARGETS_PREPARED',
        rows=labels,queries=45,source_clips=36,registered_ids=list(range(1,10)),unique_source_frames=1354,
        direct_pages=90,counts=dict(counter),per_expression={k:dict(v) for k,v in split.items()},review_sources=reviews,pins=protocols,
        supervision='Sparse body-relation presence and directly observed motion intervals/physical sequence. Not complete sign, minimum extent, exact boundaries or required repetition truth.',
        temporal_role_not_equated_with_relationship=True,intervals_not_densified=True,source_variants_preserved=True,
        independent_human_expert=False,new_model_inference=0,new_training=0,whole_goal_complete=False)
    write(U/'PROCESS_TARGETS.json',out)
    write(U/'PROCESS_LABELS_VERIFIED.json',dict(status='PASS_PROVENANCE_AXIS_SCHEMA_WITH_PHYSICAL_SCOPE',
        label_file=str(U/'PROCESS_TARGETS.json'),label_sha256=sha(U/'PROCESS_TARGETS.json'),
        files_hash_checked=len(checked),source_clips=36,queries=45,unique_source_frames=1354,pages=90,counts=dict(counter),
        repeated_ids_not_independent_people_proof=True,verified_visual_semantics=False,independent_human_truth=False,
        new_inference=0,new_training=0,whole_goal_complete=False))
    print('Validated direct FIT process labels:',dict(counter),'files',len(checked))
if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'LABEL_VERIFICATION_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
