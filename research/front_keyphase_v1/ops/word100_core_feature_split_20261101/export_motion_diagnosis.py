"""Export retrospective ordered evidence; never replace original truth or gate."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from collections import Counter
from motion_sequence import U, read, sha, write, encode, COMPONENTS

def main():
    source = U/'DATASET.json'
    dataset = read(source)
    by_id = {r['sample_id']: r for r in dataset['rows']}
    first_file = U/'review/MOTION_DISAGREEMENT_FIRST.json'
    first = read(first_file)
    expected = {(r['id'], r['hand']) for r in first}
    assert len(expected) == 28
    reviews = [U/'review/MOTION_CAUSE_A.json', U/'review/MOTION_CAUSE_B.json']
    hashes = {str(p):sha(p) for p in [source, first_file, U/'LABEL_AUDIT.json', U/'ORDERED_MOTION_CONTRACT.json', U/'motion_sequence.py', *reviews]}
    causes, output, seen, counts = Counter(), [], set(), Counter()
    for file in reviews:
        review = read(file)
        assert 'IN_PROGRESS' not in review['status'].upper()
        records=review.get('rows',review.get('records'))
        assert len(records) == 14
        for record in records:
            key = record['sample_id'], record['side']
            assert key in expected and key not in seen
            seen.add(key)
            row = by_id[key[0]]
            if 'frame_ids' in record:
                assert record['frame_ids'] == row['frame_ids']
                assert record['timestamps_us'] == row['timestamps_us']
                image,image_hash=record['image'],record['image_sha256']
            else:
                assert record['frame_range']==[row['frame_ids'][0],row['frame_ids'][-1]]
                assert len(record['viewed_images'])==1
                image,image_hash=record['viewed_images'][0]['path'],record['viewed_images'][0]['sha256']
            assert Path(image).resolve() == Path(row['image']).resolve()
            assert sha(image) == image_hash
            record_causes=record.get('plausible_causes')
            if record_causes is None:
                record_causes=[record['category_disagreement_cause'],*record['secondary_causes']]
            out = encode(row, key[1], record['events'])
            out.update(source_review=str(file), source_review_sha256=hashes[str(file)],
                original_primary_secondary_categories={k: next(r for r in first if (r['id'],r['hand'])==key)[k] for k in ('a','b')},
                diagnostic_causes=record_causes,
                unresolved=record.get('unresolved',record.get('uncertain')),
                original_model_roi_and_missing_masks_unchanged=True,
                original_image=image, original_image_sha256=image_hash)
            output.append(out)
            causes.update(set(record_causes))
            counts['hand_windows'] += 1
            counts['original_hand_frames'] += len(row['frame_ids'])
            for c in COMPONENTS:
                counts[f'{c}_direct_frame_cells'] += sum(t['components'][c]['direct'] for t in out['timeline'])
                counts[f'{c}_unknown_frame_cells'] += sum(not t['components'][c]['direct'] for t in out['timeline'])
            distinct = {r['value'] for r in out['ordered_runs']['translation'] if r['value'] is not None}
            counts['multiple_direct_translation_states'] += len(distinct)>1
    assert seen == expected and counts['original_hand_frames'] == 364
    assert all(sha(p)==h for p,h in hashes.items())
    assert read(U/'LABEL_AUDIT.json')['status']=='NOT_READY_SUPPORT_OR_REVIEW'
    result = dict(status='COMPLETE_RETROSPECTIVE_SEQUENCE_EVIDENCE_NOT_TRAINING_TRUTH',
        source_pins=hashes, counts=dict(counts), overlapping_cause_counts=dict(causes),
        selected_error_cases_not_population_sample=True,
        same_codex_review_not_independent_human_truth=True,
        original_gate_remains_failed=True, original_dataset_and_labels_modified=False,
        unseen_segments_not_declared_unimportant=True, training_started=False,
        performance_improvement_claim=None, rows=output)
    write(U/'MOTION_SEQUENCE_DIAGNOSIS.json',result)
    print(__import__('json').dumps({k:v for k,v in result.items() if k not in ('rows','source_pins')},ensure_ascii=False))

if __name__=='__main__':
    main()
