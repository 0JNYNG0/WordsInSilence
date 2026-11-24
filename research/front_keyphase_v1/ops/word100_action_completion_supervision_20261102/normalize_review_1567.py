"""Lossless schema adapter for the direct WORD1567 visual annotation.

Only copies saved annotation values. No media, model, old target, or prediction
is loaded. The original annotation and the root verifier remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

UNIT = Path(__file__).resolve().parent
SOURCE = UNIT / 'REVIEW_1567.json'
OUTPUT = UNIT / 'REVIEW_1567_COMPAT.json'
VERIFIED = UNIT / 'REVIEW_1567_COMPAT_VERIFIED.json'
EXPECTED_SOURCE_SHA256 = '0ccf4c969a252c04cb34e2a9664fdd24f840a3228ca20a8bcbc8315a981016db'
REQUIRED = {
    'id', 'clip_id', 'word_id', 'person', 'source_sha256', 'physical_query',
    'reviewed_pages', 'predicate_definitions', 'relations', 'motion_intervals',
    'sequence_links', 'review_bounds_sufficient', 'whole_word_correctness',
    'required_repetition_count',
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def preserve_or_write(path, data):
    if path.exists():
        assert path.read_bytes() == data, f'Refusing to replace different output: {path}'
    else:
        path.write_bytes(data)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def leaf_count(value):
    if isinstance(value, dict):
        return sum(leaf_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(leaf_count(v) for v in value)
    return 1


def main():
    raw = SOURCE.read_bytes()
    source_hash = sha_bytes(raw)
    assert source_hash == EXPECTED_SOURCE_SHA256
    source = json.loads(raw.decode('utf-8-sig'))
    assert 'rows' not in source and len(source['records']) == 18
    assert source['label_contract']['linguistic_correctness'] is None
    assert source['label_contract']['required_cycle_count'] is None

    # Retain every original top-level field, including the original records.
    # The new rows key is a deep copy of records, plus explicit null aliases.
    output = deepcopy(source)
    output['rows'] = []
    for index, record in enumerate(source['records']):
        row = deepcopy(record)
        row['whole_word_correctness'] = source['label_contract']['linguistic_correctness']
        row['required_repetition_count'] = source['label_contract']['required_cycle_count']
        row['annotation_lineage'] = {
            'path': str(SOURCE),
            'sha256': source_hash,
            'source_records_key': 'records',
            'source_record_index': index,
            'source_record_id': record['id'],
        }
        assert REQUIRED <= row.keys()
        assert row['word_id'] == 'WORD1567' and 1 <= row['person'] <= 9
        assert all(row[key] == value for key, value in record.items())
        output['rows'].append(row)

    output['normalization'] = {
        'type': 'lossless_saved_annotation_schema_adapter',
        'source_path': str(SOURCE),
        'source_sha256': source_hash,
        'original_annotation_unchanged': True,
        'original_top_level_fields_preserved': True,
        'original_records_preserved': True,
        'field_mapping': {
            'rows': 'records (deep copy; same order)',
            'rows[i].<each original record key>': 'records[i].<same key>, recursively unchanged',
            'rows[i].whole_word_correctness': 'label_contract.linguistic_correctness (null; no judgment asserted)',
            'rows[i].required_repetition_count': 'label_contract.required_cycle_count (null; no count asserted)',
            'rows[i].annotation_lineage': 'source file path/SHA and exact original records index/id',
        },
        'scope': 'Naming/schema compatibility only; no label interpretation, regrading, scoring, densification, or new visual review.',
        'root_verifier_modified': False,
        'root_verifier_input_note': 'Root must select REVIEW_1567_COMPAT.json for WORD1567; the original filename remains unchanged.',
        'model_forwards': 0,
        'new_training': 0,
    }
    payload = encoded(output)
    preserve_or_write(OUTPUT, payload)

    # Verify the serialized output, not only the in-memory builder.
    reread = json.loads(OUTPUT.read_text(encoding='utf-8'))
    assert all(reread[key] == value for key, value in source.items())
    source_leaves = 0
    for original, row in zip(source['records'], reread['rows']):
        assert REQUIRED <= row.keys()
        assert all(row[key] == value for key, value in original.items())
        assert row['whole_word_correctness'] is None
        assert row['required_repetition_count'] is None
        source_leaves += leaf_count(original)
    assert len(reread['rows']) == len(source['records'])
    assert sha_bytes(SOURCE.read_bytes()) == source_hash
    check = {
        'status': 'PASS_LOSSLESS_SCHEMA_MAPPING',
        'source_path': str(SOURCE),
        'source_sha256_before': source_hash,
        'source_sha256_after': sha_bytes(SOURCE.read_bytes()),
        'output_path': str(OUTPUT),
        'output_sha256': sha_bytes(OUTPUT.read_bytes()),
        'normalizer_path': str(Path(__file__).resolve()),
        'normalizer_sha256': sha_bytes(Path(__file__).read_bytes()),
        'source_records': len(source['records']),
        'output_rows': len(reread['rows']),
        'source_record_leaf_values_preserved': source_leaves,
        'source_top_level_fields_preserved': len(source),
        'unmapped_original_fields': 0,
        'dropped_records': 0,
        'changed_original_values': 0,
        'added_null_aliases': 36,
        'annotation_lineage_records': 18,
        'root_verifier_modified': False,
        'root_verifier_executed': False,
        'new_model_inference': 0,
        'new_training': 0,
        'verified_visual_semantics': False,
    }
    preserve_or_write(VERIFIED, encoded(check))
    print(json.dumps(check, ensure_ascii=False))


if __name__ == '__main__':
    main()
