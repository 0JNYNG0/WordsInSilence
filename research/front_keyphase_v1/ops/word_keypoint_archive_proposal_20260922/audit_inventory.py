"""Cross-check target file locations without reading coordinate payloads."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import struct

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plan = read(HERE / 'INSPECTION_PLAN.json')
    result = read(HERE / 'INSPECTION_RESULT.json')
    groups = read(HERE / 'CLIP_DIRECTORY.json')
    coverage = read(HERE / 'TARGET_COVERAGE.json')
    index = read(HERE / 'TARGET_MEMBER_INDEX.json')
    source = read(plan['source_label_audit'])
    assert sha(plan['source_label_audit']) == plan['source_label_audit_sha256']
    archive = Path(plan['archive'])
    before = archive.stat()
    assert (before.st_size, before.st_mtime_ns) == (plan['archive_bytes'], plan['archive_mtime_ns'])
    assert sum(g['frames'] for g in groups) == result['extensions']['json']
    assert len(groups) == 15000 and len({g['word_number'] for g in groups}) == 3000
    assert {g['signer_id'] for g in groups} == {2}
    assert Counter(g['view'] for g in groups) == dict.fromkeys(['D', 'F', 'L', 'R', 'U'], 3000)
    expected = {r['clip_id']: r for r in source['source_rows'] if r['signer_id'] == 2}
    indexed = defaultdict(list)
    for member in index['members']:
        indexed[member['clip_id']].append(member)
    assert set(indexed) == set(expected) and len(indexed) == 16
    assert len(index['members']) == 1660
    for clip, members in indexed.items():
        frames = sorted(m['frame'] for m in members)
        assert frames == list(range(len(frames)))
        assert expected[clip]['split_role'] == 'train_candidate'
    matched_rows = [r for r in coverage['rows'] if r['found']]
    assert {r['clip_id'] for r in matched_rows} == set(expected)
    assert all(not r['training_approved'] and r['semantic_target_id'] is None for r in coverage['rows'])
    bytes_read = 0
    with archive.open('rb', buffering=0) as f:
        for m in index['members']:
            f.seek(m['local_offset'])
            head = f.read(30); bytes_read += len(head)
            assert len(head) == 30 and head[:4] == b'PK\x03\x04'
            flags, method = struct.unpack_from('<2H', head, 6)
            assert (flags, method) == (m['flags'], m['method'])
            nl, el = struct.unpack_from('<2H', head, 26)
            name = f.read(nl); extra = f.read(el); bytes_read += len(name) + len(extra)
            assert len(name) == nl and len(extra) == el
            assert name.decode('utf-8' if flags & 0x800 else 'cp437') == m['name']
            assert f.tell() + m['compressed'] <= plan['directory_offset']
            if not flags & 8:
                crc, compressed, uncompressed = struct.unpack_from('<3L', head, 14)
                assert crc == m['crc32']
                assert compressed in (m['compressed'], 0xffffffff)
                assert uncompressed in (m['uncompressed'], 0xffffffff)
    after = archive.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    audit = {
        'status': 'PASS_DIRECTORY_AND_TARGET_LOCAL_HEADER_AGREEMENT',
        'target_clips': 16, 'target_local_headers_checked': len(index['members']),
        'local_header_bytes_read': bytes_read,
        'archive_size_and_mtime_unchanged': True,
        'source_label_metadata_hash_unchanged': True,
        'target_frame_indices_contiguous_and_zero_based': True,
        'all_target_rows_train_signer02': True,
        'semantic_mapping_or_training_approval_changed': False,
        'coordinate_payload_bytes_read': 0, 'decompressed_members': 0,
        'compressed_payload_crc_validated': False,
        'protected_models_or_user_recordings_read': False,
        'input_artifact_hashes': {name: sha(HERE/name) for name in ['INSPECTION_PLAN.json','INSPECTION_RESULT.json','CLIP_DIRECTORY.json','TARGET_COVERAGE.json','TARGET_MEMBER_INDEX.json','inspect_archive.py','audit_inventory.py']}
    }
    with (HERE/'INDEPENDENT_AUDIT.json').open('x', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='input_artifact_hashes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
