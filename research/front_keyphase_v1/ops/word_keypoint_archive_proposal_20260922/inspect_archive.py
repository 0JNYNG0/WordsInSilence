"""Inventory the one authorized archive. Read ZIP metadata only; never extract."""
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import struct
import time

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
ARCHIVE = Path(r'C:\Users\local-user\Downloads\수어 영상\1.Training\[라벨]02_real_word_keypoint.zip')
READ = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, value):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    request = READ(HERE / 'REQUEST.json')
    assert request['download_approved'] and request['allowed_archive_count'] == 1
    assert request['file']['archive_name'] == '02_real_word_keypoint.zip'
    assert ARCHIVE.is_file()
    before = ARCHIVE.stat()
    assert 1_000_000 < before.st_size <= 10 * 1024**3
    assert not list(ARCHIVE.parent.glob(ARCHIVE.name + '.irx*')), 'Transfer still active'
    source_pin = READ(LAB / 'ops/native_context_readiness_20260921/coordinate_plan.json')
    reader_path = next(Path(p) for p in source_pin['source_pins'] if p.endswith('zip_probe.py'))
    assert SHA(reader_path) == source_pin['source_pins'][str(reader_path)]
    spec = importlib.util.spec_from_file_location('existing_zip_metadata_reader', reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    evidence = READ(LAB / 'evidence/word_input_audit_20260918.json')
    source_audit_path = Path(evidence['audit'])
    source = READ(source_audit_path)
    rows = source['source_rows']
    assert len(rows) == 256
    wanted = {r['clip_id']: r for r in rows}
    assert len(wanted) == len(rows)
    families = defaultdict(list)
    for row in rows:
        families[row['word_number']].append(row)
    entries, directory_bytes, offset = reader.directory_location(ARCHIVE)
    write('INSPECTION_PLAN.json', {
        'at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'archive': str(ARCHIVE), 'archive_bytes': before.st_size,
        'archive_mtime_ns': before.st_mtime_ns,
        'directory_entries': entries, 'directory_bytes': directory_bytes,
        'directory_offset': offset, 'reader': str(reader_path), 'reader_sha256': SHA(reader_path),
        'source_label_audit': str(source_audit_path), 'source_label_audit_sha256': SHA(source_audit_path),
        'request_sha256': SHA(HERE / 'REQUEST.json'),
        'scope': 'Central-directory metadata only, all archive groups and existing 256 source-label rows. No decompression, coordinate payload, training, inference or new download.'
    })
    pattern = re.compile(rb'(NIA_SL_WORD(\d+)_REAL(\d+)_([A-Z]))_(\d+)_keypoints\.json$')
    groups = {}
    extensions = Counter()
    methods = Counter()
    unmatched_examples = []
    selected_members = []
    digest = hashlib.sha256()
    total_compressed = total_uncompressed = encrypted = unmatched_files = folders = 0
    started = time.monotonic()
    with ARCHIVE.open('rb', buffering=1024 * 1024) as f:
        f.seek(offset)
        for _ in range(entries):
            head = f.read(46)
            assert len(head) == 46 and head[:4] == b'PK\x01\x02', 'Invalid central header'
            nl, el, cl = struct.unpack_from('<3H', head, 28)
            name, extra, comment = f.read(nl), f.read(el), f.read(cl)
            assert len(name) == nl and len(extra) == el and len(comment) == cl
            digest.update(head + name + extra + comment)
            flags, method = struct.unpack_from('<2H', head, 8)
            crc, compressed, uncompressed = struct.unpack_from('<3L', head, 16)
            local = struct.unpack_from('<L', head, 42)[0]
            cursor = 0
            while cursor + 4 <= len(extra):
                tag, length = struct.unpack_from('<2H', extra, cursor)
                payload = extra[cursor+4:cursor+4+length]
                assert len(payload) == length
                cursor += 4 + length
                if tag == 1:
                    pos = 0
                    if uncompressed == 0xffffffff:
                        uncompressed = struct.unpack_from('<Q', payload, pos)[0]; pos += 8
                    if compressed == 0xffffffff:
                        compressed = struct.unpack_from('<Q', payload, pos)[0]; pos += 8
                    if local == 0xffffffff:
                        local = struct.unpack_from('<Q', payload, pos)[0]
            total_compressed += compressed
            total_uncompressed += uncompressed
            encrypted += bool(flags & 1)
            methods[method] += 1
            if name.endswith(b'/'):
                folders += 1
                continue
            extensions[name.rsplit(b'.', 1)[-1].decode('ascii', errors='replace')] += 1
            match = pattern.search(name.rsplit(b'/', 1)[-1])
            if not match:
                unmatched_files += 1
                if len(unmatched_examples) < 5:
                    unmatched_examples.append(name.decode('utf-8' if flags & 0x800 else 'cp437'))
                continue
            clip, word, signer, view, frame = match.groups()
            clip = clip.decode('ascii'); frame = int(frame)
            assert 0 <= frame < 100000, 'Unexpected frame index'
            if clip not in groups:
                groups[clip] = {'clip_id': clip, 'word_number': word.decode('ascii'), 'signer_id': int(signer), 'view': view.decode('ascii'), 'frames': 0, 'bits': 0, 'duplicate_frames': 0, 'first_frame': frame, 'last_frame': frame, 'compressed_bytes': 0, 'uncompressed_bytes': 0}
            g = groups[clip]
            bit = 1 << frame
            g['duplicate_frames'] += bool(g['bits'] & bit)
            g['bits'] |= bit
            g['frames'] += 1
            g['first_frame'] = min(g['first_frame'], frame)
            g['last_frame'] = max(g['last_frame'], frame)
            g['compressed_bytes'] += compressed
            g['uncompressed_bytes'] += uncompressed
            if clip in wanted:
                selected_members.append({'clip_id': clip, 'name': name.decode('utf-8' if flags & 0x800 else 'cp437'), 'frame': frame, 'flags': flags, 'method': method, 'crc32': crc, 'compressed': compressed, 'uncompressed': uncompressed, 'local_offset': local})
        assert f.tell() == offset + directory_bytes, 'Central-directory size mismatch'
    after = ARCHIVE.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    for g in groups.values():
        unique = g.pop('bits').bit_count()
        g['unique_frames'] = unique
        g['frame_indices_contiguous'] = unique == g['last_frame'] - g['first_frame'] + 1 and not g['duplicate_frames']
    by_signer = Counter()
    by_view = Counter()
    for g in groups.values():
        by_signer[g['signer_id']] += 1
        by_view[g['view']] += 1
    def role(n):
        return 'train_candidate' if 1 <= n <= 9 else 'development_only' if 11 <= n <= 13 else 'reserved_test_metadata_only' if 14 <= n <= 16 else 'unassigned_or_excluded'
    matches = []
    for row in rows:
        g = groups.get(row['clip_id'])
        matches.append({k: row[k] for k in ['clip_id', 'word_number', 'signer_id', 'raw_label', 'split_role', 'target_candidates_not_approved']} | {'found': g is not None, 'frames': g['frames'] if g else 0, 'semantic_target_id': None, 'training_approved': False})
    matched = [r for r in matches if r['found']]
    family_summary = []
    for word, label_rows in sorted(families.items()):
        found = [r for r in matched if r['word_number'] == word]
        family_summary.append({'word_number': word, 'raw_labels': sorted({r['raw_label'] for r in label_rows}), 'target_candidates_not_approved': sorted({v for r in label_rows for v in r['target_candidates_not_approved']}), 'front_signers_found': sorted(r['signer_id'] for r in found), 'train_signers_found': sorted(r['signer_id'] for r in found if r['split_role'] == 'train_candidate'), 'dev_signers_found': sorted(r['signer_id'] for r in found if r['split_role'] == 'development_only')})
    write('CLIP_DIRECTORY.json', sorted(groups.values(), key=lambda g:g['clip_id']))
    write('TARGET_MEMBER_INDEX.json', {'archive': str(ARCHIVE), 'members': selected_members, 'payloads_read': 0})
    write('TARGET_COVERAGE.json', {'families': family_summary, 'rows': matches, 'semantic_mapping_approved': False})
    result = {
        'status': 'DIRECTORY_INSPECTION_COMPLETE_PAYLOAD_AND_TRAINING_UNVERIFIED',
        'at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'archive': str(ARCHIVE), 'archive_bytes': before.st_size, 'archive_mtime_ns': before.st_mtime_ns,
        'directory_entries': entries, 'directory_bytes': directory_bytes, 'directory_sha256': digest.hexdigest(),
        'folders': folders, 'extensions': dict(extensions), 'compression_methods': dict(methods), 'encrypted_entries': encrypted,
        'compressed_member_bytes': total_compressed, 'uncompressed_member_bytes': total_uncompressed,
        'clip_groups': len(groups), 'unique_word_ids': len({g['word_number'] for g in groups.values()}),
        'clip_groups_by_signer': dict(sorted(by_signer.items())), 'clip_groups_by_view': dict(by_view),
        'signer_roles': {n: role(n) for n in sorted(by_signer)},
        'unmatched_files': unmatched_files, 'unmatched_examples': unmatched_examples,
        'duplicate_frame_groups': sum(bool(g['duplicate_frames']) for g in groups.values()),
        'noncontiguous_frame_groups': sum(not g['frame_indices_contiguous'] for g in groups.values()),
        'matched_existing_label_rows': len(matched), 'matched_families': sum(bool(f['front_signers_found']) for f in family_summary),
        'matched_rows_by_split': dict(Counter(r['split_role'] for r in matched)),
        'target_front_frames': len(selected_members), 'family_summary': family_summary,
        'directory_scan_seconds': round(time.monotonic()-started,3),
        'source_payloads_read': 0, 'reserved_payloads_read': 0, 'extracted_files': 0, 'training_runs': 0,
        'limits': ['Central directory identifies packaged clips, not actual coordinate quality or semantic equivalence.', 'CRC/decompression/full archive hash not performed; downloaded bytes and directory structure are verified.', 'Do not infer signer from archive number or treat views/frames as separate people.', 'Existing signer split and all18 targets remain unchanged; no automatic follow-on download.']
    }
    write('INSPECTION_RESULT.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['family_summary','limits']},ensure_ascii=False), flush=True)
    print(json.dumps(family_summary,ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
