"""Check saved WORD1567 partial-review provenance and witness axis only.

No model, prediction, old target, media decoder, or visual grading is invoked.
This cannot certify the semantic correctness of the Codex visual annotations.
"""
from pathlib import Path
from collections import Counter
import json
import hashlib

U = Path(__file__).resolve().parent
REPORT = U / 'REVIEW_PARTIAL_1567.json'
MANIFEST = U / 'PARTIAL_VIEWS_MANIFEST.json'
OUT = U / 'REVIEW_PARTIAL_1567_VERIFIED.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a, m = read(REPORT), read(MANIFEST)
    assert sha(MANIFEST) == a['manifest_sha256']
    meta = {v['view_id']: v for v in m['views']
            if 'WORD1567' in v['query_id'] and
            v['construction_kind_audit_only'] in ['prefix', 'suffix', 'middle_splice']}
    assert len(meta) == len(a['rows']) == 54
    assert set(meta) == {r['view_id'] for r in a['rows']}
    assert len({r['view_id'] for r in a['rows']}) == 54
    pages, extras, sources, queries = {}, {}, set(), set()
    counts, frame_slots, motion_sets = Counter(), 0, 0
    for r in a['rows']:
        q = meta[r['view_id']]
        for key in ['query_id', 'physical_query', 'pixel_view_key', 'clip_id',
                    'recorded_person_id', 'source_sha256']:
            assert r[key] == q[key], (r['view_id'], key)
        assert 1 <= r['recorded_person_id'] <= 9
        sources.add(r['clip_id']); queries.add(r['query_id'])
        assert r['reviewed_pages'] == [dict(path=p['path'], sha256=p['sha256'])
                                      for p in q['review_pages']]
        for p in r['reviewed_pages']:
            if p['path'] not in pages:
                assert sha(p['path']) == p['sha256']
                pages[p['path']] = p['sha256']
        axis = {f['output_index']: f for f in q['ordered_frame_map']}
        assert r['reviewed_output_indices'] == list(axis)
        assert list(axis) == list(range(len(axis)))
        frame_slots += len(axis)
        assert r['seam_output_indices'] == q['seam_output_indices']
        assert r['observed_process_evidence'] in [0, 1, None]
        assert r['visible_chain'] in ['yes', 'partial', 'uncertain']
        assert r['evidence_reason']
        for key in ['whole_word_correctness', 'learner_error', 'required_repetitions']:
            assert r[key] is None
        counts[str(r['observed_process_evidence'])] += 1
        for key in ['setup_witnesses', 'terminal_witnesses']:
            for w in r[key]:
                assert w['output_index'] in axis
                f = axis[w['output_index']]
                assert f['source_native_frame'] == w['source_frame']
                assert f['image_available'] and f['pixel_operation'] == 'copy'
        for w in r['motion_witnesses']:
            fs, ix = w['source_frames'], w['output_indices']
            assert len(fs) == len(ix) and len(fs) >= 2
            assert all(b == a + 1 for a, b in zip(fs, fs[1:]))
            assert all(b == a + 1 for a, b in zip(ix, ix[1:]))
            assert not set(ix[1:]) & set(q['seam_output_indices'])
            for f, i in zip(fs, ix):
                assert axis[i]['source_native_frame'] == f
                assert axis[i]['image_available'] and axis[i]['pixel_operation'] == 'copy'
            assert any(lo <= min(fs) <= max(fs) <= hi
                       for lo, hi in q['retained_contiguous_native_fragments'])
            motion_sets += 1
        assert len(r['remaining_evidence_by_fragment']) == len(q['retained_contiguous_native_fragments'])
        for item, bounds in zip(r['remaining_evidence_by_fragment'], q['retained_contiguous_native_fragments']):
            assert item['source_frame_bounds'] == bounds
            lo, hi = bounds
            expected = [i for i, f in axis.items() if lo <= f['source_native_frame'] <= hi]
            assert item['output_indices'] == expected and item['observation']
        if r['observed_process_evidence'] == 1:
            assert r['visible_chain'] == 'yes' and r['motion_witnesses']
            assert r['setup_witnesses'] and r['terminal_witnesses']
            ok = False
            for lo, hi in q['retained_contiguous_native_fragments']:
                for w in r['motion_witnesses']:
                    fs = w['source_frames']
                    setups = [x['source_frame'] for x in r['setup_witnesses'] if lo <= x['source_frame'] <= min(fs)]
                    terminals = [x['source_frame'] for x in r['terminal_witnesses'] if max(fs) <= x['source_frame'] <= hi]
                    ok |= lo <= min(fs) <= max(fs) <= hi and bool(setups) and bool(terminals)
            assert ok, 'Positive must have one native-fragment witness chain'
        for extra in r['additional_native_images_reviewed']:
            assert extra['clip_id'] == r['clip_id']
            assert any(f['source_native_frame'] == extra['source_frame'] for f in axis.values())
            if extra['path'] not in extras:
                assert sha(extra['path']) == extra['sha256']
                extras[extra['path']] = extra['sha256']
    assert len(pages) == 67 and len(sources) == 9 and len(queries) == 18
    assert frame_slots == 972 and len(extras) == 1
    assert counts == {'0': 50, '1': 4}
    report = dict(status='PASS_SAVED_PROVENANCE_SCHEMA_AND_WITNESS_AXIS',
                  review_path=str(REPORT), review_sha256=sha(REPORT),
                  manifest_path=str(MANIFEST), manifest_sha256=sha(MANIFEST),
                  verifier_path=str(Path(__file__).resolve()), verifier_sha256=sha(__file__),
                  views=54, native_queries=18, dependent_source_clips=9,
                  pages_hash_verified=67, additional_native_image_hash_verified=1,
                  reviewed_output_frame_slots=972, label_counts=dict(counts),
                  motion_witness_sets_checked=motion_sets,
                  missing_views=0, missing_reviewed_output_indices=0,
                  positive_chains_crossing_seam=0,
                  unviewed_pages_claimed_by_report=0,
                  visual_viewing_claim_basis='Reviewer tool-session record; checksum checks alone do not prove visual inspection.',
                  verified_visual_semantics=False, independent_human_expert=False,
                  new_model_inference=0, new_training=0)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
