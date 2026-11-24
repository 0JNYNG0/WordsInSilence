"""Prepare twelve training-side demonstrations; do not evaluate or retrain anything."""
import hashlib
import json
from pathlib import Path
import zipfile

UNIT = Path(__file__).resolve().parent
BASE = UNIT.parents[1]
TRAIN = BASE / 'ops/word100_training_20261013'
CACHE = Path('W:/WordsInSilence/data/AIHub_103/word100_web_references_20261013')
LABELS = ['수어', '마음에들다', '멀다', '잘하다', '믿다', '즐겁다', '감사', '수영', '공항', '별', '사계절', '냉커피']

def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()
def write(p, d): Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', 'utf-8')

def main():
    f = read(TRAIN/'FROZEN_EXPERIMENT.json')
    manifest = read(TRAIN/'MANIFEST.json')
    guard = read(TRAIN/'PRODUCTION_GUARD.json')
    assert all(sha(p) == h for p, h in guard.items())
    checkpoint = BASE/'runs/word100_tcn_v1/whole_context/best.pt'
    pinned = 'e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'
    assert sha(checkpoint) == pinned
    policy = dict(words=LABELS, selection='Everyday practice suitability judgment, not a frequency ranking or score selection.',
                  performer='lowest available training signer ID; no development or heldout clips',
                  full_classifier=100, no_retraining=True, reference_is_training_example_not_accuracy_test=True,
                  new_web='Explicit word100 research practice in existing server; legacy default model unchanged',
                  user_authorization='2026-10-13 user: use everyday words within the learned100 and downloaded videos; improve side-by-side GUI/UX.')
    # Save choice BEFORE reading videos; selection never reads per-class evaluation results.
    if not (UNIT/'SCOPE.json').exists(): write(UNIT/'SCOPE.json', policy)
    else: assert read(UNIT/'SCOPE.json') == policy
    CACHE.mkdir(parents=True, exist_ok=True)
    items = []
    archives = {}
    try:
        for label in LABELS:
            r = min((r for r in manifest if r['canonical_label'] == label and r['role'] == 'train_candidate'),
                    key=lambda r: (r['signer_id'], r['clip_id']))
            assert 1 <= r['signer_id'] <= 9
            e = r['video_entries'][0]
            record = read(Path(f['output'])/'records'/f"{r['clip_id']}.json")
            path = CACHE/f"{r['clip_id']}.mp4"
            if not path.exists():
                z = archives.setdefault(e['archive'], None)
                if z is None: z = archives[e['archive']] = zipfile.ZipFile(e['archive'])
                info = z.getinfo(e['member'])
                assert info.file_size == e['bytes'] and info.CRC == e['crc32']
                data = z.read(e['member'])
                assert hashlib.sha256(data).hexdigest() == record['source_sha256']
                with path.open('xb') as out: out.write(data)
            assert path.stat().st_size == record['source_bytes'] and sha(path) == record['source_sha256']
            items.append(dict(word_id=r['word_code'], label=label,
                              display_name='마음에 들다' if label == '마음에들다' else label,
                              clip_id=r['clip_id'], signer_id=r['signer_id'], source_role='train_candidate',
                              video_path=str(path), source_sha256=record['source_sha256'],
                              source_archive=e['archive'], source_member=e['member'],
                              duration_ms=round(r['duration_metadata']*1000), observation=record['observation']))
    finally:
        for z in archives.values():
            if z is not None: z.close()
    config = dict(schema='wis_word100_practice_v1', model_key='word100_tcn_context_v1',
                  checkpoint=str(checkpoint), checkpoint_sha256=pinned,
                  word_codes=f['classes'], labels=f['labels'], items=items,
                  reference_set_id='aihub_word100_training_demo12_v1')
    write(BASE/'config/word100_practice.json', config)
    write(UNIT/'REFERENCE_CHECK.json', dict(status='PASS', count=len(items),
          bytes=sum(Path(i['video_path']).stat().st_size for i in items),
          identical_to_prepared_source=True, signer_ids=sorted({i['signer_id'] for i in items}),
          development_or_heldout_references=0, classifier_classes=100))
    print(json.dumps(read(UNIT/'REFERENCE_CHECK.json')))

if __name__ == '__main__': main()
