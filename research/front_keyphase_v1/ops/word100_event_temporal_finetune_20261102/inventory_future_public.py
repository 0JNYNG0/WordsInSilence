"""Inventory only. No pixels, extraction, model inference, or final-test claim."""
from pathlib import Path
import json,zipfile,hashlib
U=Path(__file__).resolve().parent
ROOT=Path('W:/WordsInSilence/data/AIHub_103')
WORDS={'WORD0739','WORD1133','WORD1567','WORD2523'}
def main():
    out=U/'FUTURE_PUBLIC_INVENTORY.json';assert not out.exists()
    metadata=ROOT/'inventory_20261011/FRONT_MANIFEST.jsonl'
    rows=[]
    for line in metadata.open(encoding='utf8'):
        if not any(w in line for w in WORDS):continue
        r=json.loads(line)
        if r['filename_signer_id'] in [17,18]:rows.append(r)
    assert len(rows)==8 and all(r['metadata_clean'] for r in rows)
    archive=ROOT/'videos/수어 영상/2.Validation/[원천]01_real_word_video.zip'
    matches={r['clip_id']:[] for r in rows}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            stem=Path(info.filename).stem
            if stem in matches and info.filename.lower().endswith('.mp4'):
                matches[stem].append(dict(member=info.filename,bytes=info.file_size,crc32=info.CRC))
    result=dict(scope='Read-only public archive central directory and old public metadata, no pixels or model outputs.',
        archive=str(archive),archive_size=archive.stat().st_size,
        rows=[dict(clip_id=r['clip_id'],word_id=r['word_code'],recorded_id=r['filename_signer_id'],metadata_labels=r['labels'],matches=matches[r['clip_id']]) for r in rows],
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),downloaded=0,extracted=0,inference=0,
        independent_evaluation_eligibility='UNVERIFIED. Before use: audit previous input/training/evaluation manifests and duplicate bytes; establish frozen protocol and available physical ground truth. Recorded IDs are not visual proof of distinct people.',
        locked_14_15_media_access=False,id16_quarantine_unchanged=True,whole_goal_complete=False)
    with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('Public metadata candidates8; archive exact matches',sum(len(v) for v in matches.values()),';no extraction or inference')
if __name__=='__main__':main()
