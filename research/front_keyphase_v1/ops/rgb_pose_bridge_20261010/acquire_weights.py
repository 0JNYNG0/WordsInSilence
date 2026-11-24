"""Explicit user-approved single official weight download, no uploads."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

UNIT=Path(__file__).resolve().parent
BASE=UNIT.parents[1]
URL='https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'
DEST=BASE/'assets/rgb_encoder/mobilenet_v3_small-047dcff4.pth'


def main():
    DEST.parent.mkdir(parents=True,exist_ok=True)
    assert not (UNIT/'WEIGHTS.json').exists(), 'Already recorded; do not repeat'
    if not DEST.exists():
        partial=DEST.with_suffix('.part')
        with urlopen(URL,timeout=60) as response, partial.open('xb') as out:
            assert response.url==URL, 'Unexpected redirect'
            n=0
            while chunk:=response.read(1024*1024):
                n+=len(chunk)
                assert n<=20*1024*1024, 'Unexpected size'
                out.write(chunk)
        digest=hashlib.sha256(partial.read_bytes()).hexdigest()
        assert digest.startswith('047dcff4'), 'Official filename hash prefix mismatch'
        partial.rename(DEST)
    digest=hashlib.sha256(DEST.read_bytes()).hexdigest()
    assert digest.startswith('047dcff4')
    report={'status':'ACQUIRED_OFFICIAL_PREFIX_MATCH_FULL_HASH_PINNED', 'url':URL,
            'path':str(DEST.resolve()),'bytes':DEST.stat().st_size,'sha256':digest,
            'authorization':'AUTHORIZATION.json; user approved approximately10MB download',
            'digest_note':'Official source publishes filename prefix; full SHA256 measured locally, not an independently published full digest.',
            'scope':'Local research use; no redistribution/deployment-rights claim. Prior ENCODER_CONTRACT license review applies.'}
    with (UNIT/'WEIGHTS.json').open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__': main()
