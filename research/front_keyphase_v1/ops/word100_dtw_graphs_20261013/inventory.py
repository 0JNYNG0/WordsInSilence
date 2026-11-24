"""Correct the path-existence inventory using authoritative extraction records."""
from pathlib import Path
import json
import shutil
import hashlib
U=Path(__file__).resolve().parent; B=U.parents[1]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=read(U/'BASIS_INVENTORY.json')
assert not (U/'before/BASIS_INVENTORY_initial_path_check.json').exists()
shutil.copy2(U/'BASIS_INVENTORY.json',U/'before/BASIS_INVENTORY_initial_path_check.json')
manifest=read(B/'ops/word100_training_20261013/MANIFEST.json')
cache=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
for item in out['items']:
    rows=[x for x in manifest if x['word_code']==item['word_id'] and x['role']=='train_candidate']
    records=[read(cache/'records'/(x['clip_id']+'.json')) for x in rows]
    item['cached_observations_exist']=sum(Path(x['observation']).is_file() for x in records)
    item['observation_hashes_match']=all(sha(x['observation'])==x['observation_sha256'] for x in records)
    assert item['cached_observations_exist']==9 and item['observation_hashes_match']
out['correction']='Initial guessed directory layout was wrong; now uses each frozen extraction record, including reused cache paths.'
out['total_training_clips']=sum(x['training_clips'] for x in out['items'])
out['total_other_signer_comparisons_available']=sum(x['other_signer_comparisons_available'] for x in out['items'])
(U/'BASIS_INVENTORY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({'verified_training_clips':out['total_training_clips'],'available_pairs':out['total_other_signer_comparisons_available']}))
