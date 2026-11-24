"""Freeze the display-only scope and inventory already prepared public inputs."""
import hashlib
import json
import shutil
from pathlib import Path

U = Path(__file__).resolve().parent
B = U.parents[1]
R = B.parents[1]
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(name, value):
    (U/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', 'utf-8')

assert not (U/'SCOPE.json').exists(), 'Do not overwrite the original scope'
(U/'before').mkdir(exist_ok=True)
for name in ['practice.js','practice.css','practice.html']:
    shutil.copy2(B/'web'/name, U/'before'/name)
shutil.copy2(R/'NEW_CHAT_HANDOFF.md', U/'before/NEW_CHAT_HANDOFF.md')
guard = read(B/'ops/word100_analysis_20261013/ORIGINAL_GUARD.json')
paths = list(guard) + [str(B/'src/wis_lab/service'/p) for p in ['word100.py','word100_analysis.py','app.py']]
write('ORIGINAL_GUARD.json', {p:sha(p) for p in paths})
write('SCOPE.json', {
    'implementer':'Codex local', 'purpose':'DTW horizontal measurement graphs and criteria documentation',
    'allowed':['practice.html/css/js display','read-only public input inventory','browser QA','handoff documentation'],
    'unchanged':['TCN','DTW calculation','saved user records','reference bindings','all experiment results'],
    'success':['3 horizontal graphs with units and explicit axis limits','zero/missing/overflow distinguishable',
               'same original metric values and top5','no inferred pass line or correctness percent',
               'criteria plan separates measurement scale, normal variation and correctness threshold'],
    'display_scales':{'degrees':180,'shoulder_width_percent':100},
    'scale_meaning':'Plot extents only, not learned tolerances; numbers above extent are retained and flagged',
    'new_training':False,'camera':False,'new_user_inference':False,'new_calibration':False})
manifest=read(B/'ops/word100_training_20261013/MANIFEST.json')
practice=read(B/'config/word100_practice.json')['items']
references={x['word_id']:x for x in read(B/'config/word100_analysis_references.json')['items']}
inventory=[]
for word in practice:
    rows=[x for x in manifest if x['word_code']==word['word_id'] and x['role']=='train_candidate']
    ref=references[word['word_id']]
    cache=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
    observations=[Path(read(cache/'records'/(x['clip_id']+'.json'))['observation']) for x in rows]
    inventory.append({'word_id':word['word_id'],'label':ref['label'],
        'training_signers':sorted({x['signer_id'] for x in rows}), 'training_clips':len(rows),
        'reference_signer':ref['signer_id'],'other_signer_comparisons_available':sum(x['signer_id']!=ref['signer_id'] for x in rows),
        'cached_observations_exist':sum(p.is_file() for p in observations)})
write('BASIS_INVENTORY.json',{'source':'existing frozen manifest; training roles only', 'items':inventory,
    'note':'Existence and provenance inventory only; sign correctness/variant review and numerical baseline not executed'})
print(json.dumps({'scope':'FROZEN','inventory':inventory}, ensure_ascii=False))
