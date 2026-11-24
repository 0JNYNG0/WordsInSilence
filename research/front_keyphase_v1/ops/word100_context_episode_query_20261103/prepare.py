"""Freeze one new FIT scope diagnostic; never relabel/re-score older experiments."""
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
import av
import numpy as np
from PIL import Image, ImageDraw

U = Path(__file__).resolve().parent
O = U.parent
F = O / 'word100_direct_event_context_20261102'
P = O / 'word100_learned_guide_feedback_20261102'
R = O / 'word100_learned_rgb_motion_20261102'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def exclusive(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def prepare():
    assert not (U / 'FROZEN.json').exists(), 'Do not repeat preparation or new scope selection'
    audit = read(F / 'NATIVE_TIME_AUDIT.json')
    inputs = read(P / 'INPUT_MANIFEST.json')['rows']
    broad = {1: [42,81], 4: [65,103], 7: [35,73]}
    late = {1: [76,81], 4: [99,103], 7: [71,73]}
    rows = []
    for person in [1,4,7]:
        cid = f'NIA_SL_WORD1133_REAL{person:02d}_F'
        a = next(r for r in audit['rows'] if r['clip_id'] == cid)
        m = next(r for r in inputs if r['clip_id'] == cid)
        assert m['role'] == 'fit' and m['person'] == person
        assert sha(a['source']) == a['source_sha256']
        assert sha(m['input']) == a['native_input_sha256']
        rows.append(dict(clip_id=cid, person=person, word_id=m['word_id'], role='fit',
                         source=a['source'], source_sha256=a['source_sha256'],
                         input=m['input'], input_sha256=sha(m['input']),
                         broad_frames=broad[person], late_frames=late[person]))
    pins = [F/'CONTEXT_LABELS.json', F/'CONTEXT_VERIFIED.json', F/'NATIVE_TIME_AUDIT.json',
            F/'CONTEXT_1133.json', R/'runs/rgb_residual_42/model.pt', R/'RGB_NORMALIZATION.json',
            O/'word100_learned_motion_order_20261102/NORMALIZATION.json']
    contract = dict(created_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn='no progress: answered clarification; now execute next available safe action',
        purpose='Query learned motion occurrence inside a body-defined episode; do not gate a required motion by whether that same motion was detected',
        status='FROZEN_NEW_FIT_DIAGNOSTIC_NOT_ADOPTION_TEST', rows=rows,
        reviewed_scope_provenance='AF three actual 1133 body-context sheets and predicate reviewer late candidates; no probability-driven scope selection',
        local_review='Exactly 14 original consecutive frames in three fixed late windows; may establish absence of substantial arm-relative traverse only within each window. No minimum-duration semantic claim.',
        comparison='Frozen RGB residual seed42, one forward batch of six scopes: broad and late for each of three FIT records. Existing current-frame observations and original timestamps. No old completed evaluation rerun.',
        new_adapter='Generic explicit episode interval input with target index; output scope, probabilities, missingness, status. Target description is not supplied to the observation model.',
        restrictions=['No training, weight changes, threshold search, max-score scope search, first-terminal cutoff, dense context labels, old-label edits, DEV/holdout/private access, camera, download, external transfer, Git, web replacement',
                      'Existing S interval zeros overlap positive sequence tails and are not clean hold truth',
                      'No completion, source-clause correctness, educational correction, or generalization claim from these FIT scopes',
                      'Image context inherited from cached SignRep must be disclosed; cropped feature interval does not necessarily crop original RGB receptive field'],
        gates=dict(structural='Exact native interval; no missing rows removed; no target-shape-dependent truncation; explicit insufficient/context estimate states; unchanged model/source pins',
                   diagnostic='Report all six once. Late labels remain null until actual consecutive RGB reviewed. Short windows cannot prove complete-stroke omission sensitivity.',
                   integration='No live adoption from this diagnostic alone; no forced improvement statement'),
        pins={str(p):sha(p) for p in pins}, whole_goal_complete=False)
    exclusive(U/'FROZEN.json', contract)
    review = []
    for r in rows:
        lo, hi = r['late_frames']
        a = np.load(r['input'])
        frames = []
        with av.open(r['source']) as container:
            for i, f in enumerate(container.decode(video=0)):
                if i > hi:
                    break
                if i < lo:
                    continue
                us = round(float(f.pts * f.time_base) * 1e6)
                assert int(a['frame_id'][i]) == i and int(a['timestamp_us'][i]) == us
                im = f.to_image().convert('RGB')
                im.thumbnail((640,480))
                frames.append((i,us,im))
        assert [v[0] for v in frames] == list(range(lo,hi+1))
        sheet = Image.new('RGB', (640*3,520*((len(frames)+2)//3)), 'white')
        d = ImageDraw.Draw(sheet)
        for j,(fid,us,im) in enumerate(frames):
            x,y=(j%3)*640,(j//3)*520
            sheet.paste(im,(x,y+36))
            d.text((x+8,y+8),f"{r['clip_id']} f{fid} native {us/1e6:.6f}s",fill='black')
        dest = U / f"{r['clip_id']}_late.png"
        assert not dest.exists()
        sheet.save(dest)
        review.append(dict(clip_id=r['clip_id'],frames=[v[0] for v in frames],
                           timestamp_us=[v[1] for v in frames],path=str(dest),sha256=sha(dest)))
    exclusive(U/'REVIEW_READY.json', dict(status='14_NATIVE_FRAMES_READY_NOT_YET_REVIEWED',
                                         rows=review, frozen_sha256=sha(U/'FROZEN.json')))
    print(json.dumps(dict(scopes=6,clips=3,native_review_frames=14,model_inference=0)))

if __name__ == '__main__':
    prepare()
