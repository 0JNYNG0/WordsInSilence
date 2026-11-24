"""Diagnostic ablations, not deployable gates or candidate/threshold selection.

All four public crops receive the same predeclared transformations. The original
full-view observations are an engineering control, not human ground truth.
"""
from review import *

def main():
    target = U / 'ATTRIBUTION.json'
    assert not target.exists()
    torch.set_num_threads(4)
    plan = read(U / 'CROP_FROZEN.json')
    conditions = ['project_original_into_crop', 'crop_left_hand_wrist_missing',
                  'crop_hips_missing', 'crop_left_hand_wrist_and_hips_missing']
    write(U / 'ATTRIBUTION_FROZEN.json', dict(
        conditions=conditions, clips=[r['clip_id'] for r in plan['rows']],
        interpretation='Diagnostic interventions only. Left-removal is deliberately also applied to two-hand words; never a runtime fix.',
        no_model_or_threshold_selection=True, no_training=True,
        original_projection_is_not_gold=True, no_user_pixels_or_keypoints=True))
    model = Candidate()
    f = read(J / 'FROZEN.json')
    out = []
    for r in plan['rows']:
        z = np.load(U / (r['clip_id'] + '_crop.npz'))
        old = np.load(r['raw'])
        l,t,rr,b = r['crop']
        answers = {}
        for condition in conditions:
            xy = z['xy_px'].copy()
            score = z['score'].copy()
            if condition == 'project_original_into_crop':
                xy = (old['xy_px'] - np.array([l,t], np.float32)) * np.array([640/(rr-l),480/(b-t)],np.float32)
                score = old['score'].copy()
            if condition in ('crop_left_hand_wrist_missing','crop_left_hand_wrist_and_hips_missing'):
                xy[:, [9]+list(range(91,112))] = np.nan
            if condition in ('crop_hips_missing','crop_left_hand_wrist_and_hips_missing'):
                xy[:, [11,12]] = np.nan
            a = A.convert(xy,score,z['timestamp_us'],z['frame_id'],640,480)
            x = a['tcn280']
            with torch.inference_mode():
                logit = model.tcn(torch.from_numpy(x[None]),torch.ones(1,len(x)))[0].numpy()
            answers[condition] = dict(**ranks(logit,f,f['classes'].index(r['word_id'])), logits=logit.tolist())
        out.append(dict(clip_id=r['clip_id'], word=r['word_id'], conditions=answers))
    write(target, dict(rows=out, caveat='Four exposed development clips; diagnostic controls, not proposed runtime changes or independent performance evaluation.'))
    print(json.dumps([{ 'clip':r['clip_id'], 'ranks':{k:v['truth_rank'] for k,v in r['conditions'].items()} } for r in out]))

if __name__ == '__main__':
    main()
