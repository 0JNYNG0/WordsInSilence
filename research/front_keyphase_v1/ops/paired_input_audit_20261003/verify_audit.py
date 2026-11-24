"""Codex independent verification of locally executed paired audit; no camera/user inputs."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
FRONT = HERE.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    cfg = read(HERE / 'CONFIG_LOCAL_ONLY.json')
    rows = read(cfg['manifest'])['rows']
    result = read(HERE / 'result/RESULT.json')
    scores = np.load(HERE / 'result/SCORES.npz', allow_pickle=False)
    logits = scores['logits']
    classes, arms, ids = (scores[k].tolist() for k in ['classes', 'arms', 'sample_ids'])
    truth = np.asarray([classes.index(r['label']) for r in rows])
    assert logits.shape == (113, 4, 93)
    assert ids == [r['sample_id'] for r in rows]
    assert np.array_equal(truth, scores['true_indices'])
    assert len(set(classes)) == 93
    predictions = logits.argmax(axis=-1)
    correct = predictions == truth[:, None]
    rank = np.argsort(-logits, axis=-1, kind='stable')
    for i, r in enumerate(result['rows']):
        assert r['sample_id'] == ids[i] and r['true_index'] == truth[i]
        for a, name in enumerate(arms):
            saved = r['arms'][name]
            rival = np.max(np.delete(logits[i, a], truth[i]))
            assert saved['top1'] == predictions[i, a]
            assert saved['correct'] == bool(correct[i, a])
            assert saved['true_rank'] == int(np.flatnonzero(rank[i, a] == truth[i])[0]) + 1
            assert saved['true_margin'] == float(logits[i, a, truth[i]] - rival)
            assert saved['top2_gap'] == float(logits[i, a, rank[i,a,0]] - logits[i,a,rank[i,a,1]])
    counts = {a: int(correct[:, j].sum()) for j, a in enumerate(arms)}
    assert counts == result['summary']['per_arm_correct']
    for name, group in result['summary']['paired_comparisons'].items():
        first, second = [arms.index(a) for a in name.split('->')]
        masks = [('overall', None, np.ones(113, bool))]
        masks += [('per_label', v, np.array([r['label'] == v for r in rows])) for v in group['per_label']]
        masks += [('per_signer', v, np.array([str(r['signer_id']) == v for r in rows])) for v in group['per_signer']]
        for category, key, sel in masks:
            a, b = correct[sel, first], correct[sel, second]
            calculated = dict(n=int(sel.sum()), both_correct=int((a&b).sum()),
                              first_only_correct=int((a&~b).sum()), second_only_correct=int((~a&b).sum()),
                              both_wrong=int((~a&~b).sum()), prediction_same=int((predictions[sel,first] == predictions[sel,second]).sum()))
            saved = group[category] if key is None else group[category][key]
            assert calculated == saved
    independent = read(HERE/'INDEPENDENT_AVAILABILITY.json')
    total = result['summary']['availability_totals']['raw_common51_availability']['all']
    for a,b in [('slots','common_joint_slots'),('native','native_available'),('mediapipe','mp_available'),
                ('both','both'),('native_only','native_only'),('mp_only','mp_only')]:
        assert total[a] == independent[b]
    assert result['summary']['frames_total'] == 2789
    assert np.array_equal(logits[:,0], logits[:,1]), 'same dt must produce same native/time logits'
    hashes_before = read(HERE/'INPUT_HASHES_BEFORE.json')
    assert all(sha(p) == h for p,h in hashes_before.items())
    assert all(sha(p) == h for p,h in cfg['pins'].items())
    recorded_hashes = read(HERE/'result/SOURCE_HASHES.json')
    assert recorded_hashes['before'] == recorded_hashes['after']

    # Independently assemble three predetermined rows; scorer deliberately bypasses author's CLI/helpers.
    sys.path[:0] = cfg['python_paths']
    import torch
    import tools_prep_union93_candidate as u
    from pilot.data import build_channels
    from pilot.web_input import to_native_like
    torch.set_num_threads(2)
    payload = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    model, model_classes = u.rebuild_from_payload(payload)
    model.eval()
    assert model_classes == classes
    max_error = 0.0
    chosen = [0, len(rows)//2, len(rows)-1]
    for i in chosen:
        r=rows[i]
        with np.load(r['native_source']) as n, np.load(r['input_path']) as p, np.load(r['observation_path']) as o:
            assert np.array_equal(n['frame_index'],p['frame_id'])
            oi=np.searchsorted(o['frame_id'],p['frame_id'])
            assert np.array_equal(o['frame_id'][oi],p['frame_id'])
            _, pm=to_native_like(o['xy_px'][oi],o['usable'][oi])
            xs=[build_channels(n['xy_native'],n['coordinate_present'],n['timestamp_nominal_us'])['x'],
                build_channels(n['xy_native'],n['coordinate_present'],p['timestamp_us'])['x'],
                build_channels(n['xy_native'],n['coordinate_present']&pm,p['timestamp_us'])['x'],
                p['x'].copy()]
            for j,x in enumerate(xs):
                with torch.inference_mode():
                    y=model(torch.from_numpy(x[None].copy()),torch.ones((1,len(x)),dtype=torch.float32))[0].numpy()
                e=float(np.abs(y-logits[i,j]).max());max_error=max(max_error,e)
                assert e <= cfg['tolerance'], (r['sample_id'],arms[j],e)
    out={'status':'PASS','rows_all_metrics_recomputed':113,'classes':93,'arms':4,
         'counts':counts,'per_label_and_signer_comparisons_recomputed':True,
         'availability_matches_independent_source_sum':True,'timing_control_exactly_equal':True,
         'independent_model_rechecks':len(chosen)*4,'row_indices_predetermined':chosen,'max_logit_error':max_error,
         'input_files_unchanged':len(hashes_before),'pins_unchanged':len(cfg['pins']),
         'user_data':0,'camera':0,'training':0,'production_changes':0}
    (HERE/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False))

if __name__ == '__main__':
    main()
