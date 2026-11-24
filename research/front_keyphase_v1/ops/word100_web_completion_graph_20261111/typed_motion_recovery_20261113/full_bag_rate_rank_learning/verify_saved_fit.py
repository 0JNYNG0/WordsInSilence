"""Independent saved-file audit; root launches only after L completion.

No model object, forward pass, optimizer, feature generation, or evaluation run.
Torch is used solely to deserialize pinned CPU state dictionaries and compare
their normalization buffers. K's RNG is replayed as label/index arithmetic.
"""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import time
import traceback
import numpy as np

L = Path(__file__).resolve().parent
H = L.parent
K = H / 'temporal_rate_rank_learning'
J = H / 'splice_rank_learning'
OUT = L / 'INDEPENDENT_SAVED_FIT_VERIFICATION.json'
DETAIL = L / 'INDEPENDENT_SAVED_FIT_DETAIL.json'
FAMILIES = [2, 3, 4, 5, 7]
CHECKS = 0


def require(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=True, indent=2, allow_nan=False)
        f.write('\n')


def sourcekey(row):
    return row['clip_id'], row['view'], row['reverse'], row['family']


def build_data(meta):
    bags, original = [], []
    variants = collections.defaultdict(lambda: collections.defaultdict(list))

    def item(row, branch, rate):
        with np.load(row['path'], allow_pickle=False) as d:
            y = d['y'].copy()
            spans = d['spans'].copy()
        require(y.dtype == np.bool_ and y.ndim == 1 and y.any() and (~y).any(), 'known bag labels')
        require(spans.shape == (len(y), 2), 'candidate span index')
        require(row['split'] == 'train' and row['person'] in range(2, 8), 'public FIT split')
        require((~y).sum() <= 512, 'stored negative cap')
        b = dict(row=row, branch=branch, rate=float(rate), y=y, spans=spans,
                 positives=np.flatnonzero(y), negatives=np.flatnonzero(~y), index=len(bags))
        bags.append(b)
        return b

    for r in meta['replay_original']:
        b = item(r, 'normal', 1.)
        original.append(b)
        if r['family'] in [2, 3]:
            variants[sourcekey(r)]['normal'].append(b)
    for r in meta['replay_splice']:
        require(not r.get('skipped', False), 'original splice retained')
        variants[sourcekey(r)]['splice'].append(item(r, 'splice', 1.))
    for r in meta['bags']:
        if not r.get('skipped', False):
            variants[sourcekey(r)][r['base_kind']].append(item(r, r['base_kind'], r['rate']))
    require(len(original) == 178 and len(bags) == 460 and len(variants) == 72, 'fixed bags')
    require(sum(bool(r.get('skipped', False)) for r in meta['bags']) == 78, '78 held entries retained')
    require(len({b['row']['clip_id'] for b in bags}) == 24, '24 replay clips')
    require(len({b['row']['clip_id'] for b in bags if b['row']['family'] in [2, 3]}) == 6, '6 rate clips')
    groups = {f: [i for i, b in enumerate(original) if b['row']['family'] == f] for f in FAMILIES}
    return original, variants, bags, groups


def replay_k(seed, original, variants, bags, groups, schedule, oldindex):
    """Reproduce pinned old label sampling; no old model execution."""
    rng = np.random.default_rng(seed)
    draws = np.zeros(len(bags), np.int64)
    exposure = [np.zeros(len(b['y']), np.int32) for b in bags]
    for step in range(1200):
        selected_sources = []
        for family in FAMILIES:
            sources = rng.choice(groups[family], 2, replace=False)
            selected_sources.append([int(x) for x in sources])
            for oi in sources:
                b = original[int(oi)]
                if family in [2, 3]:
                    v = variants[sourcekey(b['row'])]
                    # Preserve K's original unsorted rate order and RNG timing.
                    selected = [v[branch][int(rng.integers(len(v[branch])))]
                                for branch in ['normal', 'splice']]
                else:
                    selected = [b]
                for b in selected:
                    neg = rng.choice(b['negatives'], min(len(b['negatives']), 56), replace=False)
                    draws[b['index']] += 1
                    exposure[b['index']][b['positives']] += 1
                    exposure[b['index']][neg] += 1
        require(selected_sources == schedule[step], 'K exact per-step source trajectory')
    for i, b in enumerate(bags):
        r = oldindex[i]
        require(r['key'] == f'bag_{i}' and r['path'] == b['row']['path']
                and r['sha256'] == b['row']['sha256'], 'K exact original bag order')
        require(r['sampled_steps'] == int(draws[i]), 'K saved sample-count reconstruction')
        require(r['base_kind'] == b['branch'] and r['rate'] == b['rate'], 'K branch/rate identity')
    return draws, exposure


def verify_l_draws(run, original, variants, bags, schedule):
    counts = np.zeros(len(bags), np.int64)
    with Path(run['source_step_draws']).open(encoding='utf-8') as f:
        for step, line in enumerate(f):
            record = json.loads(line)
            require(step < 1200 and record['step'] == step + 1, 'L step sequence')
            expected_sources = [i for family in schedule[step] for i in family]
            require([r['original_index'] for r in record['sources']] == expected_sources,
                    'L source choices equal original K trajectory')
            weights = collections.Counter()
            seen = set()
            for r, oi in zip(record['sources'], expected_sources):
                b = original[oi]
                require(all(r[k] == b['row'][k] for k in ['clip_id', 'view', 'reverse', 'family']),
                        'L source identity')
                if b['row']['family'] in [2, 3]:
                    selected = []
                    for branch in ['normal', 'splice']:
                        entries = sorted(variants[sourcekey(b['row'])][branch], key=lambda x: x['rate'])
                        selected.extend((x, .05 / len(entries)) for x in entries)
                else:
                    selected = [(b, .1)]
                require(len(r['bags']) == len(selected), 'all available rates present')
                require(math.isclose(sum(d['loss_weight'] for d in r['bags']), .1), 'source total weight')
                for d, (b, weight) in zip(r['bags'], selected):
                    i = b['index']
                    require(i not in seen, 'candidate bag used once per step')
                    seen.add(i)
                    require(d['bag_index'] == i and d['branch'] == b['branch'] and d['rate'] == b['rate'],
                            'L bag/rate order')
                    require(d['candidates'] == len(b['y']) and d['positive'] == len(b['positives'])
                            and d['negative'] == len(b['negatives']), 'full bag exposure')
                    require(math.isclose(d['loss_weight'], weight, rel_tol=1e-12), 'branch/rate weight')
                    counts[i] += 1
                    weights[b['row']['family']] += d['loss_weight']
            require(all(math.isclose(weights[f], .2) for f in FAMILIES), 'five equal family weights')
        require(step == 1199, 'all1200 steps recorded')
    return counts


def score(logits, label):
    a = np.asarray(logits, np.float64)
    top = int(np.argmax(a))
    m = float(a.max())
    total = m + math.log(float(np.exp(a - m).sum()))
    p = a[label]
    pm = float(p.max())
    positive = pm + math.log(float(np.exp(p - pm).sum()))
    return dict(top_index=top, top_full_positive=bool(label[top]),
                positive_minus_negative_margin=float(a[label].max() - a[~label].max()),
                positive_mass=float(math.exp(min(0., positive - total))),
                listwise_loss=float(max(0., total - positive)))


def summaries(rows):
    out = []
    for fields in [('seed',), ('seed', 'family'), ('seed', 'base_kind', 'rate'),
                   ('seed', 'family', 'base_kind', 'rate'), ('seed', 'person')]:
        groups = collections.defaultdict(list)
        for r in rows:
            groups[tuple(r[k] for k in fields)].append(r)
        for key, rr in sorted(groups.items()):
            cm = [[0, 0], [0, 0]]
            for r in rr:
                cm[int(r['K']['top_full_positive'])][int(r['L']['top_full_positive'])] += 1
            out.append(dict(group_by=list(fields), **dict(zip(fields, key)), bags=len(rr),
                            unique_source_clips=len({r['clip_id'] for r in rr}),
                            K_full_positive=sum(r['K']['top_full_positive'] for r in rr),
                            L_full_positive=sum(r['L']['top_full_positive'] for r in rr),
                            paired_K_rows_L_columns=cm,
                            K_mean_listwise_loss=float(np.mean([r['K']['listwise_loss'] for r in rr])),
                            L_mean_listwise_loss=float(np.mean([r['L']['listwise_loss'] for r in rr]))))
    return out


def main():
    require(not OUT.exists() and not DETAIL.exists(), 'do not overwrite completed audit')
    trained = read(L / 'TRAINING_COMPLETE.json')
    require(trained['status'] == 'FULL_BAG_RATE_TRAINING_COMPLETE_NOT_ADOPTED', 'L completed once')
    cfg = read(L / 'TRAINING_FROZEN.json')
    require(trained['training_frozen_sha256'] == sha(L / 'TRAINING_FROZEN.json'), 'training contract pin')
    require(cfg['seeds'] == [42, 43, 44] and cfg['steps'] == 1200, 'fixed protocol')
    pins = dict(cfg['pins'])
    for p in [Path(__file__), L/'TRAINING_COMPLETE.json', L/'TRAINING_FROZEN.json',
              K/'TRAINING_COMPLETE.json', K/'FIT_FAILURE_CAUSES.json']:
        pins[str(p)] = sha(p)
    for p, h in pins.items():
        require(sha(p) == h, 'frozen input/code/protection hash: ' + p)
    original, variants, bags, groups = build_data(read(K/'PREPARATION_INPUTS_COMPLETE.json'))
    schedule = read(L/'SOURCE_SCHEDULE.json')['schedules']
    oldruns = {r['seed']: r for r in read(K/'TRAINING_COMPLETE.json')['rows']}
    runs = {r['seed']: r for r in trained['rows']}
    require(len(trained['rows']) == 3 and set(runs) == set(oldruns) == {42, 43, 44}, 'three fixed seeds')
    rows, runchecks, normalization = [], [], []
    import torch  # State-dictionary reads only; no model instantiation or forward.
    for seed in [42, 43, 44]:
        r, old = runs[seed], oldruns[seed]
        require(r['steps'] == 1200, 'fixed final checkpoint')
        require((r['initial_model'], r['initial_model_sha256']) ==
                (old['initial_model'], old['initial_model_sha256']), 'same J initial model as K')
        for run in [r, old]:
            for pk, hk in [('model', 'model_sha256'), ('initial_model', 'initial_model_sha256'),
                           ('fit_predictions', 'fit_predictions_sha256')]:
                require(sha(run[pk]) == run[hk], pk + ' saved hash')
                pins[run[pk]] = run[hk]
        for pk, hk in [('fit_prediction_index', 'fit_prediction_index_sha256'),
                       ('source_step_draws', 'source_step_draws_sha256'),
                       ('candidate_exposures', 'candidate_exposures_sha256')]:
            require(sha(r[pk]) == r[hk], pk + ' saved hash')
            pins[r[pk]] = r[hk]
        state = torch.load(r['model'], weights_only=True, map_location='cpu')
        parent = torch.load(r['initial_model'], weights_only=True, map_location='cpu')
        require(set(state) == set(parent), 'architecture keys unchanged')
        require(all(state[k].shape == parent[k].shape and torch.isfinite(state[k]).all() for k in state),
                'state shapes and finiteness')
        require(torch.equal(state['mean'], parent['mean']) and torch.equal(state['std'], parent['std']),
                'actual normalization buffers preserved')
        normalization.append(dict(seed=seed, identical_parent_buffers=['mean', 'std'], model_forward=False))
        del state, parent
        index = read(r['fit_prediction_index'])
        oldpath = Path(old['model']).parent/'FIT_PREDICTION_INDEX.json'
        oldindex = read(oldpath)
        pins[str(oldpath)] = sha(oldpath)
        require(len(index) == len(oldindex) == 460, 'same460 FIT bags')
        kd, ke = replay_k(seed, original, variants, bags, groups, schedule[str(seed)], oldindex)
        ld = verify_l_draws(r, original, variants, bags, schedule[str(seed)])
        with np.load(r['candidate_exposures'], allow_pickle=False) as ex, \
                np.load(r['fit_predictions'], allow_pickle=False) as lp, \
                np.load(old['fit_predictions'], allow_pickle=False) as kp:
            require(set(ex.files) == set(lp.files) == set(kp.files) == {f'bag_{i}' for i in range(460)},
                    'exact saved array keys')
            for i, b in enumerate(bags):
                name = f'bag_{i}'
                ix = index[i]
                require(ix['key'] == name and ix['path'] == b['row']['path'] and ix['sha256'] == b['row']['sha256'],
                        'L candidate input identity')
                require(ix['sampled_steps'] == int(ld[i]), 'L bag counts match saved step draws')
                ev = ex[name]
                require(ev.shape == b['y'].shape and np.all(ev == ld[i]), 'every candidate exposed every bag draw')
                for tag, ids in [('positive', b['positives']), ('negative', b['negatives'])]:
                    require(ix[tag+'_exposure_min'] == int(ev[ids].min()) and
                            ix[tag+'_exposure_max'] == int(ev[ids].max()), 'exposure report arithmetic')
                new, before = lp[name], kp[name]
                require(new.shape == before.shape == (len(b['y']), 5), 'paired prediction dimensions')
                require(np.isfinite(new).all() and np.isfinite(before).all(), 'saved prediction finiteness')
                axis = FAMILIES.index(b['row']['family'])
                ks, ls = score(before[:, axis], b['y']), score(new[:, axis], b['y'])
                for s in [ks, ls]:
                    s['top_span'] = b['spans'][s['top_index']].astype(int).tolist()
                rows.append(dict(seed=seed, bag_index=i, path=b['row']['path'], clip_id=b['row']['clip_id'],
                                 person=b['row']['person'], view=b['row']['view'], reverse=b['row']['reverse'],
                                 family=b['row']['family'], base_kind=b['branch'], rate=b['rate'],
                                 candidates=len(b['y']), positive=len(b['positives']), negative=len(b['negatives']),
                                 K=ks, L=ls, K_bag_draws=int(kd[i]), L_bag_draws=int(ld[i]),
                                 K_old_top_exposure=int(ke[i][ks['top_index']]),
                                 L_old_top_exposure=int(ev[ks['top_index']]),
                                 K_new_top_exposure=int(ke[i][ls['top_index']]),
                                 L_new_top_exposure=int(ev[ls['top_index']]),
                                 K_negative_exposure_min=int(ke[i][b['negatives']].min()),
                                 K_negative_exposure_max=int(ke[i][b['negatives']].max())))
        require(r['minimum_bag_draws'] == int(ld.min()) and r['maximum_bag_draws'] == int(ld.max()),
                'reported bag count extrema')
        require(r['all_training_bags_sampled'] == bool((ld > 0).all()), 'reported bag coverage')
        runchecks.append(dict(seed=seed, model=str(r['model']), source_steps=1200, source_choices=12000,
                             bags=460, every_stored_candidate_exposed_on_each_bag_draw=True,
                             minimum_bag_draws=int(ld.min()), maximum_bag_draws=int(ld.max()),
                             K_old_top_exposure_in_failed_bags=[q['K_old_top_exposure'] for q in rows
                                 if q['seed'] == seed and not q['K']['top_full_positive']],
                             L_same_top_exposure_in_K_failed_bags=[q['L_old_top_exposure'] for q in rows
                                 if q['seed'] == seed and not q['K']['top_full_positive']]))
    require(len(rows) == 1380, '460bag times3seed dependent diagnostics')
    failed = read(K/'FIT_FAILURE_CAUSES.json')['failures']
    lookup = {(r['seed'], r['path']): r for r in rows}
    known_failures = []
    for f in failed:
        q = lookup[f['seed'], f['path']]
        require(q['K']['top_index'] == f['top_index'] and not q['K']['top_full_positive'],
                'earlier K failed case exactly reconstructed')
        known_failures.append(q)
    # Recheck immutable pins after arithmetic; never silently replace changed files.
    for p, h in pins.items():
        require(sha(p) == h, 'end hash unchanged: ' + p)
    grouped = summaries(rows)
    detail = dict(rows=rows, summary=grouped, prior_K_failed_cases=known_failures,
                  changed_selection=[r for r in rows if r['K']['top_index'] != r['L']['top_index']])
    write_new(DETAIL, detail)
    result = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status='SAVED_FIT_EXPOSURE_AND_ARITHMETIC_VERIFIED_NOT_G2_OR_GENERALIZATION',
                  checks=CHECKS, runs=runchecks, normalization=normalization,
                  primary_fit_summary=[r for r in grouped if r['group_by'] == ['seed']],
                  prior_K_failed_cases=len(known_failures),
                  prior_K_failures_now_positive=sum(r['L']['top_full_positive'] for r in known_failures),
                  new_fullpositive_regressions=sum(r['K']['top_full_positive'] and not r['L']['top_full_positive'] for r in rows),
                  population='24 public FIT clips,6 people; rate-expanded subset6 clips.460 correlated bags x3seeds, NOT1380 independent examples.',
                  limits=['All reported predictions are training diagnostics, not new-user accuracy.',
                          'Negative means not a reviewed full-positive extent; it is not necessarily a linguistic error.',
                          'The78 held time variants remain outside training score denominators, separately retained in preparation.',
                          'Equal optimizer steps/source schedule/initialization; larger batch FLOPs and two bundled exposure changes.',
                          'No training, feature generation, model forward, threshold selection, or semantic completion claim by this audit.'],
                  detail=str(DETAIL), detail_sha256=sha(DETAIL), pins=pins,
                  no_model_forward=True, no_training=True, no_new_evaluation=True, no_private=True,
                  no_final8=True, web_changed=False, semantic_completion=None, G2_pass=None)
    write_new(OUT, result)
    print('Saved FIT verification complete:', CHECKS, 'checks;', result['primary_fit_summary'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        p = L / f'SAVED_FIT_AUDIT_FIRST_FAILURE_{time.time_ns()}.txt'
        p.write_text(traceback.format_exc(), encoding='utf-8')
        raise
