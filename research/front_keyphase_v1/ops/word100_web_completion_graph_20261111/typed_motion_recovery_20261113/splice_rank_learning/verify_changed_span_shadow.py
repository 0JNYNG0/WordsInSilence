"""Independent saved arithmetic; no torch, model inference or evaluation rerun."""
from pathlib import Path
import collections
import hashlib
import json
import time
import traceback
import numpy as np

J = Path(__file__).resolve().parent
G = J.parent.parent
O = G.parent
E = J / 'changed_span_shadow'
L = G / 'candidate_description_supervision/local_rank_learning/evaluation'
checks = collections.Counter()


def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r): return r['seed'], r['clip_id'], r['kind']
def reqkey(cid, kind, card, span): return cid, kind, card, None if span is None else tuple(span)
def ck(group, passed):
    assert bool(passed), group
    checks[group] += 1


def shape_available(x):
    masks = x[:, 186:237] > .5
    out = []
    for j in range(7):
        points = [9, 30] if j in (5, 6) else ([9] + list(range(14, 30)) if j in (2, 4) else [30] + list(range(35, 51)))
        out.append(masks[:, points].all(1))
    return np.stack(out, 1)


def old_direction_geometry(x, ts, a, b, axis, side):
    """Independent axes0/1 reconstruction from original body-only contract."""
    assert axis in (0, 1)
    points = x[:, :102].astype(float).reshape(-1, 51, 2)
    valid = x[:, 186:237] > .5
    shoulder_ok = valid[:, 1] & valid[:, 2]
    distances = np.linalg.norm(points[:, 1] - points[:, 2], axis=1)
    scale = float(np.median(distances[shoulder_ok])) if shoulder_ok.any() else 0.
    if scale < 1e-5:
        p, v = np.zeros_like(points), np.zeros_like(valid)
    else:
        p = (points - (points[:, 1] + points[:, 2])[:, None] / 2) / scale
        v = valid & shoulder_ok[:, None]
    p[:, :, 0] *= (np.sign(p[:, 2, 0] - p[:, 1, 0]) * (1 if side == 'R' else -1))[:, None]
    elbow, wrist = (4, 6) if side == 'R' else (3, 5)
    q = p[:, wrist]
    if axis == 0:
        values = np.column_stack([-q[:, 1], q[:, 0], -q[:, 1], q[:, 0]])
        good = v[:, wrist] & np.isfinite(values).all(1)
    else:
        arm = q - p[:, elbow]
        length = np.linalg.norm(arm, axis=1)
        values = np.column_stack([-arm[:, 1]/np.maximum(length, 1e-5), arm[:, 0]/np.maximum(length, 1e-5), length, -q[:, 1]])
        good = v[:, [wrist, elbow]].all(1) & np.isfinite(values).all(1) & (length > .05)
    vv, gg, tt = values[a:b+1], good[a:b+1], ts[a:b+1].astype(float)
    coverage = float(gg.mean())
    eligible = coverage >= .8 and int(gg.sum()) >= 4 and bool(gg[0] and gg[-1])
    if not eligible:
        return np.zeros(58, np.float32), False, coverage, int(gg.sum())
    vgood, tgood = vv[gg], tt[gg]
    at = np.linspace(tgood[0], tgood[-1], 9)
    ss = np.stack([np.interp(at, tgood, vgood[:, j]) for j in range(4)], 1)
    geom = np.r_[(ss-ss[0]).ravel(), ss[0], ss[-1]-ss[0], np.abs(np.diff(ss, axis=0)).mean(0),
        np.ptp(ss, axis=0), np.eye(5)[axis], coverage].astype(np.float32)
    return geom, True, coverage, int(gg.sum())


def direction_state(p, desired, eligible):
    if not eligible: return 'observation_unknown'
    if desired == 1:
        return 'supported' if p[1] >= .65 and max(p[0], p[2]) < .65 else 'uncertain'
    other = 2-desired
    if p[desired] >= .65 and p[other] < .65 and p[1] < .65: return 'supported'
    if p[other] >= .65 and p[desired] < .35 and p[1] < .65: return 'opposite_direction'
    if p[1] >= .65: return 'movement_not_established'
    return 'uncertain'


def main():
    ck('no_saved_verification_overwrite', not (E/'VERIFICATION.json').exists())
    cfg = read(E/'FROZEN.json'); result = read(E/'RESULTS.json')
    old = read(L/'shadow_components/RESULTS.json'); newrank = read(J/'evaluation/COMPARISON.json')
    for p, h in cfg['pins'].items(): ck('frozen_input_pin', sha(p) == h)
    for p, h in read(G/'PROTECTIONS.json').items():
        ck('protected477_original_pin', cfg['pins'][p] == h and sha(p) == h)
    ck('result_frozen_pin', result['frozen_sha256'] == sha(E/'FROZEN.json'))
    ck('all_original_summary_exact', result['summary'][:len(old['summary'])] == old['summary'])
    om = {key(r): r for r in old['rows']}; rm = {key(r): r for r in newrank['rows']}
    ck('row_denominator246', len(result['rows']) == len(om) == len(rm) == 246)
    ck('no_rerun_deployment_or_completion', not result['old_predictions_rerun'] and not result['common_encoder_rerun']
        and not result['TCN_DTW_rerun'] and not result['web_changed'] and result['semantic_completion'] is None and result['G3_pass'] is None)
    ck('selector_and_observer_separate', result['observer_seed'] == 42 and result['selector_seeds'] == [42, 43, 44])
    ck('execution_plan_exact', result['execution'] == cfg['plan'])
    saved = {}
    for r in old['rows']:
        for cards in r['conditions'].values():
            for c in cards:
                kk = reqkey(r['clip_id'], r['kind'], c['card'], c['span'])
                if kk in saved: ck('old_duplicate_equal', saved[kk] == c)
                saved[kk] = c
    request_lookup = {q['request_id']: q for q in cfg['requests']}
    request_tuples = {reqkey(q['clip_id'],q['kind'],q['card'],q['span']): q for q in cfg['requests']}
    new_req = {q['request_id']: q for q in cfg['requests'] if q['action'] == 'new_request'}
    ck('55_18_37_request_denominators', len(request_lookup)==55 and len(new_req)==37 and len(cfg['requests'])-len(new_req)==18)
    ck('37_new_requests_only', set(result['new_requests']) == set(new_req))
    source_cache = {}
    point_cache = {}
    for p in result['pointwise_cache']:
        ck('pointwise_cache_pin', sha(p['path']) == p['sha256'])
        ck('pointwise_source_pin', sha(p['source']) == p['source_sha256'])
        with np.load(p['source'], allow_pickle=False) as z: x, ts = z['x'].copy(), z['ts'].copy()
        with np.load(p['path'], allow_pickle=False) as z: frames, attrs = z['frame_indices'].copy(), z['attribute_probabilities'].copy()
        sid = p['clip_id'], p['kind']; source_cache[sid] = (x, ts, shape_available(x))
        ck('pointwise_shapes_range', attrs.shape == (len(frames), 7) and np.isfinite(attrs).all() and np.all((attrs>=0)&(attrs<=1)))
        ck('pointwise_unique_ordered', np.array_equal(frames, np.unique(frames)) and np.all((frames>=0)&(frames<len(x))))
        ck('pointwise_denominator', p['selected_valid_frames']==len(frames) and p['total_source_frames']==len(x))
        point_cache[sid] = {int(f): attrs[i] for i, f in enumerate(frames)}
    cards_cfg = read(O/'word100_action_feedback_demo_20261108/CARDS.json')['words']
    family_cfg = read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
    caps = {(r['word'],r['card'],r['feature'],r['expected']):r['qualified']
        for r in read(O/'word100_guide_requirement_learning_20261111/GUIDE_CAPABILITY.json')['rows']}
    needed = collections.defaultdict(set)
    for q in new_req.values():
        if q['span'] is None: continue
        sid=q['clip_id'],q['kind'];a,b=q['span'];word=q['clip_id'].split('_')[2]
        guide=next(c for c in cards_cfg[word] if c['id']==q['card'])
        aa=a+(b-a)//2 if guide.get('shape_scope')=='late' else a
        av=source_cache[sid][2]
        for j in guide.get('features',{}): needed[sid].update(int(f)for f in np.arange(aa,b+1)[av[aa:b+1,int(j)]])
    for sid, cache in point_cache.items(): ck('only_required_valid_frames_inferred',set(cache)==needed[sid])
    movement={}
    for m in result['new_movement_requests']:
        sid=m['clip_id'],m['kind'];a,b=m['span'];x,ts,_=source_cache[sid]
        geom,ok,cov,nobs=old_direction_geometry(x,ts,a,b,m['axis'],m['moving'])
        ck('old_body_geometry_exact',np.array_equal(geom,np.asarray(m['geometry'],np.float32)))
        ck('old_geometry_eligibility',ok==m['eligible'] and abs(cov-m['info']['coverage'])<1e-12 and nobs==m['info']['observed'])
        probs=np.asarray(m['probabilities']);ck('saved_probabilities_valid',len(probs)==3 and np.isfinite(probs).all() and np.all(probs>=0) and abs(probs.sum()-1)<2e-6)
        mk=(*sid,a,b,m['axis'],m['moving']);ck('movement_unique',mk not in movement);movement[mk]=m
    ck('movement37_only',len(movement)==37)
    changes=[];feature_changes=[];new_warning=[];old_warning_lost=[];new_inference_cards={};same=0;changed=0
    for r in result['rows']:
        kk=key(r);original=om[kk]
        for mode,cs in original['conditions'].items(): ck('original4conditions_exact',r['conditions'][mode]==cs)
        before={c['card']:c for c in original['conditions']['local_full']}
        selected={c['card']:c for c in rm[kk]['conditions']['splice_rank']['cards']}
        for c in r['conditions']['splice_rank']:
            oldc=before[c['card']];sc=selected[c['card']];span=sc['chosen']['span']if sc['chosen']else None
            ck('J_span_exact_and_no_completion',c['span']==span and c['completion']is None)
            req=reqkey(r['clip_id'],r['kind'],c['card'],span)
            if span==oldc['span']:
                same+=1;ck('unchanged_span_exact',c==oldc)
            else:
                changed+=1;q=request_tuples[req]
                if q['action']=='reuse_saved_identical_request':ck('18_saved_request_reuse_exact',c==saved[req])
                else:
                    if q['request_id']in new_inference_cards:ck('new_request_reuse_across_seeds_exact',c==new_inference_cards[q['request_id']])
                    new_inference_cards[q['request_id']]=c
                changes.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],before_span=oldc['span'],after_span=span,before_status=oldc['status'],after_status=c['status'],source=q['action']))
            for a,b in zip(oldc['evidence'],c['evidence']):
                ck('evidence_identity_preserved',a['task']==b['task'] and a.get('feature')==b.get('feature') and a.get('expected')==b.get('expected') and a.get('axis')==b.get('axis') and a.get('desired')==b.get('desired'))
                if a['state']!=b['state']:feature_changes.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],before_span=oldc['span'],after_span=span,task=b['task'],before=a,after=b))
            if oldc['status']!='difference_evidence' and c['status']=='difference_evidence' and r['kind']in('full_normal','speed075','speed125'):new_warning.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card']))
            if oldc['status']=='difference_evidence' and c['status']!='difference_evidence':old_warning_lost.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],before=oldc['status'],after=c['status']))
            if r['kind']=='all_hand_missing':
                for v in c['evidence']:
                    if v['task']=='shape':ck('allmissing_shape_unknown',v['state']=='observation_unknown')
            states=[v['state']for v in c['evidence']]
            expected='association_unknown'if span is None else'difference_evidence'if 'difference'in states or'opposite_direction'in states else'supported_features_only'if states and all(v=='supported'for v in states)else'insufficient_evidence'
            ck('card_state_arithmetic',c['status']==expected)
    ck('77_changed_241_same_cards',changed==77 and same==241)
    ck('37_new_card_requests',len(new_inference_cards)==37)
    for rid,c in new_inference_cards.items():
        q=new_req[rid];sid=q['clip_id'],q['kind'];word=q['clip_id'].split('_')[2];a,b=c['span']
        guide=next(v for v in cards_cfg[word]if v['id']==c['card']);spec=next(v for v in family_cfg[word]if v['id']==c['card'])
        aa=a+(b-a)//2 if guide.get('shape_scope')=='late'else a;av=source_cache[sid][2]
        for v in c['evidence']:
            if v['task']=='shape':
                j=v['feature'];want=guide['features'][str(j)];valid=av[aa:b+1,j];ids=np.arange(aa,b+1)[valid]
                vals=np.asarray([point_cache[sid][int(f)][j]for f in ids],np.float32)
                p=float(vals.mean())if len(vals)else None;p=p if want or p is None else 1-p
                eligible=bool(valid.mean()>=.8);qualified=caps.get((word,c['card'],j,want),False)
                state='observation_unknown'if p is None or not eligible else'association_unknown'if not qualified else'supported'if p>=.65 else'difference'if p<=.35 else'uncertain'
                ck('shape_saved_arithmetic',v['span']==[aa,b]and v['expected']==want and v['probability']==p and v['state']==state and v['qualified']==qualified and v['observed_count']==int(valid.sum()))
            else:
                m=movement[(*sid,a,b,spec['axis'],spec['hands'][0])];desired=int(spec['sign']+1)
                ck('direction_saved_arithmetic',v['span']==[a,b]and v['desired']==desired and v['probabilities']==m['probabilities']and v['state']==direction_state(v['probabilities'],desired,m['eligible']))
    summary=[]
    for seed in[42,43,44]:
        for kind in sorted({r['kind']for r in result['rows']}):
            cs=[c for r in result['rows']if r['seed']==seed and r['kind']==kind for c in r['conditions']['splice_rank']];items=[v for c in cs for v in c['evidence']]
            summary.append(dict(seed=seed,condition='splice_rank',kind=kind,cards=len(cs),statuses=dict(collections.Counter(c['status']for c in cs)),shape=dict(collections.Counter(v['state']for v in items if v['task']=='shape')),direction=dict(collections.Counter(v['state']for v in items if v['task']=='direction'))))
    ck('all_new_summary_exact',summary==result['summary'][len(old['summary']):])
    ck('normal_speed_no_new_difference',len(new_warning)==0)
    normal_changes=[c for c in changes if c['kind']in('full_normal','speed075','speed125')]
    ck('normal_speed_all_card_states_preserved',all(c['before_status']==c['after_status']for c in normal_changes))
    for r in result['rows']:
        if r['kind']in('omit_final_event','mask_final_event'):
            bc={c['card']:c for c in r['conditions']['local_full']}
            for c in r['conditions']['splice_rank']:
                ck('omission_mask_not_promoted_to_feature_pass',c['status']!='supported_features_only'or bc[c['card']]['status']=='supported_features_only')
                ck('omission_mask_no_completion',c['completion']is None)
    paired=dict(rows=changes,feature_changes=feature_changes,old_difference_status_lost=old_warning_lost,
        normal_speed_new_difference=new_warning,
        meaning='Paired source/card span changes with the same frozen observer. No human error truth or model relearning claim.')
    (E/'PAIRED_CHANGES.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    out=dict(status='SAVED_ARITHMETIC_AND_PRESERVATION_PASS_NOT_G3_PASS',at=time.time(),checks=dict(checks),total_checks=sum(checks.values()),
        protected_original_files=477,rows=246,cards=318,changed_cards=77,same_cards=241,unique_changed_requests=55,saved_reuse=18,new_requests=37,
        new_pointwise_sources=len(point_cache),new_pointwise_frames=sum(len(v)for v in point_cache.values()),new_movement_requests=len(movement),
        changed_card_statuses=sum(c['before_status']!=c['after_status']for c in changes),changed_feature_states=len(feature_changes),normal_speed_new_difference=0,
        lost_difference_statuses=old_warning_lost,completion_promotions=0,allmissing_shape_unknown=True,
        limitations=['Only saved arithmetic checked; no model probabilities independently rescored.',
            'Codex separate verifier is not independent human or Korean Sign Language correctness validation.',
            'Some omission/mask opposite-direction evidence disappears after changed spans. It is not verified opposite-action truth for an absent action. Neither loss nor recovery alone proves better educational feedback.',
            'Parent reports remaining fast 33ms extent G2 regression; this semantic audit does not resolve it.'],
        sources={str(p):sha(p)for p in[Path(__file__),E/'RESULTS.json',E/'FROZEN.json',L/'shadow_components/RESULTS.json',J/'evaluation/COMPARISON.json',E/'PAIRED_CHANGES.json']},no_model_inference=True,no_web_change=True,G3_pass=None)
    (E/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print('Saved checks PASS',sum(checks.values()),'changed feature states',len(feature_changes),'new normal warnings',len(new_warning),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        (E/f'VERIFICATION_FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
