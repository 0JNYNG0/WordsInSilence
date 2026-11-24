"""Public normal and controlled observability comparison; no user inputs."""
from apple_candidate import *
import json, copy, time
from geometry_guard import hand_occlusion as legacy
import check_controls

read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def summarize(q):
    return {c['id']: c['status'] for c in q['criteria']}

def main():
    from required_role_candidate import torch
    torch.set_num_threads(2)
    old, new = RequiredRoleCandidate(), RoleEvidenceCandidate()
    rows = read(APPLE/'INPUTS.json')['rows'] + read(APPLE/'holdout/EXTRACTED.json')['rows']
    normal, controls = [], []
    for r in rows:
        path = r.get('raw', r.get('raw_path'))
        assert 'Temp' not in path and 'user_' not in path
        raw = dict(np.load(path))
        before = old.analyze(raw); after = new.analyze(raw)
        bs, ns = summarize(before), summarize(after)
        normal.append(dict(person=r['person'], role=r['role'], before=bs, after=ns,
                           no_regression=all(ns[k] == v for k, v in bs.items() if v == 'supported'),
                           no_new_warning=all(v != 'difference' or bs[k] == 'difference' for k, v in ns.items())))
        write(D/'normal'/f'p{r["person"]}.json', dict(before=before, after=after))
        if r['role'] == 'fit':
            continue
        pair = next((e for e in before['events'] if e['region'] == 2), None)
        if not pair:
            continue
        a, b = pair['context']; middle = (a + b)//2
        # Drop only finger tips on the passive hand, retaining its wrist/palm.
        # This is measurement loss, not a synthetic incorrect human motion.
        for side, hand in [('L',91), ('R',112)]:
            z = copy.deepcopy(raw)
            z['xy_px'][middle:middle + 5, np.array([4,8,12,16,20]) + hand] = np.nan
            z['score'][middle:middle + 5, np.array([4,8,12,16,20]) + hand] = 0
            o, n = old.analyze(z), new.analyze(z)
            os, ss = summarize(o), summarize(n)
            check = ss['wipe_shape'] == ns['wipe_shape'] if side == 'L' else ss['wipe_shape'] != 'supported'
            check = check and all(v != 'difference' or os[k] == 'difference' for k, v in ss.items())
            controls.append(dict(person=r['person'],case='missing_'+side+'_tips',before=os,after=ss,
                                 check=check,source='public-keypoint missingness control; not real learner error'))
            write(D/'controls'/f'p{r["person"]}_missing_{side}.json',dict(before=o,after=n))
        # Bilateral function property: own-hand validity must not depend on the other hand.
        ch = adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
        for side, offset, other in [('R',112,91),('L',91,112)]:
            test = copy.deepcopy(ch); test['point_available'][:,other:other+21] = False
            facts = hand_evidence(test,side)
            assert np.array_equal(facts['uncertain'],~test['point_available'][:,offset:offset+21].all(1))
    write(D/'VISIBILITY_RESULT.json',dict(normal=normal,controls=controls,
       normal_regressions=[x for x in normal if not x['no_regression'] or not x['no_new_warning']],
       failures=[x for x in controls if not x['check']],training_runs=0,private_data=False))
    print('NORMAL',len(normal),'REGRESSIONS',sum(not x['no_regression'] or not x['no_new_warning'] for x in normal),
          'CONTROL',len(controls),'FAIL',sum(not x['check'] for x in controls),flush=True)
    dest=D/'legacy_controls'; (dest/'controls').mkdir(parents=True,exist_ok=True)
    check_controls.main(RoleEvidenceCandidate,dest,True)

if __name__=='__main__': main()
