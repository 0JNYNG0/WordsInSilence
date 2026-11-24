from model import *
from collections import Counter

def main():
    assert not (F/'INPUT_CONTRACT.json').exists()
    actions=read(D/'ACTION_TARGETS.json');contexts=read(E/'CONTEXT_TARGETS.json');poses={r['clip_id']:r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows']};pairs=Counter();shared=[]
    for r in actions['rows']:
        # Preserve the actual source schema, not an assumed dense pose target.
        points=[p for p in poses[r['clip_id']]['points'] if p['mask']]
        physical={p['frame']:p.get('label',p.get('group')) for p in points}
        assert all(g is not None for g in physical.values())
        for p in r['points']:
            if p['frame'] in physical:pairs[(p['action_family'],physical[p['frame']])]+=1;shared.append(dict(clip_id=r['clip_id'],frame=p['frame'],action=p['action_family'],physical=physical[p['frame']]))
    rows=fit_rows();assert len(rows)==216
    old_known=sum(int((r['episode']>=0).sum()) for r in rows);role_known=sum(int((r['role']>=0).sum()) for r in rows)
    assert old_known==5102 and role_known==6390
    assert all(np.all(r['role'][(r['episode']>=0)&(r['episode']<8)]==1) for r in rows)
    actor_pairs=[dict(action=int(a),physical=int(b),count=int(n)) for (a,b),n in sorted(pairs.items())]
    pins={str(p):sha(p) for p in [D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',D/'runs/episode_42/model.pt',S/'PHYSICAL_STATE_TARGETS.json',T/'runs/current_42/model.pt',B/'PREPARED.json']}
    exclusive(F/'INPUT_CONTRACT.json',dict(at=clock(),status='PASS',previous_turn='Progress: E4training/352comparisons, separate arithmetic and19input-interpretation contradictions; no live job.',
      task='Keep action content and process role as separate simultaneous outputs. Context/rest is never a replacement label for handshape, body relation or movement.',
      action_source='D frozen9-way action/rest classifier, fixed primary episode42, not a new choice of best E condition.',
      role_targets='D directly reviewed actions->main1; directly reviewed rest->rest2; E directly reviewed approach/return->context0. Unknown stays unknown, no word/position pseudo labels.',
      differences_from_A_B='A/B used broad role labels and instantaneous posture as sequence states. This trial uses D action-process witnesses and action content, with explicit E context examples. The8old broad-transit overlaps never erase D actions.',
      overlap='Physical configuration and action ownership are distinct concurrent facts. At shared directly annotated points report both; no dense or automatic physical-to-action mapping.',
      shared_direct_witnesses=shared,action_physical_pairs=actor_pairs,
      frames_kept=True,fit_inputs=216,action_labels=old_known,role_labels=role_known,pins=pins,
      scope='36 public FIT recordedIDs01-09; no DEV targets in this contract. Same Codex annotations, not independent expert truth.',whole_goal_complete=False))
    print('Contract pass',old_known,role_known,actor_pairs)
if __name__=='__main__':main()
