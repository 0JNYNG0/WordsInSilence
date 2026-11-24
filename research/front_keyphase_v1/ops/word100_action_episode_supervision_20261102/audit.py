"""Audit existing supervision before building new action-episode targets."""
from pathlib import Path
import sys,collections
D=Path(__file__).resolve().parent;O=D.parent;B=O/'word100_learned_action_process_20261102'
sys.path.insert(0,str(B));from process_model import read,sha,exclusive,write,clock,A,S,T,P
M=O/'word100_learned_motion_order_20261102'

def main():
    assert not (D/'INPUT_AUDIT.json').exists()
    manifest=[r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['role']=='fit']
    motion=read(M/'MOTION_TARGETS.json');roles={r['clip_id']:r for r in read(A/'ROLE_TARGETS.json')['rows']};poses={r['clip_id']:r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows']}
    support=read(M/'MOTION_GUIDE_SUPPORT.json');guides=read(P/'GUIDE_SUPPORT.json');rows=[]
    for r in manifest:
        cid=r['clip_id'];assert 1<=r['person']<=9
        eps=[e for e in motion['rows'] if e['clip_id']==cid];pr=[x for x in poses[cid]['points'] if x['mask']];ar=[x for x in roles[cid]['points'] if x['mask']]
        directed=[]
        for e in eps:
            if e['kind']!='directed':continue
            targets=[i for i in range(5) if e['mask'][i] and e['y'][i]>0]
            for target in targets:
                sg=next(x for x in support if x['word_id']==r['word_id'] and x['target']==target)
                directed.append(dict(id=e['id'],target=target,physical_process=motion['targets'][target],frames=e['frames'],guide_id=sg['guide_id'],
                  source_phrases=sg['source_phrases'],original_claim=e['claim'],per_frame_action_id_verified=False,
                  targets_valid_for='Net physical motion somewhere across the reviewed episode; not exact frame-by-frame action stage or onset/offset.'))
        rows.append(dict(clip_id=cid,person=r['person'],word_id=r['word_id'],label=r['label'],source=r['source'],source_sha256=r['source_sha256'],frames=r['frames'],
          physical_pose_points=len(pr),role_points=len(ar),performed_role_points=sum(x['label']==1 for x in ar),
          motion_episodes=directed,source_guide_descriptions=[g for g in guides['guides'] if g['word_id']==r['word_id']],
          verified_framewise_action_ownership=0,
          missing='Which described action owns an otherwise identical passing or returning posture; within-action changes; temporal boundaries. Existing role/pose labels do not specify this.'))
    assert len(rows)==36 and {r['person'] for r in rows}==set(range(1,10))
    pins={str(p):sha(p) for p in [D/'PLAN.json',P/'INPUT_MANIFEST.json',A/'ROLE_TARGETS.json',S/'PHYSICAL_STATE_TARGETS.json',M/'MOTION_TARGETS.json',M/'MOTION_GUIDE_SUPPORT.json',P/'GUIDE_SUPPORT.json',B/'SEQUENCE_GAIN_AUDIT.json']}
    out=dict(at=clock(),status='SPARSE_POSE_AND_ROLE_ARE_NOT_ACTION_EPISODE_GOLD',clips=36,recorded_people=list(range(1,10)),
      physical_pose_points=sum(r['physical_pose_points'] for r in rows),role_points=sum(r['role_points'] for r in rows),directed_motion_episodes=sum(len(r['motion_episodes']) for r in rows),
      verified_framewise_action_ownership=0,rows=rows,pins=pins,
      conclusions=[
       'Existing motion supervision already describes physical from-to processes and can be retained; it is not full per-time guide-action supervision.',
       'Do not convert each motion interval into dense labels: original authorship explicitly disclaims exact onset/per-frame class.',
       'Do not count one posture as one unique action. A raised fist may be traversed during another movement or intentionally held at an endpoint.',
       'Prepare chronological public FIT review with existing source guide clauses and original times; annotate action-owned intervals or sparse witnesses, keeping boundary/semantic uncertainty separate.',
       'Exclude DEV from this input review; do not repair old labels using new candidate predictions. No predictions are displayed in the review.'
      ],new_training=0,new_private_input=0,whole_goal_complete=False)
    exclusive(D/'INPUT_AUDIT.json',out)
    print('FIT',len(rows),'pose',out['physical_pose_points'],'roles',out['role_points'],'motion episodes',out['directed_motion_episodes'],'dense action ownership',out['verified_framewise_action_ownership'],flush=True)

if __name__=='__main__':main()
