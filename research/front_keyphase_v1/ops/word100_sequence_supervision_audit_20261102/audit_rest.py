"""FIT-only admission audit. Model posteriors remain evidence, never new labels."""
from pathlib import Path
import sys, collections
J=Path(__file__).resolve().parent
sys.path.insert(0,str(J.parent/'word100_separate_action_context_20261102'))
from model import *

def main():
    assert not (J/'REST_INPUT_AUDIT.json').exists()
    rows=fit_rows(); records=[]; counts=collections.Counter()
    witness=read(J/'SEQUENCE_INPUT_AUDIT.json')['rows']
    for r, s in zip(rows,witness):
        assert (r['clip_id'],r['kind'])==(s['clip_id'],s['kind'])
        assert r['person'] in range(1,10)
        cur=r['current']; assert cur.shape==(len(r['x']),10)
        for f in np.flatnonzero(r['role']>=0):
            truth=int(r['role'][f]); pred=int(cur[f].argmax()); conf=float(cur[f].max())
            counts[f'role{truth}_known']+=1
            if pred==9: counts[f'role{truth}_T_rest_argmax']+=1
            if cur[f,9]>=.8:
                counts[f'role{truth}_T_rest_confident']+=1
                if truth==1: records.append(dict(clip_id=r['clip_id'],kind=r['kind'],frame=int(f),source_frame=int(r['mapping'][f]),action=int(r['episode'][f]),rest_probability=float(cur[f,9])))
        sampled=[int(f) for run in s['witness_runs'] for f in run['frames']]
        assert sampled==sorted(sampled)
        assert all(0<=r['episode'][f]<8 and r['available'][f] for f in sampled)
    out=dict(at=clock(),status='AUDITED_NOT_AUTOMATIC_GOLD',inputs=len(rows),counts=dict(counts),confident_rest_vs_direct_action=records,
       sequence_admission='Sparse witness sequence is not a complete video transcript. A sequence loss may only see directly reviewed sampled timepoints, ordered on the native axis. Unlabelled times and edit seams remain unlabelled. Do not apply full-video CTC from these lists.',
       rest_contract='Retain T rest posterior as a separate immutable channel. A confident rest versus confident learned main-action conflict is held and exposed, never resolved using target order or unavailable images. No posterior-generated labels.',
       scope='FIT01-09 only; repeated transformed frames are not independent observations. No DEV selection or new inference.',
       pins={str(p):sha(p) for p in [J/'SEQUENCE_INPUT_AUDIT.json',D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',B/'PREPARED.json']},whole_goal_complete=False)
    exclusive(J/'REST_INPUT_AUDIT.json',out)
    print(dict(counts));print('conflicts',len(records));print(records[:8])
if __name__=='__main__':main()
