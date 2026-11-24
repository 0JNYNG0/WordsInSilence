"""Audit what sparse annotations prove before changing sequence semantics."""
from pathlib import Path
import json,hashlib,importlib.util
from collections import Counter
from datetime import datetime,timezone,timedelta
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
D=O/'word100_action_episode_supervision_20261102'
E=O/'word100_guide_conditioned_episode_20261102'
V=O/'word100_content_role_bridge_20261102'
T=O/'word100_learned_state_context_balance_20261102'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    targets=read(D/'ACTION_TARGETS.json');contexts=read(E/'CONTEXT_TARGETS.json');rows=targets['rows']
    assert len(rows)==36 and {r['person'] for r in rows}==set(range(1,10))
    ctx={r['clip_id']:r for r in contexts['rows']};words={};coverage=[]
    for r in rows:
        main={int(p['frame']):int(p['action_family']) for p in r['points'] if p['mask'] and p['action_family']<8}
        known={int(p['frame']) for p in ctx[r['clip_id']]['points'] if p['mask']}
        assert set(main)<=known
        coverage.append(dict(clip_id=r['clip_id'],frames=r['frames'],known_main=len(main),known_process=len(known),
            process_unknown=r['frames']-len(known),first_main=min(main),last_main=max(main)))
    for word in sorted({r['word_id'] for r in rows}):
        rr=[r for r in rows if r['word_id']==word];groups=sorted({p['action_family'] for r in rr for p in r['points'] if p['mask'] and p['action_family']<8})
        edges=[]
        for a in groups:
            for b in groups:
                if a!=b and all(max(p['frame'] for p in r['points'] if p['mask'] and p['action_family']==a)<min(p['frame'] for p in r['points'] if p['mask'] and p['action_family']==b) for r in rr):edges.append([a,b])
        words[word]=dict(groups=groups,sparse_witness_precedences=edges,recorded_ids=[r['person'] for r in rr],
            full_timeline_exclusive_nonoverlap_proven=False,all_unlabelled_frames_meaningless=False,
            permitted_repeat_count_proven=False)
    # These are synthetic contract demonstrations, NOT new public-video scores.
    spec=importlib.util.spec_from_file_location('scope_sequence',T/'sequence_full.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    graph=dict(groups=[0,1,2],precedences=[[0,1],[0,2],[1,2]])
    examples={}
    for name,ids in {'ordered':[0,0,1,1,2,2], 'later_same_configuration':[0,0,1,1,2,2,1,1],
        'short_contrary':[0,0,1,1,0,2,2], 'reverse':[2,2,1,1,0,0]}.items():
        p=np.eye(10)[ids];q=m.observe(p,np.ones(len(ids),bool),graph,np.arange(len(ids))/30)
        examples[name]=dict(status=q['status'],edges=q['precedences'])
    assert examples['ordered']['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
    assert examples['later_same_configuration']['status']=='IMPORTANT_STATE_OR_ORDER_UNCONFIRMED'
    assert examples['short_contrary']['status']!='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
    assert examples['reverse']['status']=='PHYSICAL_ORDER_CONFLICT'
    files=[D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',T/'sequence_full.py',V/'VERIFIED.json',V/'CAUSE_AUDIT.json',U/'audit_contract.py']
    out=dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='CONTRACT_SCOPE_AUDITED_NO_MODEL_CHANGE',
        previous_goal_turn='VERIFIED_WAIT: session85123 polled live while answering W storage question; no duplicate process.',
        this_goal_turn='PROGRESS: completed two supervised role fine-tunes,176fixed development evaluations and separate arithmetic; then audited source/sequence scope.',
        source_coverage=coverage,total_frames=sum(r['frames'] for r in coverage),
        known_process=sum(r['known_process'] for r in coverage),known_main=sum(r['known_main'] for r in coverage),
        graphs=words,synthetic_contract_examples=examples,pins={str(p):sha(p) for p in files},
        conclusions=[
            'The sparse witnesses support relative order between witnessed action episodes, not a complete framewise rule for every later similar pose.',
            'A later similar configuration can be a return OR another meaningful action. Neither possibility can be resolved from the class ID alone.',
            'The current strict matcher intentionally holds overlaps; this is uncertainty, not proof that normal signing is wrong.',
            'Do not silently select first occurrences, crop to known labels, drop short contradictory frames, or force expected order.',
            'The next learned alignment must keep observed configurations and match individual occurrences to the guide flow; unmatched evidence must remain visible.',
            'Guide-supported relative motion, endpoint configurations, and preparation/return have different roles; a single frame class alone does not establish a full occurrence.'
        ],new_labels=0,new_training=0,new_public_inference=0,revised_old_scores=0,whole_goal_complete=False)
    save(U/'GRAPH_SCOPE_AUDIT.json',out)
    print({k:out[k] for k in ['status','total_frames','known_process','known_main']})
    print({k:v['status'] for k,v in examples.items()})

if __name__=='__main__':main()
