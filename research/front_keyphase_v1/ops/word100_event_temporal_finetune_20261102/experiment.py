"""Fixed 2x2: additional reviewed FIT labels x copied event temporal adapter."""
from pathlib import Path
import sys,copy
U=Path(__file__).resolve().parent;O=U.parent
L=O/'word100_seated_fit_transfer_20261102'
R=O/'word100_event_transition_review_20261102'
OUT=Path('W:/WordsInSilence_derived/word100_event_temporal_finetune_20261102')
sys.path.insert(0,str(L))
from transfer_data import *
from evaluate_transfer import eval_rows
from evaluate_witness import observe,behavior,NORMAL,SUPPORTED
import evaluate_episode as previous
INITIAL=DEST/'runs/seated_variation/model.pt'
CONDITIONS=['old_frozen','new_frozen','old_adapt','new_adapt']

def model_for(tag):
    m=WitnessNet();m.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True));m.blank.requires_grad_(False)
    for module in [m.actor.context,m.actor.delta]:
        for p in module.parameters():p.requires_grad_(tag.endswith('adapt'))
    return m

def data():
    base,extras=load_pairs();original=base+extras['seated_variation'];added=copy.deepcopy(original)
    overlay=read(R/'ANNOTATION_OVERLAY.json')
    for p,h in overlay['pins'].items():assert sha(p)==h
    by={r['clip_id']:r for r in overlay['rows']}; changes=[]
    for i,r in enumerate(added):
        assert r['person'] in range(1,10)
        a0=r['episode'].copy();r0=r['role'].copy();mp=np.asarray(r['mapping'],int)
        valid=r['available'].copy()
        for f in np.flatnonzero(np.abs(np.diff(mp))>3)+1:valid[max(0,f-2):min(len(valid),f+3)]=False
        points=by[r['clip_id']]['points']; pts={p['frame']:p for p in points}
        for j,f in enumerate(mp):
            if not valid[j] or f not in pts:continue
            p=pts[f];prov=p['provenance']
            if prov=='existing_derived_FIT_annotation':continue
            # New source annotations have not been directly reviewed in the
            # artificial seated view. Keep them out of these additional views.
            if i>=len(base) and prov!='resolved_native_RGB_correction_0_to_1':continue
            if prov=='resolved_native_RGB_correction_0_to_1':assert r['episode'][j]==0
            else:assert r['episode'][j]<0 and r['role'][j]<0
            r['episode'][j]=p['action_family'];r['role'][j]=p['role']
        changes.append(dict(clip_id=r['clip_id'],kind=r['kind'],action_changes=int((a0!=r['episode']).sum()),role_changes=int((r0!=r['role']).sum()),
            extra_view=i>=len(base),main_unknown=int(((r['episode']<0)&(r['role']==1)&(r0!=r['role'])).sum())))
        assert np.array_equal(r['available'],original[i]['available'])
    assert len(original)==len(added)==252
    return original,added,changes

def weights(rows,key,count):
    a=np.concatenate([r[key][r[key]>=0] for r in rows]);n=np.bincount(a,minlength=count)
    return torch.tensor(np.clip(n.mean()/np.maximum(n,1),.25,4),dtype=torch.float32)

def summary(report):
    raw=[p for p in report['raw_points'] if p['kind'] in NORMAL and p['y']<8]
    used=[p for p in report['points'] if p['kind'] in NORMAL and p['y']<8]
    ns=[s for s in report['sequences'] if s['kind'] in NORMAL]
    rev=[s for s in report['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']]
    rests=[p for p in report['role_points'] if p['kind'] in NORMAL and p['y']==2]
    return dict(raw_correct=sum(p['y']==p['pred'] for p in raw),major_known=len(raw),used_correct=sum(p['y']==p['pred'] for p in used),
        family_used={str(g):sum(p['y']==g and p['pred']==g for p in used) for g in range(8)},
        normal_order=sum(s['status']==SUPPORTED and bool(s['precedences']) for s in ns),
        clean_order=sum(s['status']==SUPPORTED and bool(s['precedences']) and not(s['evaluation_action_errors'] or s['raw_evaluation_action_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ns),
        short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in rev),reverse_false_pass=sum(s['status']==SUPPORTED for s in rev),
        false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns),rest_known=len(rests),rest_exact=sum(p['pred']==2 for p in rests),
        rest_false_main=sum(p['pred']==1 for p in rests),normal_cases=len(ns),head_median_seconds=report['head_median_seconds'])
