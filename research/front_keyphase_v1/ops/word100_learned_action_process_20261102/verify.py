"""Saved-output verification; no training, extraction or model execution."""
from process_model import *
from evaluate import NORMAL

def metric(points):
    groups={}
    for p in points:
        a=groups.setdefault(str(p['y']),dict(known=0,correct=0));a['known']+=1;a['correct']+=int(p['y']==p['pred'])
    return dict(known=len(points),correct=sum(v['correct'] for v in groups.values()),macro_recall=sum(v['correct']/v['known'] for v in groups.values())/len(groups),groups=groups)

def same(a,b):
    assert a['known']==b['known'] and a['correct']==b['correct'] and a['groups']==b['groups']
    assert abs(a['macro_recall']-b['macro_recall'])<1e-12

def main():
    assert not (B/'VERIFIED.json').exists();e=read(B/'EVALUATION.json');p=read(B/'FROZEN.json');reports={};count=0
    meta={r['key']:r for r in read(B/'EVALUATION_INPUTS.json')['rows']}
    for rr in e['reports']:
        r=read(B/'evaluations'/f"{rr['condition']}.json");reports[rr['condition']]=r
        for kind,m in r['metrics'].items():same(metric([p for p in r['points'] if p['kind']==kind]),m)
        same(metric(r['auxiliary_physical_points']),r['physical_metrics']);same(metric(r['pair_points']),r['pair_metrics'])
        assert len(r['sequences'])==len(r['feedback'])==88
        for seq,fb in zip(r['sequences'],r['feedback']):
            npz=np.load(B/'predictions'/rr['condition']/f"{seq['key']}.npz");baseline=np.load(T/'predictions/current_42'/f"{seq['key']}.npz")
            assert np.array_equal(npz['current_probability'],baseline['probability']);n=len(npz['frame_id'])
            assert seq['comparison_frames']==n and seq['comparison_frame_range']==fb['full_native_frame_range']==[0,n-1]
            assert seq['whole_sign_correctness'] is None and fb['whole_sign_correctness'] is None
            base=read(meta[seq['key']]['base_path'])
            for stage,prior in zip(fb['guide_time_evidence'],base['stages']):
                for k in ['features','relations','frame_ids','source_phrases','start_seconds','end_seconds']:
                    assert stage[k]==prior.get(k,[])
            if not npz['image_available'].all():assert seq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED' and fb['status']=='OBSERVATION_GAP_NOT_CERTIFIED'
            count+=1
    r=reports[p['primary']];normal=metric([q for q in r['points'] if q['kind'] in NORMAL]);core=normal['groups']['1']
    ss=[q for q in r['sequences'] if q['kind'] in NORMAL];ok={q['key'] for q in ss if q['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'};bad=[q['key'] for q in ss if q['status']=='PHYSICAL_ORDER_CONFLICT']
    old_report=read(A/'evaluations/temporal_42.json');before={q['key'] for q in old_report['sequences'] if q['kind'] in NORMAL and q['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'}
    rev=[q for q in r['sequences'] if q['kind']=='actual_reverse_main' and q['precedences']];detected=[q['key'] for q in rev if q['status']=='PHYSICAL_ORDER_CONFLICT'];false=[q['key'] for q in rev if q['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED']
    cuts=[q for q in r['feedback'] if q['kind']=='action_cut' and q['word']=='WORD1133']
    assert len(cuts)==1
    gates=dict(role_macro=normal['macro_recall']>=.85,core_recall=core['correct']/core['known']>=.95,normal_order=len(ok)>=21,
      normal_previous_confirmations_preserved=not(before-ok),normal_false_order=not bad,local_reverse=len(detected)>=2,reverse_false_support=not false,
      cut_motion_false_completion=all(not q['movement_confirmed'] and q['status']!='SUPPORTED_REVIEWED_ORDER_AND_MOVEMENT' and not q['combined_supported'] for q in cuts),
      image_gaps_preserved=True,prior_observations_preserved=True)
    for path,h in e['pins'].items():assert sha(path)==h
    for path,h in p['pins'].items():assert sha(path)==h
    for path,h in read(B/'PREPARED.json')['pins'].items():assert sha(path)==h
    failures=[q for q in r['points'] if q['kind'] in NORMAL and q['y']==1 and q['pred']!=1]
    out=dict(at=clock(),status='PASS_TO_ISOLATED_RUNTIME' if all(gates.values()) else 'PROCESS_COMPONENT_ONLY_NOT_READY',gates=gates,primary=p['primary'],
      normal_role=normal,normal_core_recall=core['correct']/core['known'],normal_order_before=sorted(before),normal_order_after=sorted(ok),normal_gained=sorted(ok-before),normal_lost=sorted(before-ok),
      normal_false_order=bad,local_reverse_detected=detected,local_reverse_false_supported=false,core_loss=failures,
      image_gap_cases=sum(bool(q['image_gap_frames']) for q in r['sequences']),verified_arrays=count,
      normal_combined_supported=[q['key'] for q in r['feedback'] if q['kind'] in NORMAL and q['combined_supported']],
      head_median_seconds=r['head_median_seconds'],end_to_end_seconds=None,full_semantic_correctness_claimed=False,
      same_codex_saved_arithmetic=True,independent_human_expert=False,whole_goal_complete=False)
    exclusive(B/'VERIFIED.json',out);print(out['status'],gates,flush=True);print('role',normal['correct'],normal['known'],'order',len(before),'->',len(ok),'reverse',len(detected),'coreloss',len(failures),flush=True)

if __name__=='__main__':main()
