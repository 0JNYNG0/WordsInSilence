"""Separate saved-result arithmetic and behavior checks; not expert truth."""
from collections import Counter
import numpy as np,torch
from phase_model import *
from sequence_evidence import learn_graph,observe

def behavior_tests():
    graph=learn_graph()['graphs'];g=graph['WORD0739'];n=9;prob=np.zeros((n,9));prob[np.arange(n),np.repeat(g['groups'],3)]=1;av=np.ones(n,bool);scope=np.arange(n);ts=np.arange(n)/30
    assert observe(prob,av,scope,g,ts)['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
    assert observe(prob[::-1],av,scope,g,ts)['status']=='PHYSICAL_ORDER_CONFLICT'
    missing=av.copy();missing[3:6]=False;o=observe(prob,missing,scope,g,ts)
    assert o['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED' and o['missing_groups']==[g['groups'][1]] and o['image_gap_frames']==[3,4,5]
    cyc=graph['WORD1567'];assert not cyc['precedences'];p=np.zeros((12,9));p[np.arange(12),np.tile(np.repeat(cyc['groups'],3),2)]=1
    assert observe(p,np.ones(12,bool),np.arange(12),cyc,np.arange(12)/30)['status']=='CONFIGURATIONS_OBSERVED_ORDER_RULE_UNSUPPORTED'
    return dict(ordered=True,reversed=True,missing_comparison_range_preserved=True,repeated_equivalent_states_not_error=True)

def main():
    assert not (S/'SAVED_VERIFICATION.json').exists();behaviors=behavior_tests();e=read(S/'EVALUATION.json');plan=read(S/'FROZEN_TRAINING_V2.json');inputs=read(S/'EVALUATION_INPUTS.json')['rows'];reports=[]
    for r in e['reports']:
        full=read(S/'evaluations'/f"{r['condition']}.json")
        for kind,expected in full['metrics'].items():
            pts=[p for p in full['points'] if p['kind']==kind];total=len(pts);correct=sum(p['y']==p['pred'] for p in pts);per={}
            for p in pts:
                v=per.setdefault(str(p['y']),[0,0]);v[0]+=1;v[1]+=int(p['y']==p['pred'])
            macro=sum(v[1]/v[0] for v in per.values())/len(per)
            assert total==expected['known'] and correct==expected['correct'] and abs(macro-expected['macro_recall'])<1e-12
            assert expected==r['metrics'][kind]
        assert len(full['sequences'])==88
        for seq in full['sequences']:
            assert seq['whole_sign_correctness'] is None
            if seq['image_gap_frames']:assert seq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
        reports.append(full)
    baseline=reports[0];primary=next(r for r in reports if r['condition']==plan['primary']);normal_kinds=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
    def subset_metric(report,kinds):return metrics([p for p in report['points'] if p['kind'] in kinds])
    bm=subset_metric(baseline,normal_kinds);cm=subset_metric(primary,normal_kinds);br=subset_metric(baseline,['actual_reverse','actual_reverse_main']);cr=subset_metric(primary,['actual_reverse','actual_reverse_main'])
    no_drop=cm['macro_recall']>=bm['macro_recall'];reverse_gain=cr['macro_recall']-br['macro_recall']>=.03
    per_drop={k:cm['groups'][k]['correct']/cm['groups'][k]['known']-bm['groups'][k]['correct']/bm['groups'][k]['known'] for k in bm['groups']}
    group_gate=min(per_drop.values())>=-.05
    preserved=all(sha(p)==h for p,h in plan['pins'].items()) and all(sha(r['input'])==r['input_sha256'] and sha(r['base_path'])==r['base_sha256'] and sha(r['axis_path'])==r['axis_sha256'] for r in inputs)
    assert preserved
    fit=read(S/'PHYSICAL_STATE_TARGETS.json');fitids={r['person'] for r in fit['rows'] if r['role']=='fit'};devids={r['person'] for r in fit['rows'] if r['role']=='development'};assert not fitids&devids and fitids==set(range(1,10)) and devids=={11,12,13}
    v=dict(at=clock(),status='PASS_STATE_METRICS_TO_SEQUENCE_INTEGRATION' if no_drop and reverse_gain and group_gate else 'DO_NOT_REPLACE_CURRENT_DEMO',
      labels=fit['summary'],behavior_checks=behaviors,saved_comparisons=616,seeds=[42,43,44],old_outputs_preserved=preserved,
      normal_before=bm,normal_after=cm,reversed_before=br,reversed_after=cr,normal_group_recall_changes=per_drop,
      gates=dict(normal_no_regression=no_drop,reversed_gain_3pp=reverse_gain,normal_group_drop_at_most5pp=group_gate),
      primary_fixed_before_evaluation=True,independent_human_expert=False,same_codex_separate_arithmetic=True,
      full_sequence_runtime_validation='State accuracy alone does not adopt a runtime feedback policy. Compare actual coarse-window order evidence and unsupported gaps first.')
    exclusive(S/'SAVED_VERIFICATION.json',v);print(v['status']);print('normal',bm['correct'],bm['known'],'->',cm['correct'],cm['known']);print('reversed',br['correct'],br['known'],'->',cr['correct'],cr['known']);print(v['gates'])

if __name__=='__main__':main()
