"""Separate saved-input arithmetic and preservation check, authored by root.
This is not an independent human/clinical/KSL correctness assessment.
"""
from pathlib import Path
import hashlib, json, math
import numpy as np
from evidence_route import describe_node, describe_comparison

U=Path(__file__).resolve().parent; OPS=U.parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
checks=0
def check(condition, message):
    global checks
    assert condition,message
    checks+=1

def main():
    frozen=read(U/'FROZEN.json')
    source=read(OPS/'word100_action_feedback_casebook_20261101/INPUTS.json')['rows']
    result=read(U/'MISSINGNESS.json')
    for alias,row in result['rows'].items():
        r=source[alias]
        with np.load(r['observation'],allow_pickle=False) as o, np.load(r['input'],allow_pickle=False) as a:
            check(o['frame_id'].tolist()==row['frame_ids'],'IDs')
            check(o['timestamp_us'].tolist()==row['timestamp_us'],'PTS')
            inside=[any(lo<=float(t)/1e6<=hi for lo,hi in r['intervals']) for t in o['timestamp_us']]
            check(inside==row['action_scope'],'existing annotation scope')
            check(np.isfinite(a['rgb']).all(1).tolist()==row['rgb_features_present'],'RGB finite flags')
            for side,offset in [('L',33),('R',54)]:
                counts=[sum(bool(o['observed'][i,k]) and all(math.isfinite(float(v)) for v in o['xy_px'][i,k])
                            for k in range(offset,offset+21)) for i in range(len(o['frame_id']))]
                check(counts==row['sides'][side]['observed_points'],'observed count')
                names=r['feature_names']; wrist=[names.index(side+'_hand_wrist_'+d) for d in 'xy']
                angles=[j for j,n in enumerate(names) if n.startswith(side+'_') and '_angle' in n and 'elbow' not in n]
                gates=[all(bool(a['mask'][i,j]) for j in wrist) and sum(bool(a['mask'][i,j]) for j in angles)>=10
                       for i in range(len(counts))]
                check(gates==row['sides'][side]['saved_shape_gate'],'gate unchanged')
                for scope,selection in [('full',[True]*len(counts)),('annotated_action',inside)]:
                    s=row['sides'][side]['scopes'][scope]
                    check(s['frames']==sum(selection),'scope denominator')
                    check(s['no_observed_hand_frames']==sum(v==0 and yes for v,yes in zip(counts,selection)),'absence')
                    check(s['some_observed_hand_frames']+s['no_observed_hand_frames']==s['frames'],'partition')
                    check(s['saved_shape_gate_frames']==sum(g and yes for g,yes in zip(gates,selection)),'gate count')
                    expected=[int(o['frame_id'][i]) for i in range(len(counts)) if selection[i] and counts[i]==0]
                    actual=[]
                    for run in s['no_hand_runs']:
                        ids=list(range(run['first_frame'],run['last_frame']+1)); actual+=ids
                        check(len(ids)==run['frames'],'run length')
                        index=np.flatnonzero(o['frame_id']==ids[0])[0]
                        check(run['start_seconds']==o['timestamp_us'][index]/1e6,'run PTS')
                    check(actual==expected,'runs cover exactly missing samples')
    saved=read(OPS/'word100_feedback_causal_pilot_20261101/POINT_DIAGNOSIS.json')
    routed=read(U/'EVIDENCE_CHANNELS.json')
    check(set(saved['nodes'])==set(routed['nodes']),'node denominator')
    for key,node in saved['nodes'].items():
        got=routed['nodes'][key]
        check({k:got[k] for k in node}==node,'all saved fields preserved')
        for target,p in node['probabilities'].items():
            c=got['evidence_channels'][target]
            check(c['saved_comparison_value']==node['values'][target],'value support unchanged')
            check(c['model_probability']==p,'saved posterior exact')
            check(c['educational_correction'] is None and c['request_new_recording'] is None,'no verdict')
            if p is not None:
                check(c['model_estimate']==int(p>=.5),'saved threshold only')
    original={(r['seed'],r['opportunity'],q):c for r in saved['records'] for q,c in zip(r['query_nodes'],r['comparisons'])}
    check(len(original)==len(routed['comparisons']),'comparison denominator')
    for c in routed['comparisons']:
        e=c['evidence']; key=(c['seed'],c['opportunity'],c['query_node'])
        check(e['saved_comparison']==original[key],'comparison preserved')
        check(e['educational_correction'] is None and e['request_new_recording'] is None,'unknown not correction')
        check(not e['saved_support_promoted'] and not e['model_agreement_is_correct_action'],'not inflated')
    # Deliberate edge cases: no prediction, non-finite output, missing RGB,
    # and a high-confidence contradiction must not become correctness/re-recording.
    cases=[(None,True),(float('nan'),True),(.9999,False),(.9999,True)]
    for p,rgb in cases:
        node={'probabilities':{'L_open_four':p,'L_closed_four':p},
              'support':{'L_open_four':False,'L_closed_four':False},
              'values':{'L_open_four':None,'L_closed_four':None}}
        got=describe_node(node,rgb)
        for c in got['evidence_channels'].values():
            check(c['saved_comparison_value'] is None and not c['saved_detector_gate'],'no support invention')
            check(c['educational_correction'] is None and c['request_new_recording'] is None,'no decision invention')
        comp=describe_comparison({'status':'INSUFFICIENT_OBSERVATION'},got,got,'L_open_four')
        check(comp['educational_correction'] is None and not comp['saved_support_promoted'],'agreement is not correctness')
    for p,h in frozen['pins'].items():
        check(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'protected pin '+p)
    result={'status':'PASS','checks':checks,'verifier':'root separate saved-data checker',
            'independent_human_truth':False,'new_inference':0,'new_training':0,
            'tests':'counts, exact source preservation, denominators, unavailable/contradictory model estimates do not produce correction'}
    with (U/'VERIFY.json').open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
