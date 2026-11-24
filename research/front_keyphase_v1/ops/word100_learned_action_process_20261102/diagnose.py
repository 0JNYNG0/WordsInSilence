"""Explain the saved comparison; no threshold/model selection or rerun."""
from collections import Counter
from process_model import *
from evaluate import NORMAL

def main():
    assert not (B/'DIAGNOSIS.json').exists();v=read(B/'VERIFIED.json');r=read(B/'evaluations'/f"{v['primary']}.json");a=read(A/'evaluations/temporal_42.json')
    points=[p for p in r['points'] if p['kind'] in NORMAL];oldpoints={(p['key'],p['frame']):p for p in a['points']}
    confusion=np.zeros((3,3),int)
    for p in points:confusion[p['y'],p['pred']]+=1
    differences=[]
    for p in points:
        q=oldpoints[p['key'],p['frame']]
        if p['pred']!=q['pred']:differences.append(dict(**p,before=q['pred'],before_probability=q['probability']))
    summaries={}
    for c in read(B/'FROZEN.json')['conditions']:
        for seed in [42,43,44]:
            tag=f'{c}_{seed}';q=read(B/'evaluations'/f'{tag}.json');ss=[x for x in q['sequences'] if x['kind'] in NORMAL];rev=[x for x in q['sequences'] if x['kind']=='actual_reverse_main' and x['precedences']]
            summaries[tag]=dict(native_role=[q['metrics']['native']['correct'],q['metrics']['native']['known']],normal_order=sum(x['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED' for x in ss),
              normal_false_conflicts=sum(x['status']=='PHYSICAL_ORDER_CONFLICT' for x in ss),local_reverse_detected=sum(x['status']=='PHYSICAL_ORDER_CONFLICT' for x in rev),
              auxiliary_pair_trained=c=='joint',pair_metrics=q['pair_metrics'] if c=='joint' else None)
    seq_remaining=[{k:q[k] for k in ['key','status','states','precedences','role_core_witnesses']} for q in r['sequences'] if q['kind'] in NORMAL and q['precedences'] and q['status']!='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED']
    exclusive(B/'DIAGNOSIS.json',dict(at=clock(),primary=v['primary'],normal_role_confusion=confusion.tolist(),transit_rest_errors=int(confusion[0,2]+confusion[2,0]),
      changed_points=differences,core_loss_original_moments=sorted({(p['key'].split('_F_')[0]+'_F',p['source_frame']) for p in v['core_loss']}),
      condition_summaries=summaries,remaining_sequence=seq_remaining,
      retained_movement_guard='Existing motion continuity proof is reused, not newly improved by this branch.',
      key_scope='Pair auxiliary classification learns direction between labelled physical endpoints; it cannot alone prove the intervening guide action was completed. Control pair is untrained and is not a fair accuracy baseline.',
      selection='Primary joint42 remained fixed even if another seed is better. DEV repeatedly exposed; no new independent generalization claim.'))
    print('diagnosis',summaries,flush=True)

if __name__=='__main__':main()
