"""Locate remaining causes from stored outputs, without changing scores."""
from sequence_model import *
from collections import Counter
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
def main():
    assert not (J/'DIAGNOSIS.json').exists();v=read(J/'VERIFIED.json');out={}
    for name in v['conditions']:
        r=read(J/'evaluations'/f'{name}.json');raw={(p['key'],p['frame']):p for p in r['raw_points']}
        table=Counter();native=[]
        for p in r['points']:
            if p['kind'] in NORMAL and p['y']<8 and p['pred']!=p['y']:table[(p['y'],p['pred'])]+=1
        for sq in r['sequences']:
            if sq['kind']!='native':continue
            p=[p for p in r['points'] if p['key']==sq['key'] and p['y']<8]
            native.append(dict(key=sq['key'],word=sq['word'],status=sq['status'],action_errors=sq['evaluation_action_errors'],
                direct_major_known=len(p),direct_major_correct=sum(p['y']==p['pred'] for p in p),
                correct_content_suppressed=[p['frame'] for p in p if raw[(p['key'],p['frame'])]['pred']==p['y'] and p['pred']!=p['y']],
                rest_conflicts=sq['rest_action_conflict_frames'],unexpected=sq['unexpected_action_frames'],
                role_main_witnesses=sq['role_main_witnesses'],states=sq['states'],
                previous_reviewed_core_unconfirmed=sq['previous_reviewed_core_unconfirmed']))
        out[name]=dict(normal_status_counts=dict(Counter(sq['status'] for sq in r['sequences'] if sq['kind'] in NORMAL)),
            normal_confusions=[dict(truth=a,pred=b,count=n) for (a,b),n in sorted(table.items())],native=native)
    train=read(J/'TRAINING_RESULT.json')
    direct=[dict(tag=r['tag'],start_action_loss=r['history'][0]['losses'][1],end_action_loss=r['history'][-1]['losses'][1],end_sequence_loss=r['history'][-1]['losses'][3]) for r in train['runs']]
    record=dict(at=clock(),conditions=out,fit_losses=direct,
        interpretation_limits=['Low FIT loss with DEV errors supports an observed generalization gap, not a proven single cause.',
        'Sampled-witness CTC cannot establish boundaries or absence in unlabelled time. No amount of fitting those same witnesses turns unknown intervals into exhaustive supervision.',
        'Rest probability reuse and new binary activity head are a package relative to F; CE/CTC contrast isolates only added sequence objective.',
        'Repeated DEV is development evidence, not new independent generalization proof.'],
        pins={str(p):sha(p) for p in [J/'VERIFIED.json',J/'TRAINING_RESULT.json',J/'FROZEN.json']},new_inference=0,old_scores_revised=False,whole_goal_complete=False)
    exclusive(J/'DIAGNOSIS.json',record)
    print({k:v['normal_status_counts'] for k,v in out.items()});print(direct)
if __name__=='__main__':main()
