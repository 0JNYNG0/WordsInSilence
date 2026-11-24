"""Separate saved-output arithmetic audit. Never loads a learned model."""
from prepare import *

def main():
    cfg=read(U/'FROZEN.json');results=read(U/'RESULTS.json');checks=[]
    def check(name,passed):
        checks.append(dict(name=name,passed=bool(passed)))
    check('source_and_protection_pins',all(sha(p)==h for p,h in dict(cfg['source_pins'],**cfg['protected']).items()))
    check('code_pins',all(sha(U/p)==h for p,h in read(U/'CODE_PINS.json').items()))
    check('nine_fixed_runs',len(results['runs'])==9)
    by={(r['condition'],r['seed']):r for r in results['runs']};paired=[]
    def key(c):return (c['clip_id'],c['frame'],c['side'],c['attribute'])
    for run in results['runs']:
        name=f"{run['condition']}_{run['seed']}";cells=run['cells'];summary=run['summary']
        check(name+'_unique_cells',len({key(c) for c in cells})==len(cells)==503)
        check(name+'_arithmetic',sum(c['pred']==c['truth'] for c in cells)==summary['correct'])
        check(name+'_score_threshold',all((int(c['prob']>=.5) if c['available'] else None)==c['pred'] for c in cells))
        check(name+'_confidence_arithmetic',sum(c['confident'] and not c['correct'] for c in cells)==summary['confident_wrong'])
        check(name+'_per_hand',sum(s['correct'] for s in run['by_hand'].values())==summary['correct'])
        check(name+'_per_word',sum(s['correct'] for s in run['by_word'].values())==summary['correct'])
        check(name+'_per_attribute',sum(s['total'] for s in run['by_attribute'].values())==503)
        check(name+'_checkpoint',sha(U/'runs'/name/'model.pt')==run['model_sha256'])
        if run['condition']=='ordinary_only':continue
        base=by['ordinary_only',run['seed']];bc={key(c):c for c in base['cells']}
        check(name+'_identical_evaluation',all(key(c) in bc and c['truth']==bc[key(c)]['truth'] for c in cells))
        gained=[key(c) for c in cells if c['correct'] and not bc[key(c)]['correct']]
        lost=[key(c) for c in cells if not c['correct'] and bc[key(c)]['correct']]
        gate=run['summary']['correct']>=base['summary']['correct'] and all(run['by_hand'][s]['correct']>=base['by_hand'][s]['correct'] for s in ['L','R']) and all(run['by_attribute'][str(j)]['correct']>=base['by_attribute'][str(j)]['correct'] for j in range(3))
        paired.append(dict(condition=run['condition'],seed=run['seed'],baseline=base['summary']['correct'],candidate=summary['correct'],total=503,gained=gained,lost=lost,nonregression_gate=gate))
    previews=read(U/'EXPANSION_SOURCE_PREVIEW.json')
    check('candidate_original_hashes',all(sha(r['video'])==r['sha256'] for r in previews['rows']))
    check('candidate_fit_only',all(r['record_id']==1 for r in previews['rows']))
    check('six_new_preview_only',len(previews['rows'])==6)
    passed=all(c['passed'] for c in checks)
    write(U/'VERIFICATION.json',dict(status='PASS' if passed else 'FAIL',checks=checks,count=len(checks),reviewer='same Codex, separate saved-output arithmetic and hash verification, not independent human KSL expert',no_model_rerun=True))
    write(U/'DECISION.json',dict(status='NO_CLEAR_TRANSFER_GAIN_NO_PROMOTION',paired=paired,reason='Numeric finger supervision equals baseline for two seeds and loses one known ordinary shape cell for the third; no reliable accuracy gain established. This does not test direct individual-finger correctness in ordinary words.',production_change=False,whole_feedback_accuracy_measured=False,word_scope_expansion_measured=False))
    print('VERIFY',passed,len(checks));print(json.dumps([{k:v for k,v in r.items() if k not in ['gained','lost']} for r in paired],indent=2))
    assert passed
if __name__=='__main__':main()
