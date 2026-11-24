"""Saved-row arithmetic only; no encoder or neural inference."""
from common import *
from collections import defaultdict

def main():
    d=read(U/'FIT_DIAGNOSTIC.json'); p=read(U/'FIT_DIAGNOSTIC_PROTOCOL.json')
    assert not (U/'FIT_DIAGNOSTIC_VERIFIED.json').exists()
    checks=0; signatures=[]; summary={}
    for name,model in d['models'].items():
        assert sha(U/'runs'/name/'model.pt')==p['models'][name]; checks+=1
        rows=model['rows']; groups=defaultdict(lambda:dict(total=0,correct=0)); keys=[]
        for r in rows:
            assert r['target'] in (0,1,2) and r['pred'] in (0,1,2)
            keys.append((r['clip_id'],r['frame'],r['side'],r['finger'],r['target']))
            for k in ['all',r['origin'],'new' if r['new_label'] else 'old','state_'+str(r['target']),'hand_'+r['side'],'word_'+r['word'],'person_'+str(r['person'])]:
                groups[k]['total']+=1; groups[k]['correct']+=r['target']==r['pred']
            checks+=1
        assert len(set(keys))==len(keys)==1061
        assert dict(groups)==model['groups'] and groups['new']['total']==200
        signatures.append(sorted(keys)); checks+=3
        summary[name]={k:groups[k] for k in ('all','new','state_1')}
    assert all(s==signatures[0] for s in signatures)
    write(U/'FIT_DIAGNOSTIC_VERIFIED.json',dict(status='PASS',checks=checks+1,
        method='Root reaggregation from saved cell predictions, no inference. Separate-agent verification of DEV is RESULT_VERIFIED.json.',
        source_sha256=sha(U/'FIT_DIAGNOSTIC.json'),summary=summary,
        inference='High FIT correctness with lower historical DEV correctness supports a generalization problem; it does not identify its full cause.',
        not_independent_generalization_test=True))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
