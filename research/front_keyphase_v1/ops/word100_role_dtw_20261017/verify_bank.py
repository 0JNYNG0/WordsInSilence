"""Stdlib recomputation of all new reference scales, independent of service code."""
import json,statistics,math,hashlib
from pathlib import Path
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8'))
b=read(U/'ROLE_BANK.json'); pairs=read(U/'TRAIN_PAIRS.json'); roles=read(U/'HAND_ROLES.json')['words']
checks=[]
for word,item in b['words'].items():
    assert item['required_sides']==roles[word]['required_sides']
    assert [x['signer_id'] for x in item['references']]==list(range(1,10))
    assert all(x['role']=='train_candidate' for x in item['references'])
    if len(item['required_sides'])==2:continue
    required=item['required_sides'][0]
    for key in ['handshape','location','movement']:
        perquery=[]
        for loo in item['leave_one_out']:
            vals=[]
            candidates=[p for p in pairs if p['query_clip']==loo['clip_id']]
            assert len(candidates)==8 and len({p['reference_clip'] for p in candidates})==8
            assert all(p['query_clip']!=p['reference_clip'] for p in candidates)
            for pair in candidates:
                if pair['result']['status']!='OK':continue
                metric=next(m for m in pair['result']['metrics'] if m['key']==key)
                d=next(s for s in metric['sides'] if s['side']==required)
                if d['value'] is not None and math.isfinite(d['value']) and d['support_fraction']>=.5 and d.get('query_distinct_frames',0)>=8 and d.get('reference_distinct_frames',0)>=8:vals.append(d['value'])
            value=statistics.median(vals) if len(vals)>=5 else None
            saved=next(m for m in loo['metrics'] if m['key']==key)['value'];assert value==saved
            if value is not None:perquery.append(value)
        base=item['baseline'][key];assert len(perquery)==base['n']==9
        assert statistics.median(perquery)==base['median'] and min(perquery)==base['min'] and max(perquery)==base['max']
        checks.append({'word':word,'metric':key,'nine_person_scale_exact':True})
assert len(checks)==9
for p,h in b['code_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
out={'status':'PASS','method':'separate stdlib median/support/self-exclusion verification; same Codex operator, no second agent review','checks':checks,'user_data_used':False}
(U/'BANK_INDEPENDENT_RECOMPUTE.json').write_text(json.dumps(out,indent=2),'utf-8');print('PASS: 9 scale distributions / 81 leave-one-person-out aggregates independently recomputed')
