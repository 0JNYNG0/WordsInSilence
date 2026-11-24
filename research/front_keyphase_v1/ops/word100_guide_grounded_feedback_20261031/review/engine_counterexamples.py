"""Downstream-only counterexamples. Imports only engine.py and NumPy, no model."""
from pathlib import Path
import importlib.util
import sys
import json
import hashlib
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('reviewed_downstream_engine',U/'engine.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)

def observation(alias,prob):
    prob=np.array(prob,dtype=float);n=len(prob)
    return dict(alias=alias,prob=prob,support=np.ones(prob.shape,bool),frame_ids=np.arange(n),times_us=np.arange(n)*33333,sample_indices=np.arange(n),intervals=[[0.,1.]])

targets=['R_open_four','R_index_only']
clause=dict(id='c',feature=targets[0],source_span='synthetic text',source_url='synthetic://not-a-real-source',link_status='partial_supported',expected=1)
refs=[]
for alias in ['ref1','ref2']:
    r=observation(alias,[[.9,.1],[.9,.1],[.9,.1]])
    r.update(anchors={targets[0]:[1]},reference_to_query=[0,1,2]);refs.append(r)

def run():
    findings=[]
    q=observation('nan_query',[[.9,.1],[np.nan,.1],[.9,.1]])
    out=engine.compare_clause(clause,refs,q,targets)
    try:json.dumps(out,allow_nan=False);safe=True;error=None
    except ValueError as e:safe=False;error=str(e)
    findings.append(dict(id='NONFINITE_QUERY_SERIALIZATION',pass_contract=safe,state=out['references'][0]['cells'][0]['state'],serialization_error=error,expected='Missing/nonfinite query evidence is UNOBSERVED and JSON null; source-linked output remains serializable.'))
    q=observation('bad_status_query',[[.9,.1],[.9,.1],[.9,.1]])
    unknown=dict(clause,link_status='pending_unrecognized')
    out=engine.compare_clause(unknown,refs,q,targets)
    findings.append(dict(id='UNRECOGNIZED_SOURCE_STATUS',pass_contract=out['status']=='SOURCE_LINK_WITHHELD',actual=out['status'],expected='An unrecognized source link status cannot enable source-linked assertions.'))
    q=observation('infinite_order',[[np.inf,.1],[np.inf,.1],[.1,.1],[.1,.9],[.1,.9],[.1,.1]])
    order=dict(id='order',before=targets[0],after=targets[1],link_status='proxy_only',source_span='synthetic sequence')
    out=engine.check_order(order,q,targets)
    findings.append(dict(id='INFINITE_ORDER_EVIDENCE',pass_contract=not out['before_events'] and out['status']!='DESCRIBED_ORDER_PATTERN_OBSERVED',actual=out['status'],before_events=out['before_events'],expected='Nonfinite probability cannot produce a confident positive temporal event.'))
    result=dict(status='PASS' if all(x['pass_contract'] for x in findings) else 'COUNTEREXAMPLES_FOUND',engine_sha256=hashlib.sha256((U/'engine.py').read_bytes()).hexdigest(),new_model_inference=0,training=0,video_decode=0,imports='Only engine.py (NumPy downstream functions); no run.py, feedback.py, torch, checkpoint or source data imports.',cases=findings)
    p=U/'review/ENGINE_COUNTEREXAMPLES_INITIAL.json'
    with p.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':run()
