"""Small tests for information loss, provenance and timing; numpy only."""
from pathlib import Path
import json
import numpy as np
from rate_core import rate_mapping,labels
U=Path(__file__).resolve().parent
def main():
    assert not(U/'MAPPING_TEST.json').exists()
    passed=[];ts=np.arange(12,dtype=np.int64)*33333
    for rate in [.8,1.2]:
        r=rate_mapping(ts,rate)
        assert np.all(np.diff(r['timestamp_us'])>0)and np.all(np.diff(r['base_query_index'])>=0)
        assert r['base_query_index'][0]==0 and r['base_query_index'][-1]==11
        assert abs(r['timestamp_us'][-1]-(ts[-1]-ts[0])/rate)<=.5
        passed.append(f'fixed_rate_{rate}_monotone_endpoints_duration')
    y=np.zeros((12,1),np.float32);mask=np.zeros_like(y);y[[4,5,6]]=1;mask[[0,1,4,5,6,10,11]]=1
    bands=np.zeros((12,1,2),bool);bands[3:5,0,0]=True;bands[7:9,0,1]=True
    source=dict(y=y,mask=mask,bands=bands,valid=np.ones_like(bands))
    d=labels(source,np.arange(12),np.zeros(12,int),0)
    assert {tuple(v)for v in d['spans'][d['positive']]}=={(3,7),(3,8),(4,7),(4,8)}
    passed.append('all_and_only_inherited_full_extents')
    ids=np.array([0,1,2,3,4,6,7,8,9,10,11]);d=labels(source,ids,np.zeros(len(ids),int),0)
    assert d['missing_witnesses'].tolist()==[5]and not d['positive'].any()and not d['crossing'].any()
    passed+=['deleted_witness_never_full','ordinary_sampling_gaps_not_splice_joins']
    pair={k:v.copy()for k,v in source.items()};pair['y'][5]=0;pair['mask'][5]=0
    ids=np.array([4,4,6,6,7,7]);d=labels(pair,ids,np.zeros(len(ids),int),0)
    assert not len(d['missing_witnesses'])and not d['positive'].any()and max(d['distinct_observed'])==3
    passed.append('duplicated_frames_cannot_satisfy_four_observations')
    ids=np.array([0,1,2,8,9,10,11,3,4,5,6,7]);blocks=np.array([0]*3+[1]*4+[2]*5)
    d=labels(source,ids,blocks,0,'splice')
    assert np.array_equal(d['negative'],d['eligible']&d['crossing'])
    assert {tuple(v)for v in d['spans'][d['positive']]}=={(7,11),(8,11)}
    passed+=['splice_provenance_separate_from_rate_mapping','contained_full_labels_preserved']
    assert not np.any(d['positive']&d['negative'])
    passed.append('disjoint_rank_targets')
    (U/'MAPPING_TEST.json').write_text(json.dumps(dict(status='PASS_NUMPY_ONLY',count=len(passed),checks=passed,
        no_encoder=True,no_training=True),indent=2)+'\n',encoding='utf-8')
    print('Mapping contracts PASS',len(passed))
if __name__=='__main__':main()
