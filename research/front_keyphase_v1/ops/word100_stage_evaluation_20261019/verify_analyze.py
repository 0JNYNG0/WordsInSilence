"""Independent loop implementation and saved-record accounting; no fitting."""
from common import *
from itertools import combinations_with_replacement
from collections import Counter

def scalar_cost(a,b,axis,side):
    # Independent indexing and scalar-coordinate arithmetic.
    coord_start=18+42*side if axis=='global' else 102+42*side
    mask_start=195+21*side if axis=='global' else 237+21*side
    joints=list(range(21)) if axis=='global' else list(range(1,21))
    ai=[i for i in range(len(a)) if all(a[i,mask_start+j]==1 for j in joints)]
    bi=[i for i in range(len(b)) if all(b[i,mask_start+j]==1 for j in joints)]
    if not ai or not bi:return None,len(a)-len(ai),len(b)-len(bi)
    c=np.empty((len(ai),len(bi)),float)
    for ii,i in enumerate(ai):
      for jj,j in enumerate(bi):
        c[ii,jj]=sum(((float(a[i,coord_start+2*k])-float(b[j,coord_start+2*k]))**2+(float(a[i,coord_start+2*k+1])-float(b[j,coord_start+2*k+1]))**2)**.5 for k in joints)/len(joints)
    return c,len(a)-len(ai),len(b)-len(bi)

def slow_directed(c):
    prev=list(c[0])
    for i in range(1,len(c)):
        prev=[float(c[i,j])+min(prev[:j+1]) for j in range(c.shape[1])]
    return min(prev)/len(c)

def stats(values):
    a=np.array([x for x in values if x is not None],float)
    if not len(a):return {'n':0,'median':None,'q25':None,'q75':None,'min':None,'max':None}
    return {'n':len(a),'median':float(np.median(a)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),'min':float(a.min()),'max':float(a.max())}

def main():
    assert not (U/'INDEPENDENT.json').exists()
    s=read(U/'SCOPE.json');check(s['pins'])
    g=read(U/'GUARDED_GEOMETRY.json');p=read(U/'EVIDENCE_GUARD_PROTOCOL.json')
    assert p['guard_sha256']==sha(U/'evidence_geometry.py') and p['criteria_sha256']==sha(U/'criteria.py')
    assert p['code_sha256']==sha(U/'guarded_benchmark.py') and g['protocol_sha256']==sha(U/'EVIDENCE_GUARD_PROTOCOL.json')
    rows=g['rows'];assert len(rows)==3398
    keys={(r['reference'],r['query'],r['view'],r['condition']) for r in rows};assert len(keys)==len(rows)
    assert len({r['word_code'] for r in rows})==100
    assert Counter(r['view'] for r in rows)=={'sensor':1699,'masked':1699}
    allowed={r['clip_id'] for r in s['rows']}
    assert all(r['reference'] in allowed and r['query'] in allowed for r in rows)
    axes=['global_L','global_R','local_L','local_R'];clock=0;dwell=0;gated=0;measures=0
    lookup={(r['reference'],r['view'],r['condition']):r for r in rows if r['query']==r['reference']}
    for r in rows:
      assert set(r['metrics'])==set(axes)
      for axis,m in r['metrics'].items():
        assert m['semantic_verdict'] is None;measures+=1
        if m['ordered'] is not None:assert m['ordered']+1e-12>=m['bag'] and m['order_gap']>=0
        if r['condition'].startswith('clock_'):
          before=lookup[r['reference'],r['view'],'original']['metrics'][axis]
          assert before==m;clock+=1
        if r['condition']=='dwell_samples':
          before=lookup[r['reference'],r['view'],'original']['metrics'][axis]
          assert m['ordered']==before['ordered'] and m['status']==before['status'];dwell+=1
        if r['condition']=='mask_middle':
          assert m['query_uncomparable']>0;gated+=1
    assert (measures,clock,dwell,gated)==(13592,2400,800,800)
    # Check complete small numeric samples with independent scalar distances,
    # naive DP and exhaustive paths, rather than calling the same helper twice.
    from evidence_geometry import compare_hand
    from criteria import directed
    maxerr=0.;numeric=0;exhaustive=0
    with np.load(U/'NUMERIC_SAMPLES.npz',allow_pickle=False) as z:
      for name in z.files:
        x=z[name];small=x[np.rint(np.linspace(0,len(x)-1,8)).astype(int)]
        cases=[small,small[::-1].copy(),small[np.r_[0:2,4:8]],np.repeat(small,2,axis=0)]
        for y in cases:
          for axis in ['global','local']:
            for side in [0,1]:
              c,ma,mb=scalar_cost(small,y,axis,side);r=compare_hand(small,y,axis,side)
              assert ma==r['reference_uncomparable'] and mb==r['query_uncomparable']
              if c is None:assert r['ordered'] is None
              else:
                ordered=(slow_directed(c)+slow_directed(c.T))/2
                bag=(sum(min(row) for row in c)/len(c)+sum(min(row) for row in c.T)/len(c.T))/2
                maxerr=max(maxerr,abs(ordered-r['ordered']),abs(bag-r['bag']))
                if min(c.shape)>=2:
                  cc=c[:3,:3];best=min(sum(cc[i,j] for i,j in enumerate(path))/len(cc) for path in combinations_with_replacement(range(cc.shape[1]),len(cc)))
                  assert abs(best-directed(cc))<1e-12;exhaustive+=1
              numeric+=1
    assert maxerr<1e-12
    # Hand-local anchors are insufficient even if labelled present.
    only=np.zeros((8,280),np.float32);only[:,237]=1;only[:,258]=1
    for side in (0,1):assert compare_hand(only,only,'local',side)['status']=='INSUFFICIENT'
    summary={};overlap=[];gap_mask=[]
    for view in ['sensor','masked']:
      summary[view]={}
      for axis in axes:
        summary[view][axis]={}
        for condition in [*s['conditions'],'other_signer_same_word']:
          rr=[r['metrics'][axis] for r in rows if r['view']==view and r['condition']==condition]
          summary[view][axis][condition]={'pairs':len(rr),'distance':stats([m['ordered'] for m in rr]),'order_gap':stats([m['order_gap'] for m in rr]),
            'insufficient_pairs':sum(m['ordered'] is None for m in rr),'any_uncomparable_frames':sum(m['reference_uncomparable']>0 or m['query_uncomparable']>0 for m in rr)}
        for condition in ['omit_middle','swap_first_two','repeat_middle']:
          comparable=[];examples=[]
          for word in sorted({r['word_code'] for r in rows}):
            normal=[r['metrics'][axis]['ordered'] for r in rows if r['word_code']==word and r['view']==view and r['condition']=='other_signer_same_word' and r['metrics'][axis]['ordered'] is not None]
            altered=next(r for r in rows if r['word_code']==word and r['view']==view and r['condition']==condition)
            value=altered['metrics'][axis]['ordered']
            if normal and value is not None:
              lo=value<=float(np.median(normal));comparable.append(lo)
              if lo:examples.append({'label':altered['label'],'word_code':word,'edited_distance':value,'same_word_other_signer_median':float(np.median(normal))})
          overlap.append({'view':view,'axis':axis,'condition':condition,'eligible_words':len(comparable),'edited_le_same_word_person_median':sum(comparable),'examples':examples[:6],
            'meaning':'Descriptive overlap. Not sensitivity, specificity, error detection accuracy, or a chosen acceptance threshold.'})
        equal=0
        for ref in s['references']:
          x=lookup[ref['clip_id'],view,'omit_middle']['metrics'][axis];y=lookup[ref['clip_id'],view,'mask_middle']['metrics'][axis]
          if x['ordered']==y['ordered']:equal+=1
        gap_mask.append({'view':view,'axis':axis,'omission_vs_mask_same_distance':equal,'of':100,'observation_counts_different':True})
    # Verify catalog eligibility, not create unsupported expert labels.
    catalog=read(U/'PHASE_REVIEW_CATALOG.json')['items'];assert len(catalog)==100
    assert not any(c['expert_reviewed'] for c in catalog)
    write(U/'ANALYSIS.json',{'status':'COMPLETE_CRITERIA_DIAGNOSTIC_NOT_SEMANTIC_VALIDATION','at':now(),'pairs':len(rows),'metrics':measures,
      'summary':summary,'overlap':overlap,'omission_vs_missing':gap_mask,'clock_control_pairs':600,'dwell_control_pairs':200,'masked_gap_pairs':200,
      'normal_other_signer_pairs':1598,'public_source_clips':899,'words':100,'expert_reviewed_phase_contracts':0,
      'automatic_correct_incorrect_labels':0,'no_model_inference':True,'no_training':True,'no_user_data':True,
      'guarded_sha256':sha(U/'GUARDED_GEOMETRY.json'),'first_probe_preserved_sha256':sha(U/'GEOMETRY.json')})
    check(s['pins'])
    write(U/'INDEPENDENT.json',{'status':'PASS','at':now(),'source_pins_verified':len(s['pins']),'saved_metrics_checked':measures,'numeric_hand_axis_checks':numeric,
      'exhaustive_small_paths':exhaustive,'max_numeric_error':maxerr,'clock_invariance_metrics':clock,'dwell_invariance_metrics':dwell,'missing_evidence_retained':gated,
      'zero_anchor_cases_rejected':2,'analysis_sha256':sha(U/'ANALYSIS.json'),'guarded_sha256':sha(U/'GUARDED_GEOMETRY.json'),'verifier_sha256':sha(__file__),
      'verification':'Same Codex independent scalar distance, naive/exhaustive alignment and stored accounting. Not independent agent or KSL expert review.'})
    print(json.dumps({'status':'PASS','numeric':numeric,'exhaustive':exhaustive,'maxerror':maxerr,'overlap':[{k:r[k] for k in ['view','axis','condition','eligible_words','edited_le_same_word_person_median']} for r in overlap]},ensure_ascii=False))
if __name__=='__main__':main()
