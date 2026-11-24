"""Read-only arithmetic over saved diagnostic inputs/results. No model imports."""
from pathlib import Path
import json, hashlib, datetime, collections
import numpy as np

V=Path(__file__).resolve().parent
U=V.parent
O=U.parent
W=U/'dynamic_blocks'
R=O/'word100_candidate_role_learning_20261111'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
errors=[]
def check(ok,message):
    if not ok: errors.append(message)

targets=read(V/'sequence_target/TARGETS.json')
frozen=read(W/'FROZEN.json')
inputs=read(W/'INPUTS_COMPLETE.json')['rows']
results=read(W/'RESULTS.json')
complete=read(W/'COMPLETE.json')
old={r['clip_id']:r for r in read(R/'EVALUATION_INPUTS.json')['rows'] if r['kind']=='full_normal'}
truth={r['clip_id']:r for r in targets['rows'] if r.get('candidate')}
check(sha(V/'sequence_target/TARGETS.json')==frozen['target_sha256'],'target hash')
check(sha(U/'run_dynamic_block_diagnostic.py')==frozen['code_sha256'],'driver hash')
input_checks=[]
for r in inputs:
    cid=r['clip_id']; target=truth[cid]; t=target['candidate']; z=np.load(r['path']); before=np.load(old[cid]['path']); ids=z['source_indices']; n=len(ids)
    check(sha(r['path'])==r['sha256'],cid+' input hash')
    check(sha(r['baseline'])==r['baseline_sha256'],cid+' baseline hash')
    check(sha(old[cid]['path'])==old[cid]['sha256'],cid+' old input hash')
    check(sha(target['source'])==target['source_sha256']==old[cid]['source_sha256'],cid+' public source hash')
    check(np.array_equal(np.sort(ids),np.arange(n)),cid+' exact permutation')
    check(np.array_equal(ids,t['query_to_source_frame_ids']),cid+' query map')
    check(np.array_equal(before['x'][ids],z['x']),cid+' x permutation')
    check(np.array_equal(before['ts'],z['ts']) and np.all(np.diff(z['ts'])>0),cid+' timestamps')
    source_to_query=t['source_to_query_frame_ids']; inverse=np.argsort(ids)
    if isinstance(source_to_query,dict): source_to_query=[source_to_query[str(i)] for i in range(n)]
    check(np.array_equal(inverse,source_to_query),cid+' inverse map')
    joins=np.flatnonzero(np.diff(ids)!=1)+1
    check(np.array_equal(joins,[j['query_right'] for j in t['splice_joins'] if j['synthetic_discontinuity']]),cid+' join enumeration')
    blocks=[]
    for name,b in t['blocks'].items():
        a,e=b['query_frame_span']; s,f=b['source_span']
        check(np.array_equal(ids[a:e+1],np.arange(s,f+1)),cid+' block '+name)
        blocks.append(dict(name=name,source=[s,f],query=[a,e],internal_direction_preserved=True))
    sem=t['semantic_direction_order']; witnesses={}
    for name,source_frames in sem.get('normal_witness_source_frame_ids',{}).items():
        q=inverse[source_frames].tolist(); witnesses[name]=dict(source=source_frames,query=q)
        check(q==sem['permuted_witness_query_frame_ids'][name],cid+' witnesses '+name)
    # Same x but newly recomputed context must not silently reuse permuted old embeddings.
    changes={k:dict(exactly_old_permutation=bool(np.array_equal(z[k],before[k][ids])),maximum_absolute_change=float(np.abs(z[k]-before[k][ids]).max())) for k in ['fh','z']}
    check(all(not c['exactly_old_permutation'] for c in changes.values()),cid+' refreshed contextual features')
    b=read(r['baseline'])
    input_checks.append(dict(clip_id=cid,frames=n,blocks=blocks,joins=joins.tolist(),witnesses=witnesses,context_comparison=changes,
        saved_event_quality=False,saved_motion_evidence_cards=sum(c.get('motion_evidence') is not None for c in b['cards']),
        visible_dynamic_order_target=bool(sem.get('visible_dynamic_descriptor_order_negative_gold',False)),whole_sign_gold=False))

limited=[]; detailed=[]; candidate_checks=[]
for r in results:
    cid=r['clip_id']; t=truth[cid]['candidate']; ii=np.array(t['query_to_source_frame_ids']); joins=[j['query_right'] for j in t['splice_joins'] if j['synthetic_discontinuity']]
    for card in r['cards']:
        for group in card['groups']:
            eligible=[c for c in group['candidates'] if c['eligible'] and c['score']>=.65]
            expected=max(eligible,key=lambda c:c['score'])['span'] if eligible else None
            check(expected==group['span'],f'{cid} selected peak {card["id"]}')
            if cid.endswith('WORD0739_REAL11_F') and r['seed']==42 and r['condition']=='reviewed_roles':
                candidate_checks.append(dict(card=card['id'],selected=group['span'],candidates=[dict(c,source_indices=ii[c['span'][0]:c['span'][1]+1].tolist(),crosses_splice=any(c['span'][0]<j<=c['span'][1] for j in joins)) for c in group['candidates']]))
    for p in r['pairs']:
        a,b=p['spans']; prob=p['probability']
        state='association_unknown' if a is None or b is None else 'supported' if a[1]<b[0] and prob>=.65 else 'reversed' if b[1]<a[0] and prob<=.35 else 'ambiguous'
        cross=any(s[0]<j<=s[1] for s in [a,b] if s for j in joins)
        check(state==p['state'],cid+' pair state')
        check(cross==p['selected_span_crosses_splice'],cid+' splice flag')
        if p['limited_dynamic_target']:
            ba=t['blocks']['B']['query_frame_span']; ca=t['blocks']['C']['query_frame_span']
            mapped=None if a is None or b is None else ba[0]<=a[0]<=a[1]<=ba[1] and ca[0]<=b[0]<=b[1]<=ca[1]
            check(mapped==p['spans_within_target_occurrences'],cid+' target containment')
            row=dict(condition=r['condition'],seed=r['seed'],clip=cid,state=state,mapped=mapped,splice=cross)
            limited.append(row)
            detailed.append(dict(row,selected=p['spans'],selected_source_indices=[ii[s[0]:s[1]+1].tolist() if s else None for s in [a,b]],
                complete_dynamic_witness_coverage=[all(int(q) in range(s[0],s[1]+1) for q in t['semantic_direction_order']['permuted_witness_query_frame_ids'][name]) if s else False for name,s in zip(['B','C'],[a,b])]))
check(limited==complete['limited_pairs'],'complete compact arithmetic')
check(len(results)==36 and len(limited)==27,'result denominators')

# Geometry-only diagnostic, identical normalization to body_points/wrist_series.
# No threshold, classifier or quality head is run. Source joins are audit-only.
trajectory=[]
r=next(r for r in inputs if r['clip_id']=='NIA_SL_WORD0739_REAL11_F'); z=np.load(r['path']); x=z['x']; ids=z['source_indices']
p=x[:,:102].reshape(-1,51,2).astype(float); v=x[:,186:237]>.5; shoulder=v[:,1]&v[:,2]
scale=np.median(np.linalg.norm(p[shoulder,1]-p[shoulder,2],axis=1)); p=(p-(p[:,1]+p[:,2])[:,None]/2)/scale
height=-p[:,6,1]; arm=p[:,6]-p[:,4]; upright=-arm[:,1]/np.maximum(np.linalg.norm(arm,axis=1),1e-5)
for name,span,values,good in [('selected_close',[64,70],height,v[:,6]&shoulder),('real_B_candidate',[94,101],height,v[:,6]&shoulder),('selected_raise',[71,89],upright,v[:,4]&v[:,6]&shoulder),('other_raise_candidate',[102,107],upright,v[:,4]&v[:,6]&shoulder)]:
    a,b=span; delta=np.diff(values[a:b+1]); total=float(np.abs(delta).sum()); j=int(np.argmax(np.abs(delta)))
    # Reproduce geometry availability only, not its neural direction.
    valid=good[a:b+1]; eligibility=valid.mean()>=.8 and valid.sum()>=4 and valid[0] and valid[-1]
    trajectory.append(dict(name=name,span=span,source=ids[a:b+1].tolist(),all_values=values[a:b+1].tolist(),availability_contract_pass=bool(eligibility),
        signed_change=float(values[b]-values[a]),largest_adjacent_step=float(delta[j]),largest_step_query=[a+j,a+j+1],largest_step_source=[int(ids[a+j]),int(ids[a+j+1])],
        largest_step_fraction_of_total_variation=float(abs(delta[j])/total) if total else None,
        actual_classifier_output_available=False))

summary={c:{str(s):dict(collections.Counter((f'{r["state"]}|mapped={r["mapped"]}|splice={r["splice"]}') for r in limited if r['condition']==c and r['seed']==s)) for s in [42,43,44]} for c in frozen['conditions']}
out=dict(status='SAVED_DIAGNOSTIC_ARITHMETIC_VERIFIED_G2_NOT_PASSED',reviewer='Separate Codex artifact arithmetic and code review; source RGB labels inherited from another Codex review, not independent KSL expert truth',
    created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),model_inference=False,training=False,private_access=False,web_changed=False,threshold_tuning=False,
    inputs=input_checks,result_records=len(results),limited_pair_records=len(limited),unique_limited_public_clips=3,errors=errors,summary=summary,
    limited_pair_details=detailed,real11_candidates=candidate_checks,real11_geometry_only=trajectory,
    pins=[dict(path=str(p),sha256=sha(p)) for p in [V/'sequence_target/TARGETS.json',W/'FROZEN.json',W/'INPUTS_COMPLETE.json',W/'RESULTS.json',W/'COMPLETE.json',U/'run_dynamic_block_diagnostic.py',U/'interval_pipeline.py',U/'occurrence_binding.py']])
(V/'dynamic_block_arithmetic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(errors=errors,summary=summary,geometry=trajectory),ensure_ascii=False))
assert not errors, errors
