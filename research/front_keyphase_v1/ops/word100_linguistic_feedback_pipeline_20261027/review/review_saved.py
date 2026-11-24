"""Independent saved-array review; no detectors, alignment, training or root writes."""
from pathlib import Path
import collections, hashlib, importlib.util, json, math
import numpy as np

U=Path(__file__).resolve().parents[1]; R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(x,msg):
    if not x:raise AssertionError(msg)

def main():
    p=read(U/'FEATURE_PROFILES.json'); src=read(U/'source_audit/SOURCE_ENTRY_DETAILS.json')['entries']; ix={(e['source_id'],str(e['entry_id'])):e for e in src}
    profiles=p['profiles']; links=[e for q in profiles for e in q['candidate_entries']]; claims=[c for e in links for c in e['claims']]
    check(len(profiles)==100 and len({q['word_id'] for q in profiles})==100,'profile set')
    check(all(q['verified_variant_id'] is None and q['required_features'] is None and q['acceptable_bounds'] is None and not q['training_eligible'] for q in profiles),'unknown rule retained')
    details_missing=[]; source_claim_count=collections.Counter()
    for e in links:
        raw=ix.get((e['source_id'],e['entry_id']))
        if raw is None:details_missing.append(e['source_id']+':'+e['entry_id']);check(not e['detail_available'] and not e['claims'],'missing detail claims')
        else:
            check(e['detail_available'] and e['description']==raw.get('description_text'),'detail mapping')
            for c in e['claims']:
                check(c['source']['url']==raw['url'] and c['source']['content_sha256']==raw['content_sha256'],'claim source')
                check(c['requiredness']=='unknown' and c['tolerance'] is None and not c['video_variant_verified'],'requiredness')
                source_claim_count[c['kind']]+=1
                if c['kind']=='text_mention_candidate':
                    check(c['text']==raw['description_text'],'text content')
                    for s in c['spans']:check(c['text'][s['start']:s['end']]==s['term'],'literal span')
                else:
                    rows=raw['typed_metadata'][c['source_field']]
                    if isinstance(rows,dict):rows=[rows]
                    check(any({k:x.get(k) for k in ('handId','handNm','handCode')}==c['source_value'] for x in rows),'typed literal value')
                    if c['source_field']=='signHandList':check(c['temporal_scope']=='initial_handshape_only','initial handshape')
    counts={'profiles':len(profiles),'with_lexical_candidates':sum(bool(q['candidate_entries']) for q in profiles),'candidate_entry_links':len(links),'source_claims':len(claims)}
    for k,v in counts.items():check(p['summary'][k]==v,('profile count',k))
    ready=read(U/'TRAINING_READINESS.json');check(ready['trained_model_count']==ready['gradient_updates']==0,'no training')
    check(not ready['descriptive_feature_training_gate']['pass'] and not ready['pedagogical_correctness_training_gate']['pass'],'correct gates')
    public=U/'public_comparison'; out=read(public/'RESULT.json'); old=U.parent/'word100_face_body_stage_20261027'; align=U.parent/'word100_face_relation_alignment_20261027'; frozen=read(align/'FROZEN.json'); rows=frozen['public_rows']; byid={x['clip_id']:x for x in rows}
    check(len(rows)==18 and len(out['pairs'])==36,'public dimensions')
    cache={}; maxangle=0.; maxpos=0.
    triples=[(0,1,2),(1,2,3),(2,3,4)]+[(0,b,b+1) if j==0 else (b+j-1,b+j,b+j+1) for b in (5,9,13,17) for j in range(3)]
    for row in rows:
        cid=row['clip_id']; obs=arrays(old/'data'/cid/'masked.npz'); ev=arrays(align/'data'/f'{cid}_evidence.npz'); saved=arrays(public/'data'/f'{cid}_features.npz'); xy=obs['xy_px']; valid=obs['usable']&np.isfinite(xy).all(-1)
        # Scalar implementation independent of measure.features.
        aa=np.full((len(xy),2,15),np.nan); pp=np.full((len(xy),2,3,2),np.nan)
        for t in range(len(xy)):
            for s,offset in enumerate((33,54)):
                for k,tr in enumerate(triples):
                    ids=[offset+i for i in tr]
                    if not all(valid[t,i] for i in ids):continue
                    a=xy[t,ids[0]].astype(float)-xy[t,ids[1]].astype(float); b=xy[t,ids[2]].astype(float)-xy[t,ids[1]].astype(float); na=math.hypot(*a);nb=math.hypot(*b)
                    if min(na,nb)>1e-9:aa[t,s,k]=180-math.degrees(math.acos(max(-1,min(1,float(a@b)/(na*nb)))))
                width=math.hypot(*(xy[t,11].astype(float)-xy[t,12].astype(float)))
                if valid[t,11] and valid[t,12] and width>1e-9:
                    center=(xy[t,11].astype(float)+xy[t,12].astype(float))/2
                    for k,i in enumerate((offset,15+s,13+s)):
                        if valid[t,i]:pp[t,s,k]=(xy[t,i]-center)/width
        check(np.array_equal(np.isfinite(aa),saved['angle_valid']),'angle validity');check(np.allclose(aa,saved['angle'],equal_nan=True,atol=1e-6),'scalar angles')
        check(np.array_equal(np.isfinite(pp).all(-1),saved['position_valid']),'position validity');check(np.allclose(pp,saved['position'],equal_nan=True,atol=1e-8),'body position')
        maxangle=max(maxangle,float(np.nanmax(np.abs(aa-saved['angle']))));maxpos=max(maxpos,float(np.nanmax(np.abs(pp-saved['position']))))
        plan=np.array([any(a*1e6<=t<=b*1e6 for a,b in row['intervals']) for t in obs['timestamp_us']]);check(np.array_equal(plan,saved['planned']),'independent fixed interval')
        cache[cid]=(aa,pp,ev,plan,obs)
    totals=collections.defaultdict(lambda:collections.Counter());pairkeys=set()
    for pair in out['pairs']:
        q,r=pair['query'],pair['reference'];check((q,r) not in pairkeys,'duplicate direction');pairkeys.add((q,r));qr,rr=byid[q],byid[r]
        check(qr['word']==rr['word'] and qr['signer']!=rr['signer'] and qr['source']['sha256']!=rr['source']['sha256'],'reference identity')
        qa,qp,qe,plan,qo=cache[q];ra,rp,re,_,ro=cache[r];z=arrays(public/'data'/f'{q}__{r}_comparison.npz');mapping=z['reference_row']; validmap=(mapping>=0)&(mapping<len(ra));ixmap=np.clip(mapping,0,len(ra)-1)
        original=arrays(align/'data'/f'{q}_reference_comparison.npz');j=original['reference_ids'].tolist().index(r);check(np.array_equal(mapping,original['reference_rows'][j]),'mapping unchanged')
        inputs=[('angles',qa,np.isfinite(qa),ra,np.isfinite(ra),False),('body_position',qp,np.isfinite(qp).all(-1),rp,np.isfinite(rp).all(-1),True)]
        for typ,mask in [('face_distance','available'),('face_direct','direct_comparable')]:inputs.append((typ,qe['distance'],qe[mask]&np.isfinite(qe['distance']),re['distance'],re[mask]&np.isfinite(re['distance']),False))
        for key,qv,qok,rv,rok,vec in inputs:
            qmask=qok&plan.reshape((-1,)+(1,)*(qok.ndim-1));rmask=rok[ixmap]&((plan&validmap).reshape((-1,)+(1,)*(qok.ndim-1)));both=qmask&rmask; diff=qv-rv[ixmap]; delta=np.sqrt((diff*diff).sum(-1)) if vec else np.abs(diff); delta=np.where(both,delta,np.nan);stats=pair['stats'][key]
            check(np.array_equal(both,z[key+'_both']),'both observed from raw masks');check(np.allclose(delta,z[key+'_difference'],equal_nan=True,atol=2e-6),'raw reconstructed delta')
            den=int(plan.sum()*math.prod(qok.shape[1:]));counts2={'planned':den,'query_observed':int(qmask.sum()),'reference_observed':int(rmask.sum()),'jointly_observed':int(both.sum()),'insufficient':den-int(both.sum())}
            for k,v in counts2.items():check(stats[k]==v,('pair count',key,k));totals[key][k]+=v
            vals=delta[both]
            if len(vals):
                check(abs(float(np.median(vals))-stats['median_abs_difference'])<2e-6,'median');check(abs(float(np.percentile(vals,90))-stats['p90_abs_difference'])<2e-6,'p90')
            else:check(stats['median_abs_difference'] is None and stats['p90_abs_difference'] is None,'empty no score')
        check(pair['linguistic_correctness']=='UNKNOWN' and pair['physical_contact']=='UNKNOWN','no judgment')
    for key,vals in totals.items():
        for k,v in vals.items():check(out['totals'][key][k]==v,'aggregate total')
        for c,n in [('query_coverage','query_observed'),('reference_coverage','reference_observed'),('joint_coverage','jointly_observed')]:check(abs(out['totals'][key][c]-vals[n]/vals['planned'])<1e-12,'coverage')
    pins=read(public/'BEFORE.json');check(pins==read(public/'AFTER.json'),'before after pins')
    for path,h in pins.items():check(digest(path)==h,('pin',path))
    result={'status':'PASS_SAVED_DESCRIPTIVE_EVIDENCE_CHECK','reviewer':'Codex /root/linguistic_eval_review separate from implementation','profile_counts':counts,'claim_kinds':dict(source_claim_count),'detail_missing_links':details_missing,'unique_current_profile_entry_keys':len({(x['source_id'],x['entry_id']) for x in links}),'unique_retrieved_entry_keys':len(ix),'linked_detail_entry_keys':len({(x['source_id'],x['entry_id']) for x in links if x['detail_available']}),'public_clips':len(rows),'independent_pairs':len(pairkeys),'totals':{k:dict(v) for k,v in totals.items()},'max_scalar_angle_difference_degrees':maxangle,'max_position_difference':maxpos,'unchanged_pins':len(pins),'training_gates_checked':True,'verified_error_labels':0,'claim':'Saved arithmetic and contracts verified; not human linguistic or landmark truth','reviewed_code_hashes':{str(x.relative_to(U)):digest(x) for x in [U/'feature_profiles.py',U/'validate_profiles.py',public/'run.py',public/'measure.py',public/'verify_saved.py']}}
    (R/'SAVED_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
