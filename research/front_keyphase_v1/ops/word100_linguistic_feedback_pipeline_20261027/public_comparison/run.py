from pathlib import Path
import hashlib,json,traceback,html
import numpy as np
from measure import features,compare,ANGLE_NAMES,TRIPLETS
U=Path(__file__).resolve().parent;OPS=U.parent.parent
OLD=OPS/'word100_face_body_stage_20261027';ALIGN=OPS/'word100_face_relation_alignment_20261027'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d): (U/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def contracts():
    rng=np.random.default_rng(17);xy=rng.normal(size=(6,75,2))*100;v=np.ones((6,75),bool);a=features(xy,v);checks=[]
    b=features(xy*2.7+[250,-63],v)
    assert np.allclose(a['angle'],b['angle'],equal_nan=True,atol=1e-9)
    assert np.allclose(a['position'],b['position'],equal_nan=True,atol=1e-9);checks.append('positive_scale_translation_invariance_1e-9')
    support=np.ones((6,2,15),bool);support[2,0,4]=False;c=features(xy,v,support)
    assert np.isnan(c['angle'][2,0,4]) and np.array_equal(c['angle'][2,0,[3,5]],a['angle'][2,0,[3,5]])
    assert np.array_equal(c['position'],a['position']);checks.append('independent_angle_support_retains_neighbors_and_position')
    crossed=xy.copy();crossed[:,33:54]=xy[:,33:54]+[500,0];crossed[:,54:75]=xy[:,54:75]-[500,0]
    before=crossed.copy();x=features(crossed,v);assert np.array_equal(crossed,before)
    assert np.allclose(x['angle'],a['angle'],atol=1e-9);checks.append('crossing_does_not_swap_identity_or_mutate_points')
    planned=np.ones(6,bool);m=np.arange(6);base=compare(a['angle'],a['angle_valid'],a['angle'],a['angle_valid'],m,planned)[0]
    mv=v.copy();mv[:,33:54]=False;miss=features(xy,mv);drop=compare(miss['angle'],miss['angle_valid'],a['angle'],a['angle_valid'],m,planned)[0]
    assert base['planned']==drop['planned']==180 and drop['jointly_observed']==90 and drop['insufficient']==90
    checks.append('query_hand_missing_fixed_denominator_180_observed_90')
    rr=compare(a['angle'],a['angle_valid'],miss['angle'],miss['angle_valid'],m,planned)[0]
    assert rr['planned']==180 and rr['reference_observed']==90;checks.append('reference_hand_missing_fixed_denominator')
    none=features(xy,np.zeros_like(v));empty=compare(none['angle'],none['angle_valid'],none['angle'],none['angle_valid'],m,planned)[0]
    assert empty['jointly_observed']==0 and empty['median_abs_difference'] is None;checks.append('all_missing_no_zero_difference_success')
    shifted=a['angle'].copy();shifted[:,0,4]+=12;ss,dd,_=compare(shifted,a['angle_valid'],a['angle'],a['angle_valid'],m,planned)
    assert np.allclose(dd[:,0,4],12) and np.all(dd[:,1]==0);checks.append('changed_feature_only_descriptive_delta_no_label')
    nanxy=xy.copy();nanxy[0,40]=np.nan;z=features(nanxy,v)
    assert not z['angle_valid'][0,0,4] and z['position_valid'][0,0].all();checks.append('nonfinite_joint_is_not_observed_position_kept')
    degen=xy.copy();degen[0,39]=degen[0,38];de=features(degen,v);assert not de['angle_valid'][0,0,4];checks.append('degenerate_angle_rejected')
    return checks

def svg(q,lines,planned,n):
    # Fixed examples, not selected by low errors. One angle is a measurement illustration.
    vals=[x[1] for x in lines];colors=['#172554','#059669','#b45309'];w=1000;h=380;left=65;top=60;pw=900;ph=240
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/>',f'<text x="32" y="28" font-family="sans-serif" font-size="18">{html.escape(q["label"])} · right index middle · projected 2D bend (degrees)</text>']
    for g in range(0,181,45):
        y=top+ph*(1-g/180);out.append(f'<path d="M{left} {y}H{left+pw}" stroke="#e2e8f0"/><text x="20" y="{y+4}" font-size="12">{g}</text>')
    for k,(label,v) in enumerate(lines):
        commands=[];openpath=False
        for i,x in enumerate(v):
            if not planned[i] or not np.isfinite(x):openpath=False;continue
            commands.append(('L' if openpath else 'M')+f'{left+pw*i/max(1,n-1):.2f} {top+ph*(1-x/180):.2f}');openpath=True
        out.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{colors[k]}" stroke-width="2"/>')
        out.append(f'<text x="{35+330*k}" y="335" font-size="13" fill="{colors[k]}">{html.escape(label)}</text>')
    out.append('<text x="35" y="365" font-size="12">Gaps = unavailable evidence. Cached alignment; no correctness threshold or physical joint angle claim.</text></svg>')
    return ''.join(out)

def main():
    if (U/'RESULT.json').exists():raise FileExistsError('Saved comparison must not be silently recomputed')
    protocol=U.parent/'review/EVAL_PROTOCOL.json';read(protocol)
    frozen=read(ALIGN/'FROZEN.json');rows=frozen['public_rows'];byid={r['clip_id']:r for r in rows}
    assert len(rows)==18 and {r['signer'] for r in rows}=={1,2,3}
    paths={str(protocol),str(ALIGN/'FROZEN.json'),str(U/'measure.py'),str(U/'run.py')}
    paths.update(frozen['protected'])
    for r in rows:
        cid=r['clip_id'];paths.update(map(str,[OLD/'data'/cid/'masked.npz',ALIGN/'data'/f'{cid}_evidence.npz',ALIGN/'data'/f'{cid}_reference_comparison.npz']))
    pins={p:sha(p) for p in paths};save('BEFORE.json',pins)
    for p,h in frozen['protected'].items():assert pins[p]==h,('protected_changed_before',p)
    tests=contracts();cache={};result=[];graphcount=0;disagreement=[]
    (U/'data').mkdir(exist_ok=True)
    for r in rows:
        cid=r['clip_id'];obs=arrays(OLD/'data'/cid/'masked.npz');ev=arrays(ALIGN/'data'/f'{cid}_evidence.npz')
        assert np.array_equal(obs['timestamp_us'],ev['timestamp_us']);f=features(obs['xy_px'],obs['usable'])
        planned=np.zeros(len(obs['frame_id']),bool)
        for start,end in r['intervals']:planned|=(obs['timestamp_us']>=start*1e6)&(obs['timestamp_us']<=end*1e6)
        cache[cid]=(obs,ev,f,planned)
        np.savez_compressed(U/'data'/f'{cid}_features.npz',**f,frame_id=obs['frame_id'],timestamp_us=obs['timestamp_us'],planned=planned)
    for q in rows:
        cid=q['clip_id'];qo,qe,qf,planned=cache[cid];align=arrays(ALIGN/'data'/f'{cid}_reference_comparison.npz')
        assert len(align['reference_ids'])==2
        graph=[('query '+cid,qf['angle'][:,1,4])];perrefs=[]
        for j,rid in enumerate(align['reference_ids'].tolist()):
            r=byid[rid];assert q['signer']!=r['signer'] and q['word']==r['word'] and q['source']['sha256']!=r['source']['sha256']
            ro,re,rf,_=cache[rid];m=align['reference_rows'][j];idx=np.clip(m,0,len(ro['frame_id'])-1);mapped=(m>=0)&(m<len(ro['frame_id']))
            stats={};arraysout={'query_frame_id':qo['frame_id'],'query_timestamp_us':qo['timestamp_us'],'reference_row':m,'planned':planned}
            for key,aa,av,bb,bv,vector in [('angles',qf['angle'],qf['angle_valid'],rf['angle'],rf['angle_valid'],False),('body_position',qf['position'],qf['position_valid'],rf['position'],rf['position_valid'],True),('face_distance',qe['distance'],qe['available']&np.isfinite(qe['distance']),re['distance'],re['available']&np.isfinite(re['distance']),False),('face_direct',qe['distance'],qe['direct_comparable']&np.isfinite(qe['distance']),re['distance'],re['direct_comparable']&np.isfinite(re['distance']),False)]:
                st,delta,both=compare(aa,av,bb,bv,m,planned,vector);stats[key]=st;arraysout[key+'_difference']=delta;arraysout[key+'_both']=both
            arraysout['query_angles']=qf['angle'];arraysout['reference_angles']=rf['angle'][idx];arraysout['reference_angles'][~mapped]=np.nan
            np.savez_compressed(U/'data'/f'{cid}__{rid}_comparison.npz',**arraysout)
            result.append({'query':cid,'reference':rid,'word_id':q['word'],'label':q['label'],'query_source_sha256':q['source']['sha256'],'reference_source_sha256':r['source']['sha256'],'query_interval_seconds':q['intervals'],'stats':stats,'judgment':'DESCRIPTIVE_ONLY','physical_contact':'UNKNOWN','linguistic_correctness':'UNKNOWN','alignment':'cached B mapping; actual phase correctness unverified'})
            refangle=rf['angle'][idx].copy();refangle[~mapped]=np.nan;perrefs.append(refangle);graph.append(('reference '+rid,refangle[:,1,4]))
        refsok=np.isfinite(perrefs[0])&np.isfinite(perrefs[1])&planned[:,None,None];diff=np.abs(perrefs[0]-perrefs[1]);vv=diff[refsok]
        disagreement.append({'query':cid,'reference_pair_joint_angle_slots':int(refsok.sum()),'reference_median_absolute_angle_difference_degrees':float(np.median(vv)) if len(vv) else None,'status':'INDIVIDUAL_REFERENCES_RETAINED_NO_AVERAGE_CANONICAL_OR_TOLERANCE'})
        if q['signer']==1 and graphcount<3:
            (U/f'example_{graphcount+1}.svg').write_text(svg(q,graph,planned,len(qo['frame_id'])),encoding='utf-8');graphcount+=1
    assert len(result)==36
    totals={}
    for key in ('angles','body_position','face_distance','face_direct'):
        totals[key]={k:sum(r['stats'][key][k] for r in result) for k in ('planned','query_observed','reference_observed','jointly_observed','insufficient')}
        for name,num in [('query_coverage','query_observed'),('reference_coverage','reference_observed'),('joint_coverage','jointly_observed')]:totals[key][name]=totals[key][num]/totals[key]['planned']
    after={p:sha(p) for p in pins};assert after==pins;save('AFTER.json',after)
    save('RESULT.json',{'status':'COMPLETE_DESCRIPTIVE_PUBLIC_REPLAY','clips':18,'expressions':6,'directed_pairs':36,'totals':totals,'pairs':result,'reference_disagreement':disagreement,
       'units':{'angles':'query annotation frame × hand × 15 projected angles × individual reference; fixed even when missing','body_position':'query annotation frame × hand × 3 entities(hand wrist, pose wrist, elbow) × reference','face_distance':'query annotation frame × hand × 10 entities × 6 face regions × reference','face_direct':'same fixed face-distance units, stricter saved direct_comparable mask'},
       'denominator_warning':'Technical comparison denominator, not required active-hand or rest/linguistic denominator. Annotation intervals are fixed, not trimmed by query visibility.',
       'angle_definition':{'names':ANGLE_NAMES,'triplets':TRIPLETS.tolist(),'units':'degrees of 2D projected bend; not anatomical 3D angles'},
       'position_units':'shoulder-width normalized image plane relative to shoulder midpoint; translation and positive uniform scale invariant, not view invariant',
       'face_units':'existing per-face-region normalized projected distances; direct evidence reused conservatively, contact unproven',
       'contracts':tests,'source_pins_unchanged':len(pins),'graphs':graphcount,'false_correction_rate':None,'verified_real_error_labels':0,'trained_models':0,
       'no_new_alignment':True,'no_classifier_inference':True,'no_linguistic_judgments':True,'no_user_mentor_access':True,
       'limitations':['Previously inspected development inputs, not fresh test','Cached time correspondence does not prove matched semantic stages','Usable coordinates can still be wrong','One same-source word ID does not verify a dictionary variant','Two references do not establish acceptable linguistic limits','No actual wrong-execution labels, so feedback error recall and false-correction accuracy are NA']})
    save('CONTRACT_CHECKS.json',{'status':'PASS','count':len(tests),'checks':tests,'tolerance':1e-9,'source_pins_unchanged':True,'independent_human_ground_truth':False})
    print(json.dumps({'pairs':len(result),'totals':totals,'checks':len(tests),'pins':len(pins)},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception:
        p=U/'FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
