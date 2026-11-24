"""Saved public-input diagnosis. No DP, training, detector, or runtime changes."""
import sys,os,json,hashlib,shutil,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
P=O/'word100_motion_phase_alignment_20261022';L=O/'word100_lopo_reference_range_20261022'
I=O/'word100_cross_person_motion_support_20261022'/'inputs'
EPS=1e-6
GROUPS={'angle':slice(0,15),'palm':slice(15,17),'elbow':slice(17,19),'wrist':slice(19,21)}
NORMAL=('identity','fast075','slow15','pace','pause')
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def check(pins):
    for p,v in pins.items():assert sha(p)==v,p
def channels(e):
    pos=np.concatenate([e['centre'],e['arm_xy'].reshape(-1,2,4)],2)
    rad=np.concatenate([np.repeat(e['radius'][:,:,None],2,2),np.zeros_like(pos[:,:,2:])],2)
    pv=np.concatenate([np.repeat(e['palm_valid'][:,:,None],2,2),np.repeat(e['arm_valid'],2,2)],2)
    lo=np.concatenate([e['angle_low'],pos-rad],2);hi=np.concatenate([e['angle_high'],pos+rad],2)
    valid=np.concatenate([e['angle_valid'],pv],2)&np.isfinite(lo)&np.isfinite(hi)
    return dict(lo=np.where(valid,lo,np.nan),hi=np.where(valid,hi,np.nan),valid=valid)
def expected(e,req):
    m=e['core'][:,None]&np.array([s in req for s in ('L','R')])[None,:]&~e['source_rest']
    return np.broadcast_to(m[:,:,None],(len(m),2,21)).copy()
def resample(c,ix):
    a=np.floor(ix).astype(int);b=np.ceil(ix).astype(int);w=(ix-a)[:,None,None]
    valid=c['valid'][a]&c['valid'][b]
    return dict(lo=np.where(valid,c['lo'][a]*(1-w)+c['lo'][b]*w,np.nan),hi=np.where(valid,c['hi'][a]*(1-w)+c['hi'][b]*w,np.nan),valid=valid)
def project(c,path,n,neighbors):
    lo=np.full((n,2,21),np.nan);hi=lo.copy();valid=np.zeros_like(lo,bool)
    for j in range(n):
        ids=np.unique(path[path[:,1]==j,0]);assert len(ids)
        v=c['valid'][ids];valid[j]=v.all(0)
        ls=[c['lo'][ids]];hs=[c['hi'][ids]];vs=[v]
        if neighbors:
            for step in (-1,1):
                ix=ids+step;keep=(ix>=0)&(ix<len(c['valid']));ix=ix[keep]
                ls.append(c['lo'][ix]);hs.append(c['hi'][ix]);vs.append(v[keep]&c['valid'][ix])
        vv=np.concatenate(vs);low=np.concatenate(ls);high=np.concatenate(hs)
        anyv=vv.any(0);lo[j]=np.where(anyv,np.where(vv,low,np.inf).min(0),np.nan);hi[j]=np.where(anyv,np.where(vv,high,-np.inf).max(0),np.nan)
    return dict(lo=lo,hi=hi,valid=valid)
def compare(p,t,domain):
    ok=domain&p['valid']&t['support'];gap=np.maximum(np.maximum(t['low']-p['hi'],p['lo']-t['high']),0.)
    bad=ok&(gap>EPS)
    return dict(comparable=ok,bad=bad,unknown=domain&~ok,within=ok&~bad,gap=np.where(ok,gap,np.nan))
def held_path(original,ix):
    # Evaluator-only frozen correspondence proxy. Nearest survivor, ties earlier.
    assert np.all(np.diff(ix)>0) and np.all(ix==np.rint(ix))
    allsrc=np.arange(int(ix[-1])+1);nearest=np.abs(allsrc[:,None]-ix[None,:]).argmin(1)
    path=np.column_stack([nearest[original[:,0]],original[:,1]])
    return np.unique(path,axis=0)
def count(z,inside):
    return {k:int((z[k]&inside).sum()) for k in ('comparable','bad','unknown','within')}
def contrasts(d,b,inside):
    return dict(new_bad=int((d['bad']&~b['bad']&inside).sum()),lost_observation=int((b['comparable']&~d['comparable']&inside).sum()),**count(d,inside))
def witness(altd,altb,actuald,actualb,inside):
    common=inside&altd['comparable']&altb['comparable']&actuald['comparable']&actualb['comparable']
    return common&altd['bad']&~altb['bad']&~(actuald['bad']&~actualb['bad'])
def source_info(e,c,req,deleted):
    domain=expected(e,req);target=domain[deleted];hdom=target[:,:,0]
    out=dict(deleted=deleted.tolist(),expected_hand_frames=int(hdom.sum()),raw_observed_hand_frames=int((e['observed'][deleted]&hdom).sum()),
        location_seen_hand_frames=int((e['location_seen'][deleted]&hdom).sum()),positive_rest_hand_frames=int(e['source_rest'][deleted].sum()),
        expected_angle_cells=int(target[:,:,:15].sum()),supported_angle_cells=int((target[:,:,:15]&c['valid'][deleted,:,:15]).sum()),
        raw_observed_but_angle_unsupported=int((target[:,:,:15]&e['observed'][deleted,:,None]&~c['valid'][deleted,:,:15]).sum()),
        explicitly_missing_angle_cells=int((target[:,:,:15]&e['missing'][deleted]).sum()),
        observed_motion_hand_frames=int((e['measured_change'][deleted]&hdom).sum()),groups={})
    left,right=int(deleted[0]-1),int(deleted[-1]+1)
    w=((deleted-left)/(right-left))[:,None,None]
    lo=np.minimum(c['lo'][left],c['lo'][right]);hi=np.maximum(c['hi'][left],c['hi'][right])
    chordlo=c['lo'][left]*(1-w)+c['lo'][right]*w;chordhi=c['hi'][left]*(1-w)+c['hi'][right]*w
    valid=target&c['valid'][deleted]&c['valid'][left]&c['valid'][right]
    gap=np.maximum(np.maximum(lo-c['hi'][deleted],c['lo'][deleted]-hi),0.)
    chord=np.maximum(np.maximum(chordlo-c['hi'][deleted],c['lo'][deleted]-chordhi),0.)
    for group,sl in GROUPS.items():
        m=valid[:,:,sl];gg=gap[:,:,sl][m];cc=chord[:,:,sl][m]
        out['groups'][group]=dict(expected=int(target[:,:,sl].sum()),observed=int((target[:,:,sl]&c['valid'][deleted,:,sl]).sum()),
          endpoint_eligible=int(m.sum()),outside_endpoint_hull=int((gg>EPS).sum()),outside_linear_chord=int((cc>EPS).sum()),
          max_endpoint_gap=float(gg.max()) if len(gg) else None,max_chord_gap=float(cc.max()) if len(cc) else None)
    return out
def freeze():
    assert not (U/'FROZEN.json').exists() and not (P/'RUNNING.json').exists()
    f=read(P/'FROZEN.json');check(f['pins']);check(f['protected'])
    assert read(P/'VERIFY.json')['status']=='PASS' and read(P/'SYNC_CHECK.json')['status']=='PASS'
    pins=dict(f['pins'])
    for p in [U/'study.py',P/'ANALYSIS.json',P/'RESULT.json',P/'VERIFY.json',P/'FROZEN.json',P/'FINAL_AUDIT.json']+list((P/'known').glob('*.npz'))+list((P/'cross').glob('*.npz'))+list(I.glob('*.npz')):pins[str(p)]=sha(p)
    (U/'before').mkdir()
    for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',dict(at=now(),rows=f['rows'],roles=f['roles'],folds=f['folds'],pins=pins,protected=f['protected'],
      authorization='User20261022: diagnose missedshort publicintervals: missingevidence/alignment/range; retainnormal speed/rest,C/T3F; excludeuser; nolearning beforejointreview.',
      scope='Saved27TRAIN publicclips3words9people. All81shortvariants x8anchors x2savedalignmentmethods; focus previous12oldmiss/9newmiss with overlapping factors. Identity+4normal variants reused, not new participants.',
      diagnostics='Source raw hand observed vs reliable angles/palm/arms, interval vs retained endpoint hull and source-frame-linear chord using EPS1e-6 inherited numeric difference only. All 3deletedframes retained in audit including zeroeligible. Same-source deletion response reused.',
      factorial='Actualsavedpath vs ownidentitypath remapped to nearest retained sourceframe (tieearlier); pooled8personrange vs singleanchor; queryneighbors on/off. Each setting compares deletion to its own originalidentity with same reference/mask/neighbors. No DP or fitted scores. Fixedaffectedrefregion from oldidentitypath. Factors overlap and are paired sensitivities,not unique linguistic causes.',
      witness='Alternative newbad hiddenbyactual counted only common-comparable deletion+identity across both conditions. No unavailable-as-improvement. Separate baselinealreadybad; no-new-difference is not no-warning.',
      hull='On actualpooled within-range cells, report those overlapping none of8actual contributors. Scalar convexgap diagnostic not proof of semanticerror. Min5support unchanged; singleanchor diagnostic not proposed acceptance.',
      controls='On1080normalcross savedcasesx2methods compare pooled/single and neighborson/off to expose warningtradeoffs. Normalself timing and rest leakage copied verifiedpreviousresults. No operationalcriteriachanged; gap135previousverified evidence reused.',
      limits=['Synthetic3frame core removal not linguisticessentialstage truth.','No exactcrossperson phasegroundtruth. Remappedidentitycounterfactual not correct alignment oracle.','Raw hand observed does not certify each fingerpoint; angleunsupported not always rawmissing.','Endpoint/chord gap is measured distinction only,noisepossible; zero doesnotmean stageunnecessary.','All27 people usedpreviously; no unseen-person accuracy or100wordgeneralization claim.'],
      forbidden=['training','modelinference','newDP','newMediaPipe','user/mentor','DEVheldout','camera','downloads','external','Git','C/T3F/runtime/webedit']))
    print('DIAGNOSIS FROZEN',flush=True)
def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()));(U/'cases').mkdir()
    olda=read(P/'ANALYSIS.json');look={r['clip_id']:r for r in f['rows']};samples={cid:load(I/(cid+'.npz')) for cid in look};ch={cid:channels(e) for cid,e in samples.items()}
    sources=[];pairs=[];normal=[];seq=0
    oldpairs={(x['query'],x['reference'],x['kind']):x for x in olda['omission_pairs']['cross']}
    for fold in f['folds']:
        cid=fold['query'];e=samples[cid];qc=ch[cid];req=f['roles'][look[cid]['word_code']]['required_sides']
        for kind in ('skip3_25','skip3_50','skip3_75'):
            prior=next(x for x in olda['omissions']['cross']['skip3']['members'] if x['query']==cid and x['kind']==kind)
            deleted=np.asarray(oldpairs[cid,fold['references'][0],kind]['removed_source_frames'],int)
            z=source_info(e,qc,req,deleted);z.update(query=cid,kind=kind,label=look[cid]['label'],previous=prior)
            z['self_source']={side:next(x for x in olda['omission_pairs']['known'] if x['query']==cid and x['kind']==kind)[side] for side in ('old','new')};sources.append(z)
        for rid in fold['references']:
            re=samples[rid];rc=ch[rid];domain=expected(re,req);t=load(L/'templates'/(cid+'__'+rid+'.npz'))
            assert np.array_equal(t['expected'],domain)
            anchor=dict(low=rc['lo'],high=rc['hi'],support=rc['valid']);templates={'pool':t,'anchor':anchor}
            orig=load(P/'cross'/('identity__'+cid+'__'+rid+'.npz'));ident={}
            for method,pathkey in [('old','base_path'),('new','path')]:
                for nb in (0,1):
                    ip=project(qc,orig[pathkey],len(domain),nb)
                    for tt,template in templates.items():ident[method,nb,tt]=compare(ip,template,domain)
            for kind in ('skip3_25','skip3_50','skip3_75'):
                name=kind+'__'+cid+'__'+rid;z=load(P/'cross'/(name+'.npz'));ix=z['source_index'];dqc=resample(qc,ix)
                deleted=np.asarray(oldpairs[cid,rid,kind]['removed_source_frames'],int)
                affected=np.zeros(len(domain),bool);affected[np.unique(orig['base_path'][np.isin(orig['base_path'][:,0],deleted),1])]=True
                inside=domain&affected[:,None,None]
                for method,pathkey,prefix in [('old','base_path','old_'),('new','path','')]:
                    paths={'actual':z[pathkey],'held':held_path(orig[pathkey],ix)};results={};projections={};arrays=dict(inside=inside,expected=domain,source_index=ix,deleted=deleted)
                    summary={}
                    for mode,path in paths.items():
                        arrays[mode+'_path']=path
                        for nb in (0,1):
                            pp=project(dqc,path,len(domain),nb);projections[mode,nb]=pp
                            for tt,template in templates.items():
                                k=f'{mode}_{tt}_{nb}';out=compare(pp,template,domain);results[k]=out;b=ident[method,nb,tt]
                                summary[k]=dict(contrasts(out,b,inside),original_bad=int((b['bad']&inside).sum()))
                                for v in ('comparable','bad','unknown'):arrays[k+'_'+v]=out[v]
                    act=results['actual_pool_1'];base=ident[method,1,'pool']
                    assert np.array_equal(act['bad'],z[prefix+'mismatch']) and np.array_equal(act['comparable'],z[prefix+'comparable'])
                    assert summary['actual_pool_1']['new_bad']==oldpairs[cid,rid,kind][method]['inside_raw_cells']
                    factors={}
                    for factor,k,bb in [('alignment','held_pool_1',base),('range','actual_anchor_1',ident[method,1,'anchor']),('neighbors','actual_pool_0',ident[method,0,'pool'])]:
                        w=witness(results[k],bb,act,base,inside);arrays[factor+'_witness']=w;factors[factor]=int(w.sum())
                    pp=projections['actual',1];overlap=(t['member_high']>=pp['lo']-EPS)&(t['member_low']<=pp['hi']+EPS)&t['member_complete']
                    convex=inside&act['within']&~overlap.any(0);arrays['convex_gap']=convex
                    # Projection extent at the fixedaffected reference times. This describes replacement, not skipped DP rows.
                    shifts=[];far=[]
                    for j in np.flatnonzero(affected):
                        mapped=ix[z[pathkey][z[pathkey][:,1]==j,0]];original=orig[pathkey][orig[pathkey][:,1]==j,0]
                        shifts.append(float(np.median(mapped)-np.median(original)))
                        far.append(bool(np.all((mapped<deleted[0]-1)|(mapped>deleted[-1]+1))))
                    rec=dict(name=name,query=cid,reference=rid,kind=kind,method=method,config=summary,factors=factors,
                        convex_gap_cells=int(convex.sum()),affected_reference_frames=int(affected.sum()),original_bad=summary['actual_pool_1']['original_bad'],
                        local_new_bad=summary['actual_pool_1']['new_bad'],local_unknown=summary['actual_pool_1']['unknown'],local_expected=int(inside.sum()),
                        reference_rows_far_from_deleted_neighbors=sum(far),median_source_shift=float(np.median(shifts)) if shifts else None,
                        interpretation='Paired diagnostic sensitivity; not mutually exclusive causal proof or semantic verdict.')
                    np.savez_compressed(U/'cases'/(method+'__'+name+'.npz'),**arrays)
                    pairs.append(rec)
            for kind in NORMAL:
                z=orig if kind=='identity' else load(P/'cross'/(kind+'__'+cid+'__'+rid+'.npz'));cc=resample(qc,z['source_index'])
                for method,pathkey,prefix in [('old','base_path','old_'),('new','path','')]:
                    rec=dict(query=cid,reference=rid,kind=kind,method=method,config={})
                    for nb in (0,1):
                        pp=project(cc,z[pathkey],len(domain),nb)
                        for tt,template in templates.items():
                            v=compare(pp,template,domain);rec['config'][f'{tt}_{nb}']=dict(count(v,domain),any_bad=bool(v['bad'].any()))
                            if nb==1 and tt=='pool':assert np.array_equal(v['bad'],z[prefix+'mismatch'])
                    normal.append(rec)
        seq+=1;print('PUBLIC CLIPS DIAGNOSED',seq,'/27',flush=True)
    assert len(sources)==81 and len(pairs)==1296 and len(normal)==2160
    check(f['pins']);check(f['protected'])
    write(U/'RESULT.json',dict(at=now(),sources=sources,pairs=pairs,normal=normal,unchanged_controls={'normal_timing':{k:olda['known'][k] for k in NORMAL},'rest_proxy':{c:olda['cross']['identity'][c+'_proxy'] for c in ('old','new')},'missing135_verified':str(P/'VERIFY.json')},whole_action_correct=None))
    (U/'RUNNING.json').unlink();print('DIAGNOSIS COMPLETE',flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if not (U/'FIRST_ERROR.txt').exists():(U/'FIRST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
