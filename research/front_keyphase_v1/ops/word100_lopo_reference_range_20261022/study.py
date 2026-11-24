import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,shutil,traceback,importlib.util
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];P=O/'word100_cross_person_motion_support_20261022'
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prev=module('lopo_previous',P/'study.py');C=module('lopo_candidate',U/'candidate.py')
read,load,write,sha,now,check=prev.read,prev.load,prev.write,prev.sha,prev.now,prev.check
KINDS=prev.prior.KINDS

def freeze():
    f=read(P/'FROZEN.json');check(f['pins']);check(f['protected']);assert read(P/'SYNC_CHECK.json')['status']=='PASS'
    assert not (P/'RECOVERY_RUNNING.json').exists() and not (U/'FROZEN.json').exists()
    pins=dict(f['pins'])
    for p in [P/'study.py',P/'candidate.py',P/'FROZEN.json',P/'FINAL_AUDIT.json',P/'VERIFY.json',U/'study.py',U/'candidate.py']+list((P/'cases').glob('*.npz'))+list((P/'inputs').glob('*.npz')):pins[str(p)]=sha(p)
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    folds=[]
    for q in f['rows']:
        refs=[r['clip_id'] for r in f['rows'] if r['word_code']==q['word_code'] and r['signer_id']!=q['signer_id']]
        assert len(refs)==8 and q['clip_id'] not in refs
        folds.append(dict(query=q['clip_id'],excluded_person=q['signer_id'],references=refs))
    write(U/'FROZEN.json',dict(at=now(),rows=f['rows'],roles=f['roles'],folds=folds,pins=pins,protected=f['protected'],kinds=KINDS,
      authorization='2026-10-22 user: public multiplepeople normal ranges; leave each person out; protect brief discriminating motion and missing evidence; preserve C/T3-F/user exclusion; no learning before joint review.',
      scope='Same public TRAIN3words9people27clips. 27LOPO query folds,8anchors each=216templates. 11conditionsx216=2376comparisons. 135fixedpath missingcontrols. Query/person excluded from ALL template contributors and pairwise alignment building template.',
      features='21 perhand channels:15projectedangles,2palmposition,2elbow,2posewrist. Units degrees or fixedshoulderwidth; no combined score. Exact original coordinate/mask/identity.',
      range='Each anchor timeline built using8referencepeople,anchor included,all query-person excluded. Reuse unchanged reference-to-anchor DTW identitypaths. Each person contributes only if ALL aligned samples are observed; no neighbors for range-building. Min/max of8persons observed interval bounds at each time/hand/channel,only >=5distinctpeople. Not statistical tolerance/confidence or semantic correctness. Peraxis hull does not prove wholepose/order.',
      query='Same frozenDTW band.2/weights. Project alignedquery plus immediatevalidneighbor only; primary comparison requires ALL alignedsamplesobserved. Partial evidence explicitly unknown. Compare original singleanchor and8person range on identical path/strict masks; separately count common-set resolved and unavailable cells. Any mismatch and >=3frames persist both retained; no wholeaction-pass.',
      tests='Identity+4normaltemporal transforms (1080 paths reused). Delete3frames or20%core at25/50/75% core (1296new alignments),positionsfixed not picked by outcome. Source/deletedindices evaluator-only. Deletion local support mapped via saved originalquery-anchorpath for evaluation,not exact crossperson time truth. Removedcore is not languageessentiality annotation.',
      missing='For eachquery lowest-other-ID anchor,each requiredhand: wholehand/3coreframehandgap/wholearm ->135 fixedpath controls. Reference template and expected domain unchanged. No zero filled score.',
      verification='Independent range aggregation/personexclusion, query interval/gap/state arithmetic, invariants, storedDPoptimum for new deletionpaths, normal-case savedpath pins, synthetic briefoutofrange and missing fixtures, graphs.',
      decision='Research-only even if warning drops; a decrease caused by insufficient support is not improvement. Evaluate omissions/missing with normalcases; no posthoc thresholds or class exceptions.',
      limits=['Only3words/9publicpeople used repeatedly for development; not untouched classification test.','LOPO excludes currentreference construction, not all prior feature development/training.','Existing reliablemask/positive rest/word role unchanged; min5fixed shared evidence threshold.','Minmax may hide real brief changes; failures reported, not tuned away.'],
      forbidden=['neural training','classifier inference','user/mentor','DEV/heldout','camera','download','external','Git','C/T3-F/runtime/web edit']))
    print('FROZEN 216templates 2376pairs controls135',flush=True)

def expected(e,req):
    side=np.array([s in req for s in ('L','R')]);return np.broadcast_to((e['core'][:,None]&side[None,:]&~e['source_rest'])[:,:,None],(len(e['core']),2,21)).copy()

def stats(o):
    groups={}
    for k,sl in [('angle',slice(0,15)),('palm',slice(15,17)),('elbow',slice(17,19)),('wrist',slice(19,21))]:
        common=o['comparable'][:,:,sl]&o['base_comparable'][:,:,sl]
        groups[k]={s:int(o[s][:,:,sl].sum()) for s in ['expected','comparable','unknown','reference_insufficient','query_incomplete','query_partial','mismatch','persistent','within_range','base_comparable','base_mismatch','base_persistent']}
        groups[k].update(warning=bool(o['persistent'][:,:,sl].any()),any_difference=bool(o['mismatch'][:,:,sl].any()),base_warning=bool(o['base_persistent'][:,:,sl].any()),
            common=int(common.sum()),common_old_difference=int((common&o['base_mismatch'][:,:,sl]).sum()),common_new_difference=int((common&o['mismatch'][:,:,sl]).sum()),
            resolved_observed=int((common&o['base_mismatch'][:,:,sl]&~o['mismatch'][:,:,sl]).sum()),
            old_difference_now_unknown=int((o['base_mismatch'][:,:,sl]&~o['comparable'][:,:,sl]).sum()))
    return dict(groups=groups,whole_action_correct=None)

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()))
    for s in ('templates','cases','controls','progress'):(U/s).mkdir()
    samples={r['clip_id']:prev.data(r) for r in f['rows']};lookup={r['clip_id']:r for r in f['rows']};allrows=[];controls=[];templates=[]
    channels={cid:C.channels(e) for cid,(q,e) in samples.items()}
    for fold in f['folds']:
        cid=fold['query'];qr=lookup[cid];q,e=samples[cid];req=f['roles'].get(qr['word_code'],{}).get('required_sides',['L','R'])
        weights=prev.A.C.weights_for(req,(False,False),prev.A.F.GROUPS)
        for rid in fold['references']:
            r,re=samples[rid];domain=expected(re,req);projections=[]
            for contributor in fold['references']:
                if contributor==rid:path=np.column_stack([np.arange(len(re['core'])),np.arange(len(re['core']))])
                else:path=load(P/'cases'/('identity__'+contributor+'__'+rid+'.npz'))['path']
                projections.append(C.project(channels[contributor],path,len(re['core']),False))
            template=C.pool(projections);tn=cid+'__'+rid
            np.savez_compressed(U/'templates'/(tn+'.npz'),**template,expected=domain)
            templates.append(dict(name=tn,query=cid,anchor=rid,contributors=fold['references'],people=[lookup[x]['signer_id'] for x in fold['references']]))
            for kind in KINDS:
                qq,qe,ix=prev.transform(q,e,kind);qc=C.channels(qe);_,deleted=prev.prior.mapping(len(e['core']),e['core'],kind)
                name=kind+'__'+cid+'__'+rid;oldfile=P/'cases'/(name+'.npz')
                reused=oldfile.is_file()
                if reused:old=load(oldfile);path=old['path'];cost=old['cost'];assert np.array_equal(ix,old['source_index'])
                else:
                    cost,ev=prev.A.legacy(qq,r,group_weights=weights,hand_participation=False,one_sided_mode='flat');path=np.asarray(prev.A.dp(cost,.2),int)
                o=C.compare(qc,channels[rid],template,path,domain)
                np.savez_compressed(U/'cases'/(name+'.npz'),**o,path=path,source_index=ix,deleted=deleted,cost=cost)
                allrows.append(dict(name=name,query=cid,anchor=rid,label=qr['label'],kind=kind,reused_path=reused,**stats(o)))
                if kind=='identity' and rid==fold['references'][0]:
                    for h,side in enumerate(('L','R')):
                        if side not in req:continue
                        for mode in ('wholehand','shortgap','wholearm'):
                            co={k:v.copy() for k,v in qc.items()};ids=np.arange(len(e['core']))
                            if mode=='shortgap':
                                active=np.flatnonzero(e['core']);mid=active[len(active)//2];ids=np.arange(mid-1,mid+2)
                            sl=slice(17,21) if mode=='wholearm' else slice(0,17)
                            co['valid'][ids,h,sl]=False;co['lo'][ids,h,sl]=np.nan;co['hi'][ids,h,sl]=np.nan
                            out=C.compare(co,channels[rid],template,path,domain);cn=mode+'_'+side+'__'+cid
                            np.savez_compressed(U/'controls'/(cn+'.npz'),**out,removed=ids,path=path)
                            controls.append(dict(name=cn,parent=name,mode=mode,side=side,**stats(out)))
        write(U/'progress'/(cid+'.json'),dict(at=now(),complete=len(allrows),total=2376));print('DONE',cid,len(allrows),flush=True)
    assert len(allrows)==2376 and len(templates)==216 and len(controls)==135
    write(U/'RESULT.json',dict(at=now(),rows=allrows,templates=templates,controls=controls,training=0,user_data=0))
    check(f['pins']);check(f['protected']);(U/'RUNNING.json').unlink()

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        p=U/'FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
