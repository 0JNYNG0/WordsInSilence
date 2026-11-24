"""Predeclared masks and time controls, keeping fixed TRAIN references and bounds."""
import study as S
import numpy as np,sys,traceback,time
from collections import Counter,defaultdict
X=S.R.mod('dev_replay_input_controls',S.B/'controls.py')
V=S.R.mod('dev_replay_control_arithmetic',S.B/'verify_alignment.py')
E=S.R.mod('dev_replay_control_evidence',S.B/'verify_range.py')
KINDS=('speed075','speed125','delete3','missing_L','missing_R','gap3_L','gap3_R')

def protocol():
    f=S.read(S.U/'FROZEN.json'); by=defaultdict(list)
    for p in f['pairs']:by[p['word_code']].append(p)
    selected=[min(pp,key=lambda p:(p['query'],p['reference'])) for _,pp in sorted(by.items())]
    S.write(S.U/'CONTROLS_PROTOCOL.json',dict(at=S.now(),selection=selected,end_to_end=selected[::10],kinds=KINDS,fixed_path_cases=300,end_to_end_cases=70,
      selection_rule='Smallest DEV ID and smallest named TRAIN reference for each word; every10th sorted word for end-to-end controls; no result selection.',
      interpretation='Fixed path masks test evidence arithmetic only. End-to-end controls recompute alignment and disclose short-gap/deletion escaping detection. Same-frame speed transforms are not real RGB performance or semantic sign errors. No change to bands or thresholds.',
      pins={str(p):S.sha(p) for p in [__file__,X.__file__,V.__file__,E.__file__,S.U/'FROZEN.json']}))
    print('CONTROLS_FROZEN300_FIXED_70_REALIGNED',flush=True)

def fixed_path(q,r,base):
    z={k:v.copy() for k,v in base.items()};phase=z['phase_old'];new=z['phase_new'];first=z['span_first'];last=z['span_last'];n=len(r['core']);d=z['domain']
    om=S.M.map_values(q,phase,first,last);nm=S.M.map_values(q,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)));rm=S.M.map_values(r,np.arange(n,dtype=float),np.arange(n),np.arange(n))
    od,ov,_=S.M.describe(om,d);nd,nv,_=S.M.describe(nm,d);rd,rv,_=S.M.describe(rm,d)
    src=np.where(nv,1,np.where(ov,2,0)).astype(np.uint8);ok=(src>0)&rv;po=d&om['valid']&r['available']
    z.update(source=src,point_ok=po,point_delta=np.where(po,abs(om['values']-r['values']),np.nan).astype(np.float32),sequence_ok=ok,sequence_delta=np.where(ok[...,None],abs(np.where((src==1)[...,None],nd,od)-rd),np.nan).astype(np.float32))
    return z

def independent_states(z,q,r,b):
    evidence=E.eligibility(z,q,r)
    for mode,i in [('point',0),('sequence',1)]:
        delta=z[mode+'_delta'];over=delta>b[mode+'_bound']+1e-6
        if mode=='sequence':over=over.any(-1)
        expected=np.select([~z['domain'],~z[mode+'_ok'],~evidence[i],b[mode+'_peer_count']<5,over],[0,1,2,3,5],default=4).astype(np.uint8)
        V.same(expected,z[mode+'_status'])

def run():
    p=S.read(S.U/'CONTROLS_PROTOCOL.json');S.check(p['pins']);f=S.read(S.U/'FROZEN.json');done=S.read(S.U/'DONE.json');samples=S.frozen_inputs(f);pairs={(x['query'],x['reference']):x for x in done['rows']};bounds={r['reference']:S.load(r['output']) for r in S.read(S.U/'BOUNDS_DONE.json')['rows']};counts=Counter();outputs={};realigned=[]
    for selected in p['selection']:
        qid,rid=selected['query'],selected['reference'];q=samples[qid];r=samples[rid];base=S.load(pairs[qid,rid]['output']);b=bounds[rid]
        # Single angle is selected by observability, not whether its score is favorable.
        cells=np.argwhere(base['point_ok'][:,:,:15]);assert len(cells)>0;j,h,a=map(int,cells[len(cells)//2]);lo=int(np.floor(base['phase_old'][j]));hi=int(np.ceil(base['phase_old'][j]))
        q1={k:v.copy() for k,v in q.items()};q1['available'][lo:hi+1,h,a]=False;q1['values'][lo:hi+1,h,a]=np.nan;q1['quality_after'][lo:hi+1,h,a]=False
        z=S.annotate(fixed_path(q1,r,base),q1,r,b);independent_states(z,q1,r,b)
        assert z['point_status'][j,h,a]==1
        other=np.ones((2,23),bool);other[h,a]=False
        for mode in ('point','sequence'):V.same(z[mode+'_status'][:,other],base[mode+'_status'][:,other])
        V.same(z['domain'],base['domain']);counts['fixed_single_angle_preserves_neighbors']+=1
        for kind in ('missing_L','missing_R'):
            changed,_,_,side,_=X.transformed(q,kind);z=S.annotate(fixed_path(changed,r,base),changed,r,b);independent_states(z,changed,r,b)
            for mode in ('point','sequence'):
                assert not(z[mode+'_status'][:,side,:17]>=4).any() and not(z[mode+'_status'][:,side,21:]>=4).any()
                V.same(z[mode+'_status'][:,1-side],base[mode+'_status'][:,1-side]);V.same(z[mode+'_status'][:,side,17:21],base[mode+'_status'][:,side,17:21])
            V.same(z['domain'],base['domain']);counts['fixed_missing_hand_keeps_other_hand_arm_domain']+=1
    for selected in p['end_to_end']:
        qid,rid=selected['query'],selected['reference'];q=samples[qid];r=samples[rid];old=S.load(pairs[qid,rid]['output']);b=bounds[rid]
        for kind in KINDS:
            changed,ix,cut,h,affected=X.transformed(q,kind);z=S.C.pair(changed,r,S.M);S.annotate(z,changed,r,b);independent_states(z,changed,r,b);V.same(old['domain'],z['domain']);c=Counter()
            target=(old['phase_old']>=cut.min()-2)&(old['phase_old']<=cut.max()+2)&r['core']
            for mode in ('point','sequence'):
                before=old[mode+'_status'];after=z[mode+'_status']
                for name,a,bb in S.GROUPS:
                    prev=before[:,:,a:bb];cur=after[:,:,a:bb];prefix=mode+'_'+name
                    c[prefix+'_domain']=int(old['domain'][:,:,a:bb].sum());c[prefix+'_new_excess']=int(((prev==4)&(cur==5)).sum());c[prefix+'_supported_to_unknown']=int(((prev>=4)&(cur>0)&(cur<4)).sum());c[prefix+'_beyond_to_within']=int(((prev==5)&(cur==4)).sum());c[prefix+'_changed_status']=int((prev!=cur).sum())
                c[mode+'_target_new_excess']=int(((before!=5)&(after==5))[target].sum());c[mode+'_target_lost_support']=int(((before>=4)&(after>0)&(after<4))[target].sum())
                if kind.startswith('missing'):
                    assert not(after[:,h,:17]>=4).any() and not(after[:,h,21:]>=4).any();counts['realigned_whole_hand_never_supported']+=1
            output=S.U/'controls'/(qid+'__'+rid+'__'+kind+'.npz');np.savez_compressed(output,**z)
            outputs[str(output)]=S.sha(output);realigned.append(dict(**selected,kind=kind,output=str(output),counts=dict(c),target_reference_frames=int(target.sum()),no_target_new_signal=not any(c[k] for k in ('point_target_new_excess','point_target_lost_support','sequence_target_new_excess','sequence_target_lost_support'))))
        print('CONTROLS_REALIGNED',len(realigned),70,flush=True)
    assert counts['fixed_single_angle_preserves_neighbors']==100 and counts['fixed_missing_hand_keeps_other_hand_arm_domain']==200 and len(realigned)==70
    S.check(f['protected']);S.write(S.U/'CONTROLS_DONE.json',dict(at=S.now(),fixed_path_verified=dict(counts),rows=realigned,pins=outputs,protocol_sha256=S.sha(S.U/'CONTROLS_PROTOCOL.json'),state_reconstruction='Independent eligibility and numpy.select precedence on each controlled result; masks at fixed path and realignment reported separately.',normal_crossing_claim='Original input joints and left/right assignment unchanged; not new anatomical/linguistic groundtruth.',new_detector=0,new_classifier=0))
    print('CONTROLS_DONE',dict(counts),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        p=S.U/('FIRST_ERROR_controls_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
