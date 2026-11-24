"""Independent vector-based arithmetic, never imported by the producer.

Directions are compared through cross/dot of unit vectors; relative directions
use complex products rather than subtracting producer atan2 headings. Source-axis
statistics use bincount/scatter operations, not the producer's per-path loop.
"""
import numpy as np
from collections import Counter

FINGER_LABELS=['thumb_mcp','thumb_ip']+[f+'_'+j for f in ['index','middle','ring','little'] for j in ['mcp','pip','dip']]


def degrees(v):
    a=np.degrees(np.arctan2(v[:,1],v[:,0]))
    return np.where(a>=180,a-360,a)[:,None]


def unit(v):
    norms=np.hypot(v[:,0],v[:,1])
    return np.divide(v,norms[:,None],out=np.zeros_like(v),where=(norms>1e-8)[:,None]),norms


def complex_relative(a,b):
    """Unit a divided by unit b: projected direction a relative to b."""
    return np.column_stack((b[:,0]*a[:,0]+b[:,1]*a[:,1],b[:,0]*a[:,1]-b[:,1]*a[:,0]))


def measurements(src):
    x=np.asarray(src['x'],float);m=np.asarray(src['mask'],bool);groups={}
    for side,off in [('L',0),('R',28)]:
        fcols=list(range(off+11,off+25));mask=m[:,fcols]
        groups[side+'_finger_angles14']={'values':np.where(mask,x[:,fcols],0)*180,'mask':mask,'columns':fcols,'circular':False,
            'labels':[side+'_'+f+'_angle' for f in FINGER_LABELS],'reason':np.where(mask,'OBSERVED','SOURCE_ANGLE_UNOBSERVED')}
        ac=[off+25,off+26];fc=[off+2,off+3,off+4,off+5]
        hraw=np.where(m[:,ac],x[:,ac],0);h,hn=unit(hraw);hsource=m[:,ac].all(1);hv=hsource&np.isfinite(hn)&(hn>1e-8)
        pos=np.where(m[:,fc],x[:,fc],0);f,fn=unit(pos[:,:2]-pos[:,2:]);fsource=m[:,fc].all(1);fv=fsource&np.isfinite(fn)&(fn>1e-8)
        hr=np.where(~hsource,'HAND_AXIS_SOURCE_UNOBSERVED',np.where(~hv,'DEGENERATE_HAND_AXIS','OBSERVED'))
        rr=[]
        for i in range(len(x)):
            causes=[]
            if not hv[i]:causes.append(str(hr[i]))
            if not fsource[i]:causes.append('FOREARM_SOURCE_UNOBSERVED')
            elif not fv[i]:causes.append('DEGENERATE_FOREARM')
            rr.append('+'.join(causes) if causes else 'OBSERVED')
        groups[side+'_hand_axis2d']={'values':degrees(h),'vectors':h,'mask':hv[:,None],'columns':ac,'circular':True,
            'labels':[side+'_hand_axis_heading_2d'],'reason':hr[:,None]}
        rel=complex_relative(h,f)
        groups[side+'_hand_forearm_relative2d']={'values':degrees(rel),'vectors':rel,'mask':(hv&fv)[:,None],'columns':ac+fc,'circular':True,
            'labels':[side+'_hand_axis_minus_forearm_heading_2d'],'reason':np.asarray(rr)[:,None]}
    return groups


def paired(a,b,path):
    qi,ri=np.asarray(path,int).reshape(-1,2).T
    shared=a['mask'][qi]&b['mask'][ri]
    if a['circular']:
        signed=degrees(complex_relative(a['vectors'][qi],b['vectors'][ri]))
    else:signed=a['values'][qi]-b['values'][ri]
    return signed,np.abs(signed),shared


def axis_stats(own,other,path,absolute,shared,side):
    n,d=own['mask'].shape
    ix=np.asarray(path,int).reshape(-1,2)[:,side];otherix=np.asarray(path,int).reshape(-1,2)[:,1-side]
    aligned=np.bincount(ix,minlength=n);matched=np.zeros((n,d),int);missing=np.zeros((n,d),int);counterpart=np.zeros((n,d),int)
    means=np.zeros((n,d));peaks=np.zeros((n,d))
    for k in range(d):
        matched[:,k]=np.bincount(ix,weights=shared[:,k],minlength=n).astype(int)
        missing[:,k]=np.bincount(ix,weights=own['mask'][ix,k]&~other['mask'][otherix,k],minlength=n).astype(int)
        counterpart[:,k]=np.bincount(ix,weights=other['mask'][otherix,k],minlength=n).astype(int)
        total=np.bincount(ix,weights=np.where(shared[:,k],absolute[:,k],0),minlength=n)
        means[:,k]=np.divide(total,matched[:,k],out=np.zeros(n),where=matched[:,k]>0)
        np.maximum.at(peaks[:,k],ix,np.where(shared[:,k],absolute[:,k],0))
    support=matched>0;mixed=support&(missing>0)
    statuses=np.where(mixed,'PARTIAL_SHARED_SUPPORT',np.where(support,'COMPARED','NOT_COMPARABLE'))
    reason=np.where(own['mask'],'COUNTERPART_UNOBSERVED','OWN_UNOBSERVED')
    reason=np.where(own['mask']&(aligned[:,None]==0),'NO_ALIGNED_COUNTERPART',reason)
    reason=np.where(support,'SHARED_OBSERVATION',reason);reason=np.where(mixed,'SHARED_AND_MISSING_COUNTERPART_LINKS',reason)
    counts={'possible':n*d,'own_observed':int(own['mask'].sum()),'own_unobserved':int((~own['mask']).sum()),'matched_support':int(support.sum()),
        'direct_complete':int((support&~mixed).sum()),'partial_shared':int(mixed.sum()),'not_comparable':int((~support).sum()),
        'observed_without_shared_counterpart':int((own['mask']&~support).sum())}
    return {'aligned_link_count':aligned,'matched_link_count':matched,'missing_counterpart_link_count':missing,'counterpart_observed_link_count':counterpart,
        'matched_support':support,'mixed_counterpart_missing':mixed,'mean_abs_deg':means,'max_abs_deg':peaks,'cell_status':statuses,'cell_reason':reason,'counts':counts}


def verify_evidence(q,r,path,out,check,identity,tol=1e-8):
    """check(name,boolean,id) is supplied by the audit runner."""
    qg,rg=measurements(q),measurements(r);path=np.asarray(path,int).reshape(-1,2)
    check('path_unchanged',np.array_equal(np.asarray(out['path']).reshape(-1,2),path),identity)
    check('group_identity',set(out['groups'])==set(qg),identity)
    check('gating_and_verdict_absent',not out['position_used_for_gating'] and not out['learned_shape_used_for_gating'] and out['whole_sign_verdict'] is None and out['semantic_verdict'] is None,identity)
    def exact(name,actual,expected):
        actual,expected=np.asarray(actual),np.asarray(expected)
        # JSON [] cannot encode a second dimension; group metadata supplies D.
        check(name,(actual.size==expected.size==0) or np.array_equal(actual,expected),identity)
    def numeric(name,actual,expected,mask,circular=False):
        actual=np.asarray(actual,dtype=float);expected=np.asarray(expected,float);mask=np.asarray(mask,bool)
        if actual.size==expected.size==0:actual=actual.reshape(expected.shape)
        if actual.shape!=expected.shape:
            check(name+'_shape',False,identity);return
        check(name+'_null_mask',np.array_equal(np.isnan(actual),~mask),identity)
        delta=actual[mask]-expected[mask]
        if circular:delta=(delta+180)%360-180
        check(name+'_values',bool(np.all(np.isfinite(actual[mask])) and np.all(np.abs(delta)<=tol)),identity)
    for key,a in qg.items():
        b=rg[key];g=out['groups'][key];signed,absolute,shared=paired(a,b,path);nlinks=len(path);dims=a['mask'].shape[1]
        exact(key+'_columns',g['source_columns'],a['columns']);check(key+'_units',g['units']=='degrees' and g['circular']==a['circular'],identity)
        exact(key+'_shared',g['links']['shared_support'],shared)
        numeric(key+'_signed',g['links']['signed_difference_deg'],signed,shared,a['circular'])
        numeric(key+'_absolute',g['links']['absolute_difference_deg'],absolute,shared)
        reported=np.asarray(g['links']['absolute_difference_deg'],float).reshape(nlinks,dims)
        antipodal=shared&(reported==180) if a['circular'] else np.zeros_like(shared)
        exact(key+'_antipodal',g['links']['antipodal_signed_direction_ambiguous'],antipodal)
        for side,src,own,other,name in [(0,q,a,b,'query'),(1,r,b,a,'reference')]:
            z=g[name];expected=axis_stats(own,other,path,absolute,shared,side);prefix=key+'_'+name
            exact(prefix+'_shape',z['possible_shape'],own['mask'].shape)
            exact(prefix+'_frames',z['frame_id'],src['frame_id']);exact(prefix+'_pts',z['timestamp_us'],src['timestamp_us'])
            exact(prefix+'_own_mask',z['own_observed'],own['mask']);exact(prefix+'_own_reason',z['own_reason'],own['reason'])
            exact(prefix+'_source_mask',z['original_source_mask'],np.asarray(src['mask'])[:,own['columns']])
            exact(prefix+'_labels',z['dimension_labels'],own['labels'])
            numeric(prefix+'_own_values',z['values_degrees'],own['values'],own['mask'],own['circular'])
            for field in ['aligned_link_count','matched_link_count','missing_counterpart_link_count','counterpart_observed_link_count','matched_support','mixed_counterpart_missing','cell_status','cell_reason']:
                exact(prefix+'_'+field,z[field],expected[field])
            for field in ['mean_abs_deg','max_abs_deg']:numeric(prefix+'_'+field,z[field],expected[field],expected['matched_support'])
            check(prefix+'_counts',z['counts']==expected['counts'],identity)
            check(prefix+'_own_reason_counts',z['own_reason_counts']==dict(Counter(own['reason'].ravel().tolist())),identity)
            exact(prefix+'_unaligned',z['unaligned_frame_indices'],np.flatnonzero(expected['aligned_link_count']==0))
        for dim,summary in enumerate(g['summary_by_dimension']):
            valid=shared[:,dim];v=absolute[valid,dim]
            check(key+'_summary_dimension',summary['dimension']==dim and summary['label']==a['labels'][dim] and summary['possible_links']==nlinks and summary['shared_links']==int(valid.sum()),identity)
            check(key+'_summary_mean',(summary['mean_abs_deg'] is None) if not len(v) else abs(summary['mean_abs_deg']-float(v.mean()))<=tol,identity)
            check(key+'_summary_max',(summary['max_abs_deg'] is None) if not len(v) else abs(summary['max_abs_deg']-float(v.max()))<=tol,identity)
            rv=reported[:,dim];peak=np.flatnonzero(valid&(rv==np.max(rv[valid]))) if valid.any() else np.array([],int)
            exact(key+'_reported_peak_integrity',summary['peak_link_indices'],peak)
    return sum(int(g['mask'].size) for g in qg.values())+sum(int(g['mask'].size) for g in rg.values())
