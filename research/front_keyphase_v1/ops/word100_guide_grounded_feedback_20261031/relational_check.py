"""Common 2D hand-to-opposite-forearm relation; not contact or correctness."""
import numpy as np

def forearm_relation(array,names,actor='R',other='L'):
    def points(prefix):
        ix=[names.index(prefix+'_'+dim) for dim in ['x','y']]
        return array['x'][:,ix].astype('float64'),array['mask'][:,ix].all(1)
    palm,pm=points(actor+'_palm_center');elbow,em=points(other+'_elbow');wrist,wm=points(other+'_pose_wrist')
    vector=wrist-elbow;length2=(vector*vector).sum(1)
    valid=pm&em&wm&np.isfinite(palm).all(1)&np.isfinite(vector).all(1)&(length2>1e-12)
    u=np.full(len(palm),np.nan);distance=np.full(len(palm),np.nan)
    u[valid]=((palm-elbow)[valid]*vector[valid]).sum(1)/length2[valid]
    projection=elbow+u[:,None]*vector
    distance[valid]=np.linalg.norm((palm-projection)[valid],axis=1)/np.sqrt(length2[valid])
    return dict(progress=u,perpendicular_distance=distance,valid=valid,
                meaning='u=0 elbow, u=1 pose wrist on the other forearm image axis; perpendicular distance in forearm-length units. Neither contact nor correct stroke established.')

def local_relation(relation,index,frame_ids,times_us,radius=2):
    lo=index-radius;hi=index+radius+1
    if lo<0 or hi>len(frame_ids):return dict(status='INCOMPLETE_WINDOW')
    ids=frame_ids[lo:hi];ts=times_us[lo:hi];valid=relation['valid'][lo:hi]
    if not valid.all() or not (np.diff(ids)==1).all() or not (np.diff(ts)>0).all():return dict(status='UNOBSERVED_WINDOW')
    values=relation['progress'][lo:hi];delta=np.diff(values)
    return dict(status='OBSERVED_2D_RELATION',first_frame=int(ids[0]),last_frame=int(ids[-1]),seconds=(ts/1e6).tolist(),
                progress=values.tolist(),perpendicular_distance=relation['perpendicular_distance'][lo:hi].tolist(),
                displacement=float(values[-1]-values[0]),every_step_toward_wrist=bool((delta>0).all()),every_step_toward_elbow=bool((delta<0).all()),
                actual_contact=None,linguistic_acceptability=None)

def selftest():
    names=[p+'_'+d for p in ['R_palm_center','L_elbow','L_pose_wrist'] for d in ['x','y']]
    x=np.array([[.1+i*.1,.03,0,0,1,0] for i in range(7)],float);mask=np.ones_like(x,bool)
    base=forearm_relation(dict(x=x,mask=mask),names)
    a=np.deg2rad(37);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    transformed=(x.reshape(-1,3,2)@rot.T*3+np.array([7,-4])).reshape(-1,6)
    changed=forearm_relation(dict(x=transformed,mask=mask),names)
    assert np.allclose(base['progress'],changed['progress'],atol=1e-12)
    assert np.allclose(base['perpendicular_distance'],changed['perpendicular_distance'],atol=1e-12)
    mirrored=x.reshape(-1,3,2).copy();mirrored[:,:,0]*=-1
    assert np.allclose(base['progress'],forearm_relation(dict(x=mirrored.reshape(-1,6),mask=mask),names)['progress'])
    hidden=mask.copy();hidden[3,0]=False
    miss=forearm_relation(dict(x=x,mask=hidden),names)
    assert len(miss['progress'])==7 and np.isnan(miss['progress'][3])
    assert local_relation(miss,3,np.arange(7),np.arange(7)*33333)['status']=='UNOBSERVED_WINDOW'
    assert local_relation(base,3,np.arange(7),np.arange(7)*33333)['every_step_toward_wrist']
    return dict(same_coordinate_frame_translation_scale_rotation_mirror_invariant=True,missing_not_filled=True,contact_not_inferred=True,scope='Coordinate-transform controls, not linguistic tolerance validation')
