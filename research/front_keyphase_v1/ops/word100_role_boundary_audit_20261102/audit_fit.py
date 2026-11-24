"""Single fixed-candidate FIT diagnosis, never model selection or new labels."""
from pathlib import Path
import sys,collections
U=Path(__file__).resolve().parent;O=U.parent
R=O/'word100_role_retention_20261102';sys.path.insert(0,str(R));import retention_model as rm
np=rm.np;torch=rm.torch;read=rm.read;sha=rm.sha;exclusive=rm.exclusive
OUT=Path('W:/WordsInSilence_derived/word100_role_boundary_audit_20261102')
CK=rm.OUT/'runs/new_labels_with_retention/model.pt'
def main():
    assert not (U/'FIT_DIAGNOSIS.json').exists() and not (U/'FROZEN.json').exists()
    rows,normal=rm.fit_rows();teacher=read(R/'TEACHER_INPUTS.json')['rows']
    selected=[(i,r) for i,r in enumerate(rows) if r['kind'] in ['original','seated_variation']]
    assert len(selected)==72 and len({r['clip_id'] for _,r in selected})==36 and {r['person'] for _,r in selected}==set(range(1,10))
    pins={str(p):sha(p) for p in [CK,U/'audit_fit.py',U/'verify_fit.py',R/'VERIFIED.json',R/'TEACHER_INPUTS.json',R/'retention_model.py']}
    for i,r in selected:pins[r['cache']]=r['cache_sha256'];pins[teacher[i]['teacher_path']]=teacher[i]['teacher_sha256']
    plan=dict(at=rm.clock(),candidate='new_labels_with_retention',sources=72,original_public_clips=36,recorded_ids=list(range(1,10)),pins=pins,
        question='Does the fixed primary exclude confirmed main actions on the FIT examples too, or only on development people? Compare native and seated inputs without changing labels.',
        no_training=True,no_DEV_new_inference=True,no_threshold_tuning=True,no_label_invention=True,no_private_or_locked_holdout=True,whole_goal_complete=False)
    exclusive(U/'FROZEN.json',plan);torch.set_num_threads(2);m=rm.RetainedRoleNet();m.load_state_dict(torch.load(CK,map_location='cpu',weights_only=True));m.eval()
    stats=read(rm.ex.D/'NORMALIZATION.json');reports=[]
    for i,r in selected:
        x,rgb,l,_,_=rm.ex.batch_j([r],stats);flow=rm.fl.flow_batch([r],normal)
        with torch.inference_mode():a,b=m(x,rgb,l,flow);pr=a.softmax(-1)[0].numpy();act=b.softmax(-1)[0].numpy()
        av=r['available'];rest=r['current'][:,9];main=(1-rest)*act[:,1];conflict=(rest>=.8)&(act[:,1]>=.8)&av
        eligible=~(((1-rest)*act[:,0]+rest>=.8)&av&~conflict)
        role=r['role'];known=role>=0;truth=role==1;major=known&truth;correct=(pr.argmax(1)==r['episode'])&(r['episode']>=0)&(r['episode']<8)
        old=np.load(teacher[i]['teacher_path'])['probability'];assert len(old)==len(role)
        point_errors=[]
        for f in np.flatnonzero(major&~eligible):point_errors.append(dict(frame=int(f),source_frame=int(r['mapping'][f]),action_family=int(r['episode'][f]),content_correct=bool(correct[f]),activity_main=float(act[f,1]),teacher_main=float(old[f,1]),rest=float(rest[f])))
        dest=OUT/f"fit_{i:03d}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
        np.savez_compressed(dest,content_probability=pr,activity_probability=act,current_probability=r['current'],image_available=av,source_frame_id=r['mapping'],role_label=role,action_label=r['episode'],eligible=eligible)
        reports.append(dict(index=i,clip_id=r['clip_id'],person=r['person'],kind=r['kind'],prediction=str(dest),sha256=sha(dest),teacher=teacher[i]['teacher_path'],
            frames=len(role),role_known=int(known.sum()),role_correct=int(((act.argmax(1)==truth)&known).sum()),major_known=int(major.sum()),major_excluded=int((major&~eligible).sum()),
            correct_main_excluded=int((correct&major&~eligible).sum()),old_main_class_errors=int(((old.argmax(1)!=1)&major).sum()),major_exclusion_points=point_errors))
    groups={}
    for k in ['original','seated_variation']:
        rr=[r for r in reports if r['kind']==k];groups[k]={key:sum(r[key] for r in rr) for key in ['role_known','role_correct','major_known','major_excluded','correct_main_excluded','old_main_class_errors']}
    for p,h in pins.items():assert sha(p)==h
    exclusive(U/'FIT_DIAGNOSIS.json',dict(at=rm.clock(),groups=groups,rows=reports,inferences=72,whole_goal_complete=False,scope='Training-fit diagnostic only. Cannot prove generalization or identify a new correct label.'))
    print(groups)
if __name__=='__main__':main()
