"""C7 read-only saved-input and public FIT RGB cause audit. No model execution."""
from pathlib import Path
import json, hashlib, datetime, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
sys.dont_write_bytecode = True
U=Path(__file__).resolve().parent; C6=U.parent; G=C6.parent.parent
F=[2,3,4,5,7]; AX={2:0,3:1,4:2,5:4,7:3}
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):
    with (U/name).open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    old=[C6/p for p in ['INPUTS.npz','INDEX.json','RESIDUAL_AUDIT.json','COMPARISON.json','FROZEN.json','experiment.py']]
    anns=[G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']
    res=read(old[2])['records']; rows=read(old[1])['rows']
    src={r['clip_id']:r for p in anns for r in read(p)['rows'] if r['role']=='fit'}
    used=sorted({r['clip_id'] for r in rows}); assert len(used)==24
    pins={str(p):sha(p) for p in old+anns+[Path(__file__)]}
    for cid in used:
        s=src[cid]; assert int(s['person']) in range(2,8)
        assert sha(s['source'])==s['source_sha256']; pins[s['source']]=s['source_sha256']
    write('FROZEN.json',dict(task='C7 saved-input cause audit',created=datetime.datetime.now().isoformat(),
        scope='15 FIT residual source points; all24 public FIT recordings as saved geometry controls',
        pins=pins,no_training=True,no_old_inference=True,no_dev_or_final8=True,
        original_motion_labels_preserved=True,no_web_change=True,
        planned_measures=['primary-family geometric availability','full vs upper primary-feature identity',
        'all feature identity','adjacent movement path vs before/after mean distance','RGB temporal review'],
        interpretation='Descriptive cause evidence, not newly validated feedback or expert linguistic ground truth.'))
    z=np.load(old[0],allow_pickle=False); x=z['absolute_context'].reshape(-1,3,10,5)
    by={(r['clip_id'],r['view']):r for r in rows if not r['reverse']}
    def at(cid,view,fr):
        r=by[(cid,view)]; ids=z['source_frame_id'][r['start']:r['stop']]
        ii=np.flatnonzero(ids==fr); assert len(ii)==1
        return r['start']+int(ii[0])
    changed=read(old[3])['changed_observations']; detail=[]
    for rec in res:
        cid,fr,fam=rec['clip_id'],rec['source_frame'],rec['family']; j=F.index(fam); a=AX[fam]*2
        out=dict(rec); out['views_audit']={}
        for view in ['full','upper','actual_crop']:
            r=by[(cid,view)]; idx=at(cid,view,fr); cur=x[r['start']:r['stop'],1,a,:4]
            local=idx-r['start']; lo=max(0,local-4);hi=min(len(cur),local+5)
            path=float(np.linalg.norm(np.diff(cur[lo:hi],axis=0),axis=1).sum())
            chord=float(np.linalg.norm(cur[hi-1]-cur[lo]))
            out['views_audit'][view]=dict(primary_axis_context=x[idx,:,a,:].tolist(),
                all_context_availability=x[idx,:,:,4].tolist(),
                current_primary_available=bool(x[idx,1,a,4]>.5),
                primary_source_label=int(z['label_provenance'][idx,j]),
                current_mask=float(z['mask'][idx,j]),current_y=float(z['y'][idx,j]),
                local_9frame_path=path,local_9frame_chord=chord,
                before_current_mean_distance=float(np.linalg.norm(x[idx,0,a,:4]-x[idx,1,a,:4])),
                after_current_mean_distance=float(np.linalg.norm(x[idx,2,a,:4]-x[idx,1,a,:4])),
                saved_changed=[r for r in changed if r['clip_id']==cid and r['source_frame']==fr and r['family']==fam and r['view']==view])
        fi,ui=at(cid,'full',fr),at(cid,'upper',fr)
        out['full_upper_primary_max_abs']=float(np.max(np.abs(x[fi,:,a,:]-x[ui,:,a,:])))
        out['full_upper_all_max_abs']=float(np.max(np.abs(x[fi]-x[ui])))
        out['full_upper_changed_relation_entries']=[k for k in range(10) if np.max(np.abs(x[fi,:,k]-x[ui,:,k]))>1e-6]
        detail.append(out)
    controls=[]
    for cid in used:
        rf=by[(cid,'full')]; ru=by[(cid,'upper')]; assert rf['length']==ru['length']
        for j,fam in enumerate(F):
            a=AX[fam]*2
            for k in range(rf['length']):
                fi=rf['start']+k; ui=ru['start']+k
                if not (z['mask'][fi,j] and z['mask'][ui,j]):continue
                controls.append(dict(clip_id=cid,frame=int(z['source_frame_id'][fi]),family=fam,
                    provenance=int(z['label_provenance'][fi,j]),y=float(z['y'][fi,j]),
                    primary_identical=bool(np.allclose(x[fi,:,a],x[ui,:,a],atol=1e-6,rtol=0)),
                    all_identical=bool(np.allclose(x[fi],x[ui],atol=1e-6,rtol=0))))
    write('CAUSE_AUDIT.json',dict(records=detail,control_count=len(controls),controls=controls,
        residual_all_available=all(v['current_primary_available'] for r in detail for v in r['views_audit'].values()),
        residual_primary_full_upper_identical=sum(r['full_upper_primary_max_abs']<1e-6 for r in detail),
        residual_all_full_upper_identical=sum(r['full_upper_all_max_abs']<1e-6 for r in detail)))
    # Large torso/arm crops are same public RGB, no generated anatomy or keypoints.
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    panels=[]
    for page in range(0,len(detail),3):
        canvas=Image.new('RGB',(1500,1080),'white'); draw=ImageDraw.Draw(canvas)
        for rr,r in enumerate(detail[page:page+3]):
            cid,fr=r['clip_id'],r['source_frame'];s=src[cid];cap=cv2.VideoCapture(s['source'])
            draw.text((12,rr*360+5),f"{cid}  family {r['family']}  target frame {fr}",font=font,fill='#132c46')
            frames=[]
            for col,f in enumerate([fr-4,fr-2,fr,fr+2,fr+4]):
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok,(cid,f)
                h,w=bgr.shape[:2];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                img=Image.fromarray(rgb).crop((int(w*.15),int(h*.04),int(w*.85),int(h*.88)))
                img.thumbnail((294,294));xp=col*300+(300-img.width)//2;yp=rr*360+37
                canvas.paste(img,(xp,yp));draw.text((col*300+10,rr*360+331),f"frame {f}  ({f/s['fps']:.3f}s)",font=font,fill='#132c46')
                frames.append(f)
            cap.release();panels.append(dict(clip_id=cid,target_frame=fr,frames=frames,page=f'rgb_{page//3+1}.jpg',source_sha256=s['source_sha256']))
        canvas.save(U/f'rgb_{page//3+1}.jpg',quality=94)
    write('RGB_PANELS.json',dict(rows=panels,review_status='PENDING_VIEW'))
    assert all(sha(p)==h for p,h in pins.items())
    print(json.dumps(dict(residual_points=len(detail),control_points=len(controls),
        primary_identical=sum(r['full_upper_primary_max_abs']<1e-6 for r in detail),
        all_identical=sum(r['full_upper_all_max_abs']<1e-6 for r in detail),panels=5),ensure_ascii=False))
if __name__=='__main__':main()
