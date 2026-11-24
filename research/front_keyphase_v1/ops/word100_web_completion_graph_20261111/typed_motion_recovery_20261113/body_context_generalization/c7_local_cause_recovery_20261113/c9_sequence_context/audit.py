"""C9 saved-feature/source review; no new neural inference or learning."""
from pathlib import Path
import json,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;C7=U.parent;C6=C7.parent;G=C6.parent.parent
AX={2:0,3:2,4:4,5:8,7:6}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):
    with (U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    paths=[C7/'NEXT_RESIDUALS.json',C7/'conditioned_context/COMPARISON.json',C7/'conditioned_context/INPUTS.npz',C6/'INPUTS.npz',C6/'INDEX.json',Path(__file__)]
    annotations=[G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']
    src={r['clip_id']:r for p in annotations for r in read(p)['rows']if r['role']=='fit'}
    losses=read(paths[0])['records'];assert len(losses)==11
    pins={str(p):sha(p)for p in paths+annotations}
    for cid in {r['clip_id']for r in losses}:
        s=src[cid];assert s['person']in range(2,8);assert sha(s['source'])==s['source_sha256'];pins[s['source']]=s['source_sha256']
    write('FROZEN.json',dict(scope='11 C8 remaining FIT movement regressions; raw before/current/after context audit only',pins=pins,
        no_new_inference=True,no_training=True,no_new_labels=True,no_DEV_final_private=True,
        hypothesis='Averaging may hide within-context order; actual residual failure cause must be distinguished from mere representational possibility.',
        planned_measures=['primary C6-C8 identity','local raw coordinate/mask sequence','mean delta still present','source onset/offset band membership','mean-preserving order permutation demonstration']))
    z=np.load(C6/'INPUTS.npz');old=z['relative_context'].reshape(-1,3,10,5);raw=z['absolute_context'].reshape(-1,3,10,5)[:,1]
    new=np.load(C7/'conditioned_context/INPUTS.npz')['features'].reshape(-1,3,10,5);rows=read(C6/'INDEX.json')['rows'];by={(r['clip_id'],r['view']):r for r in rows if not r['reverse']};out=[]
    for p in losses:
        cid,fr,fam=p['clip_id'],p['frame'],p['family'];a=AX[fam];views={}
        for view in ['full','upper','actual_crop']:
            row=by[cid,view];sl=slice(row['start'],row['stop']);ids=z['source_frame_id'][sl];i=row['start']+int(np.flatnonzero(ids==fr)[0]);ts=z['timestamp_us'];center=ts[i]
            select=np.flatnonzero((ts[sl]>=center-133333)&(ts[sl]<=center+133333))+row['start'];before=select[select<=i];after=select[select>=i]
            v=raw[select,a,:4].copy();available=raw[select,a,4]>.5
            # Current-conditioned q0 used by C8 only for forearm projection.
            if fam==4:v[:,0]*=raw[select,3,2]**2
            steps=np.diff(v[:,0]);pairvalid=available[:-1]&available[1:]
            meaningful=steps[pairvalid];path=float(np.abs(meaningful).sum());net=float(v[np.flatnonzero(available)[-1],0]-v[np.flatnonzero(available)[0],0])if available.any()else None
            views[view]=dict(source_frames=z['source_frame_id'][select].tolist(),relative_us=(ts[select]-center).tolist(),primary_scalar=v[:,0].tolist(),available=available.tolist(),all_four=v.tolist(),
                old_new_primary_max_abs=float(np.max(abs(old[i,:,a]-new[i,:,a]))),current_before_after=new[i,:,a].tolist(),
                valid_adjacent_scalar_path=path,valid_endpoint_change=net,
                raw_nonzero_adjacent_changes=int(np.count_nonzero(meaningful)),
                raw_sign_switches=int(np.count_nonzero(np.diff(np.sign(meaningful[np.nonzero(meaningful)]))))if np.count_nonzero(meaningful)>1 else 0)
        ev=[e for e in src[cid]['events']if e['family']==fam]
        bands=[dict(event_id=e['event_id'],onset=e['onset'],offset=e['offset'])for e in ev]
        inside=any(e['onset'][0]<=fr<=e['onset'][1]or e['offset'][0]<=fr<=e['offset'][1]for e in ev)
        out.append(dict(**p,views=views,boundary_bands=bands,in_existing_boundary_band=inside,
            teacher_label_kept='direct physical movement witness',causal_status='Not resolved merely by a nonzero raw path. No threshold or source label changed.'))
    # Algebraic counterexample using native observed values; not claimed as valid KSL.
    proof=[]
    for r in out:
        v=r['views']['full'];q=np.array(v['all_four']);m=np.array(v['available']);ts=np.array(v['relative_us']);pre=np.flatnonzero(ts<0)
        if len(pre)<2:continue
        perm=np.arange(len(q));perm[pre]=pre[::-1];q2=q[perm];m2=m[perm]
        def descriptor(x,valid):
            return np.stack([(x[keep]*valid[keep,None]).sum(0)/max(1,valid[keep].sum())for keep in [ts<=0,ts==0,ts>=0]])
        error=float(np.max(abs(descriptor(q,m)-descriptor(q2,m2))))
        assert error<1e-10
        proof.append(dict(clip_id=r['clip_id'],frame=r['frame'],mean_descriptor_max_difference=error,raw_order_values_changed=bool(np.any(q!=q2)),
            meaning='Same means can encode different internal order, a representation limitation. Synthetic reordering is not an independently valid sign or demonstrated cause of current error.'))
    write('CAUSE_AUDIT.json',dict(records=out,mean_order_counterexamples=proof,full_primary_unchanged=sum(r['views']['full']['old_new_primary_max_abs']<1e-6 for r in out),
        full_context_has_gap=sum(not all(r['views']['full']['available'])for r in out),
        boundary_points=sum(r['in_existing_boundary_band']for r in out),
        actual_mean_erasure_cause_proven=False,no_model_run=True))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for page in range(0,len(out),3):
        canvas=Image.new('RGB',(1500,1000),'white');draw=ImageDraw.Draw(canvas)
        for rr,r in enumerate(out[page:page+3]):
            cid,fr=r['clip_id'],r['frame'];draw.text((10,rr*330+5),f"{cid} family{r['family']} target{fr}",font=font,fill='#17324a');cap=cv2.VideoCapture(src[cid]['source'])
            for col,f in enumerate([fr-4,fr-2,fr,fr+2,fr+4]):
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
                h,w=im.shape[:2];pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop((int(w*.15),int(h*.04),int(w*.85),int(h*.88)));pic.thumbnail((294,260));canvas.paste(pic,(col*300+(300-pic.width)//2,rr*330+33));draw.text((col*300+8,rr*330+298),f'frame {f}',font=font,fill='#17324a')
            cap.release()
        canvas.save(U/f'rgb_{page//3+1}.jpg',quality=94)
    assert all(sha(p)==h for p,h in pins.items())
    print(json.dumps(dict(points=len(out),unchanged=sum(r['views']['full']['old_new_primary_max_abs']<1e-6 for r in out),gaps=sum(not all(r['views']['full']['available'])for r in out),boundary=sum(r['in_existing_boundary_band']for r in out),erasure_proven=False)))
if __name__=='__main__':main()
