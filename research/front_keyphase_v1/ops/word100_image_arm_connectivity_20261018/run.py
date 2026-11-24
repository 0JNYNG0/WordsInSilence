import json,hashlib,shutil,sys,os,time,traceback,zipfile,zlib
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import numpy as np,cv2
from candidate import PARAMS,propose
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_remaining_identity_audit_20261018';B=O/'word100_uncertain_hand_preservation_20261018';A=O/'word100_label_only_audit_20261018'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def prepare():
    assert not (U/'CONTEXT.json').exists();assert read(U/'CONTRACTS.json')['status']=='PASS';assert read(P/'FINAL_AUDIT.json')['status']=='PASS'
    c=read(P/'CONTEXT.json');pins=dict(c['pins']);pins.update(read(P/'FINAL_AUDIT.json')['pins'])
    for p in list(U.glob('*.py'))+[U/'CONTRACTS.json',P/'FINAL_AUDIT.json']:pins[str(p.resolve())]=sha(p)
    check(pins);rows={r['clip_id']:r for r in c['rows']};frames=defaultdict(set)
    for x in read(B/'ANNOTATED_COMPARISON.json'):frames[(x['clip_id'],x['view'])].add(x['index'])
    for x in read(B/'CROSSING_COMPARISON.json'):frames[(x['clip_id'],x['view'])].add(x['index'])
    for x in read(P/'REMAINING9.json'):
        for d in (-2,-1,0,1,2):frames[(x['clip_id'],x['view'])].add(max(0,x['index']+d))
    selected=[{'clip_id':cid,'view':v,'frames':sorted(fs)} for (cid,v),fs in sorted(frames.items())]
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
    write(U/'CONTEXT.json',{'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'rows':[rows[k] for k in sorted({x['clip_id'] for x in selected})],'selection':selected,'pins':pins,'parameters':PARAMS,
        'authorization':'User approved public image arm-to-hand evidence validation, remaining9 and normal/crossing preservation. C unchanged, no user training/tuning, training only after results reviewed.',
        'fixed_hypothesis':'Frame-local image connectivity adds independent support when Pose wrists weak. Crop to actual detector FOV first.640px image; CIELAB a/b local endpoint patch medians(radius .02shoulders,min1px), union chroma ball radius20; no fixed skin-color template.8connected color pixels, exactlyone usable elbow and exactlyone observed Hand wrist in component, shortest path <=1.5Chebyshev endpointdistance. Both shoulders/elbows usable/in-frame. No morphology/holefill. No observed hand removed. Current image only.',
        'assignment_guard':'Keep prior evidence-backed and withheld assignments. Only replace provisional existing hint when RGB supports unique anatomical elbow and destination is unoccupied by another prior assignment. Keep actual hand coordinates and masks. RGB connection is supporting machine evidence, not anatomical truth.',
        'scope':'All461prior annotatedhand events (441normal,13originalerrors,2uncertain,5additional) and315crossingframes, plus +/-2remaining9neighbors. Finite frozen public TRAIN diagnostic; NOT full899RGBreplay or blind heldout. No word/signer features or per-case exceptions.',
        'acceptance':'Remaining9 improvement only if original video anatomy agrees, all441normal and315crossing preserved. Any contrary normal/crossing result rejects candidate without retuning. Full change visual review, scalar mask/connectivity/path/decision independent verification. Pixel ablations and outside-FOV perturbation establish image dependency and no information leak.',
        'source_limits':['publicTRAIN1..9 only','crop before all pixel processing','whole native view only for native condition','no user/mentor/DEV/heldout','no detection/model training/classifier inference','no download/external transfer/camera/Git','no C/TCN/DTW/web change'],
        'roles':'Codex implementation/source visual review; independent numeric code, no separate human ground-truth reviewer'})
    print('FROZEN',len(selected),'clip/view groups',sum(len(x['frames']) for x in selected),'frames',len(pins),'pins',flush=True)

def source_video(r):
    for d in (A,B,P,O/'word100_visible_arm_duplicate_20261018',O/'word100_single_arm_recheck_20261018',O/'word100_track_revalidation_20261018'):
        p=d/'videos'/f"{r['clip_id']}.mp4"
        if p.exists():assert sha(p)==r['source']['sha256'];return p
    (U/'videos').mkdir(exist_ok=True);p=U/'videos'/f"{r['clip_id']}.mp4";src=r['source']
    assert not p.exists()
    with zipfile.ZipFile(src['archive']) as z:
        info=z.getinfo(src['member']);blob=z.read(info)
    assert len(blob)==src['bytes'] and hashlib.sha256(blob).hexdigest()==src['sha256'] and (zlib.crc32(blob)&0xffffffff)==src['crc32'];p.write_bytes(blob);return p

def main():
    c=read(U/'CONTEXT.json');check(c['pins']);assert not (U/'RESULTS.json').exists() and not (U/'RUNNING.json').exists();write(U/'RUNNING.json',{'pid':os.getpid(),'task':'selected public RGB connectivity audit'})
    for d in ('images','maps'): (U/d).mkdir()
    rows={r['clip_id']:r for r in c['rows']};groups=defaultdict(list)
    for x in c['selection']:groups[x['clip_id']].append(x)
    ann=read(B/'ANNOTATED_COMPARISON.json');truth={(x['clip_id'],x['view'],x['index'],x['raw_index']):x for x in ann};remaining={(x['clip_id'],x['view'],x['index'],x['raw_index']) for x in read(P/'REMAINING9.json')};cross={(x['clip_id'],x['view'],x['index']) for x in read(B/'CROSSING_COMPARISON.json')}
    results=[];annotations=[];crossed=[];changes=[];pins={};counts=Counter();start=time.monotonic()
    try:
        for ci,(cid,gs) in enumerate(groups.items(),1):
            r=rows[cid];assert r['role']=='train_candidate' and 1<=r['signer_id']<=9;video=source_video(r);cap=cv2.VideoCapture(str(video));assert cap.isOpened()
            for gr in gs:
                view=gr['view'];z=load(r['views'][view]['raw']);old=load(B/'results'/f'{cid}_{view}.npz')
                for f in gr['frames']:
                    assert f<len(z['raw_count']);fid=int(z['frame_id'][f]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok,(cid,f)
                    n=int(z['raw_count'][f]);w=int(z['width'][f]);h=int(z['height'][f]);p=z['pose_xy_px'][f];pm=z['pose_usable'][f];hands=z['hand_xy_px'][f,:n];mask=old['point_mask'][f,:n];sides=old['sides'][f,:n];ids=old['identity_side'][f,:n]
                    o=propose(im,w,h,p,pm,hands,mask,sides,ids);base=f'{cid}_{view}_{f:04d}';ip=U/'images'/f'{base}.png';mp=U/'maps'/f'{base}.npz';assert cv2.imwrite(str(ip),o['image']);np.savez_compressed(mp,support=o['support'],components=o['components']);pins[str(ip)]=sha(ip);pins[str(mp)]=sha(mp)
                    # A real cropped sensor cannot see pixels below the crop.
                    altered=im.copy();altered[h:]=255-altered[h:];altered[:h,w:]=255-altered[:h,w:]
                    hidden=propose(altered,w,h,p,pm,hands,mask,sides,ids)
                    assert hidden['sides']==o['sides'] and hidden['rgb_side']==o['rgb_side'] and np.array_equal(hidden['support'],o['support']);counts['outside_fov_invariance_frames']+=1
                    # Raw detector order must not create a different side.
                    rev=propose(im,w,h,p,pm,hands[::-1],mask[::-1],sides[::-1],ids[::-1]);assert rev['sides'][::-1]==o['sides'] and rev['rgb_side'][::-1]==o['rgb_side'];counts['order_invariance_frames']+=1
                    # No useful chroma information: must not create RGB sides.
                    gray=cv2.cvtColor(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR);ab=propose(gray,w,h,p,pm,hands,mask,sides,ids);assert all(v<0 for v in ab['rgb_side']);assert ab['sides']==sides.tolist();counts['grayscale_ablation_frames']+=1
                    record={'clip_id':cid,'word':r['label'],'view':view,'index':f,'frame_id':fid,'raw_count':n,'old_sides':sides.tolist(),'old_identity':ids.tolist(),'observed_points':mask.sum(1).tolist(),'sides':o['sides'],'rgb_side':o['rgb_side'],'reason':o['reason'],'pixel_reason':o['pixel_reason'],'elbows':o['elbows'],'wrists':o['wrists'],'elbow_components':o['elbow_components'],'wrist_components':o['wrist_components'],'scale':o['scale'],'centers_ab':o['centers_ab'],'paths':o['paths'],'image':str(ip),'map':str(mp),'source_video':str(video),'source_sha256':r['source']['sha256']};results.append(record);counts['frames']+=1;counts['raw_candidates']+=n;counts['preserved_observed_points']+=int(mask.sum())
                    for i in range(n):
                        counts[o['reason'][i]]+=1;k=(cid,view,f,i)
                        if o['sides'][i]!=sides[i]:changes.append({**record,'raw_index':i,'before':int(sides[i]),'after':o['sides'][i]})
                        if k in truth:
                            t=truth[k];annotations.append({**t,'before':int(sides[i]),'after':o['sides'][i],'rgb_side':o['rgb_side'][i],'reason':o['reason'][i],'pixel_reason':o['pixel_reason'][i],'remaining9':k in remaining,'image':str(ip),'record_index':len(results)-1})
                    if (cid,view,f) in cross:crossed.append({'clip_id':cid,'view':view,'index':f,'before':sides.tolist(),'after':o['sides']})
            cap.release()
            if ci%50==0:print('RGB',ci,'/',len(groups),'clips','changes',len(changes),flush=True)
        assert len(annotations)==461 and len(crossed)==315
        norms=[x for x in annotations if x['verdict']=='normal_identity'];rem=[x for x in annotations if x['remaining9']]
        summary={'status':'EXECUTED_PENDING_INDEPENDENT_VISUAL','counts':dict(counts),'selected_clips':len(groups),'normal441_preserved':sum(x['after']==x['before']==x['actual_side'] for x in norms),'normal_regressions':[x for x in norms if x['after']!=x['actual_side']],'crossing315_preserved':sum(x['before']==x['after'] for x in crossed),'remaining9_proposed_correct':sum(x['after']==x['actual_side'] for x in rem),'remaining9_proposed_wrong':sum(x['after']!=x['actual_side'] for x in rem),'changed_events':len(changes),'elapsed_seconds':time.monotonic()-start,'full899_rgb_replayed':False,'training':0,'classifier_inference':0,'user_reads':0,'C_changed':False}
        for name,data in [('RESULTS',results),('ANNOTATED',annotations),('CROSSING',crossed),('CHANGES',changes),('OUTPUT_PINS',pins),('SUMMARY',summary)]:write(U/f'{name}.json',data)
        check(c['pins']);(U/'RUNNING.json').unlink();print(json.dumps({k:v for k,v in summary.items() if k!='normal_regressions'},ensure_ascii=False),flush=True)
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'error':traceback.format_exc()})
        raise
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else main()
