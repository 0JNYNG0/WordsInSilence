"""TRAIN-only inventory; no detector, classifier forward or training."""
from common import *
import subprocess,zipfile,traceback,os
from collections import Counter
sys.path.insert(0,str(ROOT/'research/sign_model_pilot_v1/src'))
from pilot.web_input import to_native_like
from pilot.data import build_channels
B=O/'word100_body_boundary_20261017'

def frozen_setup():
    p=read(U/'PREFLIGHT.json');check(p['pins']);t=read(B/'training_CONTEXT.json')
    rows=[x for x in t['ordered_rows'] if x['role']=='train_candidate'];assert len(rows)==899 and all(x['signer_id'] in range(1,10) for x in rows)
    paths=[U/'PREFLIGHT.json',U/'inventory.py',U/'common.py',B/'training_CONTEXT.json',B/'training_core.py',
        O/'word100_fullframe_hand_merge_20261020/candidate.py',ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py',ROOT/'research/sign_model_pilot_v1/src/pilot/data.py']
    manifest=[]
    for row in rows:
        cid=row['clip_id'];rp=B/'data/records'/(cid+'.json');rec=read(rp);paths.append(rp)
        for view,cap in [('masked','native_capture'),('sensor','sensor_capture')]:
            for field in ('candidate','raw'):paths.append(Path(rec[cap][field]))
            for field in ('observation','channel_file'):paths.append(Path(rec['views'][view][field]))
            assert sha(rec['views'][view]['channel_file'])==row['views'][view]['sha256']
        manifest.append(dict(clip_id=cid,signer_id=row['signer_id'],word_code=row['word_code'],label_index=row['label_index'],record=str(rp)))
    write(U/'FROZEN.json',dict(at=now(),rows=manifest,pins={str(x.resolve()):sha(x) for x in paths},protected=read(V/'FROZEN.json')['protected'],
        audit='All899, no exclusions; exact original whole/context features, two actual available canvases, source archive header/CRC identity; missing-hand+visible-arm trigger counts only.',
        candidate='Only body-relative hand-wrist xy and associated body-valid channel may be newly filled at an originally wholly absent hand. Never alter existing values, local finger coordinates/masks, dt, span or original shoulder/hand size. No invented finger evidence.',
        matching='Existing C whole/context and masked/sensor schedule retained. Each context uses its own original frozen shoulder normalization. Sensor recovery must only see its cropped camera canvas.',
        prohibited=['detector/model forward','training','DEV score inspection','user/mentor/extra199/200','operating edits','Git','download','external']))
    print('TRAIN899 AUDIT FROZEN',flush=True)

def main():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()))
    rule=module('frozen_current_arm_rule',O/'word100_fullframe_hand_merge_20261020/candidate.py')
    core=module('frozen_C_schedule',B/'training_core.py')
    root=Path('W:/WordsInSilence/data/AIHub_103/videos');archives={}
    proc=subprocess.run(['rg','--files','--no-ignore','-g','*.zip',str(root)],capture_output=True,text=True,encoding='utf8',check=True)
    for line in proc.stdout.splitlines():archives.setdefault(Path(line).name,[]).append(Path(line))
    zips={};counts=Counter();rows=[];errors=[];by_id=Counter()
    for index,row in enumerate(f['rows']):
        cid=row['clip_id'];rec=read(row['record']);box=rec['calibration']['box_source_pixels'];src=rec['source'];arc=Path(src['archive'])
        try:
            if not arc.is_file():
                choices=archives.get(arc.name,[]);assert len(choices)==1,(str(arc),len(choices));arc=choices[0]
            if str(arc) not in zips:zips[str(arc)]=zipfile.ZipFile(arc,'r')
            entry=zips[str(arc)].getinfo(src['member']);assert entry.CRC==src['crc32'] and entry.file_size==src['bytes']
            out=dict(**row,source={**src,'resolved_archive':str(arc)},views={});viewtargets={}
            for view,capname in [('masked','native_capture'),('sensor','sensor_capture')]:
                cap=load(rec[capname]['candidate']);raw=load(rec[capname]['raw']);obs=load(rec['views'][view]['observation']);ch=load(rec['views'][view]['channel_file']);info=rec['views'][view]['feature_info']
                n=len(cap['frame_id']);assert np.array_equal(cap['frame_id'],raw['frame_id']) and np.array_equal(cap['frame_id'],obs['frame_id'])
                assert same(cap['timestamp_us'],obs['timestamp_us'])
                w=int(raw['width'][0]);h=int(raw['height'][0]);assert np.all(raw['width']==w) and np.all(raw['height']==h)
                # Current dataset cuts bottom only; fail explicitly if an offset needs conversion.
                assert box[0]==box[1]==0
                if view=='sensor':assert [w,h]==[box[2],box[3]] and rec['sensor_capture']['box']==box
                else:assert [w,h]==[rec['calibration']['source_width'],rec['calibration']['source_height']]
                nx,nv=to_native_like(obs['xy_px'],obs['usable']);scales={};spans={}
                for temporal in ('whole','context'):
                    lo,hi=map(int,info[temporal+'_span']);x=build_channels(nx[lo:hi+1],nv[lo:hi+1],obs['timestamp_us'][lo:hi+1]);old=ch['x_'+temporal]
                    assert old.dtype==x['x'].dtype and old.shape==x['x'].shape and old.tobytes()==x['x'].tobytes()
                    assert same(ch[temporal+'_frame_id'],obs['frame_id'][lo:hi+1]) and same(ch[temporal+'_ts'],obs['timestamp_us'][lo:hi+1])
                    scales[temporal]=dict(body=x['body_scale'],hands={key:float(val) if np.isfinite(val) else None for key,val in x['hand_scales'].items()})
                    spans[temporal]=[lo,hi];counts['exact_'+view+'_'+temporal+'_channel_cells']+=old.size
                targets=[]
                for i in range(n):
                    valid=cap['usable'][i,33:].reshape(2,21);pose=raw['pose_xy_px'][i];pu=raw['pose_usable'][i]
                    good,scale=rule.context(pose,pu,w,h)
                    if scale<=1:continue
                    proposal=rule.crop_boxes(pose,pu,[s for s in (0,1) if valid[s].any()],w,h)
                    sides=[]
                    for s,crop in proposal.items():
                        wrist=pose[15+s]
                        if good[15+s] and np.isfinite(wrist).all() and 0<=wrist[0]<box[2] and 0<=wrist[1]<box[3]:
                            assert not valid[s].any() and crop[0]>=0 and crop[1]>=0 and crop[2]<=w and crop[3]<=h;sides.append(s)
                    if sides:targets.append(dict(index=i,frame_id=int(cap['frame_id'][i]),sides=sides))
                out['views'][view]=dict(canvas=[w,h],original_roi=box,frames=n,original_spans=spans,original_scales=scales,
                    recovery_targets=targets,source_arrays=dict(capture=rec[capname]['candidate'],raw=rec[capname]['raw'],observation=rec['views'][view]['observation'],channels=rec['views'][view]['channel_file']))
                viewtargets[view]={(x['index'],s) for x in targets for s in x['sides']}
                counts[view+'_frames']+=n;counts[view+'_target_frames']+=len(targets);counts[view+'_target_hand_frames']+=sum(len(x['sides']) for x in targets)
                counts['views']+=1
            counts['sensor_only_target_hand_frames']+=len(viewtargets['sensor']-viewtargets['masked'])
            counts['full_only_target_hand_frames']+=len(viewtargets['masked']-viewtargets['sensor'])
            schedule=[core.schedule(cid,e,'C') for e in range(1,5)];assert set(schedule)=={(a,b) for a in ('sensor','masked')for b in ('x_whole','x_context')}
            out['first_four_epoch_schedule']=schedule;rows.append(out);counts['clips']+=1;by_id[str(row['signer_id'])]+=1
        except Exception as e:
            # Preserve and report every failed input; never silently exclude or train.
            errors.append(dict(clip_id=cid,error=repr(e)));rows.append(dict(**row,error=repr(e)));counts['failed_clips']+=1
        if (index+1)%100==0:print('AUDITED',index+1,'FAILED',len(errors),flush=True)
    for z in zips.values():z.close()
    assert len(rows)==899;check(f['pins']);check(f['protected'])
    write(U/'INVENTORY.json',dict(at=now(),status='PASS' if not errors else 'FAILED_INPUTS_TRAINING_PROHIBITED',counts=dict(counts),by_record_id=dict(by_id),rows=rows,errors=errors,
        source_check='ZIP member CRC32 and byte length checked; full SHA256 must be checked on extraction before new inference. No video decoded in inventory.',
        actual_new_detection=False,training=False,next='Freeze actual missing-wrist input preparation per available camera canvas; sensor and full-source caches are not interchangeable.'))
    (U/'RUNNING.json').unlink();print(json.dumps(dict(counts)),flush=True)

if __name__=='__main__':
    try:frozen_setup() if sys.argv[1]=='freeze' else main()
    except Exception:
        (U/('ERROR_'+now().replace(':','_')+'.txt')).write_text(traceback.format_exc(),encoding='utf8');raise
