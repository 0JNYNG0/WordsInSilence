"""Read-only DEV300 observation/source availability before any new neural scoring."""
from common import *
from collections import Counter
import zipfile
B=O/'word100_body_boundary_20261017'
def main():
    unit=O/'word100_recovery_dev300_20261026';assert not unit.exists();unit.mkdir();(unit/'before').mkdir()
    import shutil
    for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'):shutil.copyfile(p,unit/'before'/p.name)
    ctx=read(B/'training_CONTEXT.json');rows=[r for r in ctx['ordered_rows'] if r['role']=='development_only']
    assert len(rows)==300 and Counter(r['signer_id'] for r in rows)=={11:100,12:100,13:100}
    rule=module('dev_source_geometry',O/'word100_fullframe_hand_merge_20261020/candidate.py')
    result=[];pins={};zs={};total=Counter()
    for r in rows:
        rp=Path(ctx['records_directory'])/(r['clip_id']+'.json');rec=read(rp);p=load(rec['native_capture']['candidate']);raw=load(rec['native_capture']['raw']);masked=load(rec['views']['masked']['observation']);box=rec['calibration']['box_source_pixels']
        x=p['xy_px'];v=p['usable'];n=len(v);inside=np.isfinite(x).all(-1)&(x[...,0]>=box[0])&(x[...,0]<box[2])&(x[...,1]>=box[1])&(x[...,1]<box[3])
        assert same(masked['usable'],v&inside) and same(masked['xy_px'][masked['usable']],x[masked['usable']])
        assert same(raw['frame_id'],p['frame_id']) and same(raw['timestamp_us'],p['timestamp_us'])
        hv=v[:,33:].reshape(n,2,21);targets=[]
        for i in range(n):
            pose=raw['pose_xy_px'][i];usable=raw['pose_usable'][i];w,h=int(raw['width'][i]),int(raw['height'][i]);good,scale=rule.context(pose,usable,w,h)
            boxes=rule.crop_boxes(pose,usable,[s for s in (0,1) if hv[i,s].any()],w,h)
            target=[s for s in boxes if good[15+s] and box[0]<=pose[15+s,0]<box[2] and box[1]<=pose[15+s,1]<box[3]] if scale>1 else []
            if target:targets.append(dict(index=i,sides=target))
        src=rec['source'];zp=src['archive']
        if zp not in zs:zs[zp]=zipfile.ZipFile(zp,'r')
        info=zs[zp].getinfo(src['member']);assert info.CRC==src['crc32'] and info.file_size==src['bytes']
        for path in (rp,Path(rec['native_capture']['candidate']),Path(rec['native_capture']['raw']),Path(rec['native_capture']['raw']).parent/'assignment.npz',Path(rec['views']['masked']['observation']),Path(rec['views']['masked']['channel_file'])):pins[str(path)]=sha(path)
        result.append({k:r[k] for k in ('clip_id','signer_id','label_index','word_code')}|dict(record=str(rp),source=src,frames=n,box=box,full_source_missing_hand_frames=int((~hv.any(-1)).sum()),recovery_target_frames=targets))
        total.update(clips=1,frames=n,roi_compatible_frames=n,target_frames=len(targets),target_hand_frames=sum(len(t['sides']) for t in targets),source_bytes=src['bytes'])
    for z in zs.values():z.close()
    protected=read(U/'FROZEN.json')['protected'];check(protected)
    paths=[Path(__file__),O/'word100_fullframe_hand_merge_20261020/candidate.py',O/'word100_yolo_selective_wrist_20261025/candidate.py',ROOT/'Tools/models/hand_landmarker.task',O/'word100_yolo_hand_fourway_20261025/yolo_hand_best.pt']
    pins.update({str(p):sha(p) for p in paths})
    write(unit/'PREFLIGHT.json',dict(at=now(),status='READONLY_INPUT_PREFLIGHT_PASS_NO_NEW_INFERENCE',counts=dict(total),rows=result,pins=pins,protected=protected,
        purpose='Full public DEV300 before/after recovery comparison. This preflight only verifies native full-source -> fixed ROI equivalence, clocks, target counts and source ZIP member availability. No new image decoding/detectors/scores yet.',
        next='Freeze missing-side-only MP IMAGE retry followed by selective YOLO. Keep all old valid points/normal crossing. Same current-arm/full+crop agreement constants. No label/GT-driven gates. Save full-source merged evidence then same ROI. Independently verify inputs; re-use exact original masked logits when channels exactly match; evaluate frozen C13/T3F three seeds/E3 once on changed arrays. Fixed scales/span diagnostic separates normalization. Report DEV gains/regressions/latency and actual evidence, not unseen-user accuracy.',
        limits='Current cached native identity/landmarks are inherited. Selective absence fallback cannot correct already-present wrong hands; uncertain identity is not forced. Full-input arm visibility required, screen-outside hands not hallucinated. Finger shape remains quality proxy without new anatomical labels.',
        forbidden=['user/mentor','extra199/200','training','production edit','new model downloads','Git','upload']))
    print('DEV PREFLIGHT',json.dumps(dict(total)),flush=True)
if __name__=='__main__':main()
