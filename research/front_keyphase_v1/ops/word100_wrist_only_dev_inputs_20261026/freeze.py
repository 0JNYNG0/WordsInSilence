"""Prepare DEV manifest and identical-call cache; no detector or classifier call."""
from common import *
from collections import Counter
sys.path.insert(0,str(ROOT/'research/sign_model_pilot_v1/src'))
from pilot.web_input import to_native_like
from pilot.data import build_channels

def main():
    assert not (U/'FROZEN.json').exists()
    I=O/'word100_wrist_only_inputs_20261026';trainf=read(I/'FROZEN.json');check(trainf['pins']);check(trainf['protected'])
    original=read(B/'training_CONTEXT.json');vf=read(V/'FROZEN.json');check(vf['pins'])
    assert len(vf['rows'])==300
    target_rows=[r for r in original['ordered_rows'] if r['role']=='development_only']
    assert Counter(r['signer_id'] for r in target_rows)=={11:100,12:100,13:100}
    vrows={r['clip_id']:r for r in vf['rows']};assert set(vrows)=={r['clip_id'] for r in target_rows}
    selective=module('dev_input_selective',Y/'candidate.py')
    for d in ('cache','sources','records','outputs','features','frames','starts'):(U/d).mkdir(exist_ok=True)
    rows=[];cache={};pins={};counts=Counter()
    for row in target_rows:
        cid=row['clip_id'];vp=V/'records'/(cid+'.json');saved=read(vp);check(saved['outputs'])
        assert saved['context_sha256']==sha(V/'FROZEN.json')
        r=read(B/'data/records'/(cid+'.json'));src=vrows[cid]['source'];assert r['source']['sha256']==src['sha256']==saved['source_sha256']
        box=r['calibration']['box_source_pixels'];assert box[:2]==[0,0]
        entries={}
        for fr in saved['frames']:
            fid=fr['frame_id'];entry={}
            for kind in ('mp','yolo'):
                boxes=fr['boxes' if kind=='mp' else 'yolo_boxes']
                if not boxes:continue
                crops={}
                for s,b in boxes.items():
                    ds=[d for d in fr[kind+'_crop'] if int(d['requested_side'])==int(s)]
                    assert all(list(d['box'])==list(b) for d in ds)
                    crops[','.join(map(str,b))]=ds
                entry[kind]=dict(full=fr[kind+'_full'],crops=crops,source=str(vp))
            entries[str(fid)]=entry
        cp=U/'cache'/(cid+'.json')
        payload=dict(clip_id=cid,source_sha256=src['sha256'],canvas=[r['calibration']['source_width'],r['calibration']['source_height']],frames=entries)
        if cp.exists():assert read(cp)==payload
        else:write(cp,payload)
        cache[cid]=dict(file=str(cp),sha256=sha(cp),video=saved['source_video'],video_sha256=src['sha256'])
        pins[str(cp)]=sha(cp);pins[str(vp)]=sha(vp)
        for view,capname in [('masked','native_capture'),('sensor','sensor_capture')]:
            cap=load(r[capname]['candidate']);raw=load(r[capname]['raw']);obs=load(r['views'][view]['observation']);ch=load(r['views'][view]['channel_file'])
            assert sha(r['views'][view]['channel_file'])==row['views'][view]['sha256']
            assert equal(cap['frame_id'],raw['frame_id']) and equal(cap['timestamp_us'],obs['timestamp_us'])
            w,h=int(raw['width'][0]),int(raw['height'][0]);assert (raw['width']==w).all() and (raw['height']==h).all()
            if view=='sensor':assert [w,h]==box[2:]
            else:assert [w,h]==[r['calibration']['source_width'],r['calibration']['source_height']]
            info=r['views'][view]['feature_info'];spans={};scales={};nx,nv=to_native_like(obs['xy_px'],obs['usable'])
            for temporal in ('whole',):
                lo,hi=info[temporal+'_span'];span=slice(lo,hi+1);rebuilt=build_channels(nx[span],nv[span],obs['timestamp_us'][span])
                assert same(rebuilt['x'],ch['x_'+temporal]),(cid,view,temporal)
                spans[temporal]=[lo,hi];scales[temporal]=dict(body=rebuilt['body_scale'],hands={k:float(v) if np.isfinite(v) else None for k,v in rebuilt['hand_scales'].items()})
                counts['original_channel_cells_exact']+=rebuilt['x'].size
            targets=[]
            for i in range(len(cap['frame_id'])):
                if not any(lo<=i<=hi for lo,hi in spans.values()):continue
                boxes=selective.trigger(cap['xy_px'][i,33:].reshape(2,21,2),cap['usable'][i,33:].reshape(2,21),raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box)
                if boxes:targets.append(dict(index=i,frame_id=int(cap['frame_id'][i]),sides=sorted(boxes)))
            arr=dict(capture=r[capname]['candidate'],raw=r[capname]['raw'],observation=r['views'][view]['observation'],channels=r['views'][view]['channel_file'])
            ass=Path(arr['capture']).parent/'assignment.npz';assert ass.is_file()
            rows.append(dict(uid=cid+'__'+view,clip_id=cid,signer_id=row['signer_id'],view=view,
                source={**src,'resolved_archive':src['archive']},targets=targets,
                info=dict(canvas=[w,h],original_roi=box,frames=len(cap['frame_id']),original_spans=spans,original_scales=scales,source_arrays=arr),
                input_pins={p:sha(p) for p in list(arr.values())+[str(ass)]}))
            counts[view+'_target_frames']+=len(targets);counts[view+'_target_hands']+=sum(len(t['sides']) for t in targets)
            counts['views']+=1
    source=(OLD/'model_io.py').read_text('utf8').replace("p=U/'yolo_hand_best.pt'",f"p=Path({str(OLD/'yolo_hand_best.pt')!r})").replace("(U/'MODEL_SOURCE.json')",f"(Path({str(OLD/'MODEL_SOURCE.json')!r}))")
    (U/'model_io_local.py').write_text(source,encoding='utf8')
    for p in list(U.glob('*.py'))+[I/'FROZEN.json',I/'SMOKE_REVIEW.json',V/'FROZEN.json',B/'training_CONTEXT.json',OLD/'experiment.py',OLD/'MODEL_SOURCE.json',OLD/'yolo_hand_best.pt',V/'prepare.py',Y/'candidate.py',O/'word100_fullframe_hand_merge_20261020/candidate.py',ROOT/'Tools/models/hand_landmarker.task']:
        pins[str(p)]=sha(p)
    smoke=[]
    for sid in (11,12,13):
        cid=min({r['clip_id'] for r in rows if r['signer_id']==sid and r['targets']},key=lambda c:hashlib.sha256(c.encode()).hexdigest());smoke.append(cid)
    write(U/'FROZEN.json',dict(at=now(),rows=rows,cache=cache,smoke_uids=[r['uid'] for r in rows if r['clip_id'] in smoke],pins=pins,protected=trainf['protected'],
        authorization='Same public DEV300 observation preparation for the approved wrist-only recognition candidate. Already exposed development data, not independent final test. No scores examined or model changed.',
        protocol=trainf['protocol'],mp_options=trainf['mp_options'],raw_call_reuse_clarification=trainf['raw_call_reuse_clarification'],counts=dict(counts),
        no_training_until='TRAIN899 and DEV300 inputs independently verified and visually reviewed. Original whole/scale/time exact except accepted previously missing wrist bodyxy/mask.',
        cached_result_use='All previous full-source MP/YOLO raw calls reused when frame/crop match. Sensor image is different and cannot borrow full-canvas calls. No completed score re-inference.'))
    write(U/'INVENTORY_VERIFIED.json',dict(at=now(),status='PASS',counts=dict(counts),frozen_sha256=sha(U/'FROZEN.json'),new_detector_calls=0,new_scores=0,training=0))
    event(dict(stage='DEV_INPUTS_FROZEN',**counts))
if __name__=='__main__':main()
