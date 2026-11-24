from common import *

def main():
    assert not (U/'FROZEN.json').exists()
    protocol=read(P/'PREPARE_PROTOCOL.json');check(protocol['pins']);check(protocol['protected'])
    inventory=read(P/'INVENTORY.json');assert inventory['status']=='PASS' and len(inventory['rows'])==899
    prior=read(P/'FROZEN.json');check(prior['pins'])
    bridge=read(R/'FROZEN.json');check(bridge['pins'])
    original={r['clip_id']:r for r in inventory['rows']}
    index_old={r['clip_id']:r for r in read(OLD/'INDEX.json')};index_y={r['clip_id']:r for r in read(Y/'INDEX.json')}
    mph=sha(ROOT/'Tools/models/hand_landmarker.task');yh=sha(OLD/'yolo_hand_best.pt')
    assert mph=='fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
    assert yh=='9a98cb80bcb56a565132308433015c346f1e676a4bed856180856f541a9ce090'
    for d in ('cache','sources','records','outputs','features','frames','starts'):(U/d).mkdir(exist_ok=True)
    pins={str(p):sha(p) for p in [P/'PREPARE_PROTOCOL.json',P/'INVENTORY.json',P/'INVENTORY_VERIFIED.json',P/'FROZEN.json',R/'FROZEN.json',OLD/'FROZEN.json',Y/'FROZEN.json']}
    audit=[];cache={}
    for r in bridge['rows']:
        cid=r['clip_id'];rr=original[cid];assert r['rgb_sha256']==rr['source']['sha256']
        assert sha(r['rgb_path'])==r['rgb_sha256'];check(r['source_pins'])
        z=load(r['source_assets']['raw']);native=load(r['source_assets']['native']);current=load(rr['views']['masked']['source_arrays']['capture'])
        assert equal(z['frame_id'],current['frame_id']) and equal(z['timestamp_us'],current['timestamp_us'])
        path=Path(r['source_assets']['detections']);unit=O/path.relative_to(O).parts[0];uf=read(unit/'FROZEN.json')
        assert mph in uf['pins'].values(),str(unit)
        # IMAGE calls depend on pixels/model/options, not previous assignment or ROI.
        # Reuse raw calls only. Re-run reconciliation with current native data.
        entries={};nfull=ncrop=0
        for fr in read(path)['frames']:
            i=fr['index'];fid=int(z['frame_id'][i]);assert fid==fr['frame_id']
            boxes=fr.get('boxes')
            if boxes is None:
                boxes=rule.crop_boxes(z['pose_xy_px'][i],z['pose_usable'][i],fr['baseline_sides'],int(z['width'][i]),int(z['height'][i]))
            crops={}
            for side,box in boxes.items():
                ds=[d for d in fr['crop'] if int(d['requested_side'])==int(side)]
                assert all(list(d['box'])==list(box) for d in ds)
                crops[','.join(map(str,box))]=ds;ncrop+=1
            entries[str(fid)]={'mp':{'full':fr['full'],'crops':crops,'source':str(path)}};nfull+=1
        for ix in (index_old,index_y):
            if cid not in ix:continue
            rec=ix[cid];assert sha(rec['records'])==rec['records_sha256'];pins[rec['records']]=rec['records_sha256']
            for fr in read(rec['records'])['frames']:
                fid=int(z['frame_id'][fr['i']]);assert fid==fr.get('frame_id',fid)
                bucket=entries.setdefault(str(fid),{}).setdefault('yolo',{'full':fr['full'],'crops':{},'source':rec['records']})
                # Prefer the original all-frame run; later calls only fill absent crop boxes.
                for side,box in fr['boxes'].items():
                    ds=[d for d in fr['crop'] if int(d['requested_side'])==int(side)]
                    assert all(list(d['box'])==list(box) for d in ds)
                    bucket['crops'].setdefault(','.join(map(str,box)),ds)
        cp=U/'cache'/(cid+'.json');payload=dict(clip_id=cid,source_sha256=r['rgb_sha256'],canvas=r['video_dimensions'],frames=entries)
        if cp.exists():assert read(cp)==payload
        else:write(cp,payload)
        cache[cid]=dict(file=str(cp),sha256=sha(cp),video=r['rgb_path'],video_sha256=r['rgb_sha256'])
        audit.append(dict(clip_id=cid,mp_full=nfull,mp_crop=ncrop,source_frame_canvas_match=True,
            native_equal=all(equal(native[k],current[k]) for k in ('xy_px','usable','frame_id','timestamp_us')),
            roi_equal=r['roi_box_source_pixels']==rr['views']['masked']['original_roi'],
            reuse='Raw IMAGE call only, exact frame and crop; current native/ROI merge recalculated. Sensor cache unavailable unless same canvas.'))
        pins[str(cp)]=sha(cp);pins[str(unit/'FROZEN.json')]=sha(unit/'FROZEN.json')
    source=(OLD/'model_io.py').read_text('utf8').replace("p=U/'yolo_hand_best.pt'",f"p=Path({str(OLD/'yolo_hand_best.pt')!r})").replace("(U/'MODEL_SOURCE.json')",f"(Path({str(OLD/'MODEL_SOURCE.json')!r}))")
    (U/'model_io_local.py').write_text(source,encoding='utf8')
    for p in list(U.glob('*.py'))+[V/'prepare.py',V/'common.py',OLD/'experiment.py',OLD/'MODEL_SOURCE.json',OLD/'yolo_hand_best.pt',Y/'candidate.py',O/'word100_fullframe_hand_merge_20261020/candidate.py',ROOT/'Tools/models/hand_landmarker.task']:
        pins[str(p)]=sha(p)
    rows=[];target={(t['clip_id'],t['view']):t for t in protocol['targets']}
    for r in inventory['rows']:
        for v in ('masked','sensor'):
            a=r['views'][v]['source_arrays'];ass=Path(a['capture']).parent/'assignment.npz'
            rows.append(dict(uid=r['clip_id']+'__'+v,clip_id=r['clip_id'],signer_id=r['signer_id'],view=v,info=r['views'][v],source=r['source'],targets=target[(r['clip_id'],v)]['targets'],
                input_pins={p:sha(p) for p in list(a.values())+[str(ass)]}))
    # Hash selection is independent of labels, detector quality and recognition score.
    smoke=[]
    for sid in range(1,10):
        cids=sorted({r['clip_id'] for r in rows if r['signer_id']==sid and r['targets']},key=lambda c:hashlib.sha256(c.encode()).hexdigest())
        assert cids;smoke.extend(cids[:1])
    # Include a cached input pair as an explicit cache-provenance control.
    smoke.append(sorted(cache,key=lambda c:hashlib.sha256(c.encode()).hexdigest())[0]);smoke=sorted(set(smoke))
    write(U/'CACHE_AUDIT.json',dict(at=now(),rows=audit,clips=len(audit),method='Raw predictions independent of assignments; source/model/options/canvas/frame/crop mandatory, downstream outputs never reused.',sensor_reuse='Disallowed for full-source cache when canvas differs'))
    pins[str(U/'CACHE_AUDIT.json')]=sha(U/'CACHE_AUDIT.json')
    write(U/'FROZEN.json',dict(at=now(),rows=rows,smoke_uids=[r['uid'] for r in rows if r['clip_id'] in smoke],cache=cache,pins=pins,protected=protocol['protected'],
        authorization='Persistent user-approved autonomous public TRAIN/DEV improvement; no operating replacement or user recordings.',
        protocol=protocol['protocol'],mp_options=dict(num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5),
        raw_call_reuse_clarification='Pixel-identical IMAGE raw calls may be reused even when native/ROI differs; merge always recalculated. Native/ROI differences prevent downstream output reuse.',
        no_training_until='All inputs numeric/geometry/crossing and visual review complete; smoke gate before full extraction. All899 kept.'))
    event(dict(stage='FROZEN',views=len(rows),cache_clips=len(cache),smoke_views=len([r for r in rows if r['clip_id'] in smoke])))
if __name__=='__main__':main()
