"""Native fixed-pixel export for the finite, previously selected FIT inventory."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import av
from PIL import Image,ImageDraw
N=Path(__file__).resolve().parent;S=N/'support_inventory'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    selection=read(S/'SELECTION.json');assert len(selection['selected'])==12
    assert not (S/'MANIFEST.json').exists()
    write(S/'EXPORT_FROZEN.json',dict(pins={str(p):sha(p) for p in [Path(__file__),S/'PLAN_FROZEN.json',S/'SELECTION.json']}))
    rows=[]
    for c,row in enumerate(selection['selected'],1):
        source=Path(row['source']);assert sha(source)==row['source_sha256']
        alias=row['support_alias'];folder=S/'visuals'/alias;folder.mkdir(parents=True,exist_ok=False)
        wanted=set(row['frame_ids']);frames=[];crops={}
        with av.open(str(source)) as v:
            for i,f in enumerate(v.decode(video=0)):
                if i in wanted:
                    ts=round(float(f.pts*f.time_base)*1e6)
                    assert abs(ts-row['timestamps_us'][row['frame_ids'].index(i)])<=1
                    im=f.to_image().convert('RGB');w,h=im.size
                    box=[round(w*.2),round(h*.08),round(w*.8),round(h*.95)]
                    crop=im.crop(box);full=folder/f'full_{i:04}.png';fixed=folder/f'fixed_{i:04}.png'
                    im.save(full);crop.save(fixed);crops[i]=crop
                    frames.append(dict(frame_id=i,timestamp_us=ts,source_size=[w,h],crop_box=box,
                        full=str(full),full_sha256=sha(full),fixed=str(fixed),fixed_sha256=sha(fixed)))
                if i>=max(wanted):break
        assert len(frames)==13
        pairs=[]
        for pos in range(0,13,2):
            group=frames[pos:pos+2];cw,ch=crops[group[0]['frame_id']].size
            image=Image.new('RGB',(cw*2,ch+36),'#20252b');draw=ImageDraw.Draw(image)
            for j,f in enumerate(group):
                draw.text((j*cw+12,8),f"{alias} f{f['frame_id']} {f['timestamp_us']/1e6:.6f}s native1:1",fill='white')
                image.paste(crops[f['frame_id']],(j*cw,36))
            p=folder/f'pair_{pos//2:02}.png';image.save(p)
            pairs.append(dict(file=str(p),sha256=sha(p),frame_ids=[f['frame_id'] for f in group]))
        rows.append(dict(support_alias=alias,sample_id=row['sample_id'],target_hand=row['hand'],
            source=str(source),source_sha256=row['source_sha256'],person=row['person'],word_id=row['word_id'],
            role='fit',roi_bottom=row['roi_bottom'],frames=frames,pairs=pairs))
        print(json.dumps(dict(done=c,total=12,alias=alias)),flush=True)
    write(S/'MANIFEST.json',dict(status='COMPLETE',rows=rows,windows=12,native_frames=156,selection_sha256=sha(S/'SELECTION.json')))
    write(S/'BLIND_SUPPORT.json',dict(status='COMPLETE',manifest_sha256=sha(S/'MANIFEST.json'),
        rubric=str(N.parent/'word100_interval_motion_20261101/RUBRIC.json'),
        rubric_sha256=sha(N.parent/'word100_interval_motion_20261101/RUBRIC.json'),
        task='Only target hand hold presence. 1 needs >=3consecutive directly maintained positions with exactsegments.0 needs fullyobserved13 and definitely noheld>=3segment; truncatedpossiblehold/uncertainty=>null. No score/label access. Record sourceframe evidence and full13ROI separately. Coordinate nomination is not truth.',
        output_schema=dict(alias='C01',sample_id='sourceid',target_hand='R|L',hold_present='int0/int1/null',
            hold_segments=[dict(first_frame=0,last_frame=0,in_roi=True,note='direct maintained position')],
            full13_position_observable='bool|null',full13_roi='bool|null',
            uncertain_intervals=[dict(first_frame=0,last_frame=0,reason='uncertainty affecting hold judgment')],
            actual_image_viewed=True,images_viewed=[dict(file='actualimage',sha256='hash')],
            note='Actual observed evidence; do not infer physical meanings or force absence.'),
        rows=[{k:v for k,v in r.items() if k not in ['source','source_sha256','person','word_id','role']} for r in rows]))
if __name__=='__main__':main()
