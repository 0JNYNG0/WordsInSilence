"""Complete the SAME24 pilot's native views; exactly reuse prior9, export15 once."""
from pathlib import Path
import sys, json, importlib.util
sys.dont_write_bytecode = True
N = Path(__file__).resolve().parent
F = N / 'full24_native'
W = N.parent / 'word100_interval_motion_20261101'
U = N.parent / 'word100_core_feature_split_20261101'
spec = importlib.util.spec_from_file_location('native_export_util', N/'export_support.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
read, sha, write = m.read, m.sha, m.write

def main():
    support = read(N/'support_inventory/SUPPORT_RESULT.json')
    assert not support['annotation_contract_failures']
    assert all(support['metrics'][h]['eligible_agreed_absence'] > 0 for h in 'RL')
    prior = read(W/'PILOT_MANIFEST.json')
    native = read(N/'MANIFEST.json')
    done = {r['sample_id']: r for r in native['rows']}
    remaining = [r for r in prior['rows'] if r['sample_id'] not in done]
    assert len(prior['rows']) == 24 and len(done) == 9 and len(remaining) == 15
    inputs = {r['alias']:r for r in read(U/'INPUTS.json')['rows']}
    F.mkdir(exist_ok=False)
    pins = {str(p):sha(p) for p in [Path(__file__),N/'export_support.py',N/'MANIFEST.json',
        N/'review/NATIVE_A.json',N/'review/NATIVE_B.json',N/'NATIVE_DIAGNOSIS.json',
        N/'support_inventory/SUPPORT_RESULT.json',W/'PILOT_MANIFEST.json',W/'PILOT_GATES.json',
        W/'RUBRIC.json',W/'audit_presence.py',W/'PRESENCE_PILOT_AUDIT.json',U/'INPUTS.json']}
    write(F/'PLAN_FROZEN.json',dict(status='FROZEN_BEFORE_REMAINING15_REVIEW',
        purpose='Same24 native evidence-readability reassessment, not new generalization validation.',
        reuse9=sorted(done),review15=[r['sample_id'] for r in remaining],
        gates='Exactly unchanged W24 gates. Support12 is separate, NEVER added to24 denominator or class support.',
        unknown='N008 stays unknown unless evidence already saved; no repeated rescue. Unknown is not absence.',
        continuation='Only full24 gate PASS permits full288 labels; full training support and controls still required. Any gate FAIL closes this verification without broad labels/training.',
        scope='FIT only, source-native pixels, no model inference or re-extraction, no private data.',pins=pins))
    selection=[]
    for i,r in enumerate(remaining,1):
        inp=inputs[r['sample_id'].split('_')[0]]
        assert inp['role']==r['role']=='fit'
        selection.append(dict(support_alias=f'P{i:02}',sample_id=r['sample_id'],hand='both',
            source=inp['source'],source_sha256=inp['source_sha256'],person=r['person'],word_id=r['word_id'],
            roi_bottom=inp['box'][3],frame_ids=r['frame_ids'],timestamps_us=r['timestamps_us']))
    # Reuse the frozen export implementation body with a narrowly adapted count.
    # It only decodes source RGB, never detector/model data. Source bytes are pinned.
    rows=[]
    import av
    from PIL import Image,ImageDraw
    for c,r in enumerate(selection,1):
        source=Path(r['source']);assert sha(source)==r['source_sha256']
        alias=r['support_alias']; folder=F/'visuals'/alias;folder.mkdir(parents=True,exist_ok=False)
        wanted=set(r['frame_ids']);frames=[];crops={}
        with av.open(str(source)) as v:
            for i,f in enumerate(v.decode(video=0)):
                if i in wanted:
                    ts=round(float(f.pts*f.time_base)*1e6)
                    assert abs(ts-r['timestamps_us'][r['frame_ids'].index(i)])<=1
                    im=f.to_image().convert('RGB'); w,h=im.size
                    box=[round(w*.2),round(h*.08),round(w*.8),round(h*.95)]
                    crop=im.crop(box);full=folder/f'full_{i:04}.png';fixed=folder/f'fixed_{i:04}.png'
                    im.save(full);crop.save(fixed);crops[i]=crop
                    frames.append(dict(frame_id=i,timestamp_us=ts,source_size=[w,h],crop_box=box,
                        full=str(full),full_sha256=sha(full),fixed=str(fixed),fixed_sha256=sha(fixed)))
                if i>=max(wanted):break
        assert len(frames)==13 and [f['frame_id'] for f in frames]==r['frame_ids']
        pairs=[]
        for pos in range(0,13,2):
            group=frames[pos:pos+2];cw,ch=crops[group[0]['frame_id']].size
            im=Image.new('RGB',(cw*2,ch+36),'#20252b');draw=ImageDraw.Draw(im)
            for j,f in enumerate(group):
                draw.text((j*cw+12,8),f"{alias} f{f['frame_id']} {f['timestamp_us']/1e6:.6f}s native1:1",fill='white')
                im.paste(crops[f['frame_id']],(j*cw,36))
            p=folder/f'pair_{pos//2:02}.png';im.save(p)
            pairs.append(dict(file=str(p),sha256=sha(p),frame_ids=[f['frame_id'] for f in group]))
        rows.append(dict(alias=alias,sample_id=r['sample_id'],source=str(source),source_sha256=r['source_sha256'],
            person=r['person'],word_id=r['word_id'],role='fit',roi_bottom=r['roi_bottom'],frames=frames,pairs=pairs))
        print(json.dumps(dict(done=c,total=15,alias=alias)),flush=True)
    write(F/'MANIFEST.json',dict(status='COMPLETE',rows=rows,windows=15,native_frames=195,pins=pins))
    write(F/'BLIND_NATIVE.json',dict(status='COMPLETE',native_manifest_sha256=sha(F/'MANIFEST.json'),
        rubric=str(W/'RUBRIC.json'),rubric_sha256=sha(W/'RUBRIC.json'),
        task='Same exact original13 sourceframes. Review R/L down/hold presence + exactsegments, unknowns, ROI and events using unchanged rubric. No prior labels/results. Preserve normal hand crossing and uncertainty. No semantic correctness labels.',
        rows=[{k:v for k,v in r.items() if k not in ['source','source_sha256','person','word_id','role']} for r in rows]))

if __name__=='__main__':main()
