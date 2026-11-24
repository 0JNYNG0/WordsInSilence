"""Read-only native RGB export for nine predeclared fit windows. No detectors."""
from pathlib import Path
import json, hashlib, datetime, sys
sys.dont_write_bytecode = True
import av
from PIL import Image, ImageDraw

N=Path(__file__).resolve().parent
O=N.parent
W=O/'word100_interval_motion_20261101'
U=O/'word100_core_feature_split_20261101'
SIX=['N008_W1','N047_W1','S005_W1','N007_W1','S020_W0','S004_W2']
HOLD=['N022_W2','S039_W0','S020_W0','S002_W1']
IDS=sorted(set(SIX+HOLD))
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    assert not (N/'MANIFEST.json').exists(), 'Reuse completed exports; no overwrite.'
    (N/'review').mkdir(exist_ok=True)
    plan=dict(status='FROZEN_BEFORE_NATIVE_REVIEW',date='2026-11-01',
        authorization='User requested native large torso/arm review of six left ambiguities and genuine continuous-motion support, conditional learning only after valid readiness.',
        six_left=SIX,hold_support=HOLD,union_nine=IDS,role='fit only',
        native_export='Decode same exact13 frame IDs/PTS. Fixed central60% width, y8%-95%, at1:1 native pixels for every frame. Full source frames kept for crop-boundary review. No resizing/tracking/stabilization/interpolation/detection.',
        rubric=str(W/'RUBRIC.json'),rubric_sha256=sha(W/'RUBRIC.json'),
        comparison='Retrospective evidence-readability diagnosis. Preserve original24 answers/FAIL. No claim of fresh reliability or denominator dilution.',
        continuation='If ambiguities remain or no hold-negative, no training. Only after source support and unchanged requirements are demonstrated may a separate full readiness assessment authorize broad inputs and fixed learning.',
        stop_rule='Same9 first. No arbitrary target/threshold/window change, no forced negatives or additional candidate search based on model scores.',
        preserved=['C/T3F','production web/DTW','private recordings excluded','frozen prior results','source videos'],
        no_html_without_clear_model_improvement=True,
        pins={str(p):sha(p) for p in [Path(__file__),W/'COMPLETION.json',W/'PRESENCE_PILOT_AUDIT.json',W/'review/RESULT_DECISION_REVIEW.json',U/'INPUTS.json',U/'VISUAL_INDEX.json']})
    write(N/'PLAN_FROZEN.json',plan)
    write(N/'CURRENT_EXECUTION.json',dict(status='NATIVE_RGB_EXPORT',training=0,active=True))
    inputs={r['alias']:r for r in read(U/'INPUTS.json')['rows']}
    prior={r['sample_id']:r for r in read(W/'PILOT_MANIFEST.json')['rows']}
    rows=[]
    for count,sid in enumerate(IDS,1):
        row=prior[sid]; inp=inputs[sid.split('_')[0]]
        assert row['role']==inp['role']=='fit'
        source=Path(inp['source']);assert sha(source)==inp['source_sha256']
        folder=N/'visuals'/sid;folder.mkdir(parents=True,exist_ok=False)
        wanted=set(row['frame_ids']); decoded={}; frames=[]
        with av.open(str(source)) as video:
            for i,f in enumerate(video.decode(video=0)):
                if i in wanted:
                    ts=round(float(f.pts*f.time_base)*1000000)
                    expected=row['timestamps_us'][row['frame_ids'].index(i)]
                    assert abs(ts-expected)<=1,(sid,i,ts,expected)
                    im=f.to_image().convert('RGB'); width,height=im.size
                    box=[round(width*.2),round(height*.08),round(width*.8),round(height*.95)]
                    full=folder/f'full_{i:04}.png';im.save(full)
                    crop=im.crop(box);cp=folder/f'fixed_{i:04}.png';crop.save(cp)
                    decoded[i]=crop
                    frames.append(dict(frame_id=i,timestamp_us=ts,crop_box=box,source_size=[width,height],
                        full=str(full),full_sha256=sha(full),fixed=str(cp),fixed_sha256=sha(cp)))
                if i>=max(wanted):break
        assert len(frames)==13 and [f['frame_id'] for f in frames]==row['frame_ids']
        assert len({tuple(f['crop_box']) for f in frames})==1
        pairs=[]
        for start in range(0,13,2):
            group=frames[start:start+2];cw,ch=decoded[group[0]['frame_id']].size
            sheet=Image.new('RGB',(cw*2,ch+36),'#20252b');draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(group):
                draw.text((j*cw+12,8),f"{sid}  f{f['frame_id']}  {f['timestamp_us']/1e6:.6f}s  native1:1 fixed view",fill='white')
                sheet.paste(decoded[f['frame_id']],(j*cw,36))
            p=folder/f'pair_{start//2:02}.png';sheet.save(p)
            pairs.append(dict(file=str(p),sha256=sha(p),frame_ids=[f['frame_id'] for f in group]))
        rows.append(dict(alias=f'R{count:02}',sample_id=sid,source=str(source),source_sha256=inp['source_sha256'],
            old_montage=row['image'],old_montage_sha256=row['image_sha256'],person=row['person'],word_id=row['word_id'],role='fit',
            roi_bottom=inp['box'][3],frames=frames,pairs=pairs))
        print(json.dumps(dict(done=count,total=9,sample=sid)),flush=True)
    write(N/'MANIFEST.json',dict(status='COMPLETE',windows=9,source_frames=117,rows=rows,pins=plan['pins']))
    blind=[{k:v for k,v in r.items() if k not in ['source','source_sha256','old_montage','old_montage_sha256','person','word_id','role']} for r in rows]
    write(N/'BLIND_NATIVE.json',dict(status='COMPLETE',rows=blind,rubric=plan['rubric'],rubric_sha256=plan['rubric_sha256'],
        instructions='Read actual native fixed RGB pairs, all13frames. Full native frame PNG available if crop omits anatomy. Do not read earlier answers/outcome. Same rubric, do not force hold absence. Provide presence/segments/full13 observability/ROI/unknown/events and exact visual cause. This is a selected diagnostic, not unseen validation.'))
    (N/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(status='NATIVE_EXPORT_COMPLETE_REVIEW_NEXT',active=False,training=0)),encoding='utf8')

if __name__=='__main__':main()
