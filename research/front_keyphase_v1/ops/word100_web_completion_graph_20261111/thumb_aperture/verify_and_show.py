"""Separate raw-pixel arithmetic verifies coordinate restoration and frame lineage."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw

U=Path(__file__).resolve().parent;G=U.parent;O=G.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def angle(o,a,b):
    va=a-o;vb=b-o
    return float(np.degrees(np.arccos(np.clip((va@vb)/(np.linalg.norm(va)*np.linalg.norm(vb)),-1.,1.))))

assert not (U/'VERIFICATION.json').exists()
r=read(U/'RESULT.json');raw={};records={};checks=[]
for cid in {z['clip_id'] for z in r['rows']}:
    rec=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')
    assert rec['role']=='fit' and rec['person'] in [1,4,7]
    assert sha(rec['raw_path'])==rec['raw_sha256']
    with np.load(rec['raw_path'],allow_pickle=False) as z:
        raw[cid]={k:z[k].copy() for k in ['xy_px','timestamp_us','frame_id']}
    records[cid]=rec
for q in r['rows']:
    z=raw[q['clip_id']];ix=np.flatnonzero(z['frame_id']==q['frame_id']);assert len(ix)==1
    i=int(ix[0]);assert int(z['timestamp_us'][i])==q['source_timestamp_us']
    assert records[q['clip_id']]['source_sha256']==q['source_sha256']
    start=91 if q['hand']=='L' else 112
    p=z['xy_px'][i,start:start+21].astype(np.float64)
    scale=np.median(np.sqrt(((p[[5,9,13,17]]-p[0])**2).sum(1)))
    expected=[np.sqrt(((p[4]-p[j])**2).sum())/scale for j in [8,12,16,20]]
    expected.extend([angle(p[0],p[4],p[8]),angle(p[5],p[4],p[8])])
    actual=q['thumb_to_fingertips_norm']+[q['thumb_to_index_wrist_angle_deg'],q['thumb_to_index_mcp_angle_deg']]
    err=np.abs(np.array(expected)-np.array(actual))
    assert err[:4].max()<1e-4 and err[4:].max()<.002,(q['clip_id'],q['frame_id'],err.tolist())
    checks.append(dict(clip_id=q['clip_id'],hand=q['hand'],frame_id=q['frame_id'],source_timestamp_exact=True,distance_max_error=float(err[:4].max()),angle_max_error_deg=float(err[4:].max())))
chosen={1:[37,49],4:[83,101],7:[42,56]}
sheet=Image.new('RGB',(1320,1200),'white');draw=ImageDraw.Draw(sheet)
panels=[]
for row,person in enumerate([1,4,7]):
    for col,f in enumerate(chosen[person]):
        qs=[q for q in r['rows'] if q['record_id']==person and q['frame_id']==f]
        assert len(qs)==2
        image=Image.open(qs[0]['image']).convert('RGB');d=ImageDraw.Draw(image)
        texts=[]
        for q in qs:
            z=raw[q['clip_id']];i=int(np.flatnonzero(z['frame_id']==f)[0]);start=91 if q['hand']=='L' else 112
            points=z['xy_px'][i,start:start+21];color='#ffb000' if q['hand']=='L' else '#00aaff'
            for j in [4,8,12,16,20]:
                x,y=points[j]-np.array([630,220]);d.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='black',width=1)
            x1,y1=points[4]-np.array([630,220]);x2,y2=points[8]-np.array([630,220]);d.line((x1,y1,x2,y2),fill=color,width=2)
            texts.append(f"{q['hand']}: thumb-index / palm = {q['thumb_to_fingertips_norm'][0]:.2f}, angle {q['thumb_to_index_wrist_angle_deg']:.1f}deg")
        x,y=col*660,row*400
        draw.text((x+6,y+3),f'FIT{person:02} f{f} {qs[0]["observed_state"]}',fill='black')
        draw.text((x+6,y+16),' | '.join(texts),fill='black')
        sheet.paste(image,(x,y+40));p=U/f'FIT{person:02}_f{f}_rgb_rtmw.png';image.save(p);panels.append(str(p))
sp=U/'RGB_RTMw_COMPARISON.png';sheet.save(sp)
out=dict(status='PASS',review='Separate same-Codex raw-pixel arithmetic, not independent human coordinate truth',hand_frames=len(checks),scalar_features_checked=len(checks)*6,all_source_frame_timestamp_hash_mappings_verified=True,checks=checks,result_sha256=sha(U/'RESULT.json'),raw_records=[dict(clip_id=c,raw_path=z['raw_path'],raw_sha256=z['raw_sha256']) for c,z in records.items()],comparison_sheet=str(sp),comparison_sha256=sha(sp),panels=panels,training_ran=False,dev_access=False)
(U/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],hand_frames=out['hand_frames'],scalars=out['scalar_features_checked'],max_distance_error=max(z['distance_max_error'] for z in checks),max_angle_error=max(z['angle_max_error_deg'] for z in checks)),ensure_ascii=False))
