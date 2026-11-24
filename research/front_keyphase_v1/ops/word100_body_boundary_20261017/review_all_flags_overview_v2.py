"""Append immutable compact overview pages from completed per-window sheets.

Five windows per page; each row has original start/peak/end and peak native,
masked and sensor hand details. This is a review aid, never automatic approval.
"""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_all_flags_build as b
from PIL import Image,ImageDraw

def tile(material,frame,col,kind):
    match=next(x for x in material['images'] if frame in x['displayed_frame_ids'])
    assert b.sha(match['path'])==match['sha256']
    row=match['displayed_frame_ids'].index(frame)
    y=110+425*row+(24 if kind=='full' else 207)
    with Image.open(match['path']) as im:return im.crop((320*col,y,320*(col+1),y+(180 if kind=='full' else 200)))

def build():
    coverage=b.read(b.U/'review_flag_coverage_RESULT.json');jobs=b.plan_windows(coverage)
    outputs=[]
    for batch in (1,2,3):
        expected=[j for j in jobs if j['batch']==batch]
        for at in range(0,len(expected),5):
            group=expected[at:at+5];number=at//5+1
            paths=[b.OUT/(j['window_id']+'.json') for j in group]
            if not all(p.exists() for p in paths):continue
            materials=[]
            try:materials=[b.read(p) for p in paths]
            except json.JSONDecodeError:continue # Writer has not committed this window yet.
            assert all(m['job']==j for m,j in zip(materials,group))
            metadata=b.OUT/f'batch_{batch}_overview_v2_{number:03d}.json'
            image=b.OUT/f'batch_{batch}_overview_v2_{number:03d}.jpg'
            if metadata.exists():
                r=b.read(metadata);assert b.sha(image)==r['image_sha256']
                assert all(b.sha(p)==h for p,h in r['material_pins'].items())
                outputs.append(str(metadata));continue
            assert not image.exists(),'Uncommitted overview image preserved'
            canvas=Image.new('RGB',(1920,70+280*len(group)),(14,21,31));d=ImageDraw.Draw(canvas)
            d.text((10,5),f'묶음 {batch} / 개요 {number:03d} / 원본 시작 · 최대손실 원본 · native 확대 · masked 확대 · sensor 확대 · 원본 끝',font=b.font(24),fill='white')
            d.text((10,39),'NM=전체검출→범위, MS=범위→실제 잘린 검출. 작거나 애매하면 개별 W번호 상세 페이지/추가 프레임 필수. 생성≠검토완료.',font=b.font(21),fill=(255,219,130))
            rows=[]
            for row,m in enumerate(materials):
                j=m['job'];selected=m['selected_frames'];start=selected[0]['frame_id'];end=selected[-1]['frame_id']
                def loss(f):
                    counts=f['valid_hand_counts'];scores=[]
                    for condition in j['conditions']:
                        if f['frame_id'] not in condition['focus_frame_ids']:continue
                        side='L' if condition['side']=='left' else 'R'
                        old,new=('native','masked') if condition['comparison']=='native_to_mask' else ('masked','sensor')
                        scores.append(max(counts[old][side]-counts[new][side],0))
                    return max(scores,default=-1)
                peak=max(selected,key=lambda f:(loss(f),-f['frame_id']))['frame_id']
                y=70+280*row
                cond=', '.join(('NM' if c['comparison']=='native_to_mask' else 'MS')+':'+c['side'] for c in j['conditions'])
                d.text((8,y+1),f"{j['window_id']}  {j['label']}  {j['clip_id']}  f{j['window']['start_frame_id']}–{j['window']['end_frame_id']}  {cond}",font=b.font(22),fill='white')
                specs=[(start,0,'full','RGB 시작'),(peak,0,'full','RGB 최대손실'),(peak,1,'zoom','native 확대'),(peak,2,'zoom','masked 확대'),(peak,3,'zoom','sensor 확대'),(end,0,'full','RGB 끝')]
                for col,(frame,sourcecol,kind,label) in enumerate(specs):
                    stamp=next(f for f in selected if f['frame_id']==frame)
                    canvas.paste(b.letterbox(tile(m,frame,sourcecol,kind),320,200),(col*320,y+33))
                    d.text((col*320+5,y+238),f'{label} f{frame} {stamp["source_relative_pts_us"]/1e6:.3f}s',font=b.font(18),fill=(215,225,237))
                rows.append(dict(window_id=j['window_id'],clip_id=j['clip_id'],record_sha256=m['record_sha256'],
                    overview_displayed_frame_ids=sorted(set([start,peak,end])),
                    actual_overlay_compared_frame_ids=[peak],
                    source_only_context_frames=sorted(set([start,end])),
                    detail_material=str(b.OUT/(j['window_id']+'.json')),
                    detail_images=m['images'],review_status='PENDING',semantic_pass=False))
            canvas.save(image,quality=92,subsampling=0)
            result=dict(status='COMPACT_OVERVIEW_CREATED_PENDING_ACTUAL_VIEW',batch=batch,page=number,
                image=str(image),image_sha256=b.sha(image),material_pins={str(p):b.sha(p) for p in paths},
                rows=rows,training_allowed=False,semantic_pass=False,
                only_these_three_frames_shown=True,all_window_frames_reviewed=False,
                peak_selection='Maximum point-count loss on the actually flagged side/comparison among selected focus frames; earlier overview v1 preserved and not used for review.',
                note='Each frozen window has its own detailed source/native/masked/sensor temporal sheet. An overview row is not a substitute where motion or support-hand interpretation is unclear.',
                generator_sha256=b.sha(__file__))
            b.write_new(metadata,result);outputs.append(str(metadata))
            print(json.dumps(dict(batch=batch,page=number,windows=[r['window_id'] for r in rows],image=str(image)),ensure_ascii=False),flush=True)
    print(json.dumps(dict(compact_metadata_files=len(outputs),files=outputs),ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');build()
