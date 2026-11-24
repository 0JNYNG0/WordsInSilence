"""Prepare all frozen flag windows for human review; never grants semantic PASS.

Original TRAIN RGB bytes + saved native/masked/sensor observations only.
No detector, classifier, training, DEV image, or source mutation.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile
sys.dont_write_bytecode=True
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont

U=Path(__file__).resolve().parent
OUT=U/'review_all_flags_materials'
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
MODES=('RGB','native','masked','sensor')
MIN_FREE=8*1024**3
MAX_OUTPUT=6*1024**3
FONT={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write_new(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def font(n):
    if n not in FONT:FONT[n]=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    return FONT[n]

def loadobs(p):
    with np.load(p,allow_pickle=False) as z:
        d={k:z[k] for k in ('xy_px','observed','usable','timestamp_us','frame_id')}
    d['valid']=d['observed']&d['usable']&np.isfinite(d['xy_px']).all(-1)
    return d

def plan_windows(coverage):
    jobs=[]
    for row in coverage['rows']:
        for local,w in enumerate(row['merged_review_windows_with_300ms_context'],1):
            wid=f'W{len(jobs)+1:04d}'
            conditions=[]
            window_ids=set(w['frame_ids'])
            for c in row['conditions']:
                touched=sorted(window_ids&set(c['focus_frame_ids']))
                if touched:
                    conditions.append(dict(comparison=c['comparison'],side=c['side'],
                        trigger_long_absence=c['trigger_long_absence'],trigger_net_palm_drop=c['trigger_net_palm_drop'],
                        focus_frame_ids=touched,max_newly_absent_seconds=c['max_newly_absent_seconds'],net_palm5_drop_pp=c['net_palm5_drop_pp']))
            assert conditions
            jobs.append(dict(window_id=wid,batch=(len(jobs)%3)+1,clip_id=row['clip_id'],label=row['label'],
                local_window=local,window=w,conditions=conditions,priority_tier=row['priority_tier'],
                record_sha256=row['record_sha256'],source=row['source'],bottom_pixels=row['bottom_pixels']))
    assert len(jobs)==coverage['merged_review_window_count']==685
    assert len({j['window_id'] for j in jobs})==685
    return jobs

def select_frames(job,obs):
    fid=obs['native']['frame_id'];indices={int(f):i for i,f in enumerate(fid)}
    start=indices[job['window']['start_frame_id']];end=indices[job['window']['end_frame_id']]
    reasons=defaultdict(set)
    def add(i,why):
        if start<=i<=end:reasons[int(i)].add(why)
    for i in np.linspace(start,end,7).round().astype(int):add(i,'uniform_start_end_context')
    all_counts=np.stack([obs[v]['valid'][:,33:75].sum(1) for v in ('native','masked','sensor')],1)
    changes=np.abs(np.diff(all_counts,axis=0)).sum(1)
    for i in sorted(range(start,max(start,end)),key=lambda k:(-changes[k],k))[:2]:
        if changes[i]>0:
            add(i,'largest_observation_transition_before');add(i+1,'largest_observation_transition_after')
    loss=np.maximum(all_counts[:,0]-all_counts[:,1],0)+np.maximum(all_counts[:,1]-all_counts[:,2],0)
    add(start+int(np.argmax(loss[start:end+1])),'maximum_total_point_loss')
    # Each flagged comparison/side overlapping this window gets a representative
    # loss frame and the ends of its focus span. This is window coverage, not all-frame viewing.
    for c in job['conditions']:
        ii=[indices[f] for f in c['focus_frame_ids']]
        name=c['comparison']+'/'+c['side']
        add(ii[0],name+'_first_loss');add(ii[-1],name+'_last_loss')
        base=33 if c['side']=='left' else 54
        old,new=('native','masked') if c['comparison']=='native_to_mask' else ('masked','sensor')
        delta=obs[old]['valid'][:,base:base+21].sum(1)-obs[new]['valid'][:,base:base+21].sum(1)
        add(max(ii,key=lambda k:(delta[k],-k)),name+'_maximum_side_loss')
    return sorted(reasons),{str(int(fid[i])):sorted(reasons[i]) for i in sorted(reasons)}

def source_bytes(cid,row,rec):
    candidates=[Path(row['retained_video'])] if row.get('retained_video') else []
    candidates += [U/'review_sources'/f'{cid}.mp4',U/'actual_geometry_review_sources'/f'{cid}.mp4']
    s=rec['source']
    for p in candidates:
        if p.is_file():
            data=p.read_bytes()
            assert len(data)==s['bytes'] and hashlib.sha256(data).hexdigest()==s['sha256']
            return data,{'method':'existing_exact_source_cache','path':str(p)}
    with zipfile.ZipFile(s['archive']) as z:
        info=z.getinfo(s['member']);assert info.CRC==s['crc32'] and info.file_size==s['bytes']
        data=z.read(info)
    assert len(data)==s['bytes'] and hashlib.sha256(data).hexdigest()==s['sha256']
    return data,{'method':'frozen_train_zip_member_in_memory','archive':s['archive'],'member':s['member']}

def decode(data,selected,ts):
    frames={};pts={};origin=None
    with av.open(io.BytesIO(data)) as video:
        for i,f in enumerate(video.decode(video=0)):
            assert f.pts is not None,'Missing source PTS'
            absolute=float(f.pts*f.time_base)*1e6
            if origin is None:origin=absolute
            relative=round(absolute-origin)
            assert i<len(ts) and abs(relative-int(ts[i]))<=1,('Source PTS mismatch',i,relative,int(ts[i]))
            if i in selected:
                frames[i]=f.to_image().convert('RGB')
                pts[i]=dict(source_absolute_pts_us=round(absolute),source_relative_pts_us=relative,observation_timestamp_us=int(ts[i]))
            if i>=max(selected):break
    assert set(frames)==set(selected)
    return frames,pts,origin

def overlay(rgb,obs,i,bottom):
    im=rgb.copy();d=ImageDraw.Draw(im)
    # Full original image remains visible for review. Sensor's excluded region
    # is NOT a model input; the boundary line marks its actual observation limit.
    d.line((0,bottom,im.width,bottom),fill=(255,224,0),width=5)
    if obs is not None:
        xy=obs['xy_px'][i];good=obs['valid'][i]
        for base,body,color in ((33,(11,13,15),(50,255,90)),(54,(12,14,16),(0,205,255))):
            for a,b in zip(body,body[1:]):
                if good[a] and good[b]:d.line(tuple(xy[a])+tuple(xy[b]),fill=color,width=4)
            for a,b in EDGES:
                a+=base;b+=base
                if good[a] and good[b]:d.line(tuple(xy[a])+tuple(xy[b]),fill=color,width=3)
            for p in range(base,base+21):
                if good[p]:
                    x,y=xy[p];d.ellipse((x-5,y-5,x+5,y+5),fill=color)
    return im

def zoom_box(job,obs,i,size):
    w,h=size;points=[]
    for side in {c['side'] for c in job['conditions']}:
        base,body=(33,15) if side=='left' else (54,16)
        for mode in ('native','masked','sensor'):
            o=obs[mode];good=o['valid'][i]
            points.extend(o['xy_px'][i,base:base+21][good[base:base+21]].tolist())
            if good[body]:points.append(o['xy_px'][i,body].tolist())
    if points:
        p=np.asarray(points);x0,y0=p.min(0)-70;x1,y1=p.max(0)+70
        cx,cy=(x0+x1)/2,(y0+y1)/2
        bw=max(x1-x0,240);bh=max(y1-y0,200)
        # Fixed aspect ratio avoids distortion of hand geometry.
        bw=max(bw,bh*16/10);bh=bw*10/16
        x0,x1=cx-bw/2,cx+bw/2;y0,y1=cy-bh/2,cy+bh/2
    else:x0,y0,x1,y1=w*.25,h*.15,w*.75,h*.9
    return tuple(int(v) for v in (max(0,x0),max(0,y0),min(w,x1),min(h,y1)))

def letterbox(im,width,height):
    im=im.copy();im.thumbnail((width,height),Image.Resampling.LANCZOS)
    out=Image.new('RGB',(width,height),(8,12,18));out.paste(im,((width-im.width)//2,(height-im.height)//2));return out

def render_page(job,indices,obs,frames,pts,page,total):
    tw,fh,zh=320,180,200;rh=425;header=110
    image=Image.new('RGB',(tw*4,header+rh*len(indices)),(14,21,31));d=ImageDraw.Draw(image)
    d.text((10,5),f"{job['window_id']}  묶음{job['batch']}  {job['label']}  {job['clip_id']}  [{page}/{total}]",font=font(22),fill='white')
    d.text((10,36),f"구간 f{job['window']['start_frame_id']}–{job['window']['end_frame_id']} / 하단 {job['bottom_pixels']}px / "+', '.join(c['comparison']+':'+c['side'] for c in job['conditions']),font=font(16),fill=(215,225,237))
    d.text((10,61),'원본 전체 | 전체검출 native | 범위만 masked | 실제 잘린 입력 sensor   / 초록=L 하늘=R 노랑=관측 하단',font=font(16),fill=(215,225,237))
    d.text((10,84),'각 칸 위=전체 화면, 아래=손 주변 확대. sensor의 선 아래 RGB는 검토용 문맥이며 모델에는 제공되지 않음.',font=font(15),fill=(255,219,130))
    metadata=[]
    for row_i,i in enumerate(indices):
        y=header+row_i*rh;box=zoom_box(job,obs,i,frames[i].size)
        counts={v:{s:int(obs[v]['valid'][i,b:b+21].sum()) for s,b in [('L',33),('R',54)]} for v in ('native','masked','sensor')}
        for col,mode in enumerate(MODES):
            im=overlay(frames[i],None if mode=='RGB' else obs[mode],i,job['bottom_pixels'])
            x=col*tw
            d.text((x+4,y+1),f"{mode}  f{int(obs['native']['frame_id'][i])} {pts[i]['source_relative_pts_us']/1e6:.3f}s"+(f" L{counts[mode]['L']} R{counts[mode]['R']}" if mode!='RGB' else ''),font=font(14),fill='white')
            image.paste(letterbox(im,tw,fh),(x,y+24));image.paste(letterbox(im.crop(box),tw,zh),(x,y+fh+27))
        metadata.append(dict(frame_id=int(obs['native']['frame_id'][i]),**pts[i],valid_hand_counts=counts,zoom_box_pixels=list(box)))
    return image,metadata

def selftest():
    ts=np.arange(20,dtype=np.int64)*33333;fid=np.arange(20)
    valid=np.ones((20,75),bool);a={'valid':valid.copy(),'frame_id':fid,'timestamp_us':ts}
    b={'valid':valid.copy(),'frame_id':fid,'timestamp_us':ts};b['valid'][5:10,33:54]=False
    obs={'native':a,'masked':b,'sensor':b}
    job={'window':{'start_frame_id':2,'end_frame_id':15},'conditions':[{'comparison':'native_to_mask','side':'left','focus_frame_ids':list(range(5,10))}]}
    selected,reasons=select_frames(job,obs)
    assert 2 in selected and 15 in selected and 5 in selected and 9 in selected
    assert any('largest_observation_transition' in x for vals in reasons.values() for x in vals)
    assert all(2<=i<=15 for i in selected) and len(selected)==len(set(selected))
    assert min(job['conditions'][0]['focus_frame_ids']) in selected
    print('PASS five pure sampling assertions; no source RGB read')

def main(limit=None):
    start=time.perf_counter();OUT.mkdir(exist_ok=True)
    context_path=U/'PREP_CONTEXT.json';coverage_path=U/'review_flag_coverage_RESULT.json'
    context=read(context_path);coverage=read(coverage_path);jobs=plan_windows(coverage)
    context_sha=sha(context_path);coverage_sha=sha(coverage_path)
    assert not (U/'PREP_RUNNING.json').exists() and coverage['training_allowed'] is False
    rows={r['clip_id']:r for r in context['ordered_rows'] if r['role']=='train_candidate'}
    assert len(rows)==899
    expected={j['window_id']:j for j in jobs};byclip=defaultdict(list)
    for j in jobs:byclip[j['clip_id']].append(j)
    free=shutil.disk_usage(U).free;assert free>=MIN_FREE,'Insufficient free disk budget'
    runpins={'context':context_sha,'coverage':coverage_sha,'generator':sha(__file__)}
    generated=[];reused=0;clips=0;output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    for cid,clipjobs in byclip.items():
        if limit is not None and clips>=limit:break
        pending=[]
        for job in clipjobs:
            p=OUT/f"{job['window_id']}.json"
            if p.exists():
                old=read(p);assert old['pins']==runpins and old['job']==job
                assert all(sha(img['path'])==img['sha256'] for img in old['images'])
                generated.append(old);reused+=1
            else:pending.append(job)
        if not pending:continue
        assert shutil.disk_usage(U).free>=MIN_FREE and output_bytes<MAX_OUTPUT
        row=rows[cid];rp=U/'data/records'/f'{cid}.json';rec=read(rp)
        assert rec['role']=='train_candidate' and 1<=rec['signer_id']<=9
        assert sha(rp)==pending[0]['record_sha256']
        obs={};opins={}
        for view in ('native','masked','sensor'):
            p=Path(rec['views'][view]['observation']);assert sha(p)==rec['outputs'][str(p)]
            opins[str(p)]=rec['outputs'][str(p)];obs[view]=loadobs(p)
        ts=obs['native']['timestamp_us'];fid=obs['native']['frame_id']
        assert np.array_equal(fid,np.arange(len(fid)))
        for v in obs:assert np.array_equal(ts,obs[v]['timestamp_us']) and np.array_equal(fid,obs[v]['frame_id'])
        selections={j['window_id']:select_frames(j,obs) for j in pending}
        all_selected=set(i for selection,_ in selections.values() for i in selection)
        data,method=source_bytes(cid,row,rec);frames,pts,origin=decode(data,all_selected,ts);del data
        for job in pending:
            selection,reasons=selections[job['window_id']];images=[];shown=[]
            pages=(len(selection)+3)//4
            for page,at in enumerate(range(0,len(selection),4),1):
                image,meta=render_page(job,selection[at:at+4],obs,frames,pts,page,pages)
                p=OUT/f"{job['window_id']}_p{page:02d}.jpg";assert not p.exists(),'Uncommitted partial output preserved; inspect first failure'
                image.save(p,quality=90,subsampling=0);output_bytes+=p.stat().st_size
                assert output_bytes<MAX_OUTPUT,'Output disk budget exceeded'
                images.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,displayed_frame_ids=[r['frame_id'] for r in meta]));shown+=meta
            result=dict(status='MATERIAL_CREATED_PENDING_HUMAN_REVIEW',job=job,pins=runpins,
                record_path=str(rp),record_sha256=sha(rp),observation_pins=opins,
                source_sha256=rec['source']['sha256'],source_read=method,source_absolute_pts_origin_us=origin,
                PTS_relative_alignment_max_error_us=max(abs(pts[i]['source_relative_pts_us']-int(ts[i])) for i in selection),
                selected_frames=shown,selection_reasons=reasons,images=images,
                all_window_frames_viewed=False,semantic_pass=False,training_allowed=False,
                next_if_ambiguous='Request additional frame IDs within this exact window using review_all_flags_extra.py, then record actual viewed frames and a separate disposition.')
            write_new(OUT/f"{job['window_id']}.json",result);generated.append(result)
        clips+=1
        print(json.dumps(dict(stage='REVIEW_MATERIAL_CLIP',clip_id=cid,windows_done=len(generated),total_windows=685,output_megabytes=round(output_bytes/1024**2,1)),ensure_ascii=False),flush=True)
    assert sha(context_path)==context_sha and sha(coverage_path)==coverage_sha
    if limit is not None:
        print(json.dumps(dict(status='BOUNDED_MATERIAL_PREPARATION',windows=len(generated),new_clips=clips,all_window_approval=False),ensure_ascii=False));return
    final=[read(OUT/f'{wid}.json') for wid in expected]
    assert {r['job']['window_id'] for r in final}==set(expected)
    batches=[]
    for batch in (1,2,3):
        entries=[r for r in final if r['job']['batch']==batch]
        ledger=dict(status='TEMPLATE_PENDING_HUMAN_REVIEW',batch=batch,coverage_sha256=coverage_sha,training_allowed=False,
            windows=[dict(window_id=r['job']['window_id'],clip_id=r['job']['clip_id'],record_sha256=r['record_sha256'],
                material_path=str(OUT/f"{r['job']['window_id']}.json"),material_sha256=sha(OUT/f"{r['job']['window_id']}.json"),
                images=r['images'],review_status='PENDING',actual_viewed_frame_ids=[],
                interval_dispositions=[dict(comparison=c['comparison'],side=c['side'],focus_frame_ids=c['focus_frame_ids'],
                    observation='',interpretation='',unresolved='',disposition='PENDING') for c in r['job']['conditions']],
                extra_frames_requested=[],reviewer=None,reviewed_at_utc=None) for r in entries])
        lp=OUT/f'batch_{batch}_ledger_TEMPLATE.json';write_new(lp,ledger)
        hp=OUT/f'batch_{batch}.html';assert not hp.exists()
        contents=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>전체 표시구간 검토</title><style>body{font:16px sans-serif;background:#111827;color:#eee;margin:24px}article{margin:24px 0;padding:14px;border:1px solid #475569}img{display:block;width:100%;max-width:1280px;margin:8px 0}a{color:#93c5fd}summary{cursor:pointer;font-size:20px}small{color:#cbd5e1}</style>',f'<h1>표시구간 검토 묶음 {batch} / {len(entries)}구간</h1><p>자료 생성은 검토 완료가 아닙니다. 모든 구간을 확인하고 별도 ledger 사본에 실제 관찰과 미해결 사항을 기록하세요.</p>']
        for r in entries:
            j=r['job'];contents.append(f"<article id='{j['window_id']}'><details><summary>{j['window_id']} · {html.escape(j['label'])} · {j['clip_id']} · f{j['window']['start_frame_id']}–{j['window']['end_frame_id']}</summary>")
            contents.append('<p>'+html.escape(', '.join(c['comparison']+' / '+c['side'] for c in j['conditions']))+'</p>')
            for image in r['images']:contents.append(f"<a href='{Path(image['path']).name}'><img loading='lazy' src='{Path(image['path']).name}'></a>")
            contents.append(f"<small>기록 <a href='{j['window_id']}.json'>프레임·PTS·핀·선택 근거</a> / 애매한 경우 추가 프레임 요청</small></details></article>")
        hp.write_text('\n'.join(contents),encoding='utf-8')
        batches.append(dict(batch=batch,windows=len(entries),clips=len({r['job']['clip_id'] for r in entries}),images=sum(len(r['images']) for r in entries),html=str(hp),ledger_template=str(lp)))
    report=dict(status='ALL_685_WINDOWS_MATERIALIZED_PENDING_HUMAN_REVIEW',created_utc=datetime.now(timezone.utc).isoformat(),
        pins=runpins,windows=685,clips=len(byclip),batches=batches,total_images=sum(len(r['images']) for r in final),
        displayed_frames=sum(len(r['selected_frames']) for r in final),output_bytes=output_bytes,
        initial_free_bytes=free,final_free_bytes=shutil.disk_usage(U).free,seconds=time.perf_counter()-start,
        detector_runs=0,classifier_runs=0,training_runs=0,development_videos=0,semantic_pass=False,training_allowed=False,
        manual_work='685 windows divided 229/228/228. Review time is not measured; 15–45 seconds/window would be about 2.9–8.6 person-hours, plus ambiguous follow-up. Parallel reviewers reduce elapsed time, not total work.')
    write_new(U/'review_all_flags_RESULT.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--limit-clips',type=int)
    args=parser.parse_args()
    try:
        if args.self_test:selftest()
        else:main(args.limit_clips)
    except Exception as e:
        p=U/'review_all_flags_FIRST_ERROR.json'
        if not p.exists():write_new(p,dict(at_utc=datetime.now(timezone.utc).isoformat(),error=repr(e),traceback=traceback.format_exc(),generator_sha256=sha(__file__)))
        raise
