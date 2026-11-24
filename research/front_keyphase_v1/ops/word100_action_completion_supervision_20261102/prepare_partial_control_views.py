"""Prepare auditable native-frame views, not labels or inference inputs.

All image overlays are review-only. Future model inputs must be decoded from
actual transformed original video, never from these sheets or cached features.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, statistics, traceback, os
from PIL import Image, ImageDraw, ImageFont

U=Path(__file__).resolve().parent
OUT=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/partial_views')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def key(d): return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def fragments(ids):
    out=[]
    for f in ids:
        if not out or out[-1][1]+1!=f: out.append([f,f])
        else: out[-1][1]=f
    return out

def main():
    assert not (U/'PARTIAL_VIEWS_MANIFEST.json').exists()
    assert not (U/'PARTIAL_VIEWS_INTENT.json').exists(), 'Inspect interrupted preparation before resuming'
    con=read(U/'PARTIAL_CONTROL_CONTRACT.json')
    for p in con['inputs']: assert sha(p['path'])==p['sha256']
    targets=read(U/'PROCESS_TARGETS.json')['rows']
    review={r['id']:r for r in read(U/'PROCESS_REVIEW_MANIFEST.json')['rows']}
    eligible=[]; skipped=[]
    for r in targets:
        assert 1<=r['person']<=9 and r['role']=='fit' and len(r['sequence_links'])==1
        seq=r['sequence_links'][0];lo,hi=r['review_bounds']
        a=max(seq['setup_frames']) if seq['setup_frames'] else None
        b=min(seq['terminal_frames']) if seq['terminal_frames'] else None
        ok=r['review_bounds_sufficient'] and seq['visible_chain']=='yes' and a is not None and b is not None and lo<=a<b<=hi and b-a-1>=1
        if ok: eligible.append(r)
        else: skipped.append(dict(query_id=r['id'],original_chain=seq,reason='No supported non-overlapping anchors; preserve native partial/unknown/variant.'))
    assert len(targets)==45 and len(eligible)==44 and len(skipped)==1
    sentinels=[]
    for q in sorted({r['physical_query'] for r in eligible}):
        for person in [1,9]:
            xs=sorted([r for r in eligible if r['physical_query']==q and r['person']==person],key=lambda r:r['id'])
            assert xs;sentinels.append(xs[0]['id'])
    views=[];media={};validated=set()
    def add(r,kind,ids,speed=1.,blank=()):
        meta=review[r['id']];fs={f['frame']:f for f in meta['frames']}
        lo,hi=r['review_bounds'];assert set(fs)==set(range(lo,hi+1))
        assert ids and ids==sorted(set(ids)) and all(f in fs for f in ids)
        period=statistics.median([fs[f+1]['timestamp_us']-fs[f]['timestamp_us'] for f in range(lo,hi)])
        mapping=[];t=0.;seams=[]
        for j,f in enumerate(ids):
            src=fs[f]
            if src['path'] not in validated:
                assert sha(src['path'])==src['sha256'];validated.add(src['path'])
            if j:
                adjacent=f==ids[j-1]+1
                t+=(src['timestamp_us']-fs[ids[j-1]]['timestamp_us'] if adjacent else period)/speed
                if not adjacent:seams.append(j)
            mapping.append(dict(output_index=j,source_native_frame=f,source_native_timestamp_us=src['timestamp_us'],
                source_frame_path=src['path'],source_frame_sha256=src['sha256'],view_timestamp_us=round(t),
                pixel_operation='blackout' if f in blank else 'copy',image_available=f not in blank))
        digest=key(dict(source_sha256=r['source_sha256'],frames=[(m['source_native_frame'],m['pixel_operation'],m['view_timestamp_us']) for m in mapping]))
        accounting=[];kept=set(ids)
        for m in r['motion_intervals']:
            whole=set(range(m['start'],m['end']+1));ret=whole&kept
            accounting.append(dict(original_interval=[m['start'],m['end']],original_label_audit_only=m['label'],
                retained_native_fragments=fragments(sorted(ret)),removed_native_spans=fragments(sorted(whole-ret)),
                retention='fully_retained' if ret==whole else ('wholly_removed' if not ret else 'partially_retained'),
                remainder_label=None,blackout_frames=sorted(ret&set(blank))))
        seq=r['sequence_links'][0]
        entry=dict(view_id=r['id']+'__'+kind,query_id=r['id'],clip_id=r['clip_id'],recorded_person_id=r['person'],role='fit',
            physical_query=r['physical_query'],source_path=r['source'],source_sha256=r['source_sha256'],
            original_review_path=r['source_annotation'],original_review_sha256=r['source_annotation_sha256'],sequence_link_index=0,
            reviewed_bounds=[lo,hi],a=max(seq['setup_frames']) if seq['setup_frames'] else None,b=min(seq['terminal_frames']) if seq['terminal_frames'] else None,
            construction_kind_audit_only=kind,pixel_view_key=digest,
            query_review_key=key([digest,r['physical_query'],key(r['predicate_definitions'])]),ordered_frame_map=mapping,
            retained_contiguous_native_fragments=fragments(ids),removed_native_spans=fragments(sorted(set(fs)-kept)),
            seam_output_indices=seams,motion_retention_audit_only=accounting,labels_status='pending_direct_review',labels=None,
            original_scope_variant=r.get('actual_terminal_variant'),speed=speed,
            actual_retimed_video_prepared=False,model_input_prepared=False,review_overlay_not_model_input=True)
        views.append(entry)
        if digest not in media:media[digest]=entry
    elig_ids={r['id'] for r in eligible}
    for r in targets:
        lo,hi=r['review_bounds'];seq=r['sequence_links'][0];add(r,'native',list(range(lo,hi+1)))
        if r['id'] not in elig_ids:continue
        a=max(seq['setup_frames']);b=min(seq['terminal_frames']);left=list(range(lo,a+1));right=list(range(b,hi+1))
        add(r,'prefix',left);add(r,'suffix',right);add(r,'middle_splice',left+right)
        if r['id'] in sentinels:
            add(r,'speed_0_8',list(range(lo,hi+1)),.8);add(r,'speed_1_25',list(range(lo,hi+1)),1.25)
            anchors=seq['setup_frames']+seq['motion_interval']+seq['terminal_frames']
            add(r,'intact_context_trim',list(range(min(anchors),max(anchors)+1)))
            ms=sorted([m for m in r['motion_intervals'] if m['label']==1 and seq['motion_interval'][0]<=m['start']<m['end']<=seq['motion_interval'][1]],key=lambda m:(m['start'],m['end']))
            assert ms,'No positive motion support; do not invent blackout span'
            add(r,'missing_rgb',list(range(lo,hi+1)),blank=range(ms[0]['start'],ms[0]['end']+1))
    assert len(views)==217 and len(sentinels)==10
    pins={str(U/n):sha(U/n) for n in ['PARTIAL_CONTROL_CONTRACT.json','PROCESS_TARGETS.json','PROCESS_REVIEW_MANIFEST.json','PROCESS_INPUT_BINDING.json','prepare_partial_control_views.py']}
    write(U/'PARTIAL_VIEWS_INTENT.json',dict(at=now(),status='FROZEN_REVIEW_VIEWS_NO_TARGET_LABELS',pins=pins,views=views,
        skipped=skipped,sentinels=sentinels,new_inference=0,new_training=0))
    OUT.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
    pages={};counts=Counter()
    for i,(digest,v) in enumerate(media.items()):
        maps=v['ordered_frame_map'];page_list=[]
        # All retained frames are shown; sheets never stand in for speed playback.
        for j in range(0,len(maps),25):
            part=maps[j:j+25];cw,ch=270,212
            sheet=Image.new('RGB',(cw*5,76+ch*((len(part)+4)//5)),'#f5f6f8');d=ImageDraw.Draw(sheet)
            d.text((8,5),v['clip_id']+' | shared pixel/time view '+digest[:12],font=font,fill='#111827')
            d.text((8,28),'FULL original frame | ALL retained frames | source f / new t | NOT labels',font=font,fill='#111827')
            for k,m in enumerate(part):
                x=(k%5)*cw+4;y=76+(k//5)*ch
                if m['image_available']:
                    with Image.open(m['source_frame_path']) as src:
                        im=src.copy();im.thumbnail((cw-8,ch-31))
                else:im=Image.new('RGB',(cw-8,ch-31),'#808080')
                sheet.paste(im,(x,y+29));d.text((x,y),f"f{m['source_native_frame']} / {m['view_timestamp_us']/1e6:.3f}s",font=font,fill='#111827')
                if m['output_index'] in v['seam_output_indices']:d.rectangle((x,y,x+cw-9,y+ch-4),outline='#d92d20',width=4)
            dest=OUT/(digest[:20]+f'_p{j//25+1}.jpg');sheet.save(dest,quality=92)
            page_list.append(dict(path=str(dest),sha256=sha(dest),output_indices=[m['output_index'] for m in part],source_native_frames=[m['source_native_frame'] for m in part]))
        pages[digest]=page_list
        if i%20==0:
            write(U/'PARTIAL_VIEWS_PROGRESS.json',dict(at=now(),status='RENDERING_UNLABELLED_REVIEW_SHEETS',pid=os.getpid(),done=i+1,total=len(media)))
            print('review views',i+1,'/',len(media),flush=True)
    for v in views:
        v['review_pages']=pages[v['pixel_view_key']];v['native_comparison_pages']=review[v['query_id']]['pages'];counts[v['construction_kind_audit_only']]+=1
    write(U/'PARTIAL_VIEWS_MANIFEST.json',dict(at=now(),status='PREPARED_PENDING_DIRECT_VIEW_REVIEW',views=views,skipped=skipped,
        counts=dict(counts),queries=45,source_clips=36,unique_pixel_time_views=len(media),pages=sum(map(len,pages.values())),
        intent_sha256=sha(U/'PARTIAL_VIEWS_INTENT.json'),pins=pins,new_inference=0,new_training=0,
        actual_transformed_videos_prepared=0,whole_goal_complete=False))
    write(U/'PARTIAL_VIEWS_PROGRESS.json',dict(at=now(),status='PREPARED_PENDING_DIRECT_VIEW_REVIEW',active=None,done=len(media),total=len(media)))
    print('Prepared',dict(counts),'unique',len(media),'no labels or inference',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'PARTIAL_VIEWS_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
