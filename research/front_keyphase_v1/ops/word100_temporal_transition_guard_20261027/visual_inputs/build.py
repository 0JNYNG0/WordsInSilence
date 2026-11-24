"""Frozen13-frame RGB windows for direct/temporal evidence review, no inference."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,random,traceback
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;OPS=U.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(n,d):(U/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sheet(images,order,labels):
    out=Image.new('RGB',(1920,1600),(243,245,248));draw=ImageDraw.Draw(out);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    for j,idx in enumerate(order):
        im=images[idx].copy();im.thumbnail((472,360),Image.Resampling.LANCZOS)
        x=j%4*480+(480-im.width)//2;y=j//4*400+36
        out.paste(im,(x,y));draw.text((j%4*480+10,j//4*400+5),labels[idx],font=font,fill=(20,30,45))
    return out
def focus_box(xy,valid,selected,side,w,h):
    ids=np.r_[np.arange(33+side*21,54+side*21),13+side,15+side]
    p=xy[selected][:,ids];ok=valid[selected][:,ids]&np.isfinite(p).all(-1);pp=p[ok]
    if not len(pp):return (0,0,w,h),'no_observed_target_points_fullframe_fallback'
    low=pp.min(0)-90;high=pp.max(0)+90;center=(low+high)/2;size=np.maximum(high-low,[360,360])
    lo=np.maximum(0,np.floor(center-size/2)).astype(int);hi=np.minimum([w,h],np.ceil(center+size/2)).astype(int)
    return (int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])),'target_hand_and_arm_union_90px_margin_min360'

def main():
    if (U/'INDEX.json').exists():raise FileExistsError('Completed frozen export is reused')
    fp=U.parent/'FROZEN.json';f=read(fp);rows=f['rows'];assert len(rows)==18 and f['dense_offsets_frames']==list(range(-6,7))
    pins={str(fp):sha(fp),str(U/'build.py'):sha(U/'build.py')}
    for p,h in f['pins'].items():assert sha(p)==h;pins[p]=h
    shared=read(OPS/'word100_shared_feature_bundle_20261027/FROZEN.json');records={r['clip_id']:r for r in shared['rows']}
    groups=defaultdict(list)
    for r in rows:
        p=Path(r['source']);assert p.exists(),('Frozen source missing; do not substitute',str(p))
        assert sha(p)==r['source_sha256'];pins[str(p)]=r['source_sha256'];groups[r['clip_id']].append(r)
    done=[];outputs={};source_metadata={};maxpts=0
    for clip,case_rows in groups.items():
        r0=case_rows[0];sr=records[clip];rp=Path(sr['record']);assert sha(rp)==sr['record_sha256'];record=read(rp)
        assert record['source']['sha256']==r0['source_sha256'];op=Path(record['views']['native']['observation']);sp=op.with_suffix('.sidecar.json');sc=read(sp)
        assert sha(op)==sc['observation_sha256'] and sc['source_sha256']==r0['source_sha256']
        for p in (rp,op,sp):pins[str(p)]=sha(p)
        with np.load(op,allow_pickle=False) as z:fid=z['frame_id'].copy();ts=z['timestamp_us'].copy();xy=z['xy_px'].copy();valid=z['usable'].copy()
        assert np.array_equal(fid,np.arange(len(fid)))
        needed=set()
        for case in case_rows:
            chosen=[case['center_frame_id']+d for d in f['dense_offsets_frames']]
            assert min(chosen)>=0 and max(chosen)<len(fid)
            assert abs(int(ts[case['center_frame_id']])-case['center_pts_us'])<=1
            needed.update(chosen)
        images={};pixel_hashes={};actual=[];first=None
        with av.open(r0['source']) as c:
            for n,frame in enumerate(c.decode(c.streams.video[0])):
                if first is None:first=frame.pts
                t=int(round(float((frame.pts-first)*frame.time_base)*1e6));actual.append(t)
                if n in needed:
                    rgb=frame.to_ndarray(format='rgb24');images[n]=Image.fromarray(rgb);pixel_hashes[n]=hashlib.sha256(rgb.tobytes()).hexdigest()
        assert len(actual)==len(fid);err=int(np.max(np.abs(np.asarray(actual)-ts)));assert err<=1;maxpts=max(maxpts,err)
        source_metadata[clip]={'source':r0['source'],'source_sha256':r0['source_sha256'],'frames':len(actual),'decoded_unique_selected_frames':len(needed),'pts_max_error_us':err}
        for case in case_rows:
            cid=case['case_id'];folder=U/cid;folder.mkdir(exist_ok=True);chosen=[case['center_frame_id']+d for d in f['dense_offsets_frames']]
            seedhash=hashlib.sha256((pins[str(fp)]+'|'+cid+'|fixed-neutral-v1').encode()).hexdigest();rng=random.Random(int(seedhash[:16],16))
            codes=list(range(1,14));rng.shuffle(codes);labels=[f'K{x:02d}' for x in codes]
            shuffled=list(range(13));rng.shuffle(shuffled);assert shuffled!=list(range(13))
            side={'L':0,'R':1}[case['side']];w,h=images[chosen[0]].size;crop,policy=focus_box(xy,valid,chosen,side,w,h)
            fulls=[];crops=[];meta=[]
            for j,frameid in enumerate(chosen):
                label=labels[j];full=images[frameid];detail=full.crop(crop);pf=folder/(label+'_full.png');pc=folder/(label+'_focus.png');full.save(pf);detail.save(pc)
                with Image.open(pf) as check:assert hashlib.sha256(np.asarray(check.convert('RGB')).tobytes()).hexdigest()==pixel_hashes[frameid]
                with Image.open(pc) as check:assert np.array_equal(np.asarray(check),np.asarray(full)[crop[1]:crop[3],crop[0]:crop[2]])
                outputs[str(pf)]=sha(pf);outputs[str(pc)]=sha(pc);fulls.append(full);crops.append(detail)
                meta.append({'neutral_image_id':label,'image_key':label,'frame_id':frameid,'offset_frames':frameid-case['center_frame_id'],'actual_pts_us':actual[frameid],'observation_timestamp_us':int(ts[frameid]),'full_file':str(pf),'focus_file':str(pc),'rgb_sha256':pixel_hashes[frameid]})
            ordered=folder/'ordered_contact.png';shufflepath=folder/'shuffled_contact.png';fullsheet=folder/'full_ordered_contact.png';center=folder/'center_only.png'
            sheet(crops,list(range(13)),labels).save(ordered);sheet(crops,shuffled,labels).save(shufflepath);sheet(fulls,list(range(13)),labels).save(fullsheet);crops[6].save(center)
            for p in (ordered,shufflepath,fullsheet,center):outputs[str(p)]=sha(p)
            out={**case,'source_sha_verified':True,'display_only_focus_crop_xyxy':list(crop),'focus_crop_policy':policy,'raw_frame_size':[w,h],'seed_sha256':seedhash,'neutral_id_assignment':'randomized independently of time order using frozen hash and case ID','ordered_image_ids':labels,'shuffled_image_ids':[labels[i] for i in shuffled],'center_image_id':labels[6],
                 'ordered_contact':str(ordered),'shuffled_contact':str(shufflepath),'full_ordered_contact':str(fullsheet),'center_only':str(center),'frames':meta}
            done.append(out);write('PROGRESS.json',{'completed_windows':len(done),'total_windows':18,'rows':done})
            print(json.dumps({'case':cid,'windows':len(done),'ordered':str(ordered),'shuffled':str(shufflepath)},ensure_ascii=False),flush=True)
    done.sort(key=lambda r:r['case_id']);assert len(done)==18 and sum(len(r['frames']) for r in done)==234
    write('SOURCE_PINS_BEFORE.json',pins);after={p:sha(p) for p in pins};assert after==pins;write('SOURCE_PINS_AFTER.json',after);write('OUTPUT_PINS.json',outputs)
    write('INDEX.json',{'status':'COMPLETE','frozen_source':str(fp),'windows':18,'frames_per_window':13,'full_rgb_png':234,'focus_rgb_png':234,'contact_sheets':54,'center_only_png':18,'rows':done,'source_metadata':source_metadata,
      'no_new_detector':True,'no_training':True,'no_new_dtw':True,'no_user_mentor_access':True,'no_original_writes':True,
      'blinding':'Contact images have only randomized K identifiers. ordered shows temporal order; shuffled conceals frame order. This mapping file reveals time/group and must be withheld for direct static image labeling.',
      'crop_warning':'Display-only generous target hand/arm crop may omit face or other hand. Full original PNG and full_ordered sheet retained. Crop is never a model input or truth label.'})
    write('VERIFY.json',{'status':'PASS','source_pins_unchanged':len(pins),'source_videos':len(groups),'windows':18,'full_rgb_png':234,'focus_png':234,'source_sha_matches_frozen':True,'pts_max_error_us':maxpts,'all_full_png_rgb_matches_decode':True,'all_focus_png_exact_original_pixel_crop':True,'fixed_seed_shuffle':True,'same_agent_self_check':True})

if __name__=='__main__':
    try:main()
    except Exception:
        p=U/'FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
