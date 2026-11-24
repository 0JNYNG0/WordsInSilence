"""Saved FIT radial-distance audit and largest-amplitude hold RGB inspection."""
from audit import *
import cv2
from PIL import Image,ImageDraw,ImageFont
R=U/'rgb_review'

def main():
    assert not(U/'RADIAL_HOLD_AUDIT.json').exists()
    a=read(U/'RESULTS.json');groups=defaultdict(list)
    for r in a['rows']:
        if r['measurement']['eligible']:groups[r['label_kind'],r['view'],r['reverse']].append(r)
    summary=[]
    for key,rr in sorted(groups.items()):
        ex=1 if key[2] else -1
        summary.append(dict(label_kind=key[0],view=key[1],reverse=key[2],rows=len(rr),radial_delta=numbers([r['measurement']['wrist_distance_delta'] for r in rr]),
            radial_sign_supports_toward_wrist_occurrence=sum(ex*r['measurement']['wrist_distance_delta']>0 for r in rr) if key[0]!='hold' else None,
            radial_sign_opposes_toward_wrist_occurrence=sum(ex*r['measurement']['wrist_distance_delta']<0 for r in rr) if key[0]!='hold' else None,
            projected_radial_direction_disagreement=sum(r['measurement']['delta']*r['measurement']['wrist_distance_delta']>0 for r in rr),
            projection_conflict_radial_support=sum(((-1 if key[2] else 1)*r['measurement']['delta']<0)and ex*r['measurement']['wrist_distance_delta']>0 for r in rr) if key[0]!='hold' else None))
    hold=max((r for r in a['rows']if r['label_kind']=='hold' and r['view']=='full' and not r['reverse'] and r['measurement']['eligible']),key=lambda r:abs(r['measurement']['delta']))
    cid=hold['clip_id'];aa=hold['source_span'][0]+hold['span'][0];bb=hold['source_span'][0]+hold['span'][1]
    ss=[r for p in[GRAPH/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',PRIOR/'additional_source_review/SOURCE_EPISODES_ADDED.json']for r in read(p)['rows']if r['clip_id']==cid];assert len(ss)==1;src=ss[0];assert src['person']in range(2,8) and sha(src['source'])==src['source_sha256']
    cr=next(r for p in[old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json']for r in read(p)['rows']if r['clip_id']==cid)
    ready=next(r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['clip_id']==cid);assert ready['person']in range(2,8)
    ch=np.load(ready['channel']);cz=np.load(cr['path']);fw=int(ch['source_width']);fh=int(ch['source_height']);roi=[int(fw*.32),int(fh*.29),int(fw*.68),int(fh*.63)]
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'))
    guard=old.load('_radial_hold_guard',OPS/'word100_observation_guard_20261108/guard.py');ad=old.load('_radial_hold_ad',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    cl=guard.quarantine(cz['xy_px'],cz['score'],cz['timestamp_us'],640,480);cv=ad.convert(cl['xy_px'],cz['score'],cz['timestamp_us'],cz['frame_id'],640,480)
    fp=ch['native67'];cp=cv['native67'].copy();l,t,right,bottom=cr['crop'];cp[:,:,0]=cp[:,:,0]/640*(right-l)+l;cp[:,:,1]=cp[:,:,1]/480*(bottom-t)+t
    cap=cv2.VideoCapture(src['source']);assert cap.isOpened();tiles=[];frames=list(range(max(0,aa-2),min(src['frames']-1,bb+2)+1));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(tuple(roi));dd=ImageDraw.Draw(im)
        for pts,col in[(fp[f],'#17456f'),(cp[f],'#d17920')]:
            xy=[tuple(pts[k]-np.asarray(roi[:2]))for k in[6,7,4]];dd.line(xy[:2],fill=col,width=4)
            for x,y in xy:dd.ellipse((x-5,y-5,x+5,y+5),fill=col)
        im.thumbnail((460,270));tile=Image.new('RGB',(480,310),'white');tile.paste(im,((480-im.width)//2,34));ImageDraw.Draw(tile).text((12,7),f'frame {f}'+(' | reviewed hold' if hold['source_span'][0]<=f<=hold['source_span'][1] else ''),font=font,fill='#14243a');tiles.append(tile)
    cap.release();cols=4;sheet=Image.new('RGB',(1920,60+310*((len(tiles)+3)//4)),'#e4e9ef');ImageDraw.Draw(sheet).text((12,10),cid+f' | largest full hold delta [{aa},{bb}] | blue full / orange crop',font=font,fill='#14243a')
    for k,im in enumerate(tiles):sheet.paste(im,((k%cols)*480,60+(k//cols)*310))
    dest=R/(cid+'_largest_hold_coordinate_zoom.jpg');sheet.save(dest,quality=94)
    flips=[dict(clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],original_span=r['original_span'],projected_delta=r['measurement']['delta'],radial_delta=r['measurement']['wrist_distance_delta'])for r in a['sign_conflicts']]
    out=dict(at=stamp(),scope='FIT-only same saved partial/full/hold candidates. Distance approach is a complementary observation, not automatic along-arm/contact truth.',summary=summary,projection_sign_conflicts=flips,
        largest_hold=dict(clip_id=cid,original_span=[aa,bb],source_span=hold['source_span'],measurement=hold['measurement'],source=src['source'],source_sha256=src['source_sha256'],sheet=str(dest),sheet_sha256=sha(dest),frames=frames,roi=roi),
        pins={str(p):sha(p)for p in[Path(__file__),U/'RESULTS.json',Path(ready['channel']),Path(cr['path'])]},new_model_inference=False)
    write(U/'RADIAL_HOLD_AUDIT.json',out)
    print(json.dumps(dict(summary=summary,flips=flips,largest_hold=out['largest_hold']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
