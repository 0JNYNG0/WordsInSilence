"""Audit the meaning of inherited clause negatives, not new pose inference."""
from audit_saved import *
import cv2
from PIL import Image,ImageDraw,ImageFont
def main():
    files=[G/'episode_stage_contrast/RGB_CONTRAST_LABELS.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json',G/'candidate_description_supervision/core.py',G/'candidate_description_supervision/prepare_source_boundary.py',G/'candidate_description_supervision/prepare_rank_learning.py',C6.parent/'body_progress_witness_learning/prepare.py']
    co={r['clip_id']:r for r in read(files[0])['rows']}
    co.update({r['clip_id']:r for r in read(files[1])['contrasts']})
    sources={r['clip_id']:r for p in [G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',files[1]] for r in read(p)['rows'] if r['role']=='fit'and r['person']in range(2,8)}
    z=dict(np.load(C6/'INPUTS.npz'));idx=read(C6/'INDEX.json')['rows'];raw=z['absolute_context'].reshape(-1,3,10,5)[:,1]
    records=[]
    for r in idx:
        if r['view']!='full'or r['reverse']:continue
        cid=r['clip_id'];contrast=co[cid];src=sources[cid]
        for i in range(r['start'],r['stop']):
            fr=int(z['source_frame_id'][i]);sel=np.arange(max(r['start'],i-4),min(r['stop'],i+5))
            for j,fam in enumerate(F):
                code=int(z['label_provenance'][i,j])
                if code not in[2,3,4]:continue
                kinds=[k for k in['entry','returning','other_held_presentation']if fr in contrast[k]]
                other=[e['event_id']for e in src['events']if e['family']!=fam and fr in e['progression_witness_frames']]
                if other:kinds.append('other_clause_progression')
                if code in[3,4]:kinds.append('own_terminal_hold'if code==3 else'other_terminal_hold')
                a=raw[sel,AX[fam]];v=a[:,4]>.5;adj=v[:-1]&v[1:];delta=np.diff(a[:,:4],axis=0)
                records.append(dict(clip_id=cid,person=r['person'],frame=fr,family=fam,code=code,negative_provenance=kinds,other_clause_events=other,
                    frames=z['source_frame_id'][sel].tolist(),raw_primary_context=a.tolist(),valid_adjacent_path4=np.abs(delta[adj]).sum(0).tolist(),
                    raw_right_wrist_xy=raw[sel,0,:2].tolist(),right_wrist_observed=(raw[sel,0,4]>.5).tolist(),raw_left_wrist_xy=raw[sel,1,:2].tolist(),left_wrist_observed=(raw[sel,1,4]>.5).tolist(),
                    physical_hold_ground_truth_provided=False if code==2 else 'Earlier source held posture; must separately confirm this measured relation at this time'))
    count=collections.Counter(k for r in records if r['code']==2 for k in r['negative_provenance'])
    unclassified=[r for r in records if r['code']==2 and not r['negative_provenance']]
    # Source examples predetermined from reviewed entry/return categories, not classifier scores.
    panels=[]
    for word in['WORD0739','WORD2523','WORD1133','WORD1567']:
        for person in[2,4,6]:
            cid=f'NIA_SL_{word}_REAL{person:02}_F';src=sources[cid];r=next(r for r in idx if r['clip_id']==cid and r['view']=='full'and not r['reverse'])
            anchors=[]
            for k in['entry','returning']:
                eligible=[p for p in records if p['clip_id']==cid and k in p['negative_provenance']]
                if eligible:
                    p=min(eligible,key=lambda p:(p['frame'],p['family']));anchors.append(dict(kind=k,frame=p['frame'],old_family=p['family'],old_code=p['code']))
            if src['hold_sample_frames']:anchors.append(dict(kind='held_posture_review',frame=src['hold_sample_frames'][len(src['hold_sample_frames'])//2]))
            if not anchors:continue
            font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21);canvas=Image.new('RGB',(1600,len(anchors)*330),'white');draw=ImageDraw.Draw(canvas);cap=cv2.VideoCapture(src['source']);assert cap.isOpened();assert sha(src['source'])==src['source_sha256']
            for row,a in enumerate(anchors):
                fs=[max(0,min(src['frames']-1,a['frame']+v))for v in[-4,-2,0,2,4]];a['review_frames']=fs
                draw.text((10,row*330+3),f"{cid} {a['kind']} centre {a['frame']}",font=font,fill='#16304a')
                for col,fr in enumerate(fs):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,im=cap.read();assert ok
                    h,w=im.shape[:2];pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop((int(.14*w),int(.04*h),int(.86*w),int(.90*h)));pic.thumbnail((310,270));canvas.paste(pic,(col*320+(320-pic.width)//2,row*330+32));draw.text((col*320+10,row*330+303),str(fr),font=font,fill='#16304a')
            cap.release();path=U/f'label_contract_{word}_{person:02}.jpg';canvas.save(path,quality=93);panels.append(dict(clip_id=cid,source=src['source'],source_sha256=src['source_sha256'],path=str(path),sha256=sha(path),anchors=anchors))
    write('LABEL_CONTRACT_AUDIT.json',dict(status='CODE_SEMANTICS_CONFIRMED_SOURCE_PANELS_AWAIT_REVIEW',code2_native=len([r for r in records if r['code']==2]),provenance_counts=dict(count),unclassified=unclassified,
        finding='category2 is NOT_THIS_CLAUSE, not physical immobility. Entry/return and other-action motion were admitted negative for an action-specific classifier; later readout name body_progress did not change them.',
        no_crossword_negative_auto_assignment='Targets only iterate source groups; absent families in other words have mask0, not forced negative. The concrete mismatch is within-source entry/return/other-clause, plus family-word confounding.',
        old_metrics_unchanged=True,new_physical_task_requires_direct_target=True,records=records,panels=panels,pins={str(p):sha(p)for p in files}))
    print(json.dumps(dict(code2_native=sum(r['code']==2 for r in records),provenance=dict(count),unclassified=len(unclassified),panels=len(panels))))
if __name__=='__main__':main()
