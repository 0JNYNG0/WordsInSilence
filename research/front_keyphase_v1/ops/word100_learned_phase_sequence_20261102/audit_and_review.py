"""Explicit public-only evidence inventory and broad physical-state review sheets."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib
import numpy as np,av
from PIL import Image,ImageDraw,ImageFont
S=Path(__file__).resolve().parent;O=S.parent;P=O/'word100_learned_guide_feedback_20261102';M=O/'word100_learned_motion_order_20261102';V=O/'word100_learned_body_relation_20261102';N=O/'word100_learned_motion_invariance_20261102'
sys.path.insert(0,str(N));from prepare_real_fit import read,sha,exclusive,clock

def main():
    assert not (S/'SOURCE_AUDIT.json').exists()
    manifest=read(P/'INPUT_MANIFEST.json');rows=manifest['rows'];fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
    assert len(fit)==36 and len(dev)==12 and {r['person'] for r in fit}==set(range(1,10))
    assert {r['person'] for r in dev}=={11,12,13}
    motions=read(M/'MOTION_TARGETS.json')['rows'];relations=read(V/'RELATION_TARGETS.json')['rows']
    known=lambda rr:sum(sum(l['target_mask']) for r in rr for l in r['labels'])
    audit=dict(at=clock(),previous_goal_turn='progress:32 locked inputs,192 comparisons,36 gap diagnostics,4 actual order edits and isolated demo8807 with verified fresh public analysis8.35s',
        fit_people=list(range(1,10)),development_people=[11,12,13],fit_clips=len(fit),development_clips=len(dev),
        fit_stage_provenance=dict(Counter(r['stage_supervision'] for r in fit)),fit_exact_center_labels=sum(len(r['labels']) for r in fit),fit_known_attribute_targets=known(fit),
        development_exact_center_labels=sum(len(r['labels']) for r in dev),development_known_attribute_targets=known(dev),
        motion_episodes=dict(Counter(r['role'] for r in motions)),
        role_of_motion_labels='Net direction over reviewed interval, not per-frame movement truth or required repeat count.',
        phase_interval_warning='An interval may contain a changing hand or alternating height. Its description is not a constant per-frame configuration label.',
        new_review='Broad visible physical configurations only, at original native frames. Unknown and transitions stay unlabelled. Direct prior feature labels may help but DTW/neural predictions are not viewed.',
        no_new_inference=True,no_training=True,no_private_access=True,no_locked14_15_access=True,
        pins={str(p):sha(p) for p in [P/'INPUT_MANIFEST.json',P/'GUIDE_SUPPORT.json',M/'MOTION_TARGETS.json',V/'RELATION_TARGETS.json']})
    exclusive(S/'SOURCE_AUDIT.json',audit)
    records=[];images={}; font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    for r in rows:
        assert sha(r['source'])==r['source_sha256']
        mm=[m for m in motions if m['clip_id']==r['clip_id']]
        frames={l['center_frame'] for l in r['labels']}
        for m in mm:
            lo,hi=m['frames'];frames.update([lo,(lo+hi)//2,hi])
        frames=sorted(frames);assert len(frames)<=14
        sampled={}
        with av.open(r['source']) as c:
            for i,f in enumerate(c.decode(video=0)):
                if i in frames:
                    im=f.to_image();w,h=im.size
                    # Fixed central upper-body view, also save full frames for any doubt.
                    dest=S/'review_frames'/r['clip_id']/f'f{i:04d}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest,quality=94)
                    box=(int(.20*w),0,int(.80*w),min(h,int(r['box'][3])))
                    crop=im.crop(box);crop.thumbnail((310,260));sampled[i]=crop
            assert i+1==r['frames']
        assert len(sampled)==len(frames)
        images[r['clip_id']]=sampled
        records.append(dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],role=r['role'],source=r['source'],source_sha256=r['source_sha256'],frames=frames,
            known_center_labels=[dict(frame=l['center_frame'],y=l['y'],mask=l['target_mask']) for l in r['labels']],
            reviewed_motion_episodes=[dict(frames=m['frames'],kind=m['kind'],y=m['y'],mask=m['mask']) for m in mm]))
    sheets=[]
    for word in sorted({r['word_id'] for r in rows}):
        for people in [[1,2,3],[4,5,6],[7,8,9],[11,12,13]]:
            selected=[r for r in records if r['word_id']==word and r['person'] in people];cols=max(len(r['frames']) for r in selected)
            sheet=Image.new('RGB',(cols*315,3*300+38),'#e6edf1');draw=ImageDraw.Draw(sheet);draw.text((10,8),f'{word} recorded IDs {people} | Original frames; no stage/model predictions',font=font,fill='black')
            for j,r in enumerate(selected):
                for k,fr in enumerate(r['frames']):
                    x=k*315;y=38+j*300;draw.text((x+4,y+3),f"ID{r['person']:02} f{fr}",font=font,fill='black');im=images[r['clip_id']][fr];sheet.paste(im,(x+(310-im.width)//2,y+32))
            path=S/'review_sheets'/f'{word}_{people[0]:02}_{people[-1]:02}.jpg';path.parent.mkdir(exist_ok=True);sheet.save(path,quality=93)
            sheets.append(dict(word=word,people=people,path=str(path),sha256=sha(path)))
    exclusive(S/'REVIEW_MANIFEST.json',dict(rows=records,sheets=sheets,phase_labels_added=False,independent_human_expert=False,
        scope='Side-by-side chronology for explicit broad physical-state labeling, not dense or linguistic gold. Full frames retained for crop doubts.'))
    print(json.dumps(audit,ensure_ascii=False));print('Prepared',len(records),'public clips;',len(sheets),'review sheets')

if __name__=='__main__':main()
