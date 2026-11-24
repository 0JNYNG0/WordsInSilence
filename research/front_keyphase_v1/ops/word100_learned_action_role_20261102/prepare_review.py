"""Chronological source evidence; no prediction-generated role labels."""
from pathlib import Path
import sys,av,shutil
from PIL import Image,ImageDraw,ImageFont
A=Path(__file__).resolve().parent;O=A.parent;S=O/'word100_learned_phase_sequence_20261102';T=O/'word100_learned_state_context_balance_20261102';P=O/'word100_learned_guide_feedback_20261102'
sys.path.insert(0,str(O/'word100_learned_motion_invariance_20261102'));from prepare_real_fit import read,sha,exclusive,clock,write

def main():
    assert not (A/'REVIEW_MANIFEST.json').exists();assert shutil.disk_usage(A).free>1024**3
    exclusive(A/'REVIEW_INTENT.json',dict(at=clock(),previous_goal_turn='progress:36actual local inputs,216FITinput audit,8phase trainings including2pilots,704saved comparisons and visual diagnosis; state accuracy improved, full sequence incomplete',
      review='Sparse visual action-role/transition review. Existing source cards are partial associations, not dense semantic truth. Do not label every intermediate movement unimportant.',
      role_intent=['evidence of the referenced configuration in this action','approaching/changing/returning through configurations, still retained for movement analysis','visible lowered rest'],
      no_private_access=True,no_locked14_15=True,new_training=False,fit_record_ids=list(range(1,10)),development_record_ids=[11,12,13]))
    targets={r['clip_id']:r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows']};records=[];imgs={}
    for r in read(P/'INPUT_MANIFEST.json')['rows']:
        assert sha(r['source'])==r['source_sha256'];points=targets[r['clip_id']]['points'];known=sorted([p for p in points if p['mask']],key=lambda p:p['frame']);first,last=known[0]['frame'],known[-1]['frame']
        selected={max(0,first-d) for d in [12,6,3,0]}|{min(r['frames']-1,last+d) for d in [0,3,6,12]}
        selected|={p['frame'] for p in points if not p['mask']}
        for x,y in zip(known,known[1:]):
            if x['group']!=y['group'] and y['frame']-x['frame']>=3:selected.add((x['frame']+y['frame'])//2)
        selected=sorted(selected);assert len(selected)<=16
        found={};saved=[]
        with av.open(r['source']) as c:
            for i,f in enumerate(c.decode(video=0)):
                if i in selected:
                    full=f.to_image();w,h=full.size;p=A/'review_frames'/r['clip_id']/f'f{i:04d}.jpg';p.parent.mkdir(parents=True,exist_ok=True)
                    small=full.copy();small.thumbnail((1280,720));small.save(p,quality=93)
                    im=full.crop((int(.22*w),0,int(.78*w),h));im.thumbnail((295,295));found[i]=im;saved.append(dict(frame=i,path=str(p),sha256=sha(p)))
            assert i+1==r['frames']
        assert len(found)==len(selected);imgs[r['clip_id']]=found
        records.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role=r['role'],source=r['source'],source_sha256=r['source_sha256'],
          frames=selected,samples=saved,previous_known={str(p['frame']):p['group'] for p in known},new_labels_assigned=False))
        write(A/'REVIEW_PROGRESS.json',dict(at=clock(),done=len(records),total=48))
    sheets=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    for word in sorted({r['word_id'] for r in records}):
        for people in [[1,2,3],[4,5,6],[7,8,9],[11,12,13]]:
            rr=[r for r in records if r['word_id']==word and r['person'] in people];out=Image.new('RGB',(8*300,6*330+35),'#edf2f5');draw=ImageDraw.Draw(out)
            draw.text((8,5),f'{word} IDs{people}: native chronology; selected transitions; no model predictions',font=font,fill='black')
            for j,r in enumerate(rr):
                for k,i in enumerate(r['frames']):
                    x=(k%8)*300;y=35+(j*2+k//8)*330
                    known=' [prior anchor]' if str(i) in r['previous_known'] else ''
                    draw.text((x+2,y+3),f"ID{r['person']:02} f{i}{known}",font=font,fill='black');im=imgs[r['clip_id']][i];out.paste(im,(x+(300-im.width)//2,y+31))
            p=A/'review_sheets'/f'{word}_{people[0]:02}_{people[-1]:02}.jpg';p.parent.mkdir(exist_ok=True);out.save(p,quality=94);sheets.append(dict(word=word,people=people,path=str(p),sha256=sha(p)))
    exclusive(A/'REVIEW_MANIFEST.json',dict(at=clock(),rows=records,sheets=sheets,reviewed=False,labels_assigned=False))
    print('Prepared',len(records),'public clips',sum(len(r['frames']) for r in records),'moments',len(sheets),'sheets')

if __name__=='__main__':main()
