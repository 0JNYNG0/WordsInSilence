"""One fixed public candidate per clip, no inherited occurrence-zero labels."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
import numpy as np
import av
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent
M=O/'word100_learned_motion_order_20261102'
P=O/'word100_learned_guide_feedback_20261102'
AG=O/'word100_context_episode_query_20261103'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def exclusive(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

QUERIES={0:'Right hand travels along the presented left forearm toward its wrist',
         1:'Hands move apart from each other',2:'Right hand rises relative to the left',
         3:'Right hand descends from face region toward torso',4:'Right hand falls relative to the left'}

def main():
    assert not (U/'REVIEW_PREP_STARTED.json').exists(),'Check actual prior process/output before resuming'
    input_rows={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']}
    source=read(M/'MOTION_TARGETS.json')
    previous={r['clip_id']:r for r in read(AG/'LATE_VISUAL_REVIEW.json')['rows']}
    candidates=[]
    for old in source['rows']:
        if old['kind']!='stable':continue
        r=input_rows[old['clip_id']]
        assert r['person'] in (list(range(1,10)) if r['role']=='fit' else [11,12,13])
        reuse=previous.get(r['clip_id'])
        bounds=reuse['fixed_native_frame_interval'] if reuse else old['frames']
        queries=[j for j in range(5) if old['mask'][j]]
        candidates.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role=r['role'],
            source=r['source'],source_sha256=r['source_sha256'],input=r['input'],input_sha256=r['input_sha256'],
            frames=bounds,queries=[dict(index=j,definition=QUERIES[j]) for j in queries],
            previous_local_review=str(AG/'LATE_VISUAL_REVIEW.json') if reuse else None,
            candidate_selection='Previously fixed candidate coordinates only; old targets/scores not copied',
            label=None,whole_clause_truth=None,educational_error=None))
    assert len(candidates)==48 and sum(r['previous_local_review'] is not None for r in candidates)==3
    sources={r['source_sha256']:r['role'] for r in candidates}
    assert len(sources)==48,'Duplicate source images must be quarantined before splitting'
    exclusive(U/'REVIEW_SCOPE.json',dict(previous_goal_turn='progress: fixed-scope test, saved verification, untrained local head implemented',
        status='FROZEN_ONE_PASS_OCCURRENCE_CONTROL_REVIEW',rows=candidates,
        guidance='Review all native frames of the fixed interval. Substantial target-direction occurrence1, clear absence0, unresolvednull. Tiny common body settling is not a distinct stroke. Local absence is not a wrong sign. Do not shift windows to obtain0.',
        limits='45 new windows (33FIT+12DEV), AG3 reuse, no re-view of positives or predictions. No word/person special thresholds. No training until dataset and code gates.',
        pins={str(p):sha(p) for p in [P/'INPUT_MANIFEST.json',M/'MOTION_TARGETS.json',AG/'LATE_VISUAL_REVIEW.json',AG/'NEXT_LEARNING_DECISION.json',U/'PLAN_RESOLUTION.json']}))
    exclusive(U/'REVIEW_PREP_STARTED.json',dict(pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
    folder=U/'review_sheets';folder.mkdir(exist_ok=True)
    ready=[]
    for r in candidates:
        assert sha(r['source'])==r['source_sha256'] and sha(r['input'])==r['input_sha256']
        a=np.load(r['input']);lo,hi=r['frames']
        if r['previous_local_review']:
            old=previous[r['clip_id']]
            ready.append(dict(**r,sheet=old['reviewed_sheet'],native_frames=old['actually_viewed_native_frames'],
                              timestamp_us=old['native_timestamps_us'],reused_verified_review=True))
            continue
        frames=[]
        with av.open(r['source']) as c:
            for i,f in enumerate(c.decode(video=0)):
                if i>hi:break
                if i<lo:continue
                us=round(float(f.pts*f.time_base)*1e6)
                assert int(a['frame_id'][i])==i and int(a['timestamp_us'][i])==us
                im=f.to_image().convert('RGB');im.thumbnail((640,360));frames.append((i,us,im))
        assert [i for i,_,_ in frames]==list(range(lo,hi+1))
        sheet=Image.new('RGB',(640*3,400*((len(frames)+2)//3)),'white');d=ImageDraw.Draw(sheet)
        for k,(i,us,im) in enumerate(frames):
            x,y=k%3*640,k//3*400;sheet.paste(im,(x,y+32));d.text((x+8,y+7),f"{r['clip_id']} f{i} {us/1e6:.6f}s",fill='black')
        dest=folder/(r['clip_id']+'.png');assert not dest.exists();sheet.save(dest)
        ready.append(dict(**r,sheet=dict(path=str(dest),sha256=sha(dest)),native_frames=[i for i,_,_ in frames],timestamp_us=[t for _,t,_ in frames],reused_verified_review=False))
        (U/'REVIEW_PROGRESS.json').write_text(json.dumps(dict(done=len(ready),total=48,last=r['clip_id'])),encoding='utf-8')
    for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
        exclusive(U/(word+'_REVIEW_QUEUE.json'),dict(rows=[r for r in ready if r['word_id']==word],
            old_labels_removed=True,predictions_read=False,not_independent_KSL_expert_truth=True))
    exclusive(U/'REVIEW_READY.json',dict(status='ALL_FIXED_NATIVE_WINDOWS_READY',rows=ready,
        new_windows=45,reused_reviews=3,new_frame_cells=sum(len(r['native_frames']) for r in ready if not r['reused_verified_review'])))
    exclusive(U/'REVIEW_PREP_FINISHED.json',dict(pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),status='COMPLETE'))
    print('Prepared45 fixed native windows; reused3 prior direct reviews; model execution0')

if __name__=='__main__':main()
