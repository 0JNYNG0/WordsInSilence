from prepare import *
from guard import quarantine
def main():
    plan=read(U/'INPUT_PLAN.json');rows={r['clip_id']:r for r in plan['rows']};results=[]
    for item in read(O/'word100_rtmw_mediapipe_compare_20261108/VISUAL_INDEX.json'):
        r=rows.get(item['clip_id']);mask=item.get('target_mask',[])
        if r is None or not mask:continue
        z=np.load(r['cache'] or U/'crops'/(r['clip_id']+'.npz'));old=np.load(r['raw']);g=quarantine(z['xy_px'],z['score'],z['timestamp_us'],640,480);i=item['frame'];l,t,rr,b=r['crop']
        for side,cols,start in [(0,[0,2],91),(1,[1,3],112)]:
            if not any(j<len(mask) and mask[j] and item['targets'][j]>.5 for j in cols):continue
            q=old['xy_px'][i,start:start+21];inside=((q[:,0]>=l)&(q[:,0]<rr)&(q[:,1]>=t)&(q[:,1]<b)).sum()
            if inside>=18:results.append(dict(clip=r['clip_id'],frame=i,side=side,preserved=not bool(g['hand_quarantined'][i,side])))
    bad=[r for r in results if not r['preserved']]
    write(U/'VISIBLE_CROP_CHECK.json',dict(rows=results,failed=bad,status='FAIL' if bad else 'PASS',scope='Existing positive RGB shape centers that project mostly inside new crop; not every possible normal motion'))
    print(json.dumps(dict(count=len(results),failed=bad)));assert not bad
if __name__=='__main__':main()
