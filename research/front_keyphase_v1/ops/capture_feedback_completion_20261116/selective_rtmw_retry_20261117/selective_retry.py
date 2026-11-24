"""Bounded re-observation, never trajectory interpolation or target-shape search."""
import copy
import time
import numpy as np
import cv2

VERSION = 'selective_front_rtmw_retry_v1'
MAX_CALLS = 16
BUDGET = 1.2


def visible(p, w, h):
    return np.isfinite(p).all(-1) & (p[..., 0] >= 0) & (p[..., 0] < w) & (p[..., 1] >= 0) & (p[..., 1] < h)


def spans(mask):
    ids = np.flatnonzero(mask)
    return [a for a in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if len(a)]


def requests(raw, initial):
    xy = raw['xy_px']; w = int(raw['source_width']); h = int(raw['source_height'])
    ts = raw['timestamp_us']; av = visible(xy, w, h)
    needed = {hand: np.zeros(len(xy), bool) for hand in ['L', 'R']}
    unknown_hands=set()
    for card in initial.get('cards', []):
        if card.get('status') != 'unknown':
            continue
        unknown_hands.update(h for h in card.get('hands',[]) if h in needed)
        for a, b in card.get('query_spans', []):
            if not (0 <= a <= b < len(xy)):
                continue
            for hand in card.get('hands', []):
                if hand in needed:
                    needed[hand][a:b+1] = True
    # Missing observations may truncate the detected event itself. Search only
    # short gaps touching its observed boundary, not just its surviving core.
    lesson=initial.get('lesson_acceptance',{})
    neighborhoods=[]
    for ev in lesson.get('events',[]):
        neighborhoods.append(ev.get('context',[ev['start'],ev['end']]))
    for ev in lesson.get('movement_evidence',{}).get('movement_candidates',[]):
        neighborhoods.append(ev['source_span'])
    for c in lesson.get('criteria',[]):
        if c.get('status')=='unknown' and c.get('observed_span'):
            neighborhoods.append(c['observed_span'])
    for a,b in neighborhoods:
        if not (0 <= a <= b < len(xy)):continue
        near=(ts>=ts[a]-300000)&(ts<=ts[b]+300000)
        for hand in unknown_hands:needed[hand] |= near
    candidates = []; rejected = []
    for hand, start in [('L', 91), ('R', 112)]:
        # Group over the complete time axis BEFORE intersecting important spans;
        # clipping a long disappearance to an event must not turn it into a short gap.
        missing = ~av[:, start:start+21].all(1)
        for run in spans(missing):
            a, b = int(run[0]), int(run[-1])
            if not needed[hand][a:b+1].any():
                continue
            reason = None
            if len(run) > 6 or a == 0 or b == len(xy)-1 or ts[b+1]-ts[a-1] > 300000:
                reason = 'long_or_unbounded_gap'
            elif not av[a-1, start:start+21].all() or not av[b+1, start:start+21].all():
                reason = 'missing_brackets'
            if reason:
                rejected.append(dict(hand=hand, span=[a,b], reason=reason)); continue
            own = xy[[a-1,b+1], start:start+21]
            palm = float(np.median(np.linalg.norm(own[:,9]-own[:,0], axis=1)))
            if not np.isfinite(palm) or palm < 4:
                rejected.append(dict(hand=hand, span=[a,b], reason='tiny_anchor')); continue
            opposite = 112 if start == 91 else 91
            blocked = False
            for i in range(a-1,b+2):
                alpha = float((ts[i]-ts[a-1])/(ts[b+1]-ts[a-1]))
                # Interpolation is used only as a rejection envelope, NEVER as an observation.
                envelope = own[0]*(1-alpha)+own[1]*alpha
                if (envelope < 2).any() or (envelope[:,0] > w-3).any() or (envelope[:,1] > h-3).any():
                    blocked = True; break
                other = xy[i,opposite:opposite+21][av[i,opposite:opposite+21]]
                if len(other) >= 3:
                    area,_ = cv2.intersectConvexConvex(cv2.convexHull(envelope.astype('float32')), cv2.convexHull(other.astype('float32')))
                    if area > 0:
                        blocked = True; break
            if blocked:
                rejected.append(dict(hand=hand, span=[a,b], reason='projected_overlap_or_frame_edge')); continue
            pts = xy[a-1:b+2,:11].reshape(-1,2)
            pts = pts[visible(pts,w,h)]
            hands = xy[a-1:b+2,91:].reshape(-1,2); hands=hands[visible(hands,w,h)]
            pts = np.concatenate([pts,hands,own.reshape(-1,2)])
            shoulder = np.linalg.norm(xy[a-1,5]-xy[a-1,6])
            if not np.isfinite(shoulder) or shoulder < 5:
                rejected.append(dict(hand=hand, span=[a,b], reason='missing_body_anchor')); continue
            margin = .35*shoulder
            lo = np.maximum(pts.min(0)-margin,0); hi=np.minimum(pts.max(0)+margin,[w,h])
            box = np.r_[lo,hi].astype(np.float32)
            candidates.append(dict(hand=hand,start=start,span=[a,b],frames=list(range(a-1,b+2)),palm=palm,box=box.tolist()))
    candidates.sort(key=lambda x:(len(x['frames']),x['span'][0],x['hand']))
    return candidates, rejected


class VideoFrames:
    """Read exact recorded PTS; bounded selected-image cache, no pose replay."""
    def __init__(self, path, raw, transform=None):
        self.path=str(path); self.raw=raw; self.transform=transform; self.decoded=0
    def read(self, ids, deadline):
        import av
        wanted={int(self.raw['timestamp_us'][i]):i for i in ids}
        result={}
        with av.open(self.path) as container:
            stream=container.streams.video[0]
            start=min(wanted)
            container.seek(int(start / 1e6 / float(stream.time_base)),stream=stream,backward=True)
            for frame in container.decode(stream):
                if time.perf_counter()>deadline:
                    break
                self.decoded+=1
                if frame.pts is None:
                    continue
                pts=round(float(frame.pts*frame.time_base)*1e6)
                if pts > max(wanted)+2:
                    break
                matches=[k for k in wanted if abs(k-pts)<=2]
                if matches:
                    im=frame.to_ndarray(format='bgr24')
                    if self.transform is not None:im=self.transform(im)
                    i=wanted[matches[0]]
                    if im.shape[:2] != (int(self.raw['source_height']),int(self.raw['source_width'])):
                        raise ValueError('Frame dimensions differ from original observations')
                    result[i]=im
                if len(result)==len(ids):break
        return result


def retry(raw, initial, reader, engine, infer, word):
    began=time.perf_counter(); candidates,rejected=requests(raw,initial)
    meta=dict(version=VERSION,requested=bool(candidates),pose_calls=0,replaced_points=0,accepted=False,rejected=rejected,read_frames=0,reason='not_needed',seconds=0.)
    if not candidates:
        out=dict(initial);out['selective_retry']=meta;meta['seconds']=time.perf_counter()-began;return out,raw
    candidate=candidates[0];a,b=candidate['span'];start=candidate['start'];palm=candidate['palm']
    meta.update(hand=candidate['hand'],span=[a,b],reason='pending')
    def done(reason, output=initial, fixed=raw):
        meta.update(reason=reason,seconds=time.perf_counter()-began)
        result=dict(output);result['selective_retry']=meta
        return result,fixed
    # A gap plus both visible bracketing frames are always processed together.
    if 2*len(candidate['frames'])>MAX_CALLS:return done('pose_budget')
    deadline=began+BUDGET
    try:
        images=reader.read(candidate['frames'],deadline-.4)
        meta['read_frames']=len(images);meta['decoded_frames']=reader.decoded
        if len(images)!=len(candidate['frames']):return done('missing_frame_or_read_budget')
        predictions={};w=int(raw['source_width']);h=int(raw['source_height']);box=np.array(candidate['box'],np.float32)
        center=(box[:2]+box[2:])/2;extent=(box[2:]-box[:2])*1.1
        second=np.r_[np.maximum(center-extent/2,0),np.minimum(center+extent/2,[w,h])]
        for i in candidate['frames']:
            if time.perf_counter()>deadline-.4:return done('pose_time_budget')
            im=images[i]; hand_outputs=[]
            for bb in [box,second]:
                kp,sc=engine.pose(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),bboxes=[bb])
                meta['pose_calls']+=1
                p=kp[0,start:start+21];score=sc[0,start:start+21]
                if not visible(p,w,h).all() or not np.isfinite(score).all() or (score<.3).any():return done('weak_new_observation')
                hand_outputs.append(p)
            one,two=hand_outputs
            delta=np.linalg.norm(one-two,axis=1)/palm
            if delta.mean()>.12 or delta.max()>.35:return done('crop_disagreement')
            original=raw['xy_px'][i,start:start+21];ok=visible(original,w,h)
            if ok.any() and np.max(np.linalg.norm(one[ok]-original[ok],axis=1)/palm)>.35:return done('contradicts_observed_points')
            if i in [a-1,b+1] and np.mean(np.linalg.norm(one-original,axis=1)/palm)>.2:return done('bracket_disagreement')
            # Require visual content at the hand: rejects blank/covered controls;
            # this is not proof that every finger is visible.
            low=np.maximum(np.floor(one.min(0)-2),0).astype(int);high=np.minimum(np.ceil(one.max(0)+3),[w,h]).astype(int)
            patch=im[low[1]:high[1],low[0]:high[0]]
            if patch.size==0 or float(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).std())<4:return done('no_hand_image_detail')
            predictions[i]=(one,score)
        fixed={k:np.array(v,copy=True) for k,v in raw.items()}
        for i in range(a,b+1):
            missing=~visible(raw['xy_px'][i,start:start+21],w,h)
            p,score=predictions[i]
            fixed['xy_px'][i,start:start+21][missing]=p[missing]
            fixed['score'][i,start:start+21][missing]=score[missing]
            meta['replaced_points']+=int(missing.sum())
        if time.perf_counter()>deadline-.25:return done('no_recheck_budget')
        newer=infer(fixed,word)
        before={c['id']:c for c in initial['lesson_acceptance']['criteria']}
        after={c['id']:c for c in newer['lesson_acceptance']['criteria']}
        # Preserve both correct and incorrect established evidence, including its
        # localized spans. Unknown may become either supported OR difference.
        for cid,c in before.items():
            d=after.get(cid)
            if d is None:return done('criterion_contract_changed')
            if c['status']!='unknown' and c['status']!=d['status']:return done('existing_verdict_changed')
            if c['status']=='difference' and c.get('difference_spans')!=d.get('difference_spans'):return done('existing_error_location_changed')
        recovered=[cid for cid,c in before.items() if c['status']=='unknown' and after[cid]['status']!='unknown']
        if not recovered:return done('no_criterion_recovered')
        if time.perf_counter()>deadline:return done('total_time_budget')
        # Word recognition is deliberately kept on its original observation path.
        newer['tcn_top3']=copy.deepcopy(initial['tcn_top3'])
        meta.update(accepted=True,recovered_criteria=recovered,original_tcn_preserved=True)
        return done('accepted_reobservation',newer,fixed)
    except Exception as exc:
        meta['error']=type(exc).__name__+': '+str(exc)
        return done('retry_failed_keep_original')
