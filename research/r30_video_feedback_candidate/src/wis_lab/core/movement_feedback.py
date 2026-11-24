"""Observed movement evidence, separate from lexical features and execution correctness.

Reversal hysteresis/run rules follow the reviewed pilot motion_events_v3, using its frozen
0.05 shoulder-width / 0.25s defaults. This module uses the original HAND-wrist trajectory,
never switches between pose and hand detectors, and never fills a missing sample.
Direction changes are NOT repetition counts. Preparation/recovery may be in the visibility
span; partial counts are neither lower nor upper bounds on an unobserved true movement.
"""
from __future__ import annotations
import numpy as np
from .features import GROUP_ORDER, GROUPS

SCHEMA = 'observed_movement_v1'
TURN_AMPLITUDE = 0.05
MAX_GAP_SECONDS = 0.25
MAX_RATE_GAP_SECONDS = 0.2
MIN_RUN_FRAMES = 3
MIN_OBSERVED_FRAMES = 8
MIN_OBSERVED_FRACTION = 0.5


def movement_source(features, present, timestamp_us) -> dict:
    """Keep full-rate movement and separate validity of wrist and fine-hand changes.

    A zero rate is produced only from two observed identical inputs. Unknown shape never
    becomes zero, and missing fingers do not erase a measurable wrist velocity.
    Adds metadata only; all legacy arrays and masks are left untouched.
    """
    offsets, cursor = {}, 0
    for g in GROUP_ORDER:
        offsets[g] = slice(cursor, cursor + GROUPS[g][0])
        cursor += GROUPS[g][0]
    ts = np.asarray(timestamp_us, dtype=np.int64)
    dt = np.diff(ts) / 1e6
    rate_time_ok = (dt > 0) & (dt <= MAX_RATE_GAP_SECONDS)
    hands = {}
    for side in ('L', 'R'):
        pos = features[:, offsets['handpos_'+side]]
        shape = features[:, offsets['shape_'+side]]
        pos_ok = present[:, GROUP_ORDER.index('handpos_'+side)] & np.isfinite(pos).all(axis=1)
        shape_ok = present[:, GROUP_ORDER.index('shape_'+side)] & np.isfinite(shape).all(axis=1)
        wrist_speed = [None] * len(ts)
        shape_rate = [None] * len(ts)
        for i in range(1, len(ts)):
            if not rate_time_ok[i-1]:
                continue
            if pos_ok[i-1] and pos_ok[i]:
                wrist_speed[i] = float(np.linalg.norm(pos[i]-pos[i-1]) / dt[i-1])
            if shape_ok[i-1] and shape_ok[i]:
                shape_rate[i] = float(np.linalg.norm(shape[i]-shape[i-1]) / dt[i-1])
        hands[side] = {
            'wrist_xy': [p.tolist() if ok else None for p,ok in zip(pos,pos_ok)],
            'wrist_speed_sw_per_second': wrist_speed,
            'shape_rate_hand_local_per_second': shape_rate,
        }
    return {'schema':SCHEMA, 'timestamp_us':ts.tolist(), 'hands':hands,
            'coordinate_source':'mediapipe_hand_wrist_body_normalized',
            'sampling':'original_visibility_span_before_DTW_sampling'}


def _runs(valid, clock):
    idx = np.flatnonzero(valid)
    if not len(idx):
        return []
    delta = np.diff(clock[idx])
    cuts = np.flatnonzero((np.diff(idx)!=1) | (delta>MAX_GAP_SECONDS) | (delta<=0)) + 1
    return [r for r in np.split(idx,cuts) if len(r)]


def _turns(y):
    turns, direction = 0, 0
    lo = hi = ext = float(y[0])
    for v in map(float,y[1:]):
        if direction == 0:
            lo,hi = min(lo,v),max(hi,v)
            if hi-v >= TURN_AMPLITUDE:
                direction,ext = -1,v
            elif v-lo >= TURN_AMPLITUDE:
                direction,ext = 1,v
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext-v >= TURN_AMPLITUDE:
                turns,direction,ext = turns+1,-1,v
        else:
            if v < ext:
                ext = v
            elif v-ext >= TURN_AMPLITUDE:
                turns,direction,ext = turns+1,1,v
    return turns


def _rates(values, total_frames):
    known = [v for v in values[1:] if v is not None and np.isfinite(v)]
    possible = max(total_frames-1,0)
    return {'observed_steps':len(known), 'possible_steps':possible,
            'unknown_steps':possible-len(known),
            'mean':float(np.mean(known)) if known else None,
            'status':'OBSERVED' if known else 'UNAVAILABLE',
            'completeness':'COMPLETE' if known and len(known)==possible else 'PARTIAL'}


def describe(source) -> dict:
    if not isinstance(source,dict) or source.get('schema') != SCHEMA:
        return {'status':'UNAVAILABLE', 'reason':'original movement observation not retained'}
    ts = np.asarray(source['timestamp_us'],dtype=np.int64)
    n = len(ts)
    clock = (ts-ts[0])/1e6 if n else np.array([],dtype=float)
    hands = {}
    for side in ('L','R'):
        raw = source['hands'][side]
        xy = np.array([p if p is not None else [np.nan,np.nan]
                       for p in raw['wrist_xy']],dtype=float).reshape(n,2)
        valid = np.isfinite(xy).all(axis=1)
        runs = _runs(valid,clock)
        countable = [r for r in runs if len(r)>=MIN_RUN_FRAMES]
        observed = int(valid.sum())
        usable = sum(map(len,countable))
        fraction = observed/n if n else 0
        enough = observed>=MIN_OBSERVED_FRAMES and fraction>=MIN_OBSERVED_FRACTION and usable>=MIN_OBSERVED_FRAMES
        complete = len(runs)==1 and observed==n
        hands[side] = {
            'status':'OBSERVED' if enough else 'INSUFFICIENT_OBSERVATION',
            'completeness':('COMPLETE' if complete else 'PARTIAL') if enough else 'UNCOUNTABLE',
            'observed_frames':observed,'total_frames':n,'observed_fraction':fraction,
            'runs':len(runs),'countable_runs':len(countable),
            'observed_seconds':float(sum(clock[r[-1]]-clock[r[0]] for r in runs)),
            'horizontal_turns':sum(_turns(xy[r,0]) for r in countable) if enough else None,
            'vertical_turns':sum(_turns(xy[r,1]) for r in countable) if enough else None,
            'wrist_speed':_rates(raw['wrist_speed_sw_per_second'],n),
            'shape_change':_rates(raw['shape_rate_hand_local_per_second'],n),
        }
    return {'status':'AVAILABLE','hands':hands,'source':source['coordinate_source'],
            'scope':source['sampling']}


def compare_sources(query_source, reference_source) -> dict:
    q,r = describe(query_source),describe(reference_source)
    return {'schema':SCHEMA,'status':'AVAILABLE' if q['status']==r['status']=='AVAILABLE' else 'UNAVAILABLE',
            'query':q,'reference':r,
            'settings':{'turn_amplitude_shoulder_widths':TURN_AMPLITUDE,
                        'max_run_gap_seconds':MAX_GAP_SECONDS,'max_rate_gap_seconds':MAX_RATE_GAP_SECONDS},
            'execution_assessment':'UNCONFIGURED','accepted':None,
            'caveats':['Direction reversals are not repetitions.',
                       'Partial counts are not bounds on a true total; do not use as an error/pass rule.',
                       'Visibility span may contain preparation and recovery.',
                       'Speed differences and different units are descriptive, not a correctness score.']}
