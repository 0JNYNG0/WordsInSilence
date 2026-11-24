"""Bounded RAM-only operational reasons, never stores pixels or coordinates."""
from collections import deque
import copy

CATEGORIES=('missing_keypoints','shape_uncertain','shape_mismatch','rule_unmet',
            'motion_uncertain','observation_delay','progress')

class DiagnosticLedger:
    def __init__(self,history_limit=5,event_limit=24):
        self.history=deque(maxlen=history_limit);self.event_limit=event_limit;self.current=None
        self.first_ts=None;self.last_stamp=None;self.last_code=None;self.last_reset=None

    def begin(self,source,generation):
        if self.current is not None:
            self.end('replaced')
            self.history.append(copy.deepcopy(self.current))
        self.current=dict(source=source,generation=int(generation),ended=False,end_reason=None,
            counts={k:0 for k in CATEGORIES},code_counts={},observed_updates=0,held_updates=0,
            resets=0,last_reason=None,recent_events=[],metric_summary={},stage_counts={},
            stage_first_seen_ms={},subcause_counts={},motion_subcause_scope='per hand query, not per frame or error')
        self.first_ts=None;self.last_stamp=None;self.last_code=None;self.last_reset=None

    def observe(self,observation,metrics):
        if self.current is None or self.current['ended']:return
        d=observation.get('diagnostic') or {}
        ts=observation.get('assessed_timestamp_us');basis='assessed'
        if not isinstance(ts,(int,float)):
            # Capture/dimension resets deliberately discard the assessment buffer.
            # Retain their reason at the received-time metric, without any pixels.
            elapsed=metrics.get('elapsed_seconds')
            if not d.get('reset_reason') or not isinstance(elapsed,(int,float)):return
            ts=round(elapsed*1_000_000);basis='received_metric'
        stamp=(basis,ts)
        if stamp==self.last_stamp:return
        self.last_stamp=stamp
        if self.first_ts is None:self.first_ts=ts
        category=d.get('category')
        if category not in CATEGORIES:return
        # Explicit allowlist: never retain model tensors, point arrays, images or arbitrary input.
        code=str(d.get('code') or category)[:100]
        message=str(d.get('message') or '')[:180]
        reason=dict(category=category,code=code,message=message,
            stage=int(observation.get('stage_index',0)),held_progress=d.get('held_progress') is True,
            reset_reason=str(d.get('reset_reason'))[:100] if d.get('reset_reason') else None,
            relative_ms=round((ts-self.first_ts)/1000,1),time_basis=basis)
        if d.get('missing_parts'):
            reason['missing_parts']=[str(p)[:60] for p in d['missing_parts'][:16]]
        c=self.current;c['observed_updates']+=1;c['counts'][category]+=1
        c['code_counts'][code]=c['code_counts'].get(code,0)+1
        stage=str(reason['stage'])
        c['stage_first_seen_ms'].setdefault(stage,reason['relative_ms'])
        stage_count=c['stage_counts'].setdefault(stage,{k:0 for k in CATEGORIES})
        stage_count[category]+=1
        # Scalar operational causes only. Never retain probabilities, trajectories,
        # model features, coordinates or raw input. Old sessions cannot be backfilled.
        def add_cause(key):
            c['subcause_counts'][key]=c['subcause_counts'].get(key,0)+1
        if category=='motion_uncertain':
            queries=observation.get('action_direction_check',[])
            if isinstance(queries,dict):queries=[queries]
            for query in queries[:2]:
                if not isinstance(query,dict):continue
                hand=query.get('moving');axis=query.get('axis')
                if hand not in ('L','R') or axis not in range(5):continue
                why=query.get('reason')
                allowed={'model_uncertain','insufficient_observed_geometry','no_current_prefix',
                         'endpoint_outside_received_history','interval_not_yet_assessed','no_observed_start'}
                if why in allowed:sub=why
                elif not query.get('eligible'):sub='query_unavailable'
                elif query.get('state')==1:sub='model_hold'
                elif query.get('state') in (0,2):sub='model_direction_available'
                else:sub='model_unresolved'
                add_cause(f'stage{stage}.motion.{hand}.axis{axis}.{sub}')
        if category in ('shape_uncertain','shape_mismatch','missing_keypoints'):
            hand_states=observation.get('shape_diagnostic',{})
            for hand,detail in hand_states.items():
                if hand not in ('L','R') or not isinstance(detail,dict):continue
                allowed=('model_disagreement','model_confidence_insufficient','geometry_unavailable',
                         'points_unavailable','palm_axis_degenerate','finger_chain_degenerate','model_decided')
                cause=detail.get('observation_reason')
                if cause in allowed:add_cause(f'stage{stage}.shape.{hand}.hand.{cause}')
                for finger,cause in zip(('index','middle','ring','little'),detail.get('finger_reasons',[])):
                    if cause in allowed:add_cause(f'stage{stage}.shape.{hand}.{finger}.{cause}')
                for finger in detail.get('unavailable_fingers',[]):
                    if finger in ('index','middle','ring','little'):add_cause(f'stage{stage}.shape.{hand}.{finger}.unavailable')
                for finger in detail.get('unknown_fingers',[]):
                    if finger in ('index','middle','ring','little'):add_cause(f'stage{stage}.shape.{hand}.{finger}.model_unresolved')
                for finger in detail.get('contrary_fingers',[]):
                    if finger in ('index','middle','ring','little'):add_cause(f'stage{stage}.shape.{hand}.{finger}.predicted_contrary')
        if reason['held_progress']:c['held_updates']+=1
        # Count transition events, not every displayed frame of the same reset reason.
        reset=reason['reset_reason']
        if reset and reset!=self.last_reset:c['resets']+=1
        self.last_reset=reset
        signature=(category,code,reason['stage'],reason['held_progress'],reset)
        if signature!=self.last_code:
            c['recent_events'].append(reason.copy());c['recent_events']=c['recent_events'][-self.event_limit:]
        self.last_code=signature;c['last_reason']=reason
        for src,dest in [('displayed_observation_age_ms','max_observation_age_ms'),('rtmw_ms','max_rtmw_ms'),('observation_ms','max_observation_ms')]:
            value=metrics.get(src)
            if isinstance(value,(int,float)) and 0<=value<1e7:
                c['metric_summary'][dest]=max(c['metric_summary'].get(dest,0),round(float(value),1))

    def end(self,reason):
        if self.current is not None and not self.current['ended']:
            self.current['ended']=True;self.current['end_reason']=str(reason)[:40]

    def snapshot(self):
        return copy.deepcopy(dict(session=self.current,recent_sessions=list(self.history),
            retention='RAM only; current and five previous sessions; cleared by server restart',
            scope='Observation update counts, not semantic error counts or accuracy',
            reset_count_scope='Distinct consecutive runs of reset conditions, including pre-start conditions; not lost-progress count',
            video_saved=False,coordinates_saved=False,training_use=False))
