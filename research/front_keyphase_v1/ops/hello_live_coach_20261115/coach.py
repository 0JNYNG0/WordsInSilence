"""Guided exercise with learned handshape AND existing learned action direction.

Not a newly trained SEN0354 recognizer or a full linguistic correctness verdict.
Each output assesses the center of seven ALREADY received frames (3-frame delay).
The model's bidirectional context is confined to those received frames. No future
clip normalization, reference timing, word score or target pose fills observations.
"""
from pathlib import Path
from collections import deque
from copy import deepcopy
import sys,time
import numpy as np
import torch
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'basic_demo_feedback_20261115'))
from shape_consensus import ShapeConsensus
sys.path.insert(0,str(O/'word100_rtmw_shared_input_20261108'))
import adapter

HAND={'L':list(range(91,112)),'R':list(range(112,133))}
FINGERS={'L':[list(range(k,k+4))for k in [96,100,104,108]],
         'R':[list(range(k,k+4))for k in [117,121,125,129]]}
FINGER_KO=['검지','중지','약지','새끼손가락']
HAND_KO={'L':'왼손','R':'오른손'}
BODY_PARTS={5:'left_shoulder',6:'right_shoulder',7:'left_elbow',8:'right_elbow',9:'left_body_wrist',10:'right_body_wrist'}
FINGER_PARTS=['index','middle','ring','little']
# Requirements belong to a movement stage, not to a word-name exception.
# Thumb landmarks are not inspected by the admitted long-finger checks.
STAGE_CONTRACT=(
    dict(kind='forearm_sweep',body=(5,6,7,9,10),shape={'R':0},hand_wrists=('R',)),
    dict(kind='paired_vertical_repeat',body=(5,6,7,8,9,10),shape={'L':1,'R':1},hand_wrists=('L','R')),
)
GAP_PAUSE_US=250_000
_DEFAULT_ACTION=object()

class Coach:
    def __init__(self,model=None,action=_DEFAULT_ACTION,stage_contract=None,gap_pause_us=GAP_PAUSE_US):
        torch.set_num_threads(2)
        self.model=model if model is not None else ShapeConsensus()
        if action is _DEFAULT_ACTION:
            from action_runtime import ActionObserver
            action=ActionObserver()
        # None is a comparison/test seam only. The live server always uses default.
        self.action=action
        self.stage_contract=deepcopy(STAGE_CONTRACT if stage_contract is None else stage_contract)
        if len(self.stage_contract)!=2 or [s['kind']for s in self.stage_contract]!=['forearm_sweep','paired_vertical_repeat']:
            raise ValueError('Expected forearm sweep and paired repeat stage contracts')
        self.gap_pause_us=int(gap_pause_us)
        if not 0<self.gap_pause_us<=300_000:raise ValueError('Gap pause must be bounded by 300 ms')
        self.reset()

    def reset(self):
        if self.action is not None:self.action.reset()
        self.action_output=None
        self.frames=deque(maxlen=7);self.last_ts=None;self.last_assessed=None
        self.dimensions=None
        self.stage_index=0;self.completed=[False,False];self.events=[]
        self.start_q=None;self.prev_q=None;self.max_q=None;self.start_ts=None;self.stroke_start=None
        self.turn='down';self.high=None;self.low=None;self.high_pair=None;self.low_pair=None;self.repeats=0
        self.good_shape_since=None;self.invalid_since=None
        self.max_q_ts=None;self.high_ts=None;self.low_ts=None
        self.sweep_observed_travel=0.;self.sweep_motion_start=None;self.sweep_endpoint_ready=True
        self.gap_since=None;self.gap_expired=False;self.last_good_observation_ts=None
        self.last_output=self._base('waiting','편 손을 반대쪽 팔 위에 올려 보세요.')
        return self.last_output

    def _base(self,phase,message):
        category='motion_uncertain'if phase=='action_uncertain'else'rule_unmet'if phase in ('waiting','preparing','correction')else'progress'
        return dict(stage_index=self.stage_index,completed=self.completed.copy(),phase=phase,message=message,
            checks=[],highlights=[],events=self.events.copy(),repeats=self.repeats,
            completion_scope='관측한 핵심 동작의 단계 연습',whole_word_correct=None,
            limitations=['엄지·정확한 손바닥 방향·실제 접촉은 이 시험에서 완전 검증하지 않습니다.'],
            model_role='학습된 손모양과 시간 합성곱·GRU 방향 관측 + 설명 기반 단계·반복 완료 규칙',
            learned_direction_axes=[2,0],diagnostic_only_direction_axes=[],
            assessed_timestamp_us=self.last_assessed,observation_lag_ms=None,
            diagnostic=dict(category=category,code=phase,message=message,held_progress=False))

    def _clear_pending(self):
        self.start_q=None;self.prev_q=None;self.max_q=None;self.start_ts=None
        self.stroke_start=None;self.high=None;self.low=None;self.high_pair=None;self.low_pair=None;self.turn='down';self.repeats=0
        self.max_q_ts=None;self.high_ts=None;self.low_ts=None
        self.sweep_observed_travel=0.;self.sweep_motion_start=None;self.sweep_endpoint_ready=True

    def _has_pending(self):
        return self.start_q is not None or self.high is not None or self.repeats>0

    def _diagnose(self,out,category,code,**details):
        out['diagnostic']=dict(category=category,code=code,message=out['message'],held_progress=False)
        out['diagnostic'].update(details)
        return out

    def _pause(self,t,category,code,message,missing_parts=None):
        """Hold confirmed progress briefly; missing time never creates movement."""
        if self.gap_since is None:
            self.gap_since=self.last_good_observation_ts if self.last_good_observation_ts is not None else t
        elapsed=max(0,int(t)-self.gap_since)
        reset_reason=None
        if elapsed>self.gap_pause_us and not self.gap_expired:
            self._clear_pending();self.gap_expired=True;reset_reason='observation_gap_exceeded'
        out=self._base('unknown',message)
        out['diagnostic']=dict(category=category,code=code,message=message,
            held_progress=bool(self._has_pending()and not self.gap_expired),gap_duration_ms=round(elapsed/1000,1))
        if missing_parts:out['diagnostic']['missing_parts']=missing_parts
        if reset_reason:out['diagnostic']['reset_reason']=reset_reason
        return out

    def _resume(self,t):
        recovering=self.gap_since is not None
        reset_reason=None
        if recovering and t-self.gap_since>self.gap_pause_us and not self.gap_expired:
            self._clear_pending();reset_reason='observation_gap_exceeded'
        self.gap_since=None;self.gap_expired=False;self.last_good_observation_ts=t
        return recovering,reset_reason

    def _required_observation(self,p,w,h,contract):
        valid=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
        missing=[BODY_PARTS.get(i,'body_'+str(i))for i in contract['body']if not valid[i]]
        for hand in contract['hand_wrists']:
            side='left'if hand=='L'else'right'
            if not valid[HAND[hand][0]]:missing.append(side+'_hand_wrist')
        for hand in contract['shape']:
            side='left'if hand=='L'else'right'
            for name,indices in zip(FINGER_PARTS,FINGERS[hand]):
                if not valid[indices].all():missing.append(side+'_'+name)
        sc=float(np.linalg.norm(p[5]-p[6]))
        if not np.isfinite(sc)or sc<=max(12,w*.02):missing.append('shoulder_scale')
        if missing:return sc,missing,'required_points_unavailable'
        for hand in contract['hand_wrists']:
            base=HAND[hand][0];pose=9 if hand=='L'else 10
            if np.linalg.norm(p[base]-p[pose])>.28*sc:
                missing.append(('left'if hand=='L'else'right')+'_hand_body_wrist_alignment')
        return sc,missing,'inconsistent_hand_anchor'if missing else None

    def _motion(self,start,t,axis,moving,desired):
        if self.action is None:
            return True,dict(test_baseline_without_action=True)
        if start is None:
            return False,dict(eligible=False,state=-1,reason='no_observed_start')
        evidence=self.action.movement(start,t,axis,moving,'L'if moving=='R'else'R')
        return bool(evidence['eligible']and evidence['state']==desired),evidence

    def update(self,xy,score,timestamp_us,width,height):
        t=int(timestamp_us);p=np.asarray(xy,np.float32);s=np.asarray(score,np.float32)
        if p.shape!=(133,2)or s.shape!=(133,):raise ValueError('Expected RTMW133 points/scores')
        if self.last_ts is not None and t<=self.last_ts:raise ValueError('Nonmonotonic timestamps')
        reset_reason=None
        if self.last_ts is not None and t-self.last_ts>self.gap_pause_us:reset_reason='capture_gap_exceeded'
        if self.dimensions is not None and self.dimensions!=(width,height):reset_reason='source_dimensions_changed'
        if reset_reason:
            self.frames.clear();self._clear_pending();self.last_assessed=None
            self.gap_since=None;self.gap_expired=False;self.last_good_observation_ts=None
        self.dimensions=(width,height)
        # Observe all received motion before any shape check can return early.
        if self.action is not None:
            self.action_output=self.action.update(p,s,t,int(width),int(height))
        self.last_ts=t;self.frames.append((p.copy(),s.copy(),t,int(width),int(height)))
        if len(self.frames)<7:
            self.last_output=self._base('observing','손과 팔을 확인하고 있어요.')
            self._diagnose(self.last_output,'observation_delay','warming_up')
            if reset_reason:self.last_output['diagnostic']['reset_reason']=reset_reason
            self.last_output['action_model']=self.action_output
            return self.last_output
        f=list(self.frames);p,s,at,w,h=f[3];self.last_assessed=at
        # No delayed assessment beyond 300 ms: slow/gapped capture cannot accumulate completion.
        if t-at>300000 or f[-1][2]-f[0][2]>600000:
            self._clear_pending();out=self._base('unknown','관측이 늦어졌어요. 손을 잠깐 보여 주세요.')
            self.gap_since=None;self.gap_expired=False;self.last_good_observation_ts=None
            self._diagnose(out,'observation_delay','assessment_too_old',reset_reason='assessment_too_old')
        else:
            begin=time.perf_counter()
            try:
                z=adapter.convert(np.stack([r[0]for r in f]),np.stack([r[1]for r in f]),
                    np.array([r[2]for r in f],np.int64),np.arange(7),w,h)
                shapes={hand:self.model.sequence(z['tcn280'],hand)for hand in ['L','R']}
                obs={hand:dict(state=int(v['states'][3]),fingers=v['fingers'][3].astype(int).tolist(),
                    available=v['available'][3].astype(bool).tolist(),
                    reason=str(v['reasons'][3])if 'reasons'in v else None,
                    finger_reasons=v['finger_reasons'][3].tolist()if 'finger_reasons'in v else [],
                    geometry_reasons=v['geometry_reasons'][3].tolist()if 'geometry_reasons'in v else [])for hand,v in shapes.items()}
                out=self.consume_observation(p,at,w,h,obs)
            except (ValueError,RuntimeError,adapter.PilotDataError) as e:
                # Coordinate normalization failure is uncertainty, never a semantic error.
                out=self._pause(at,'missing_keypoints','normalization_unavailable','손과 양쪽 어깨가 보이도록 해 주세요.')
                out['observation_issue']=type(e).__name__
            out['observer_ms']=round((time.perf_counter()-begin)*1000,2)
        out['assessed_timestamp_us']=at;out['observation_lag_ms']=round((t-at)/1000,1)
        out['action_model']=self.action_output
        if self.action_output is not None:
            self.action_output['admitted_direction_axes']=[2,0]
            self.action_output['diagnostic_only_direction_axes']=[]
            self.action_output['greeting_stage_training']=False
            if self.stage_index>=1:
                self.action_output['message']='학습한 모델로 두 손의 위아래 움직임을 확인하고 있어요.'
            else:
                self.action_output['message']='학습한 모델로 팔을 따라 움직인 방향을 확인하고 있어요.'
        out['assessed_xy']=[[float(x)if np.isfinite(x)else None for x in pt]for pt in p]
        self.last_output=out;return out

    def consume_observation(self,p,t,w,h,obs):
        """State transition seam for tests. Live calls ONLY use learned observations."""
        if self.stage_index>=2:
            out=self._base('complete','두 주요 동작을 확인했어요.')
            out['checks']=[dict(id='exercise',label='두 단계 연습',state='ok')];return out
        contract=self.stage_contract[self.stage_index]
        sc,missing,issue=self._required_observation(p,w,h,contract)
        if missing:
            return self._pause(t,'missing_keypoints',issue,'손과 팔이 잘 보이지 않아요. 다시 보여 주세요.',missing)
        if contract['kind']=='forearm_sweep'and float((p[9]-p[7])@(p[9]-p[7]))<(.15*sc)**2:
            return self._pause(t,'missing_keypoints','forearm_anchor_unavailable','반대쪽 팔이 잘 보이도록 해 주세요.',['left_forearm_axis'])
        unusable=[]
        for hand in contract['shape']:
            for j,reason in enumerate(obs.get(hand,{}).get('geometry_reasons',[])):
                if reason in ('palm_axis_degenerate','finger_chain_degenerate'):
                    unusable.append(hand+'_'+FINGER_PARTS[j])
        if unusable:
            out=self._pause(t,'missing_keypoints','hand_geometry_unusable',
                '손가락 관절을 구분하지 못했어요. 손이 잘 보이도록 잠깐 보여 주세요.',unusable)
            out['shape_diagnostic']={hand:dict(observation_reason=obs.get(hand,{}).get('reason'),
                finger_reasons=obs.get(hand,{}).get('finger_reasons',[]))for hand in contract['shape']}
            return out
        shape_states={};highlights=[];bad=[]
        for hand,expected in contract['shape'].items():
            item=obs.get(hand,dict(state=-1,fingers=[-1]*4));state=item['state'];shape_states[hand]='ok'if state==expected else'unknown'if state<0 else'wrong'
            for j,pred in enumerate(item['fingers']):
                desired=1 if expected==0 else 0
                hs='unknown'if pred<0 else'ok'if pred==desired else'wrong'
                highlights.append(dict(indices=FINGERS[hand][j],state=hs))
                if hs=='wrong':bad.append((hand,j,expected))
            # Known finger contradiction vetoes a coarse whole-hand acceptance.
            desired=1 if expected==0 else 0
            if any(v>=0 and v!=desired for v in item['fingers']):shape_states[hand]='wrong'
        checks=[dict(id=hand+'_shape',label=HAND_KO[hand]+(' 펴기'if expected==0 else' 구부리기'),state=shape_states[hand])for hand,expected in contract['shape'].items()]
        shape_ok=all(x=='ok'for x in shape_states.values())
        if not shape_ok:
            was_moving=self._has_pending()
            wrong=[hand for hand,state in shape_states.items()if state=='wrong']
            if wrong:
                self._clear_pending();self.gap_since=None;self.gap_expired=False;self.last_good_observation_ts=None
                if bad:
                    hand,j,expected=bad[0];message=f"{HAND_KO[hand]} {FINGER_KO[j]}를 {'펴'if expected==0 else'구부려'} 주세요."
                else:
                    hand=wrong[0];expected=contract['shape'][hand];message=f"{HAND_KO[hand]}을 {'펴'if expected==0 else'구부려'} 주세요."
                out=self._base('correction'if was_moving else'preparing',message)
                self._diagnose(out,'shape_mismatch','required_shape_mismatch',reset_reason='known_shape_mismatch')
                if not was_moving:
                    for check in checks:
                        if check['state']=='wrong':check['state']='pending'
                    for high in highlights:
                        if high['state']=='wrong':high['state']='pending'
            else:out=self._pause(t,'shape_uncertain','required_shape_uncertain','손가락 모양 판정이 불확실해요. 참고 모양을 잠깐 유지해 주세요.')
            details={}
            for hand,expected in contract['shape'].items():
                item=obs.get(hand,{});fingers=item.get('fingers',[-1]*4);available=item.get('available',[True]*4)
                desired=1 if expected==0 else 0
                details[hand]=dict(
                    observation_reason=item.get('reason'),finger_reasons=item.get('finger_reasons',[]),
                    unavailable_fingers=[name for name,ok in zip(FINGER_PARTS,available)if not ok],
                    unknown_fingers=[name for name,pred,ok in zip(FINGER_PARTS,fingers,available)if ok and pred<0],
                    contrary_fingers=[name for name,pred in zip(FINGER_PARTS,fingers)if pred>=0 and pred!=desired])
            out.update(checks=checks,highlights=highlights,shape_diagnostic=details);return out
        recovering,reset_reason=self._resume(t)
        if self.stage_index==0:
            out=self._sweep(p,t,sc,checks,highlights,recovering)
        else:out=self._repeat(p,t,sc,checks,highlights,recovering)
        if reset_reason:out['diagnostic']['reset_reason']=reset_reason
        out['hand_observation']=obs
        return out

    def _sweep(self,p,t,sc,checks,highlights,recovering=False):
        elbow=p[7];wrist=p[9];axis=wrist-elbow;den=float(axis@axis)
        if den<(.15*sc)**2:
            return self._pause(t,'missing_keypoints','forearm_anchor_unavailable','반대쪽 팔이 잘 보이도록 해 주세요.',['left_forearm_axis'])
        tip=p[[120,124,128,132]].mean(0)
        q=float((tip-elbow)@axis/den);gap=float(np.linalg.norm(tip-(elbow+q*axis))/sc)
        torso=(p[5]+p[6])/2;level=float((tip[1]-torso[1])/sc)
        on_arm=gap<=.30 and -.15<=q<=1.40 and -.1<level<1.35
        start=on_arm and q<=.55
        highlights.append(dict(indices=[7,9],state='ok'if on_arm else'pending'))
        phase='waiting';message='편 손을 반대쪽 팔 위쪽에 올려 주세요.';reset_reason=None;resumed=False
        if self.start_q is None:
            if start:
                self.start_q=q;self.start_ts=t;self.prev_q=q;self.max_q=q;self.max_q_ts=t
                self.sweep_motion_start=t;self.sweep_observed_travel=0.;self.sweep_endpoint_ready=True
                phase='moving';message='손을 편 채 손목 쪽으로 쓸어 주세요.'
            elif on_arm and q>.55:phase='preparing';message='손목보다 팔 위쪽에서 시작해 주세요.'
        elif not on_arm:
            self._clear_pending();reset_reason='forearm_position_unmet';message='팔을 따라 손목 쪽으로 이어서 움직여 주세요.'
        elif recovering:
            # Keep already observed travel, but never add a displacement through
            # the gap or query learned direction across that unobserved stroke.
            if self.max_q is not None and q<self.max_q-.18:
                self._clear_pending();phase='action_uncertain';reset_reason='recovery_position_reversed'
                message='손의 위치가 바뀌어 흐름을 다시 확인할게요. 팔 위쪽에서 시작해 주세요.'
            else:
                self.prev_q=q;self.max_q=max(self.max_q,q);self.max_q_ts=t;self.sweep_motion_start=t
                # An endpoint crossed while hidden must be observed again.
                self.sweep_endpoint_ready=q<.85
                phase='moving';resumed=True;message='손이 다시 보여요. 팔을 따라 움직임을 이어 주세요.'
        else:
            phase='moving';message='손을 편 채 손목 쪽으로 쓸어 주세요.'
            if self.max_q is not None and q<self.max_q-.18:
                supported,direction=self._motion(self.max_q_ts,t,2,'R',0)
                self._clear_pending();phase='correction'if supported else'action_uncertain'
                reset_reason='observed_reverse_motion'
                message='손목 쪽으로 쓸어 주세요. 방향을 바꿔 보세요.'if supported else'움직임의 방향을 확인하기 어려워요. 팔 위쪽에서 다시 시작해 주세요.'
            else:
                self.sweep_observed_travel+=max(0.,q-self.prev_q)
                if q<.85:self.sweep_endpoint_ready=True
                if q>=.85 and q-self.start_q>=.40 and self.sweep_observed_travel>=.40 and self.sweep_endpoint_ready:
                    supported,direction=self._motion(self.sweep_motion_start,t,2,'R',2)
                    if supported:
                        self.events.append(dict(stage=0,kind='sweep_observed',start_us=self.start_ts,end_us=t,
                            evidence='learned open moving long fingers + learned temporal direction + observed forearm travel and endpoint',
                            action_direction=direction,start_fraction=round(self.start_q,3),end_fraction=round(q,3),
                            observed_forward_travel=round(self.sweep_observed_travel,3)))
                        self.completed[0]=True;self.stage_index=1;self._clear_pending()
                        phase='stage_complete';message='첫 동작을 확인했어요. 이제 두 손을 주먹으로 만들어 주세요.'
                    else:
                        phase='action_uncertain';message='손목 쪽으로 움직인 흐름을 확인하고 있어요.'
            self.prev_q=q
            if self.start_q is not None and q>=self.max_q:self.max_q=q;self.max_q_ts=t
        checks+= [dict(id='start',label='팔 위쪽에서 시작',state='ok'if self.start_q is not None or self.completed[0]else'pending'),
                  dict(id='sweep',label='손목 쪽으로 이동',state='ok'if self.completed[0]else'pending')]
        if phase=='correction':highlights.append(dict(indices=[7,9,112],state='wrong'))
        elif phase=='action_uncertain':highlights.append(dict(indices=[7,9,112],state='unknown'))
        out=self._base(phase,message);out.update(checks=checks,highlights=highlights,geometry=dict(forearm_fraction=round(q,3),forearm_gap=round(gap,3),observed_forward_travel=round(self.sweep_observed_travel,3)))
        if resumed:self._diagnose(out,'progress','observation_resumed',held_progress=True)
        if reset_reason:out['diagnostic']['reset_reason']=reset_reason
        if 'direction'in locals():out['action_direction_check']=direction
        return out

    def _repeat(self,p,t,sc,checks,highlights,recovering=False):
        torso=(p[5]+p[6])/2;wr=(p[[91,112]]-torso)/sc
        aligned=abs(float(wr[0,1]-wr[1,1]))<=.22
        separated=.30<=float(np.linalg.norm(wr[0]-wr[1]))<=1.50
        level=bool((wr[:,1]>-.15).all()and(wr[:,1]<.90).all())
        if not(aligned and separated and level):
            self._clear_pending();out=self._base('correction','두 주먹을 가슴 앞에 나란히 놓아 주세요.')
            self._diagnose(out,'rule_unmet','paired_position_unmet',reset_reason='paired_position_unmet')
            highlights.append(dict(indices=[7,9,8,10],state='wrong'))
        else:
            y=float(wr[:,1].mean());phase='moving';message='두 주먹을 함께 위아래로 움직여 주세요.'
            pair=wr[:,1].copy()
            if self.high is None:
                self.high=y;self.low=y;self.high_pair=pair.copy();self.low_pair=pair.copy();self.stroke_start=t;self.high_ts=t;self.low_ts=t
            if recovering:
                # Preserve only completed strokes and the required next direction.
                # A hidden upward return cannot arm a second downward stroke.
                self.high=y;self.low=y;self.high_pair=pair.copy();self.low_pair=pair.copy()
                self.stroke_start=t;self.high_ts=t;self.low_ts=t
                message='두 손이 다시 보여요. 위아래 움직임을 이어 주세요.'
            elif self.turn=='down':
                # Use the latest genuinely observed extremum: a hold has zero
                # displacement and must not pad the subsequent movement query.
                if y<=self.high:self.high=y;self.high_pair=pair.copy();self.high_ts=t;self.stroke_start=t
                # Each downward stroke needs measured amplitude; a held pose cannot count.
                if y-self.high>=.060 and float((pair-self.high_pair).min())>=.045:
                    pair_direction=[self._motion(self.high_ts,t,0,hand,0)for hand in ['L','R']]
                    if all(ok for ok,_ in pair_direction):
                        self.repeats+=1;self.low=y;self.low_pair=pair.copy();self.low_ts=t;self.turn='up'
                        self.events.append(dict(stage=1,kind='downstroke_observed',start_us=self.stroke_start,end_us=t,number=self.repeats,
                            action_direction=[e for _,e in pair_direction],
                            evidence='learned flexed long fingers + learned direction for BOTH wrists + paired downward displacement'))
                        message=f'{self.repeats}번 움직임을 확인했어요.'
                    else:phase='action_uncertain';message='내리는 방향을 아직 확인하지 못했어요. 두 주먹을 함께 아래로 움직여 주세요.'
                if self.repeats>=2:
                    self.completed[1]=True;self.stage_index=2;phase='complete';message='두 주요 동작을 확인했어요.'
            else:
                if y>=self.low:self.low=y;self.low_pair=pair.copy();self.low_ts=t
                # Separate upward return is mandatory before another downward stroke.
                if self.low-y>=.035 and float((self.low_pair-pair).min())>=.025:
                    pair_direction=[self._motion(self.low_ts,t,0,hand,2)for hand in ['L','R']]
                    if all(ok for ok,_ in pair_direction):
                        self.turn='down';self.high=y;self.high_pair=pair.copy();self.high_ts=t;self.stroke_start=t
                    else:phase='action_uncertain';message='올리는 방향을 아직 확인하지 못했어요. 두 주먹을 함께 위로 움직여 주세요.'
            out=self._base(phase,message);out['geometry']=dict(mean_wrist_height=round(y,3),next_direction=self.turn)
            if recovering:self._diagnose(out,'progress','observation_resumed',held_progress=bool(self.repeats))
            if 'pair_direction'in locals():out['action_direction_check']=[e for _,e in pair_direction]
            if phase=='action_uncertain':highlights.append(dict(indices=[7,9,8,10],state='unknown'))
        checks += [dict(id='paired_position',label='가슴 앞에 나란히',state='ok'if aligned and separated and level else'wrong'),
                   dict(id='repeat',label=f'위아래 움직임 {self.repeats}/2',state='ok'if self.completed[1]else'pending')]
        out.update(checks=checks,highlights=highlights)
        return out
