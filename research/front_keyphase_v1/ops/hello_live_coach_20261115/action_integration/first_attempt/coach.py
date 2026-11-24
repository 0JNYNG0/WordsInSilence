"""Guided exercise with learned handshape AND existing learned action direction.

Not a newly trained SEN0354 recognizer or a full linguistic correctness verdict.
Each output assesses the center of seven ALREADY received frames (3-frame delay).
The model's bidirectional context is confined to those received frames. No future
clip normalization, reference timing, word score or target pose fills observations.
"""
from pathlib import Path
from collections import deque
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
_DEFAULT_ACTION=object()

class Coach:
    def __init__(self,model=None,action=_DEFAULT_ACTION):
        torch.set_num_threads(2)
        self.model=model if model is not None else ShapeConsensus()
        if action is _DEFAULT_ACTION:
            from action_runtime import ActionObserver
            action=ActionObserver()
        # None is a comparison/test seam only. The live server always uses default.
        self.action=action
        self.reset()

    def reset(self):
        if self.action is not None:self.action.reset()
        self.action_output=None
        self.frames=deque(maxlen=7);self.last_ts=None;self.last_assessed=None
        self.stage_index=0;self.completed=[False,False];self.events=[]
        self.start_q=None;self.prev_q=None;self.max_q=None;self.start_ts=None;self.stroke_start=None
        self.turn='down';self.high=None;self.low=None;self.high_pair=None;self.low_pair=None;self.repeats=0
        self.good_shape_since=None;self.invalid_since=None
        self.max_q_ts=None;self.high_ts=None;self.low_ts=None
        self.last_output=self._base('waiting','편 손을 반대쪽 팔 위에 올려 보세요.')
        return self.last_output

    def _base(self,phase,message):
        return dict(stage_index=self.stage_index,completed=self.completed.copy(),phase=phase,message=message,
            checks=[],highlights=[],events=self.events.copy(),repeats=self.repeats,
            completion_scope='관측한 핵심 동작의 단계 연습',whole_word_correct=None,
            limitations=['엄지·정확한 손바닥 방향·실제 접촉은 이 시험에서 완전 검증하지 않습니다.'],
            model_role='학습된 손모양 + 기존 시간 합성곱·GRU 방향 관측 + 설명 기반 단계·완료 규칙',
            assessed_timestamp_us=self.last_assessed,observation_lag_ms=None)

    def _clear_pending(self):
        self.start_q=None;self.prev_q=None;self.max_q=None;self.start_ts=None
        self.stroke_start=None;self.high=None;self.low=None;self.high_pair=None;self.low_pair=None;self.turn='down';self.repeats=0
        self.max_q_ts=None;self.high_ts=None;self.low_ts=None

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
        if self.last_ts is not None and t-self.last_ts>300000:
            self.frames.clear();self._clear_pending()
        # Observe all received motion before any shape check can return early.
        if self.action is not None:
            self.action_output=self.action.update(p,s,t,int(width),int(height))
        self.last_ts=t;self.frames.append((p.copy(),s.copy(),t,int(width),int(height)))
        if len(self.frames)<7:
            self.last_output=self._base('observing','손과 팔을 확인하고 있어요.')
            self.last_output['action_model']=self.action_output
            return self.last_output
        f=list(self.frames);p,s,at,w,h=f[3];self.last_assessed=at
        # No delayed assessment beyond 300 ms: slow/gapped capture cannot accumulate completion.
        if t-at>300000 or f[-1][2]-f[0][2]>600000:
            self._clear_pending();out=self._base('unknown','관측이 늦어졌어요. 손을 잠깐 보여 주세요.')
        else:
            begin=time.perf_counter()
            try:
                z=adapter.convert(np.stack([r[0]for r in f]),np.stack([r[1]for r in f]),
                    np.array([r[2]for r in f],np.int64),np.arange(7),w,h)
                shapes={hand:self.model.sequence(z['tcn280'],hand)for hand in ['L','R']}
                obs={hand:dict(state=int(v['states'][3]),fingers=v['fingers'][3].astype(int).tolist())for hand,v in shapes.items()}
                out=self.consume_observation(p,at,w,h,obs)
            except (ValueError,RuntimeError,adapter.PilotDataError) as e:
                # Coordinate normalization failure is uncertainty, never a semantic error.
                self._clear_pending();out=self._base('unknown','손과 양쪽 어깨가 보이도록 해 주세요.')
                out['observation_issue']=type(e).__name__
            out['observer_ms']=round((time.perf_counter()-begin)*1000,2)
        out['assessed_timestamp_us']=at;out['observation_lag_ms']=round((t-at)/1000,1)
        out['action_model']=self.action_output
        out['assessed_xy']=[[float(x)if np.isfinite(x)else None for x in pt]for pt in p]
        self.last_output=out;return out

    def consume_observation(self,p,t,w,h,obs):
        """State transition seam for tests. Live calls ONLY use learned observations."""
        valid=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
        required=[5,6,7,8,9,10]+HAND['L']+HAND['R']
        sc=float(np.linalg.norm(p[5]-p[6]));quality=bool(valid[required].all()and sc>max(12,w*.02))
        if quality:
            for base,pose in [(91,9),(112,10)]:
                if np.linalg.norm(p[base]-p[pose])>.28*sc:quality=False
        if not quality:
            self._clear_pending();return self._base('unknown','손과 팔이 잘 보이지 않아요. 다시 보여 주세요.')
        if self.stage_index>=2:
            out=self._base('complete','두 주요 동작을 확인했어요.')
            out['checks']=[dict(id='exercise',label='두 단계 연습',state='ok')];return out
        expected=0 if self.stage_index==0 else 1
        shape_states={};highlights=[];bad=[]
        for hand in ['L','R']:
            item=obs[hand];state=item['state'];shape_states[hand]='ok'if state==expected else'unknown'if state<0 else'wrong'
            for j,pred in enumerate(item['fingers']):
                desired=1 if expected==0 else 0
                hs='unknown'if pred<0 else'ok'if pred==desired else'wrong'
                highlights.append(dict(indices=FINGERS[hand][j],state=hs))
                if hs=='wrong':bad.append((hand,j))
            # Known finger contradiction vetoes a coarse whole-hand acceptance.
            desired=1 if expected==0 else 0
            if any(v>=0 and v!=desired for v in item['fingers']):shape_states[hand]='wrong'
        checks=[dict(id=hand+'_shape',label=('왼손'if hand=='L'else'오른손')+(' 펴기'if expected==0 else' 구부리기'),state=shape_states[hand])for hand in ['L','R']]
        shape_ok=all(x=='ok'for x in shape_states.values())
        if not shape_ok:
            was_moving=self.start_q is not None if self.stage_index==0 else self.high is not None
            self._clear_pending()
            if bad:
                hand,j=bad[0];message=f"{'왼손'if hand=='L'else'오른손'} {FINGER_KO[j]}를 {'펴'if expected==0 else'구부려'} 주세요."
                out=self._base('correction'if was_moving else'preparing',message)
                if not was_moving:
                    for check in checks:
                        if check['state']=='wrong':check['state']='pending'
                    for high in highlights:
                        if high['state']=='wrong':high['state']='pending'
            else:out=self._base('unknown','손가락 모양을 확인하고 있어요. 손이 잘 보이게 해 주세요.')
            out.update(checks=checks,highlights=highlights);return out
        if self.stage_index==0:
            out=self._sweep(p,t,sc,checks,highlights)
        else:out=self._repeat(p,t,sc,checks,highlights)
        out['hand_observation']=obs
        return out

    def _sweep(self,p,t,sc,checks,highlights):
        elbow=p[7];wrist=p[9];axis=wrist-elbow;den=float(axis@axis)
        if den<(.15*sc)**2:
            self._clear_pending();return self._base('unknown','반대쪽 팔이 잘 보이도록 해 주세요.')
        tip=p[[120,124,128,132]].mean(0)
        q=float((tip-elbow)@axis/den);gap=float(np.linalg.norm(tip-(elbow+q*axis))/sc)
        torso=(p[5]+p[6])/2;level=float((tip[1]-torso[1])/sc)
        on_arm=gap<=.30 and -.15<=q<=1.40 and -.1<level<1.35
        start=on_arm and q<=.55
        highlights.append(dict(indices=[7,9],state='ok'if on_arm else'pending'))
        phase='waiting';message='편 손을 반대쪽 팔 위쪽에 올려 주세요.'
        if self.start_q is None:
            if start:
                self.start_q=q;self.start_ts=t;self.prev_q=q;self.max_q=q;self.max_q_ts=t;phase='moving';message='손을 편 채 손목 쪽으로 쓸어 주세요.'
            elif on_arm and q>.55:phase='preparing';message='손목보다 팔 위쪽에서 시작해 주세요.'
        elif not on_arm:
            self._clear_pending();message='팔을 따라 손목 쪽으로 이어서 움직여 주세요.'
        else:
            phase='moving';message='손을 편 채 손목 쪽으로 쓸어 주세요.'
            if self.max_q is not None and q<self.max_q-.18:
                supported,direction=self._motion(self.max_q_ts,t,2,'R',0)
                self._clear_pending();phase='correction'if supported else'action_uncertain'
                message='손목 쪽으로 쓸어 주세요. 방향을 바꿔 보세요.'if supported else'움직임의 방향을 확인하기 어려워요. 팔 위쪽에서 다시 시작해 주세요.'
            elif q>=.85 and q-self.start_q>=.40:
                supported,direction=self._motion(self.start_ts,t,2,'R',2)
                if supported:
                    self.events.append(dict(stage=0,kind='sweep_observed',start_us=self.start_ts,end_us=t,
                        evidence='learned open long fingers + learned temporal direction + projected forearm endpoints',
                        action_direction=direction,start_fraction=round(self.start_q,3),end_fraction=round(q,3)))
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
        out=self._base(phase,message);out.update(checks=checks,highlights=highlights,geometry=dict(forearm_fraction=round(q,3),forearm_gap=round(gap,3)))
        if 'direction'in locals():out['action_direction_check']=direction
        return out

    def _repeat(self,p,t,sc,checks,highlights):
        torso=(p[5]+p[6])/2;wr=(p[[91,112]]-torso)/sc
        aligned=abs(float(wr[0,1]-wr[1,1]))<=.22
        separated=.30<=float(np.linalg.norm(wr[0]-wr[1]))<=1.50
        level=bool((wr[:,1]>-.15).all()and(wr[:,1]<.90).all())
        if not(aligned and separated and level):
            self._clear_pending();out=self._base('correction','두 주먹을 가슴 앞에 나란히 놓아 주세요.')
            highlights.append(dict(indices=[7,9,8,10],state='wrong'))
        else:
            y=float(wr[:,1].mean());phase='moving';message='두 주먹을 함께 위아래로 움직여 주세요.'
            pair=wr[:,1].copy()
            if self.high is None:
                self.high=y;self.low=y;self.high_pair=pair.copy();self.low_pair=pair.copy();self.stroke_start=t;self.high_ts=t;self.low_ts=t
            if self.turn=='down':
                if y<self.high:self.high=y;self.high_pair=pair.copy();self.high_ts=t;self.stroke_start=t
                # Each downward stroke needs measured amplitude; a held pose cannot count.
                if y-self.high>=.060 and float((pair-self.high_pair).min())>=.045:
                    pair_direction=[self._motion(self.high_ts,t,0,hand,0)for hand in ['L','R']]
                    if all(ok for ok,_ in pair_direction):
                        self.repeats+=1;self.low=y;self.low_pair=pair.copy();self.low_ts=t;self.turn='up'
                        self.events.append(dict(stage=1,kind='downstroke_observed',start_us=self.stroke_start,end_us=t,number=self.repeats,
                            action_direction=[e for _,e in pair_direction],
                            evidence='learned flexed long fingers + learned direction for BOTH wrists + paired downward displacement'))
                        message=f'{self.repeats}번 움직임을 확인했어요.'
                    else:phase='action_uncertain';message='두 손이 함께 내려가는 흐름을 확인하고 있어요.'
                if self.repeats>=2:
                    self.completed[1]=True;self.stage_index=2;phase='complete';message='두 주요 동작을 확인했어요.'
            else:
                if y>self.low:self.low=y;self.low_pair=pair.copy();self.low_ts=t
                # Separate upward return is mandatory before another downward stroke.
                if self.low-y>=.035 and float((self.low_pair-pair).min())>=.025:
                    pair_direction=[self._motion(self.low_ts,t,0,hand,2)for hand in ['L','R']]
                    if all(ok for ok,_ in pair_direction):
                        self.turn='down';self.high=y;self.high_pair=pair.copy();self.high_ts=t;self.stroke_start=t
                    else:phase='action_uncertain';message='두 손을 함께 올리는 흐름을 확인하고 있어요.'
            out=self._base(phase,message);out['geometry']=dict(mean_wrist_height=round(y,3),next_direction=self.turn)
            if 'pair_direction'in locals():out['action_direction_check']=[e for _,e in pair_direction]
            if phase=='action_uncertain':highlights.append(dict(indices=[7,9,8,10],state='unknown'))
        checks += [dict(id='paired_position',label='가슴 앞에 나란히',state='ok'if aligned and separated and level else'wrong'),
                   dict(id='repeat',label=f'위아래 움직임 {self.repeats}/2',state='ok'if self.completed[1]else'pending')]
        out.update(checks=checks,highlights=highlights)
        return out
