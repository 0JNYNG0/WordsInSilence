"""Application progression only. No model/threshold tuning or sign correctness claim."""
import math, secrets
from feedback_acceptance import feedback_gate

SUPPORTED = {'WORD1567':'조상','WORD2523':'이집트','WORD0739':'순종','WORD1133':'잘하다'}
TARGETS = [dict(id='SEN0354',label='안녕하세요',recognition=False,feedback=False,reason='AIHub 기준 · 다인 학습 자료 준비 필요'),dict(id='WORD0099',label='사과',recognition=False,feedback=False,reason='별도 후보는 미채택 · 세부 정답 준비 필요'),dict(id='WORD1290',label='감사',recognition=True,feedback=False,reason='기존 TCN 인식 가능 · 세부 피드백 준비 필요')]

class QuestProgress:
    def __init__(self): self.reset()
    def reset(self):
        self.generation=secrets.token_hex(8);self.steps=list(SUPPORTED);self.index=0;self.seen=set();self.completions=0
    def state(self):
        return dict(generation=self.generation,steps=[dict(id=w,label=SUPPORTED[w]) for w in self.steps],index=self.index,completed=self.index==len(self.steps),completion_events=self.completions,basis='TCN target match plus qualified observed lesson requirements; unknown completion cannot advance')
    def consume(self,attempt,result):
        outcome=dict(advanced=False,reason='no_match')
        aid=attempt['id']
        if aid in self.seen:return dict(advanced=False,reason='duplicate')
        self.seen.add(aid)
        if attempt['mode']!='quest' or attempt['generation']!=self.generation:return dict(advanced=False,reason='non_quest_or_stale')
        if self.index>=len(self.steps) or attempt['target']!=self.steps[self.index]:return dict(advanced=False,reason='stale_target')
        ranks=result.get('tcn_top3') or []
        if result.get('source')!='user_inference_only':return dict(advanced=False,reason='not_live_capture')
        if result.get('stop_reason')=='time_limit':return dict(advanced=False,reason='time_limit_retry')
        if len(result.get('timestamp_us',[]))<4 or not ranks:return dict(advanced=False,reason='insufficient_observation')
        p=ranks[0].get('probability')
        if not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:return dict(advanced=False,reason='invalid_score')
        if ranks[0].get('word')!=attempt['target']:return outcome
        feedback=feedback_gate(result,attempt['target'])
        if not feedback['accepted']:return dict(advanced=False,reason=feedback['reason'])
        self.index+=1
        if self.index==len(self.steps):self.completions+=1
        return dict(advanced=True,reason=feedback['reason'],completed=self.index==len(self.steps))
