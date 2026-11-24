/* Presentation timing only. Never changes model observations or stage completion. */
(function(root){
  'use strict';
  const DEFAULTS=Object.freeze({candidateMs:350,minVisibleMs:1400,uncertainMs:600,correctionFallbackMs:1200});
  class CoachMessagePresenter {
    constructor(options={}){this.options={...DEFAULTS,...options};this.reset();}
    reset(){this.context=null;this.current=null;this.candidate=null;this.uncertainSince=null;this.correctionSince=null;this.lastSeq=-1;}
    commit(message,phase,now){
      if(!this.current || this.current.message!==message || this.current.phase!==phase)
        this.current={message,phase,shownAt:now};
      this.candidate=null;
      return {...this.current};
    }
    update(input,now){
      const context=String(input.generation ?? 0)+'|'+String(input.source ?? '');
      if(context!==this.context){this.reset();this.context=context;}
      // App also rejects stale HTTP responses; this keeps delayed text from reviving them.
      const seq=Number.isFinite(input.seq)?input.seq:this.lastSeq;
      if(seq<this.lastSeq)return this.current?{...this.current}:{message:input.initialMessage,phase:'waiting'};
      this.lastSeq=seq;
      const state=input.state || 'ready';
      if(['ready','preparing','opening','countdown','error'].includes(state)){
        this.candidate=null;this.uncertainSince=null;this.correctionSince=null;
        const messages={ready:input.initialMessage,preparing:'분석기를 준비하고 있어요.',opening:'카메라를 연결하고 있어요.',countdown:'손을 내리고 편하게 준비해 주세요.',error:'분석을 이어가지 못했어요. 아래 오류 안내를 확인해 주세요.'};
        return this.commit(messages[state],state,now);
      }
      const phase=input.phase || 'waiting',message=input.message || input.initialMessage;
      if(state==='done'&&phase!=='complete'){
        this.uncertainSince=null;this.correctionSince=null;
        return this.commit('이번 관측이 끝났어요. 확인하지 못한 동작은 다시 연습해 주세요.','ended',now);
      }
      if(phase==='stage_complete'||phase==='complete'){
        this.uncertainSince=null;this.correctionSince=null;
        return this.commit(message,phase,now);
      }
      if(!this.current)this.current={message:input.initialMessage,phase:'waiting',shownAt:now-this.options.minVisibleMs};
      if(phase==='unknown'||phase==='action_uncertain'){
        this.correctionSince=null;
        if(this.uncertainSince===null)this.uncertainSince=now;
        this.candidate=null;
        if(now-this.uncertainSince>=this.options.uncertainMs){
          const safeMessage='움직임을 조금 더 살펴보고 있어요. 손과 팔을 잘 보여 주세요.';
          return this.commit(safeMessage,phase,now);
        }
        return {...this.current};
      }
      this.uncertainSince=null;
      if(phase==='correction'){
        if(this.correctionSince===null)this.correctionSince=now;
      }else this.correctionSince=null;
      if(this.current.message===message&&this.current.phase===phase){this.candidate=null;return {...this.current};}
      const key=phase+'|'+message;
      if(this.candidate?.key!==key)this.candidate={key,message,phase,since:now};
      if(now-this.candidate.since>=this.options.candidateMs&&now-this.current.shownAt>=this.options.minVisibleMs)
        return this.commit(message,phase,now);
      // Persistent corrections with changing detail still need a useful, steady cue.
      if(this.correctionSince!==null&&now-this.correctionSince>=this.options.correctionFallbackMs&&now-this.current.shownAt>=this.options.minVisibleMs)
        return this.commit('표시된 손·팔 부분을 참고 동작에 맞춰 주세요.','correction',now);
      return {...this.current};
    }
  }
  const DIAGNOSTIC_CATEGORIES=Object.freeze({
    missing_keypoints:{label:'관절 확인 중',countLabel:'관절 누락',tone:'neutral'},
    shape_uncertain:{label:'손모양 확인 중',countLabel:'손모양 불확실',tone:'neutral'},
    shape_mismatch:{label:'손모양 교정 필요',countLabel:'손모양 조건 미충족',tone:'wrong'},
    rule_unmet:{label:'진행 조건 확인',countLabel:'진행 조건 미충족',tone:'neutral'},
    motion_uncertain:{label:'움직임 확인 중',countLabel:'움직임 불확실',tone:'neutral'},
    observation_delay:{label:'관측 지연',countLabel:'관측 지연',tone:'neutral'},
    progress:{label:'동작 관측 중',countLabel:'진행 관측',tone:'neutral'}
  });
  const count=value=>Number.isFinite(value)&&value>=0?Math.floor(value):0;
  function diagnosticSummary(diagnostics={}){
    diagnostics=diagnostics || {};
    const session=diagnostics.session || null;
    const reasonRows=Object.entries(DIAGNOSTIC_CATEGORIES).filter(([key])=>key!=='progress');
    const rowsFor=s=>reasonRows.map(([category,info])=>({category,label:info.countLabel,count:count(s?.counts?.[category])}));
    const previous=Array.isArray(diagnostics.recent_sessions)?diagnostics.recent_sessions.at(-1):null;
    const describe=s=>s?`총 관측 ${count(s.observed_updates)}회 · 진행 관측 ${count(s.counts?.progress)}회 · 진행 유지 ${count(s.held_updates)}회 · 재시작 조건 ${count(s.resets)}회`:'아직 관측 기록이 없어요.';
    return {rows:rowsFor(session),current:describe(session),lastReason:session?.last_reason?.message || '관측을 시작하면 확인하지 못한 이유가 나타나요.',
      previous:previous?`${previous.source==='public_replay'?'공개 영상':'카메라'} · ${describe(previous)} · ${rowsFor(previous).map(row=>`${row.label} ${row.count}회`).join(' · ')}`:'아직 끝난 연습이 없어요.'};
  }
  function coachTone(observation={},state='ready'){
    const category=observation.diagnostic?.category;
    // Actual unmet conditions stay visible, even when another item is uncertain.
    if(state==='error')return 'neutral';
    if(category==='shape_mismatch'||observation.diagnostic?.confirmed_error===true||(category==='rule_unmet'&&observation.phase==='correction')||observation.checks?.some(c=>c.state==='wrong'))return 'wrong';
    if(['missing_keypoints','shape_uncertain','motion_uncertain','observation_delay','rule_unmet'].includes(category)||['unknown','action_uncertain'].includes(observation.phase)||observation.checks?.some(c=>c.state==='unknown'))return 'neutral';
    return observation.completed?.length===2&&observation.completed.every(v=>v===true)?'ok':'neutral';
  }
  class DiagnosticPresenter {
    constructor(options={}){this.options={candidateMs:350,minVisibleMs:1400,...options};this.reset();}
    reset(){this.context=null;this.generation=-1;this.lastSeq=-1;this.current=null;this.candidate=null;}
    update(input,now){
      const generation=Number.isFinite(input.generation)?input.generation:0;
      if(generation<this.generation)return this.current?{...this.current}:{hidden:true,tone:'neutral'};
      const context=String(generation)+'|'+String(input.source ?? '');
      if(context!==this.context){this.reset();this.context=context;this.generation=generation;}
      const seq=Number.isFinite(input.seq)?input.seq:this.lastSeq;
      if(seq<this.lastSeq)return this.current?{...this.current}:{hidden:true,tone:'neutral'};
      this.lastSeq=seq;
      if(!['live','replay'].includes(input.state)||['complete','stage_complete'].includes(input.phase)){
        this.current={hidden:true,tone:'neutral',shownAt:now};this.candidate=null;return {...this.current};
      }
      const diagnostic=input.diagnostic || {};
      const category=Object.hasOwn(DIAGNOSTIC_CATEGORIES,diagnostic.category)?diagnostic.category:'unknown';
      const info=DIAGNOSTIC_CATEGORIES[category] || {label:'관측 확인 중',tone:'neutral'};
      const tone=category==='rule_unmet'&&(input.phase==='correction'||diagnostic.confirmed_error===true)?'wrong':info.tone;
      const next={hidden:false,category,label:info.label,tone,held:diagnostic.held_progress===true,shownAt:now};
      // Never keep an old correction label over a newly uncertain observation.
      if(!this.current||this.current.hidden||(this.current.tone==='wrong'&&next.tone==='neutral')){
        this.current=next;this.candidate=null;return {...this.current};
      }
      this.current.held=next.held;
      if(this.current.category===category&&this.current.tone===tone){this.candidate=null;return {...this.current};}
      if(this.candidate?.category!==category||this.candidate?.tone!==tone)this.candidate={...next,since:now};
      if(now-this.candidate.since>=this.options.candidateMs&&now-this.current.shownAt>=this.options.minVisibleMs){this.current=next;this.candidate=null;}
      return {...this.current};
    }
  }
  function completedActionObservation(model,observation){
    return observation?.phase==='complete'&&observation.completed?.length===2&&observation.completed.every(v=>v===true)
      &&model?.status==='observed'&&model.calls>0&&Array.isArray(model.admitted_direction_axes)
      &&model.admitted_direction_axes.includes(2)&&model.admitted_direction_axes.includes(0);
  }
  const exports={CoachMessagePresenter,DEFAULTS,completedActionObservation,DiagnosticPresenter,DIAGNOSTIC_CATEGORIES,diagnosticSummary,coachTone};
  if(typeof module!=='undefined'&&module.exports)module.exports=exports;
  else Object.assign(root,exports);
})(typeof window!=='undefined'?window:globalThis);
