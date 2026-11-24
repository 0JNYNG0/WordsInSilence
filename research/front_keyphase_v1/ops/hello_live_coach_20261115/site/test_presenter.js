'use strict';
const assert=require('node:assert/strict');
const {CoachMessagePresenter,DEFAULTS}=require('./presenter.js');
const rows=[];
function test(name,fn){fn();rows.push({name,passed:true});}
const base={state:'live',generation:1,source:'public_replay',seq:1,initialMessage:'준비',phase:'moving',message:'손목 쪽으로 쓸어 주세요.'};
test('ordinary_instruction_requires_stable_candidate',()=>{
  const p=new CoachMessagePresenter();
  assert.equal(p.update(base,0).message,'준비');
  assert.equal(p.update(base,349).message,'준비');
  assert.equal(p.update(base,350).message,base.message);
});
test('brief_error_does_not_flip_main_sentence',()=>{
  const p=new CoachMessagePresenter();p.update(base,0);p.update(base,350);
  assert.equal(p.update({...base,seq:2,phase:'correction',message:'검지를 펴 주세요.'},500).message,base.message);
  assert.equal(p.update({...base,seq:3},600).message,base.message);
});
test('persistent_correction_respects_minimum_reading_time',()=>{
  const p=new CoachMessagePresenter();p.update(base,0);p.update(base,350);
  const wrong={...base,seq:2,phase:'correction',message:'검지를 펴 주세요.'};
  p.update(wrong,500);
  assert.equal(p.update(wrong,850).message,base.message);
  assert.equal(p.update(wrong,1750).message,wrong.message);
});
test('brief_missing_observation_does_not_replace_text',()=>{
  const p=new CoachMessagePresenter();p.update(base,0);p.update(base,350);
  const unknown={...base,seq:2,phase:'unknown',message:'다른 프레임 안내'};
  assert.equal(p.update(unknown,400).message,base.message);
  assert.equal(p.update(unknown,999).message,base.message);
  assert.equal(p.update(unknown,1000).phase,'unknown');
});
test('prolonged_action_uncertainty_is_not_called_wrong',()=>{
  const p=new CoachMessagePresenter();const unknown={...base,phase:'action_uncertain'};
  p.update(unknown,0);const shown=p.update(unknown,600);
  assert.equal(shown.phase,'action_uncertain');
  assert.match(shown.message,/살펴보고/);
});
test('stage_and_exercise_completion_immediate',()=>{
  const p=new CoachMessagePresenter();p.update(base,0);
  assert.equal(p.update({...base,seq:2,phase:'stage_complete',message:'첫 동작 확인'},1).message,'첫 동작 확인');
  assert.equal(p.update({...base,seq:3,phase:'complete',message:'두 동작 확인'},2).message,'두 동작 확인');
});
test('generation_reset_never_keeps_old_success',()=>{
  const p=new CoachMessagePresenter();p.update({...base,phase:'complete',message:'완료'},0);
  assert.equal(p.update({...base,generation:2,seq:0},1).message,'준비');
});
test('source_switch_never_keeps_old_public_result',()=>{
  const p=new CoachMessagePresenter();p.update({...base,phase:'complete',message:'완료'},0);
  assert.equal(p.update({...base,source:'camera',seq:0},1).message,'준비');
});
test('ready_countdown_service_error_are_immediate',()=>{
  for(const state of ['ready','countdown','error']){
    const p=new CoachMessagePresenter();p.update({...base,phase:'complete',message:'완료'},0);
    const out=p.update({...base,seq:2,state},1);
    assert.equal(out.phase,state);assert.notEqual(out.message,'완료');
  }
});
test('stale_sequence_cannot_revive_old_message',()=>{
  const p=new CoachMessagePresenter();p.update({...base,seq:10,phase:'complete',message:'완료'},0);
  assert.equal(p.update({...base,seq:9,phase:'correction',message:'이전 오류'},10).message,'완료');
});
test('unfinished_end_does_not_keep_movement_instruction',()=>{
  const p=new CoachMessagePresenter();p.update(base,0);p.update(base,350);
  assert.equal(p.update({...base,state:'done',seq:2},351).phase,'ended');
});
test('display_layer_does_not_mutate_model_input',()=>{
  const p=new CoachMessagePresenter();const input=Object.freeze({...base});
  p.update(input,0);p.update(input,500);assert.deepEqual(input,base);
});
test('alternating_finger_corrections_wait_for_stable_instruction',()=>{
  const p=new CoachMessagePresenter();p.update(base,0);p.update(base,350);
  for(let i=0;i<20;i++){
    const out=p.update({...base,seq:i+2,phase:'correction',message:i%2?'검지를 펴 주세요.':'중지를 펴 주세요.'},500+i*200);
    assert.equal(out.message,i<7?base.message:'표시된 손·팔 부분을 참고 동작에 맞춰 주세요.');
  }
});
console.log(JSON.stringify({passed:true,presentation_defaults:DEFAULTS,model_thresholds_changed:false,checks:rows},null,2));
