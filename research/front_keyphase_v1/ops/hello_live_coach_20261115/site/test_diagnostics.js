'use strict';
// Presentation contract only; never opens devices or runs a model.
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {DiagnosticPresenter,diagnosticSummary,coachTone}=require('./presenter.js');
const rows=[];
function test(name,fn){fn();rows.push({name,passed:true});}
const base={state:'live',generation:1,source:'camera',seq:1,phase:'unknown',diagnostic:{category:'missing_keypoints',held_progress:true}};
test('reason_categories_are_stable_across_brief_changes',()=>{
  const p=new DiagnosticPresenter();assert.equal(p.update(base,0).category,'missing_keypoints');
  const shape={...base,seq:2,diagnostic:{category:'shape_uncertain'}};
  p.update(shape,500);assert.equal(p.update(shape,850).category,'missing_keypoints');
  assert.equal(p.update(shape,1400).category,'shape_uncertain');
});
test('detailed_frame_messages_do_not_change_category_hint',()=>{
  const p=new DiagnosticPresenter();
  for(let i=0;i<20;i++)assert.equal(p.update({...base,seq:i,diagnostic:{...base.diagnostic,message:'different '+i}},i*200).label,'관절 확인 중');
});
test('uncertain_observations_never_use_correction_tone',()=>{
  for(const category of ['missing_keypoints','shape_uncertain','motion_uncertain','observation_delay','unrecognized']){
    const p=new DiagnosticPresenter();assert.equal(p.update({...base,diagnostic:{category}},0).tone,'neutral');
    assert.equal(coachTone({phase:'unknown',diagnostic:{category}}),'neutral');
  }
});
test('confirmed_shape_error_and_rule_correction_stay_distinct',()=>{
  for(const category of ['shape_mismatch','rule_unmet']){
    const p=new DiagnosticPresenter();assert.equal(p.update({...base,phase:'correction',diagnostic:{category}},0).tone,'wrong');
    assert.equal(coachTone({phase:'correction',diagnostic:{category},checks:[{state:'unknown'}]}),'wrong');
  }
});
test('unmet_rule_without_confirmed_correction_is_neutral',()=>{
  const p=new DiagnosticPresenter();assert.equal(p.update({...base,phase:'waiting',diagnostic:{category:'rule_unmet'}},0).tone,'neutral');
  assert.equal(coachTone({phase:'waiting',diagnostic:{category:'rule_unmet'}}),'neutral');
  assert.equal(coachTone({phase:'waiting',completed:[true,true],diagnostic:{category:'rule_unmet'}}),'neutral');
});
test('uncertainty_immediately_removes_old_correction_label',()=>{
  const p=new DiagnosticPresenter();p.update({...base,phase:'correction',diagnostic:{category:'shape_mismatch'}},0);
  assert.equal(p.update({...base,seq:2},50).tone,'neutral');
  assert.equal(p.update({...base,seq:2},50).category,'missing_keypoints');
});
test('hold_hint_clears_without_wait_when_server_releases_hold',()=>{
  const p=new DiagnosticPresenter();assert.equal(p.update(base,0).held,true);
  assert.equal(p.update({...base,seq:2,diagnostic:{category:'missing_keypoints',held_progress:false}},50).held,false);
});
test('stale_sequence_and_generation_cannot_restore_old_reason',()=>{
  const p=new DiagnosticPresenter();p.update({...base,generation:3,seq:8,diagnostic:{category:'shape_uncertain'}},0);
  assert.equal(p.update({...base,generation:3,seq:7},2000).category,'shape_uncertain');
  assert.equal(p.update({...base,generation:2,seq:90},2000).category,'shape_uncertain');
});
test('new_generation_source_or_reset_clear_previous_hold',()=>{
  for(const changed of [{generation:2},{source:'public_replay'}]){
    const p=new DiagnosticPresenter();p.update(base,0);
    const out=p.update({...base,...changed,seq:0,diagnostic:{category:'progress'}},1);
    assert.equal(out.category,'progress');assert.equal(out.held,false);
  }
  const p=new DiagnosticPresenter();p.update(base,0);p.reset();assert.equal(p.update({...base,diagnostic:{}},1).held,false);
});
test('ready_end_and_completion_clear_hint_immediately',()=>{
  for(const changed of [{state:'ready'},{state:'done'},{state:'error'},{phase:'stage_complete'},{phase:'complete'}]){
    const p=new DiagnosticPresenter();p.update(base,0);assert.equal(p.update({...base,...changed,seq:2},1).hidden,true);
  }
});
test('uncertainty_and_error_cannot_be_promoted_to_green_completion',()=>{
  assert.equal(coachTone({completed:[true,true],phase:'unknown',diagnostic:{category:'shape_uncertain'}}),'neutral');
  assert.equal(coachTone({completed:[true,true],phase:'correction',diagnostic:{category:'rule_unmet'}}),'wrong');
});
test('summary_uses_current_and_latest_completed_session_counts_only',()=>{
  const summary=diagnosticSummary({session:{observed_updates:9,counts:{missing_keypoints:3,progress:6},held_updates:2,resets:1,last_reason:{message:'손 관절을 확인하고 있어요.'}},recent_sessions:[{source:'camera',observed_updates:98},{source:'public_replay',observed_updates:12,counts:{shape_uncertain:5}}]});
  assert.equal(summary.rows.length,6);assert.equal(summary.rows[0].count,3);
  assert.match(summary.current,/총 관측 9회/);assert.match(summary.current,/진행 유지 2회/);
  assert.match(summary.current,/재시작 조건 1회/);assert.ok(!summary.current.includes('진행 초기화'));
  assert.match(summary.previous,/공개 영상/);assert.match(summary.previous,/총 관측 12회/);assert.match(summary.previous,/손모양 불확실 5회/);assert.ok(!summary.previous.includes('98'));
});
test('missing_or_malformed_summary_has_safe_empty_counts',()=>{
  assert.equal(diagnosticSummary(null).rows.every(row=>row.count===0),true);
  assert.equal(diagnosticSummary({session:{counts:{missing_keypoints:-2,shape_uncertain:'10'}}}).rows.every(row=>row.count===0),true);
});
test('detail_labels_do_not_claim_error_frequency_or_accuracy',()=>{
  const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
  const coach=html.match(/<section class="coach"[\s\S]*?<\/section>/)[0];
  const details=html.match(/<details class="scope-details"[\s\S]*?<\/details>/)[0];
  assert.ok(coach.includes('diagnosticHint'));assert.ok(!coach.includes('diagnosticCounts'));assert.ok(!coach.includes('diagnosticLastReason'));
  assert.ok(details.includes('지금 확인 못한 이유'));assert.ok(details.includes('관측 횟수이며 동작 오류 횟수나 정확도가 아닙니다.'));
  assert.ok(details.includes('서버 메모리에만'));assert.ok(!/^<details[^>]*\sopen(?:\s|>|=)/.test(details));
  assert.equal((coach.match(/aria-live=/g)||[]).length,1,'Only the stable main instruction is a live announcement');
});
console.log(JSON.stringify({passed:true,scope:'Synthetic presenter and markup tests only; not camera or recognition accuracy.',checks:rows},null,2));
