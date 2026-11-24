'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {completedActionObservation}=require('./presenter.js');
const model=Object.freeze({status:'observed',calls:125,admitted_direction_axes:[2,0]});
const completed=Object.freeze({phase:'complete',completed:[true,true]});
const cases=[
  ['confirmed_two_stage_completion',model,completed,true],
  ['unfinished_end',model,{phase:'ended',completed:[true,false]},false],
  ['moving_after_one_stage',model,{phase:'moving',completed:[true,false]},false],
  ['complete_text_without_two_confirmed_stages',model,{phase:'complete',completed:[true,false]},false],
  ['uncertain_model',{...model,status:'uncertain'},completed,false],
  ['model_never_called',{...model,calls:0},completed,false],
  ['only_one_admitted_direction',{...model,admitted_direction_axes:[2]},completed,false],
  ['missing_observation',undefined,undefined,false]
];
const rows=cases.map(([name,m,o,expected])=>{assert.equal(completedActionObservation(m,o),expected);return {name,passed:true};});
assert.equal(model.calls,125);assert.deepEqual(completed.completed,[true,true]);
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const coach=html.match(/<section class="coach"[\s\S]*?<\/section>/)[0];
const details=html.match(/<details class="scope-details"[\s\S]*?<\/details>/)[0];
assert.ok(!coach.includes('actionObservation'));assert.ok(!coach.includes('actionModelDetail'));
assert.ok(details.includes('actionObservation'));assert.ok(details.includes('actionModelDetail'));
assert.ok(!details.match(/^<details[^>]*\sopen(?:\s|>|=)/));
assert.ok(coach.includes('coachMessage'));assert.ok(coach.includes('id="checks"'));
rows.push({name:'technical_model_status_in_closed_details_only',passed:true},{name:'main_coaching_message_and_current_checks_preserved',passed:true});
console.log(JSON.stringify({passed:true,presentation_only:true,model_result_changed:false,checks:rows},null,2));
