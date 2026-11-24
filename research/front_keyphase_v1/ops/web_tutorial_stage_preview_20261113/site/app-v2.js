import {StageRenderer,model,I} from './renderer.js';
import {Hologram} from './hologram.js';
import {Quest} from './quest.js';
const $=s=>document.querySelector(s),clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x)},mix=(a,b,t)=>a+(b-a)*t;
let scene='intro',busy=false,loaded=false,reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
let raf,last=performance.now(),typingTimer=0,typing=false,fullText='',disposed=false;
let robotX=0,robotY=1.48,robotZ=0,robotScale=.65,robotYaw=0,zoom=0,targetY=1.15,dark=0;
let npcVisible=false,npcReveal=0,npcYaw=0,gesture=false,gestureTime=0,transitionMode='off',transitionP=0;
let renderer,robot,npc,optics;
const events=[];const record=event=>{events.push({event,time:performance.now()});if(events.length>100)events.shift();};
const quest=new Quest(['인사','사과','5'],detail=>{
  record('quest-complete:'+detail.source);
  window.dispatchEvent(new CustomEvent('quest:complete',{detail}));
  $('#event-log').textContent='미리보기: 퀘스트 완료 이벤트 1회 발생';
  say('고마워요! 사과 다섯 개, 잘 받았어요.','NPC','실전 대화');
});
function say(text,speaker='MOA',role='당신의 수어 연습 친구'){
  clearInterval(typingTimer);fullText=text;typing=true;$('#speaker').textContent=speaker;$('#speaker-role').textContent=role;$('#line').textContent='';
  const chars=Array.from(text);let i=0;typingTimer=setInterval(()=>{i+=reduced?3:1;$('#line').textContent=chars.slice(0,i).join('');if(i>=chars.length)finishText();},reduced?15:39);
}
function finishText(){clearInterval(typingTimer);$('#line').textContent=fullText;typing=false;}
function button(text){$('#next-label').textContent=text;}
function label(type,title,caption){$('#room-type').textContent=type;$('#room-title').textContent=title;$('#room-caption').textContent=caption;}
function setScene(name,nav=name){scene=name;document.body.dataset.scene=name;$('.steps').querySelectorAll('button').forEach(b=>b.classList.toggle('selected',b.dataset.scene===nav));record(name);}
function animate(ms,fn){return new Promise(resolve=>{const start=performance.now();function tick(now){if(disposed){resolve();return;}const p=clamp((now-start)/ms);fn(p);if(p<1)requestAnimationFrame(tick);else resolve();}requestAnimationFrame(tick);});}
function lock(value){busy=value;for(const el of document.querySelectorAll('#next,.steps button,#reset,#gesture,#rotate-left,#rotate-right,#zoom'))el.disabled=value;}
function hidePanels(){for(const id of ['practice-heading','quest-panel','self-view','preview-controls','avatar-controls'])$('#'+id).hidden=true;$('#quest-success').hidden=true;}
function cleanFx(){transitionMode='off';transitionP=0;$('#npc-scan').style.opacity=0;optics?.clear();}
function neutralRobot(){robotX=0;robotY=1.48;robotZ=0;robotScale=.65;robotYaw=0;}
function stopGesture(){gesture=false;gestureTime=0;npcYaw=0;$('#gesture').textContent='동작 시험 보기';$('#gesture-note').textContent='공개 ‘조상’ 동작 적용 시험 · 교육용 검수 전';$('#zoom').textContent='＋';}
function intro(){
  if(busy)return;quest.reset();hidePanels();cleanFx();stopGesture();neutralRobot();npcVisible=false;dark=0;zoom=0;targetY=1.15;
  document.body.classList.remove('dark','focus-scene');$('.room-label').hidden=false;$('#dialogue').hidden=false;
  setScene('intro');label('TUTORIAL ROOM','처음 만나는, 새로운 대화','말 대신 손으로, 함께 시작해요.');say('안녕! 나는 모아야. 여기서는 우리, 손으로 이야기해 볼까?');button('함께 시작하기');$('#status').textContent='연출 미리보기 · 실제 분석 미연결';
}
async function openPractice(){
  if(busy||!loaded)return;quest.reset();finishText();lock(true);hidePanels();stopGesture();npcVisible=false;neutralRobot();zoom=0;targetY=1.15;dark=0;
  document.body.classList.remove('focus-scene','dark');$('.room-label').hidden=false;$('#dialogue').hidden=false;
  setScene('projecting','practice');label('HOLOGRAM PROJECTOR','모아가 공간을 열고 있어요.','눈빛을 따라, 새로운 연습 공간으로.');say('좋아. 내가 연습할 공간을 열어 줄게!');button('연습실을 여는 중');
  await animate(reduced?100:700,()=>{});finishText();transitionMode='enter';
  await animate(reduced?350:3000,p=>{
    transitionP=p;dark=ease((p-.16)/.60);robotYaw=Math.sin(p*Math.PI)*.035;
    const move=ease((p-.68)/.32);robotX=mix(0,1.13,move);robotZ=mix(0,-.85,move);robotY=mix(1.48,1.31,move);robotScale=mix(.65,.32,move);zoom=mix(0,1.7,move);targetY=mix(1.15,1.49,move);
    if(p>.43)document.body.classList.add('dark');
    if(p>.73){npcVisible=true;npcReveal=ease((p-.73)/.27);}
  });
  cleanFx();dark=1;npcVisible=true;npcReveal=1;robotYaw=0;document.body.classList.add('dark','focus-scene');
  $('.room-label').hidden=true;$('#practice-heading').hidden=false;$('#self-view').hidden=false;$('#avatar-controls').hidden=false;
  setScene('practice');say('가운데 캐릭터를 함께 살펴보자. 네 모습은 아래 작은 화면에서 볼 수 있어.');button('연습 마치기');$('#status').textContent='캐릭터 동작 시험 · 카메라/분석 연결 전';lock(false);
}
async function returnRoom(){
  if(busy)return;lock(true);stopGesture();hidePanels();$('#dialogue').hidden=true;$('.room-label').hidden=true;document.body.classList.remove('focus-scene');
  setScene('deprojecting','practice');transitionMode='exit';
  const sx=robotX,sy=robotY,sz=robotZ,ss=robotScale,cam=zoom,ty=targetY;
  await animate(reduced?240:1250,p=>{
    transitionP=p;const e=ease(p);dark=1-e;npcReveal=1-ease(p/.65);robotX=mix(sx,0,e);robotY=mix(sy,1.48,e);robotZ=mix(sz,0,e);robotScale=mix(ss,.65,e);zoom=mix(cam,0,e);targetY=mix(ty,1.15,e);
    if(p>.45)document.body.classList.remove('dark');
  });
  cleanFx();dark=0;npcVisible=false;neutralRobot();zoom=0;targetY=1.15;$('.room-label').hidden=false;$('#dialogue').hidden=false;
  setScene('return','intro');label('READY FOR A CONVERSATION','이제, 직접 이야기해 볼까요?','인사하고, 필요한 것을 부탁해요.');say('이번에는 실제 대화처럼 해 보자. 인사하고, 사과 다섯 개를 부탁하는 거야.');button('실전 대화 시작');$('#status').textContent='실전 퀘스트 연출 준비';lock(false);
}
function renderQuest(){
  const nodes=[...$('#quest-track').querySelectorAll('[data-step]')],links=[...$('#quest-track').querySelectorAll('.connector')];
  nodes.forEach((el,i)=>{const done=i<quest.index;el.classList.toggle('done',done);el.classList.toggle('current',!quest.completed&&i===quest.index);el.querySelector('.quest-circle').textContent=done?'✓':String(i+1);if(i===quest.index&&!quest.completed)el.setAttribute('aria-current','step');else el.removeAttribute('aria-current');});
  links.forEach((el,i)=>el.classList.toggle('filled',i<quest.index));
  $('#quest-prompt').textContent=quest.completed?'모든 부탁을 전했어요.':['먼저 인사를 건네 보세요.','이제 필요한 물건, 사과를 표현해 보세요.','마지막으로 수량 5를 표현해 보세요.'][quest.index];
  $('#quest-prompt').hidden=quest.completed;
  $('#simulate-success').disabled=quest.completed||!!quest.pending;
  $('#simulate-success').textContent=quest.completed?'모든 단계 완료':`‘${quest.words[quest.index]}’ 성공 연출 확인 →`;
  $('#quest-success').hidden=!quest.completed;
}
function resetQuest(){quest.reset();$('#event-log').textContent='';renderQuest();if(scene==='quest')say('안녕하세요! 어떤 물건이 얼마나 필요한가요?','NPC','실전 대화');}
async function startQuest(){
  if(busy||!loaded)return;lock(true);finishText();hidePanels();cleanFx();stopGesture();quest.reset();npcVisible=false;neutralRobot();zoom=0;targetY=1.15;dark=0;
  document.body.classList.remove('dark','focus-scene');$('.room-label').hidden=false;$('#dialogue').hidden=false;
  setScene('quest-arriving','quest');label('FIRST QUEST','연습에서, 대화로','모아는 곁에서 지켜볼게요.');say('인사, 사과, 다섯. 위에 있는 순서를 따라 천천히 이야기해 봐.');button('친구를 부르는 중');
  await animate(reduced?150:1100,p=>{const e=ease(p);robotX=mix(0,1.13,e);robotZ=mix(0,-.85,e);robotY=mix(1.48,1.31,e);robotScale=mix(.65,.32,e);zoom=mix(0,1.7,e);targetY=mix(1.15,1.53,e);});
  npcVisible=true;npcReveal=0;
  await animate(reduced?220:1250,p=>{npcReveal=ease(p);$('#npc-scan').style.opacity=reduced?0:Math.sin(p*Math.PI)*.5;$('#npc-scan').style.top=(73-55*p)+'%';});
  cleanFx();npcReveal=1;document.body.classList.add('focus-scene');$('.room-label').hidden=true;
  for(const id of ['quest-panel','self-view','preview-controls'])$('#'+id).hidden=false;
  setScene('quest');resetQuest();button('연습으로 돌아가기');$('#status').textContent='실전 UI 미리보기 · 성공은 시험 버튼으로 확인';lock(false);
}
async function simulateSuccess(){
  if(scene!=='quest'||busy)return;
  const ticket=quest.accept(quest.words[quest.index]);if(!ticket)return;
  $('#simulate-success').disabled=true;$('#quest-reset').disabled=true;
  const nodes=[...$('#quest-track').querySelectorAll('[data-step]')],links=[...$('#quest-track').querySelectorAll('.connector')];
  nodes[ticket.index].classList.remove('current');nodes[ticket.index].classList.add('done');nodes[ticket.index].querySelector('.quest-circle').textContent='✓';
  record('preview-word-success:'+ticket.word);
  await animate(reduced?90:250,()=>{});
  if(ticket!==quest.pending||scene!=='quest'){ $('#quest-reset').disabled=false;return; }
  if(links[ticket.index])links[ticket.index].classList.add('filled');
  await animate(reduced?150:760,()=>{});
  if(ticket!==quest.pending||scene!=='quest'){ $('#quest-reset').disabled=false;return; }
  quest.settle(ticket);renderQuest();$('#quest-reset').disabled=false;
  if(!quest.completed){say(ticket.index===0?'반가워요. 무엇이 필요한가요?':'사과군요. 몇 개 드릴까요?','NPC','실전 대화');}
}
function next(){if(busy||!loaded)return;if(typing){finishText();return;}if(scene==='intro')openPractice();else if(scene==='practice')returnRoom();else if(scene==='return')startQuest();else if(scene==='quest')openPractice();}
$('#next').addEventListener('click',next);$('#reset').addEventListener('click',intro);$('#home').addEventListener('click',e=>{e.preventDefault();intro();});
$('.steps').addEventListener('click',e=>{const b=e.target.closest('[data-scene]');if(!b||busy||!loaded)return;({intro,practice:openPractice,quest:startQuest})[b.dataset.scene]();});
function updateReduced(){document.body.classList.toggle('reduced',reduced);$('#motion').setAttribute('aria-pressed',String(reduced));$('#motion').textContent=reduced?'기본 효과로':'효과 줄이기';}
$('#motion').addEventListener('click',()=>{reduced=!reduced;updateReduced();});updateReduced();
$('#rotate-left').addEventListener('click',()=>npcYaw-=.26);$('#rotate-right').addEventListener('click',()=>npcYaw+=.26);
$('#zoom').addEventListener('click',()=>{zoom=zoom>1.8?1.7:2.0;$('#zoom').textContent=zoom>1.8?'−':'＋';});
$('#gesture').addEventListener('click',()=>{if(scene!=='practice')return;gesture=!gesture;gestureTime=0;$('#gesture').textContent=gesture?'동작 멈추기':'동작 시험 보기';record(gesture?'public-gesture-play':'public-gesture-stop');});
$('#simulate-success').addEventListener('click',simulateSuccess);$('#quest-reset').addEventListener('click',()=>{if(scene==='quest'&&!quest.pending)resetQuest();});
window.addEventListener('keydown',e=>{if(e.repeat)return;if(e.code==='Space'&&!['BUTTON','INPUT','TEXTAREA'].includes(document.activeElement.tagName)){e.preventDefault();next();}if(scene==='practice'&&!busy){if(e.code==='ArrowLeft'){e.preventDefault();npcYaw-=.16;}if(e.code==='ArrowRight'){e.preventDefault();npcYaw+=.16;}}});
window.addEventListener('pagehide',()=>{disposed=true;cancelAnimationFrame(raf);clearInterval(typingTimer);});
window.addEventListener('error',e=>record('error:'+e.message));window.addEventListener('unhandledrejection',e=>record('error:'+String(e.reason)));
async function start(){
  try{
    renderer=new StageRenderer($('#world'));optics=new Hologram($('#optics'));
    [robot,npc]=await Promise.all([renderer.load('assets/robot.glb'),renderer.load('assets/npc.glb')]);
    loaded=true;document.body.classList.add('ready');$('#loading').setAttribute('aria-hidden','true');intro();last=performance.now();
    function frame(now){
      if(disposed)return;const dt=(now-last)/1000;last=now;const t=now/1000;
      renderer.begin([0,1.72,4.5-zoom],[0,targetY,0],dark);
      const bob=reduced?0:Math.sin(t*1.3)*.02,tilt=reduced?0:Math.sin(t*.9)*.018;
      const narrow=innerWidth<760, focus=scene==='practice'||scene==='quest';
      const rx=narrow?robotX*.33:robotX, rs=narrow&&focus?robotScale*.64:robotScale, ry=narrow&&focus?robotY+.28:robotY;
      const robotRoot=model(rx,ry+bob,robotZ,rs,robotYaw,tilt);
      renderer.draw(renderer.shadow,model(rx,0,robotZ,.8),I(),null,2,1,[1,1,1,1]);renderer.asset(robot,robotRoot);
      if(npcVisible){
        if(gesture)gestureTime=Math.min(gestureTime+dt,npc.duration);
        if(gesture&&gestureTime>=npc.duration){gesture=false;$('#gesture').textContent='동작 시험 보기';}
        renderer.pose(npc,gestureTime);renderer.draw(renderer.shadow,model(0,0,0),I(),null,2,1,[1,1,1,npcReveal]);renderer.asset(npc,model(0,narrow?.62:0,0,narrow?.82:1.22,npcYaw),npcReveal);
      }
      optics.set(transitionMode,transitionP,reduced);
      // Eye centers on this MOA mesh, transformed with the same root as the face.
      const eyePoints=[[-.23,-.15,.70],[.23,-.15,.70]].map(p=>[0,1,2].map(r=>robotRoot[r]*p[0]+robotRoot[4+r]*p[1]+robotRoot[8+r]*p[2]+robotRoot[12+r]));
      optics.draw(eyePoints.map(p=>renderer.project(p)),t);
      raf=requestAnimationFrame(frame);
    }
    raf=requestAnimationFrame(frame);record('assets-ready');
  }catch(e){console.error(e);$('#error').hidden=false;$('#error p').textContent=e.message;record('error:'+e.message);}
}
Object.defineProperty(window,'previewState',{get:()=>({scene,busy,loaded,reduced,typing,npcVisible,gesture,quest:{index:quest.index,pending:!!quest.pending,completed:quest.completed},events:[...events]})});
start();
