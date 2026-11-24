import {StageRenderer,model,I} from './renderer.js';
import {Hologram} from './hologram.js';
import {signDemos} from './sign-demos/catalog.js';

const $=s=>document.querySelector(s),clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x)},mix=(a,b,t)=>a+(b-a)*t;
let captureLocked=false,previewActive=false,lastDraw=0,mood='idle',moodUntil=0;
let scene='intro',busy=false,loaded=false,reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
let raf,last=performance.now(),typingTimer=0,typing=false,fullText='',disposed=false;
let robotX=0,robotY=1.48,robotZ=0,robotScale=.65,robotYaw=0,robotPitch=0,zoom=0,targetY=1.15,dark=0;
let npcVisible=false,npcReveal=0,npcYaw=0,gesture=false,gestureTime=0,transitionMode='off',transitionP=0;
let renderer,robot,npc,optics;
let layoutDirty=true,layoutKey='',angleRects=[];
const dirtyLayout=()=>{layoutDirty=true;};
window.addEventListener('resize',dirtyLayout);
new ResizeObserver(dirtyLayout).observe($('#dialogue'));
new ResizeObserver(dirtyLayout).observe($('#angle-views'));
function placeCoach(rx,ry,rs){
  const key=[innerWidth,innerHeight,zoom,targetY,scene,rx,ry,rs].join(':');
  if(!layoutDirty&&key===layoutKey)return;
  layoutDirty=false;layoutKey=key;
  if(scene==='practice'||scene==='quest'){
    // Anchor to the visible robot, without making the text bob while reading.
    const [x,bottom]=renderer.project([rx,ry-1.04*rs,robotZ]);
    const box=$('#dialogue'),width=box.getBoundingClientRect().width,margin=innerWidth<760?10:24;
    const left=Math.max(margin,Math.min(innerWidth-margin-width,x-width*.44));
    const top=bottom+20;
    box.style.setProperty('--coach-left',left+'px');box.style.setProperty('--coach-top',top+'px');
    box.style.setProperty('--coach-tail',(x-left-12)+'px');
    box.style.setProperty('--coach-room',Math.max(150,innerHeight-top-margin)+'px');
    box.dataset.anchor=JSON.stringify({x,bottom,tail:x-left});
  }
  if(scene==='practice'&&!$('#angle-views').hidden){
    const rect=$('#angle-views').getBoundingClientRect();
    $('#avatar-controls').style.setProperty('top',(rect.bottom+12)+'px','important');
    angleRects=['angle-left','angle-right'].map(id=>$('#'+id).getBoundingClientRect());
  }else angleRects=[];
}
let demoAsset=null,demoWord='',demoTicket=0,gestureSpeed=.5,demoPlayed=false;const demoCache=new Map();
const avatarButtons='#gesture,#rotate-left,#rotate-right,#zoom,#gesture-restart,#gesture-speed,#sign-compare';
function updateAvatarControls(){const supported=!!signDemos[demoWord]||demoWord==='WORD1567',blocked=busy||captureLocked||scene!=='practice';document.querySelectorAll(avatarButtons).forEach(el=>el.disabled=blocked||!supported||(!demoAsset&&demoWord!=='WORD1567'));$('#sign-compare').disabled=blocked||!signDemos[demoWord];}
async function selectDemo(word){if(!loaded||word===demoWord)return;demoWord=word;demoPlayed=false;const ticket=++demoTicket;gesture=false;gestureTime=0;demoAsset=null;npcYaw=0;const entry=signDemos[word];$('#gesture').textContent='동작 재생';$('#gesture-note').textContent=entry?entry.label+' 시범을 준비하고 있어요.':word==='WORD1567'?'공개 조상 동작 적용 시험 · 교육용 검수 전':'이 단어는 공개 참고 영상을 이용해 주세요.';updateAvatarControls();if(!entry)return;try{if(!demoCache.has(word))demoCache.set(word,renderer.load(entry.url).catch(e=>{demoCache.delete(word);throw e}));const asset=await demoCache.get(word);if(ticket!==demoTicket)return;demoAsset=asset;gestureTime=entry.preview;$('#gesture-note').textContent=entry.steps.map(s=>s.label).join(' → ')+' · 원본과 함께 확인해요.';$('#avatar-controls').dataset.word=word;updateAvatarControls();}catch(e){if(ticket===demoTicket){$('#gesture-note').textContent='시범을 불러오지 못했어요. 공개 참고 영상을 이용해 주세요.';updateAvatarControls();}}}
const events=[];const record=event=>{events.push({event,time:performance.now()});if(events.length>100)events.shift();};
function say(text,speaker='MOA',role='당신의 수어 연습 친구'){
  clearInterval(typingTimer);fullText=text;typing=true;$('#speaker').textContent=speaker;$('#speaker-role').textContent=role;$('#line').textContent='';
  const chars=Array.from(text);let i=0;typingTimer=setInterval(()=>{i+=reduced?3:1;$('#line').textContent=chars.slice(0,i).join('');if(i>=chars.length)finishText();},reduced?15:39);
}
function finishText(){clearInterval(typingTimer);$('#line').textContent=fullText;typing=false;}
function button(text){$('#next-label').textContent=text;}
function label(type,title,caption){$('#room-type').textContent=type;$('#room-title').textContent=title;$('#room-caption').textContent=caption;}
function setScene(name,nav=name){scene=name;document.body.dataset.scene=name;$('.steps').querySelectorAll('button').forEach(b=>b.classList.toggle('selected',b.dataset.scene===nav));record(name);window.dispatchEvent(new CustomEvent('stage:scene',{detail:{scene:name}}));}
function animate(ms,fn){return new Promise(resolve=>{const start=performance.now();function tick(now){if(disposed){resolve();return;}const p=clamp((now-start)/ms);fn(p);if(p<1)requestAnimationFrame(tick);else resolve();}requestAnimationFrame(tick);});}
function lock(value){busy=value;for(const el of document.querySelectorAll('#next,.steps button,#reset'))el.disabled=value||captureLocked;updateAvatarControls();}
function hidePanels(){for(const id of ['practice-heading','quest-panel','self-view','live-controls','avatar-controls','angle-views'])$('#'+id).hidden=true;$('#quest-success').hidden=true;}
function cleanFx(){transitionMode='off';transitionP=0;$('#npc-scan').style.opacity=0;optics?.clear();}
function neutralRobot(){robotX=0;robotY=1.48;robotZ=0;robotScale=.65;robotYaw=0;robotPitch=0;}
function stopGesture(){gesture=false;gestureTime=signDemos[demoWord]?.preview||0;npcYaw=0;$('#gesture').textContent='동작 재생';$('#zoom').textContent='＋';}
function intro(){
  if(busy||captureLocked)return;hidePanels();cleanFx();stopGesture();neutralRobot();npcVisible=false;dark=0;zoom=0;targetY=1.15;
  document.body.classList.remove('dark','focus-scene');$('.room-label').hidden=false;$('#dialogue').hidden=false;
  setScene('intro');label('TUTORIAL ROOM','처음 만나는, 새로운 대화','말 대신 손으로, 함께 시작해요.');say('나는 모아야. 오늘은 사과를 부탁하고, 고마운 마음을 전하는 말을 함께 배워 보자.');button('함께 시작하기');$('#status').textContent='로컬 웹캠 테스트 · 사용자 촬영 학습 제외';
}
async function openPractice(){
  if(busy||captureLocked||!loaded)return;finishText();lock(true);hidePanels();stopGesture();npcVisible=false;neutralRobot();zoom=0;targetY=1.15;dark=0;
  document.body.classList.remove('focus-scene','dark');$('.room-label').hidden=false;$('#dialogue').hidden=false;
  setScene('projecting','practice');label('PRACTICE ROOM','함께 연습하러 가 볼까?','');say('좋아. 내가 연습할 공간을 열어 줄게!');button('연습실을 여는 중');
  finishText();transitionMode='enter';
  await animate(reduced?500:4300,p=>{
    transitionP=p;dark=ease((p-.62)/.06);
    // Preserve the approach; the bottom-to-top fade begins at the upward turn.
    if(p<.07){robotPitch=.24*ease(p/.07);}
    else if(p<.55){
      const a=Math.pow(clamp((p-.07)/.48),1.42),b=1-a;
      const bez=(v0,v1,v2,v3)=>b*b*b*v0+3*b*b*a*v1+3*b*a*a*v2+a*a*a*v3;
      robotX=bez(0,-.62,.58,.40);robotY=bez(1.48,.22,.48,5.2);robotZ=bez(0,1.6,2.0,.45);
      robotScale=.65;robotYaw=-.10*Math.sin(a*Math.PI);robotPitch=.24+.18*Math.sin(a*Math.PI)-.92*ease((a-.38)/.62);
    }else if(p<.80){robotY=5.2;robotPitch=0;}
    else{const a=ease((p-.80)/.19);robotX=mix(3.0,1.13,a);robotY=mix(1.94,1.61,a)+.10*Math.sin(a*Math.PI);robotZ=mix(-1.25,-.85,a);robotScale=mix(.28,.32,a);robotYaw=mix(-.9,-.55,a);robotPitch=-.12*Math.sin(a*Math.PI);}
    const move=ease((p-.66)/.12);zoom=mix(0,1.7,move);targetY=mix(1.15,1.49,move);
    if(p>.68)document.body.classList.add('dark');
    if(p>.72){npcVisible=true;npcReveal=ease((p-.72)/.10);}

  });
  cleanFx();dark=1;npcVisible=true;npcReveal=1;robotYaw=-.55;document.body.classList.add('dark','focus-scene');
  $('.room-label').hidden=true;$('#practice-heading').hidden=false;$('#self-view').hidden=false;$('#avatar-controls').hidden=false;$('#angle-views').hidden=false;
  $('#dialogue').hidden=false;setScene('practice');say('먼저 가운데 동작을 살펴봐. 준비되면 Space를 눌러 줘. 끝나고 손을 내리면 내가 확인해 줄게.');button('연습 마치기');$('#status').textContent='기존 관절 모델 연결 · 교정 안내는 관측 근거 범위';lock(false);
}
async function returnRoom(){
  if(busy||captureLocked)return;lock(true);stopGesture();hidePanels();$('#dialogue').hidden=true;$('.room-label').hidden=true;document.body.classList.remove('focus-scene');
  setScene('deprojecting','practice');transitionMode='exit';
  const sx=robotX,sy=robotY,sz=robotZ,ss=robotScale,cam=zoom,ty=targetY;
  await animate(reduced?240:1250,p=>{
    transitionP=p;const e=ease(p);dark=1-e;npcReveal=1-ease(p/.65);robotX=mix(sx,0,e);robotY=mix(sy,1.48,e);robotZ=mix(sz,0,e);robotScale=mix(ss,.65,e);zoom=mix(cam,0,e);targetY=mix(ty,1.15,e);
    if(p>.45)document.body.classList.remove('dark');
  });
  cleanFx();dark=0;npcVisible=false;neutralRobot();zoom=0;targetY=1.15;$('.room-label').hidden=false;$('#dialogue').hidden=false;
  setScene('return','intro');label('READY FOR A CONVERSATION','이제, 직접 이야기해 볼까요?','배운 단어를 순서대로 표현해요.');say('이번에는 사과를 부탁한 뒤 감사 인사를 전해 보자. 필요한 동작이 확인되면 다음 말로 넘어갈 거야.');button('실전 대화 시작');$('#status').textContent='사과 요청 → 감사 인사';lock(false);
}
async function startQuest(){
  if(busy||captureLocked||!loaded)return;lock(true);finishText();hidePanels();cleanFx();stopGesture();npcVisible=false;neutralRobot();zoom=0;targetY=1.15;dark=0;
  document.body.classList.remove('dark','focus-scene');$('.room-label').hidden=false;$('#dialogue').hidden=false;
  setScene('quest-arriving','quest');label('FIRST QUEST','연습에서, 대화로','모아는 곁에서 지켜볼게요.');say('위에 있는 단어부터 하나씩 표현해 봐. 이번에는 실제 촬영 결과로 진행할 거야.');button('친구를 부르는 중');
  await animate(reduced?150:1100,p=>{const e=ease(p);robotX=mix(0,1.13,e);robotZ=mix(0,-.85,e);robotY=mix(1.48,1.53,e);robotScale=mix(.65,.32,e);zoom=mix(0,1.7,e);targetY=mix(1.15,1.53,e);});
  npcVisible=true;npcReveal=0;
  await animate(reduced?220:1250,p=>{npcReveal=ease(p);$('#npc-scan').style.opacity=reduced?0:Math.sin(p*Math.PI)*.5;$('#npc-scan').style.top=(73-55*p)+'%';});
  cleanFx();npcReveal=1;document.body.classList.add('focus-scene');$('.room-label').hidden=true;
  for(const id of ['quest-panel','self-view','live-controls'])$('#'+id).hidden=false;
  robotYaw=-.55;setScene('quest');button('연습으로 돌아가기');$('#status').textContent='단어 인식과 필요한 동작을 함께 확인해요';lock(false);
}
function next(){if(busy||captureLocked||!loaded)return;if(typing){finishText();return;}if(scene==='intro')openPractice();else if(scene==='practice')returnRoom();else if(scene==='return')startQuest();else if(scene==='quest')openPractice();}
$('#next').addEventListener('click',next);$('#reset').addEventListener('click',intro);$('#home').addEventListener('click',e=>{e.preventDefault();intro();});
$('.steps').addEventListener('click',e=>{const b=e.target.closest('[data-scene]');if(!b||busy||captureLocked||!loaded)return;({intro,practice:openPractice,quest:startQuest})[b.dataset.scene]();});
function updateReduced(){document.body.classList.toggle('reduced',reduced);$('#motion').setAttribute('aria-pressed',String(reduced));$('#motion').textContent=reduced?'기본 효과로':'효과 줄이기';}
$('#motion').addEventListener('click',()=>{reduced=!reduced;updateReduced();});updateReduced();
$('#rotate-left').addEventListener('click',()=>npcYaw-=.26);$('#rotate-right').addEventListener('click',()=>npcYaw+=.26);
$('#zoom').addEventListener('click',()=>{zoom=zoom>1.8?1.7:2.0;$('#zoom').textContent=zoom>1.8?'−':'＋';});
$('#gesture').addEventListener('click',()=>{if(scene!=='practice'||captureLocked)return;if(!demoPlayed){gestureTime=0;demoPlayed=true;}gesture=!gesture;$('#gesture').textContent=gesture?'동작 멈춤':'동작 재생';record(gesture?'public-gesture-play':'public-gesture-stop');});
$('#gesture-restart').onclick=()=>{gestureTime=0;gesture=true;$('#gesture').textContent='동작 멈춤';};
$('#gesture-speed').onclick=()=>{gestureSpeed=gestureSpeed===.5?1:.5;$('#gesture-speed').textContent=gestureSpeed+'배';};
$('#sign-compare').onclick=()=>{if(captureLocked||!signDemos[demoWord])return;gesture=false;$('#gesture').textContent='동작 재생';$('#sign-frame').src='sign-demo.html?word='+encodeURIComponent(demoWord);$('#sign-dialog').showModal();};
$('#sign-close').onclick=()=>$('#sign-dialog').close();$('#sign-dialog').addEventListener('close',()=>$('#sign-frame').src='about:blank');
window.addEventListener('coach:target',e=>{if(e.detail.scene==='practice')selectDemo(e.detail.word);updateAvatarControls();});
window.addEventListener('keydown',e=>{if(e.repeat||captureLocked||document.querySelector('dialog[open]')||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.code==='Space'&&!['practice','quest'].includes(scene)&&!['BUTTON'].includes(document.activeElement.tagName)){e.preventDefault();next();}if(scene==='practice'&&!busy){if(e.code==='ArrowLeft'){e.preventDefault();npcYaw-=.16;}if(e.code==='ArrowRight'){e.preventDefault();npcYaw+=.16;}}});
window.addEventListener('pagehide',()=>{disposed=true;cancelAnimationFrame(raf);clearInterval(typingTimer);});
window.addEventListener('error',e=>record('error:'+e.message));window.addEventListener('unhandledrejection',e=>record('error:'+String(e.reason)));
async function start(){
  try{
    renderer=new StageRenderer($('#world'));optics=new Hologram($('#optics'));
    [robot,npc]=await Promise.all([renderer.load('assets/robot.glb'),renderer.load('assets/npc.glb')]);
    loaded=true;document.body.classList.add('ready');$('#loading').setAttribute('aria-hidden','true');intro();last=performance.now();
    function frame(now){
      if(disposed)return;
      // RTMW shares the integrated GPU with WebGL. Keep the last background
      // image during capture instead of competing with observation frames.
      const pause3D=captureLocked||document.hidden||!!document.querySelector('dialog[open]');
      document.body.dataset.renderMode=pause3D?'observation-priority':'animated';
      if(pause3D){last=now;lastDraw=now;raf=requestAnimationFrame(frame);return;}
      // Watching a sign needs fluid motion; idle preview stays inexpensive.
      // Countdown/recording/analysis still pause all 3D above.
      const interval=1000/(gesture?30:previewActive?12:60);
      if(now-lastDraw<interval-1){raf=requestAnimationFrame(frame);return;}
      lastDraw=now-(Math.max(interval,now-lastDraw)%interval);
      const dt=(now-last)/1000;last=now;const t=now/1000;
      renderer.begin([0,1.72,4.5-zoom],[0,targetY,0],dark);
      const bob=reduced?0:Math.sin(t*1.3)*.02,tilt=reduced?0:Math.sin(t*.9)*.018;
      const narrow=innerWidth<760, focus=scene==='practice'||scene==='quest';
      const rx=narrow?robotX*.33:robotX, rs=narrow&&focus?robotScale*.64:robotScale, ry=narrow&&focus?robotY+.06:robotY;
      placeCoach(rx,ry,rs);
      const response=now<moodUntil?Math.sin((moodUntil-now)*.009)*.06:0;
      const robotRoot=model(rx,ry+bob+(mood==='success'?Math.abs(response):0),robotZ,rs,robotYaw+(typing?.025*Math.sin(t*4):0),tilt+(mood==='uncertain'?response:response*.3),robotPitch);
      renderer.draw(renderer.shadow,model(rx,0,robotZ,.8),I(),null,2,1,[1,1,1,1]);renderer.asset(robot,robotRoot);
      if(npcVisible){
        const actor=scene==='practice'&&demoAsset?demoAsset:npc;
        if(gesture)gestureTime=(gestureTime+Math.min(dt,.1)*gestureSpeed)%(actor.duration||1);
        const poseTime=scene==='practice'&&(demoAsset||demoWord==='WORD1567')?gestureTime:0;
        renderer.pose(actor,poseTime);renderer.draw(renderer.shadow,model(0,0,0),I(),null,2,1,[1,1,1,npcReveal]);renderer.asset(actor,model(0,narrow?.62:0,0,narrow?.82:1.22,npcYaw),npcReveal);
        $('#avatar-controls').dataset.time=poseTime.toFixed(3);
      }
      const rp=(x,y,z)=>renderer.project([0,1,2].map(r=>robotRoot[12+r]+x*robotRoot[r]+y*robotRoot[4+r]+z*robotRoot[8+r]));
      // Mesh GLB is Y-up: the bottom projector is at -1.04, below the face.
      const neck=rp(0,-1.04,0),bodyCenter=rp(0,-.28,0),bodyLeft=rp(-.9,-.28,0),bodyTop=rp(0,.49,0);
      if(scene==='practice'&&demoAsset&&angleRects.length){for(const [i,side] of [[0,-1],[1,1]]){renderer.inset([side*1.26,1.5,1.26],[0,1.28,0],angleRects[i],1,()=>renderer.asset(demoAsset,model(0,0,0,1.12)));}}
      optics.set(transitionMode,transitionP,reduced);
      optics.draw([neck,bodyCenter,bodyLeft,bodyTop],t);
      raf=requestAnimationFrame(frame);
    }
    raf=requestAnimationFrame(frame);record('assets-ready');
  }catch(e){console.error(e);$('#error').hidden=false;$('#error p').textContent=e.message;record('error:'+e.message);}
}

window.addEventListener('coach:say',e=>say(e.detail.text,e.detail.speaker||'MOA'));
window.addEventListener('coach:lock',e=>{captureLocked=!!e.detail;lock(busy);});
window.addEventListener('coach:stop-gesture',stopGesture);
window.addEventListener('coach:preview',e=>{previewActive=!!e.detail;});
window.addEventListener('coach:reaction',e=>{mood=e.detail;moodUntil=performance.now()+2400;});

start();

