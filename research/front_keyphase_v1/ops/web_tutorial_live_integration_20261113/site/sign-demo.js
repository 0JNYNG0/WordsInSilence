import {StageRenderer,model} from './renderer.js';
import {signDemos} from './sign-demos/catalog.js';
const $=id=>document.getElementById(id),word=new URLSearchParams(location.search).get('word')||'WORD0099',entry=signDemos[word],video=$('source');
let renderer,asset,yaw=0,zoom=false,ready=false,disposed=false,raf;
const end=()=>Math.min(asset?.duration||0,video.duration||0);
function sync(){if(!ready)return;const t=Math.min(video.currentTime,end());$('seek').value=Math.round(t/end()*1000);$('time').textContent=t.toFixed(2)+' / '+end().toFixed(2)+'초';$('play').textContent=video.paused?'함께 재생':'멈춤';$('avatar').dataset.time=t.toFixed(3);$('avatar').dataset.word=word;}
function seek(t){video.pause();video.currentTime=Math.max(0,Math.min(t,end()));sync();}
async function play(){try{if(video.currentTime>=end()-.04)video.currentTime=0;await video.play();$('state').textContent='원본 영상의 시간에 캐릭터를 맞춰 재생해요.';}catch(e){$('state').textContent='재생 버튼을 다시 눌러 주세요.';}sync();}
$('play').onclick=()=>video.paused?play():(video.pause(),sync());$('restart').onclick=()=>{video.currentTime=0;play()};$('speed').onclick=()=>{video.playbackRate=video.playbackRate===.5?1:.5;$('speed').textContent=video.playbackRate===.5?'느리게 · 0.5배':'기본 속도 · 1배';$('speed').setAttribute('aria-pressed',String(video.playbackRate===.5));};
$('loop').onchange=()=>video.loop=$('loop').checked;$('seek').oninput=e=>seek(Number(e.target.value)/1000*end());$('left').onclick=()=>yaw-=.26;$('right').onclick=()=>yaw+=.26;$('front').onclick=()=>yaw=0;$('zoom').onclick=()=>{zoom=!zoom;$('zoom').textContent=zoom?'전체 상체':'확대';};
video.addEventListener('error',()=>{$('state').textContent='원본 영상을 읽지 못했어요. 서버 연결을 확인해 주세요.'});video.addEventListener('seeked',sync);video.addEventListener('pause',sync);video.addEventListener('play',sync);
document.addEventListener('visibilitychange',()=>{if(document.hidden)video.pause()});window.addEventListener('pagehide',()=>{disposed=true;video.pause();cancelAnimationFrame(raf)});
window.addEventListener('keydown',e=>{if(!ready||e.repeat||['INPUT','BUTTON','A'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click()}else if(e.code==='ArrowLeft'){e.preventDefault();yaw-=.16}else if(e.code==='ArrowRight'){e.preventDefault();yaw+=.16}});
async function start(){try{
 if(!entry)throw Error('준비되지 않은 단어예요. 사과나 감사를 선택해 주세요.');$('title').textContent=entry.label+' · 원본과 캐릭터';document.title=entry.label+' · 원본과 캐릭터';$('note').textContent=entry.note;document.querySelector(`nav a[href="?word=${word}"]`).setAttribute('aria-current','page');
 renderer=new StageRenderer($('avatar'));const metadata=new Promise((resolve,reject)=>{video.addEventListener('loadedmetadata',resolve,{once:true});video.addEventListener('error',()=>reject(Error('원본 영상 로딩 실패')),{once:true})});video.src=entry.reference;video.loop=true;video.playbackRate=.5;[asset]=await Promise.all([renderer.load(entry.url),metadata]);ready=true;
 for(const step of entry.steps){const b=document.createElement('button'),title=document.createElement('strong'),text=document.createElement('span');title.textContent=step.label;text.textContent=step.text;b.append(title,text);b.onclick=()=>seek(step.time);$('steps').append(b)}
 video.currentTime=entry.preview;$('play').disabled=false;$('restart').disabled=false;$('state').textContent='함께 재생하거나, 아래 동작을 골라 살펴보세요.';
 function frame(){if(disposed)return;if(!document.hidden){renderer.pose(asset,Math.min(video.currentTime,end()));renderer.begin([0,1.33,zoom?1.05:1.65],[0,1.20,0],0);renderer.asset(asset,model(0,0,0,1,yaw));sync();}raf=requestAnimationFrame(frame)}raf=requestAnimationFrame(frame);
 }catch(e){$('state').textContent=e.message;$('title').textContent='시범을 불러오지 못했어요.';console.error(e)}}start();
