const $=id=>document.getElementById(id);
const STATUS={similar:'확인한 특징이 비슷함',difference:'차이 확인',uncertain:'확인 필요 · 오답 아님'};
let catalog,data,active=0,cueIndex=0,mode=null,focusEnd=0,loadSerial=0;
const near=(a,x)=>{let lo=0,hi=a.length-1;while(lo<hi){let m=(lo+hi)>>1;if(a[m]<x)lo=m+1;else hi=m}return lo&&Math.abs(a[lo-1]-x)<Math.abs(a[lo]-x)?lo-1:lo};
const duration=d=>d.timestamp_us.at(-1)/1e6;
const card=()=>data?.cards[active];const cue=()=>card()?.cues[cueIndex];
const spans=()=>cue()?.spans||card()?.spans||[];
const inSpans=(i,s)=>s.some(([a,b])=>i>=a&&i<=b);
function pause(){mode=null;$('query').pause();$('ref').pause();}
function frame(){return near(data.timestamp_us,$('query').currentTime/data.playback_rate*1e6)}
function queryTime(i){return data.timestamp_us[i]/1e6*data.playback_rate}
function visibleSpans(){return spans().length?spans():[]}
function refCard(){return data.reference.cards.find(c=>c.id===card().id)}
function refSpans(){const ids=cue()?.card_ids||[card().id];return data.reference.cards.filter(c=>ids.includes(c.id)).flatMap(c=>c.spans).sort((a,b)=>a[0]-b[0])}
function seekRef(i){
 const ss=visibleSpans(),rs=refSpans();if(!ss.length||!rs.length)return;
 const a=ss[0][0],b=ss.at(-1)[1],c=rs[0][0],d=rs.at(-1)[1];
 const ri=Math.round(c+Math.min(1,Math.max(0,(i-a)/Math.max(1,b-a)))*(d-c));
 $('ref').currentTime=data.reference.timestamp_us[ri]/1e6;
}
function draw(id,xy,w,h,ink,hands,emphasis){
 const el=$(id),ratio=devicePixelRatio||1,W=Math.round(el.clientWidth*ratio),H=Math.round(el.clientHeight*ratio);if(el.width!==W)el.width=W;if(el.height!==H)el.height=H;
 const c=el.getContext('2d');c.clearRect(0,0,W,H);if(!$('overlay').checked||!xy)return;
 const s=Math.min(W/w,H/h),ox=(W-w*s)/2,oy=(H-h*s)/2;
 const good=i=>xy[i]&&xy[i].every(Number.isFinite)&&xy[i][0]>=0&&xy[i][0]<w&&xy[i][1]>=0&&xy[i][1]<h;
 const line=(a,b,col,width)=>{if(!good(a)||!good(b))return;c.strokeStyle=col;c.lineWidth=width*ratio;c.beginPath();c.moveTo(ox+xy[a][0]*s,oy+xy[a][1]*s);c.lineTo(ox+xy[b][0]*s,oy+xy[b][1]*s);c.stroke()};
 line(5,6,'#c4d4e299',1);
 for(const hand of ['L','R']){
  const selected=hands.includes(hand)&&emphasis,col=selected?ink:'#a7b5c26b',a=hand==='L'?[5,7,9]:[6,8,10],root=hand==='L'?91:112;
  line(a[0],a[1],col,selected?3:1);line(a[1],a[2],col,selected?3:1);
  for(const f of [1,5,9,13,17]){line(root,root+f,col,selected?2:1);for(let k=f;k<f+3;k++)line(root+k,root+k+1,col,selected?2:1)}
  if(selected)for(let j=root;j<root+21;j++)if(good(j)){c.fillStyle=col;c.beginPath();c.arc(ox+xy[j][0]*s,oy+xy[j][1]*s,1.5*ratio,0,Math.PI*2);c.fill()}
 }
}
function sync(){
 if(!data)return;const i=frame(),r=near(data.reference.timestamp_us,$('ref').currentTime*1e6),s=visibleSpans(),on=inSpans(i,s),c=card();
 const hands=cue()?.hands||c.hands,ink=c.status==='difference'?'#ffac55':c.status==='similar'?'#72b6ee':'#bdc6d2';
 draw('query-pose',data.xy[i],data.width,data.height,ink,hands,on);
 draw('ref-pose',data.reference.xy[r],data.reference.width,data.reference.height,'#72b6ee',hands,inSpans(r,refSpans()));
 $('seek').value=Math.round(data.timestamp_us[i]/1e6/duration(data)*10000);$('time').textContent=(data.timestamp_us[i]/1e6).toFixed(2)+' / '+duration(data).toFixed(2)+'초';
 $('query-info').textContent=`시험 ${i+1} / ${data.frame_count}프레임 · ${on?(c.status==='uncertain'?'확인할 후보 구간':'선택한 근거 구간'):'강조 구간 밖'}${data.kind==='continuous_full_reverse'?' · 원본 '+(data.source_mapping[i]+1)+'프레임':''}`;
 $('ref-info').textContent=`참고 ${r+1} / ${data.reference.timestamp_us.length}프레임 · `+(data.reference.timestamp_us[r]/1e6).toFixed(2)+'초';
 if(mode==='focus'&&i>=focusEnd){const end=focusEnd;pause();$('query').currentTime=queryTime(end);seekRef(end)}
 if(mode==='all'&&i>=data.frame_count-1){pause();$('query').currentTime=queryTime(data.frame_count-1)}
}
function bands(){
 $('bands').replaceChildren();const add=(s,status)=>{const el=document.createElement('span');el.className=status;el.style.left=data.timestamp_us[s[0]]/1e6/duration(data)*100+'%';el.style.width=Math.max(.3,(data.timestamp_us[s[1]]-data.timestamp_us[s[0]])/1e6/duration(data)*100)+'%';$('bands').append(el)};
 for(const s of card().spans)add(s,card().status==='difference'?'uncertain':card().status);
 if(card().status==='difference')for(const q of card().cues)for(const s of q.spans)add(s,'difference');
}
function select(index,ci=0){
 pause();active=index;cueIndex=ci;const c=card(),q=cue();
 $('message-box').className='message '+c.status;$('badge').textContent=STATUS[c.status];$('observation').textContent=q?.message||c.message;
 $('action').textContent=q?'다시 해 볼 방법: '+q.action:c.status==='similar'?'확인한 특징의 결과예요. 동작 전체의 완료 판정은 아니에요.':'참고 동작: '+c.instruction;
 $('cue-buttons').replaceChildren();for(const [j,v]of c.cues.entries()){const b=document.createElement('button');b.textContent=({direction:'이동 방향',shape:'손모양',order:'행동 순서'})[v.kind];b.classList.toggle('selected',ci===j);b.onclick=()=>select(index,j);$('cue-buttons').append(b)}
 [...$('cards').children].forEach((el,j)=>el.classList.toggle('selected',index===j));bands();const s=visibleSpans();$('focus').disabled=!s.length;
 if(s.length){const i=Math.floor((s[0][0]+s[0][1])/2);$('query').currentTime=queryTime(i);seekRef(i)}
 else{$('query').currentTime=0;$('ref').currentTime=0}
 sync();
}
async function load(id){
 const serial=++loadSerial;pause();$('workspace').setAttribute('aria-busy','true');$('error').hidden=true;
 try{
  const found=catalog.cases.find(c=>c.id===id);if(!found)throw Error('지원하지 않는 공개 결과입니다.');
  const response=await fetch(found.url);if(!response.ok)throw Error('결과를 읽지 못했습니다.');const d=await response.json();if(serial!==loadSerial)return;
  data=d;$('word').value=d.word;fillCases(d.word,d.id);history.replaceState(null,'','#'+id);
  $('source-note').textContent=`${d.label} · 공개 기록 ${d.record_id} · ${d.kind_label} — `+(d.kind==='normal'?'편집하지 않은 원본입니다.':d.kind==='continuous_full_reverse'?'정상 원본을 거꾸로 재생한 시험입니다. 실제 사용자 촬영이 아닙니다.':d.kind==='all_hands_missing'?'영상은 그대로이며, 분석 입력의 손 관절을 제거한 시험입니다.':'공개 원본의 재생 속도 또는 길이를 바꾼 시험입니다.');
  $('query-title').textContent=d.kind==='normal'?'분석한 공개 동작':'분석한 시험 동작';$('query-subtitle').textContent=d.kind_label;
  const ready=v=>new Promise((resolve,reject)=>{v.addEventListener('loadeddata',resolve,{once:true});v.addEventListener('error',()=>reject(Error('영상 재생에 실패했습니다.')),{once:true})});
  const p1=ready($('query')),p2=ready($('ref'));$('query').src=d.video;$('ref').src=d.reference.video;$('query').playbackRate=d.playback_rate;
  $('cards').replaceChildren();for(const [i,c]of d.cards.entries()){const b=document.createElement('button');b.className=c.status;const title=document.createElement('strong');title.textContent=c.instruction;const s=document.createElement('small');s.textContent=STATUS[c.status];b.append(title,s);b.onclick=()=>select(i);$('cards').append(b)}
  $('provenance').textContent='근거: G4 공개 검증 저장 결과 · '+d.clip_id+' · 2026-11-14. 학습·튜닝에 사용한 사용자 영상은 없습니다.';
  await Promise.all([p1,p2]);if(serial!==loadSerial)return;select(Math.max(0,d.cards.findIndex(c=>c.status==='difference')));$('workspace').setAttribute('aria-busy','false');
 }catch(e){if(serial!==loadSerial)return;$('error').textContent=e.message;$('error').hidden=false;$('workspace').setAttribute('aria-busy','false')}
}
function fillCases(word,id){$('case').replaceChildren();for(const c of catalog.cases.filter(c=>c.word===word)){const o=document.createElement('option');o.value=c.id;o.textContent=`기록 ${c.record_id} · ${c.kind_label}`;$('case').append(o)}$('case').value=id}
$('focus').onclick=()=>{
 const ss=visibleSpans(),rs=refSpans();if(!ss.length)return;pause();const a=ss[0][0],b=Math.max(...ss.map(s=>s[1])),slow=$('slow').checked?.5:1;focusEnd=b;mode='focus';$('query').currentTime=queryTime(a);$('query').playbackRate=data.playback_rate*slow;
 if(rs.length){const c=rs[0][0],d=Math.max(...rs.map(s=>s[1]));$('ref').currentTime=data.reference.timestamp_us[c]/1e6;const qdur=(data.timestamp_us[b]-data.timestamp_us[a])/1e6,rdur=(data.reference.timestamp_us[d]-data.reference.timestamp_us[c])/1e6;$('ref').playbackRate=Math.min(4,Math.max(.0625,rdur/Math.max(.03,qdur)*slow));$('ref').play().catch(()=>{})}
 $('query').play().catch(()=>pause());
};
$('play').onclick=()=>{pause();mode='all';const slow=$('slow').checked?.5:1;$('query').currentTime=0;$('ref').currentTime=0;$('query').playbackRate=data.playback_rate*slow;$('ref').playbackRate=Math.min(4,Math.max(.0625,duration(data.reference)/duration(data)*slow));$('query').play().catch(()=>pause());$('ref').play().catch(()=>{})};
$('pause').onclick=pause;$('seek').oninput=()=>{pause();const i=near(data.timestamp_us,Number($('seek').value)/10000*data.timestamp_us.at(-1));$('query').currentTime=queryTime(i);seekRef(i);sync()};
function step(delta){pause();const i=Math.min(data.frame_count-1,Math.max(0,frame()+delta));$('query').currentTime=queryTime(i);seekRef(i)}
$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('overlay').onchange=sync;$('zoom').oninput=()=>{document.documentElement.style.setProperty('--zoom',$('zoom').value);sync()};
for(const v of [$('query'),$('ref')]){v.addEventListener('seeked',sync);v.addEventListener('timeupdate',sync);v.addEventListener('error',()=>{$('error').textContent='영상을 읽지 못했습니다. 페이지를 다시 열어 주세요.';$('error').hidden=false})}
function tick(){if(mode)sync();requestAnimationFrame(tick)}requestAnimationFrame(tick);window.addEventListener('resize',sync);window.addEventListener('pagehide',pause);
$('case').onchange=()=>load($('case').value);$('word').onchange=()=>{const first=catalog.cases.find(c=>c.word===$('word').value&&c.kind==='normal');load(first.id)};
for(const b of document.querySelectorAll('[data-example]'))b.onclick=()=>load(b.dataset.example);
const response=await fetch('catalog.json');catalog=await response.json();
for(const w of [...new Set(catalog.cases.map(c=>c.word))]){const o=document.createElement('option');o.value=w;o.textContent=catalog.cases.find(c=>c.word===w).label;$('word').append(o)}
for(const t of catalog.limits){const li=document.createElement('li');li.textContent=t;$('limits').append(li)}
await load(catalog.cases.some(c=>c.id===location.hash.slice(1))?location.hash.slice(1):catalog.default);
