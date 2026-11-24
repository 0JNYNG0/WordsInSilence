const $=id=>document.getElementById(id);
let result=null,ref=null,active=-1,desired=null,referenceOnly=false,focusCue=null,clipEnd=null;
const near=(a,x)=>{let lo=0,hi=a.length-1;while(lo<hi){let m=(lo+hi)>>1;if(a[m]<x)lo=m+1;else hi=m}return lo&&Math.abs(a[lo-1]-x)<Math.abs(a[lo]-x)?lo-1:lo};
const view=c=>result?.coaching?.notes?.find(n=>n.card_id===c?.id)?.status||c?.display_status;
const color=c=>view(c)==='difference'?'#d4863d':view(c)==='similar'?'#2fb693':'#96a4b8';
function current(i){return active>=0?result.cards[active]:result.cards.find(c=>c.query_indices&&i>=c.query_indices[0]&&i<=c.query_indices[1]);}
function refIndex(i){const pairs=(result.path||[]).filter(p=>p[0]===i);return Math.min(ref.timestamp_us.length-1,pairs.length?Math.round(pairs.reduce((s,p)=>s+p[1],0)/pairs.length):i)}
function draw(id,xy,w,h,ink,hands,fingers){
 const el=$(id),c=el.getContext('2d');el.width=el.clientWidth*devicePixelRatio;el.height=el.clientHeight*devicePixelRatio;c.clearRect(0,0,el.width,el.height);if(!xy)return;
 const s=Math.min(el.width/w,el.height/h),ox=(el.width-w*s)/2,oy=(el.height-h*s)/2;
 const good=i=>xy[i]&&xy[i][0]>=0&&xy[i][0]<w&&xy[i][1]>=0&&xy[i][1]<h;
 const edges=[],names={1:'thumb',5:'index',9:'middle',13:'ring',17:'little'},detail=!!fingers?.length;
 for(const hand of hands||['L','R']){
  const a=hand==='L'?[5,7,9]:[6,8,10],start=hand==='L'?91:112;
  edges.push([a[0],a[1],!detail],[a[1],a[2],!detail]);
  for(const f of [1,5,9,13,17]){const mark=!detail||fingers.includes(names[f]);edges.push([start,start+f,mark]);for(let k=f;k<f+3;k++)edges.push([start+k,start+k+1,mark]);}
 }
 for(const [a,b,mark]of edges)if(good(a)&&good(b)){
  c.strokeStyle=mark?ink:'#96a4b8';c.lineWidth=(mark?3:1.3)*devicePixelRatio;c.beginPath();c.moveTo(ox+xy[a][0]*s,oy+xy[a][1]*s);c.lineTo(ox+xy[b][0]*s,oy+xy[b][1]*s);c.stroke();
 }
}
function sync(){if(!result||!ref||!$('feedback-dialog').open||referenceOnly)return;const q=$('query-video'),rv=$('ref-video'),i=near(result.timestamp_us,q.currentTime*1e6),ri=refIndex(i),card=current(i);desired=ref.timestamp_us[ri]/1e6;if(!rv.seeking&&Math.abs(rv.currentTime-desired)>.025)rv.currentTime=desired;let ink=color(card);const visibleSpans=focusCue?.spans||card?.query_spans;const hands=focusCue?.hands||card?.hands;if(visibleSpans?.length&&!visibleSpans.some(([a,b])=>i>=a&&i<=b))ink='#96a4b8';if(view(card)==='difference'&&card.difference_spans?.length&&!card.difference_spans.some(([a,b])=>i>=a&&i<=b))ink='#96a4b8';draw('query-skeleton',result.query_xy?.[i],result.width,result.height,ink,hands,focusCue?.fingers);draw('ref-skeleton',ref.xy[near(ref.timestamp_us,rv.currentTime*1e6)],ref.width,ref.height,'#508cd8',hands,focusCue?.fingers);$('feedback-seek').value=Math.round(q.currentTime/(result.timestamp_us.at(-1)/1e6)*1000);$('feedback-time').textContent=q.currentTime.toFixed(2)+'초'}
function selectCard(i){focusCue=null;clipEnd=null;$('replay-focus').hidden=true;active=i;const c=result.cards[i];desired=null;setReferenceOnly(!c.query_seconds);$('query-video').pause();$('ref-video').pause();$('feedback-message').textContent=result.coaching?.notes?.find(n=>n.card_id===c.id)?.message||c.message||'이 구간은 직접 확인해 주세요.';const diff=view(c)==='difference';$('feedback-instruction').textContent=(diff?'다시 해 볼 방법: ':'참고 동작 안내: ')+(c.instruction||c.title)+(c.query_seconds?'':' · 사용자 영상의 대응 구간은 아직 확인하지 못했어요.');[...$('feedback-cards').children].forEach((el,k)=>el.classList.toggle('selected',i===k));if(c.query_seconds){$('query-video').currentTime=c.query_seconds[0];sync()}else if(c.reference){$('ref-video').currentTime=ref.timestamp_us[c.reference[0]]/1e6}}
function showCue(cue){
 const i=result.cards.findIndex(c=>cue.card_ids.includes(c.id));if(i<0)return;
 selectCard(i);focusCue=cue;setReferenceOnly(false);$('query-video').pause();
 $('feedback-message').textContent=cue.start_seconds.toFixed(1)+'초 부근 · '+cue.observation;
 $('feedback-instruction').textContent='다시 해 볼 방법: '+cue.action;
 $('query-video').currentTime=cue.start_seconds;$('replay-focus').hidden=false;sync();
}
$('replay-focus').onclick=()=>{if(!focusCue)return;setReferenceOnly(false);clipEnd=Math.min(result.timestamp_us.at(-1)/1e6,focusCue.end_seconds+.25);$('query-video').currentTime=Math.max(0,focusCue.start_seconds-.15);$('query-video').play().catch(()=>{});};
export async function openFeedback(data,reference){result=data;ref=reference;active=-1;$('feedback-seek').value=0;$('feedback-time').textContent='0.00초';desired=null;focusCue=null;clipEnd=null;$('replay-focus').hidden=true;$('coaching-summary').textContent=data.coaching?.summary||'';$('coaching-actions').replaceChildren();for(const cue of data.coaching?.cues||[]){const b=document.createElement('button');b.textContent=cue.start_seconds.toFixed(1)+'초 · '+({direction:'이동 방향',shape:'손모양',finger:'손가락',order:'행동 순서'}[cue.kind]||'행동')+' 확인';b.onclick=()=>showCue(cue);$('coaching-actions').append(b);}$('feedback-title').textContent=data.label+' · 확인할 순간';$('feedback-source').textContent=data.source==='user_inference_only'?'내 촬영의 실제 분석 결과 · 관측된 특징 범위의 안내':(data.public_example_name||'공개 자료')+' · G6 연결 확인 결과 · 사용자 촬영이 아닙니다';$('query-caption').textContent=data.source==='user_inference_only'?'내 촬영':data.synthetic?'공개 편집 시험 영상':'공개 비교 영상';$('query-video').src=data.video;$('ref-video').src=ref.video;$('feedback-cards').replaceChildren();$('feedback-bands').replaceChildren();const dur=data.timestamp_us.at(-1)/1e6;for(const [i,c]of data.cards.entries()){const b=document.createElement('button');b.className=view(c)||'uncertain';const title=document.createElement('strong');title.textContent=c.title;const s=document.createElement('small');s.textContent=({similar:'확인한 특징이 비슷함',difference:'차이 관측 · 다시 확인',uncertain:'확인 부족 · 오답 아님',optional:'선택적 마무리'})[view(c)]||'확인 필요';b.append(title,s);b.onclick=()=>selectCard(i);$('feedback-cards').append(b);if(c.query_seconds){const el=document.createElement('span');el.className=view(c)==='difference'&&c.difference_spans?.length?'uncertain':view(c);el.style.left=c.query_seconds[0]/dur*100+'%';el.style.width=Math.max(.5,(c.query_seconds[1]-c.query_seconds[0])/dur*100)+'%';$('feedback-bands').append(el);if(view(c)==='difference')for(const [a,b]of c.difference_spans||[]){const d=document.createElement('span');d.className='difference';d.style.left=data.timestamp_us[a]/1e6/dur*100+'%';d.style.width=Math.max(.5,(data.timestamp_us[b]-data.timestamp_us[a])/1e6/dur*100)+'%';$('feedback-bands').append(d)}}}$('feedback-dialog').showModal();if(data.coaching?.cues?.length)showCue(data.coaching.cues[0]);else if(data.cards.length){let first=data.cards.findIndex(c=>view(c)==='difference');if(first<0)first=data.cards.findIndex(c=>view(c)==='uncertain');selectCard(Math.max(0,first))}}
$('feedback-close').onclick=()=>$('feedback-dialog').close();$('feedback-dialog').addEventListener('close',()=>{$('query-video').pause();$('ref-video').pause()});$('compare-play').onclick=()=>{clipEnd=null;active=-1;setReferenceOnly(false);$('query-video').currentTime=0;$('query-video').play().catch(()=>{});$('feedback-message').textContent='전체 동작을 시간 순서대로 살펴보세요.';$('feedback-instruction').textContent='색은 관측 결과이며, DTW 연결 자체가 동작의 정답을 뜻하지는 않아요.'};$('compare-stop').onclick=()=>{clipEnd=null;$('query-video').pause();};$('feedback-seek').oninput=()=>{clipEnd=null;setReferenceOnly(false);$('query-video').pause();$('query-video').currentTime=Number($('feedback-seek').value)/1000*result.timestamp_us.at(-1)/1e6;sync()};$('query-video').addEventListener('seeked',sync);$('ref-video').addEventListener('seeked',()=>{if(desired!==null&&!$('ref-video').seeking&&Math.abs($('ref-video').currentTime-desired)>.025)$('ref-video').currentTime=desired;else if(result&&ref)draw('ref-skeleton',ref.xy[near(ref.timestamp_us,$('ref-video').currentTime*1e6)],ref.width,ref.height,'#508cd8',focusCue?.hands||current(near(result.timestamp_us,$('query-video').currentTime*1e6))?.hands,focusCue?.fingers)});
function tick(){if(!$('query-video').paused){sync();if(clipEnd!==null&&$('query-video').currentTime>=clipEnd){$('query-video').pause();$('query-video').currentTime=clipEnd;clipEnd=null;}}requestAnimationFrame(tick)}requestAnimationFrame(tick);

// Loading a different video can reset a seek issued before its metadata arrives.
// Reapply the selected observed instant after load; this changes display only.
$('query-video').addEventListener('loadedmetadata',()=>{
 if(!result||!$('feedback-dialog').open)return;
 const at=focusCue?.start_seconds??(active>=0?result.cards[active]?.query_seconds?.[0]:0)??0;
 $('query-video').currentTime=at;sync();
});
$('ref-video').addEventListener('loadeddata',()=>{
 if(!result||!ref||!$('feedback-dialog').open)return;
 if(referenceOnly&&active>=0){const a=result.cards[active].reference?.[0];if(a!==undefined)$('ref-video').currentTime=ref.timestamp_us[a]/1e6;}
 else sync();
});

function setReferenceOnly(value){
 referenceOnly=value;
 let note=document.getElementById('query-unmatched');
 if(!note){note=document.createElement('div');note.id='query-unmatched';note.textContent='이 행동에 해당하는 구간을 확인하지 못했어요. 전체 재생으로 직접 살펴보세요.';$('query-video').parentElement.append(note);}
 note.hidden=!value;$('query-video').hidden=value;$('query-skeleton').hidden=value;
}

