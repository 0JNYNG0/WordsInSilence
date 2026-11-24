from common import *
assert read(U/'CONNECT_GATE.json')['passed']
p=O/'web_tutorial_live_integration_20261113/site/feedback.js';s=p.read_text('utf-8-sig')
start=s.index('function draw(');end=s.index('\nfunction sync()',start)
s=s[:start]+'''function draw(id,xy,w,h,ink,hands,fingers){
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
}'''+s[end:]
s=s.replace("color(card);const visibleSpans", "color(card);const visibleSpans")
s=s.replace("ink,hands);draw('ref-skeleton'", "ink,hands,focusCue?.fingers);draw('ref-skeleton'")
s=s.replace("'#508cd8',hands);", "'#508cd8',hands,focusCue?.fingers);")
s=s.replace("?.hands)});", "?.hands,focusCue?.fingers)});")
s=s.replace("{direction:'이동 방향',shape:'손모양',order:'행동 순서'}", "{direction:'이동 방향',shape:'손모양',finger:'손가락',order:'행동 순서'}")
p.write_text(s,'utf-8')
server=O/'web_tutorial_live_integration_20261113/server.py';s=server.read_text('utf-8-sig');s=s.replace("기존 RTMW TCN + G4 통합 판정 / G6 촬영 연결","기존 RTMW TCN + 검증된 방향·손가락 보완 / G6 촬영 연결");server.write_text(s,'utf-8')
files=[p,server,O/'word100_live_g6_20261114/engine.py',O/'word100_live_g6_20261114/g6_runtime_adapter.py']
write(U/'CONNECTION.json',dict(files={str(p):dict(before=read(U/'BEFORE_PINS.json')[str(p)],after=sha(p))for p in files},scope='Conditional handshape-detail refinement and verified direction supplement; oldTCN/DTW/capture/modelsunchanged',physical_camera_started=False))
print('connected',len(files))
