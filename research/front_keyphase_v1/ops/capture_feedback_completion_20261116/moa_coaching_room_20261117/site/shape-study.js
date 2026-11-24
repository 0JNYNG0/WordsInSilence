import {StageRenderer,model,I} from './renderer.js';
const wisps=[0,1,2,3].map(seed=>{const c=document.createElement('canvas');c.width=c.height=192;const g=c.getContext('2d');
 for(let j=0;j<15;j++){const a=j*2.399+seed,r=10+Math.sqrt(j/15)*34,x=96+Math.cos(a)*r,y=96+Math.sin(a)*r,s=26+(j%4)*6;
  const v=g.createRadialGradient(x-s*.20,y-s*.25,0,x,y,s);v.addColorStop(0,'rgba(91,113,140,.9)');v.addColorStop(.38,'rgba(48,70,100,.87)');v.addColorStop(.76,'rgba(25,43,70,.56)');v.addColorStop(1,'rgba(25,43,70,0)');g.fillStyle=v;g.fillRect(x-s,y-s,s*2,s*2);}
 return c;});
function smoke(c,origin,kind){const w=c.clientWidth,h=c.clientHeight;c.width=w*1.5;c.height=h*1.5;const g=c.getContext('2d');g.scale(1.5,1.5);const [cx,sy]=origin,base=h*.83,span=base-sy;
 function puff(x,y,r,i,alpha=.82){g.save();g.globalAlpha=alpha;g.translate(x,y);g.rotate(i*.61);g.drawImage(wisps[i%4],-r,-r,r*2,r*2);g.restore();}
 if(kind==='a'){
  for(let row=0;row<17;row++){const t=row/16,half=12+w*.30*Math.pow(t,1.15),n=Math.max(1,Math.ceil(half/19));
   for(let k=-n;k<=n;k++){const x=cx+(k/n)*half,y=sy+span*t+Math.sin(k*1.7+row)*7,r=26+48*t;puff(x,y,r,row*31+k+n,.77);}}
 }else{
  // A short rising stem joins a broad, billowing base, with no smooth tube.
  for(let row=0;row<13;row++){const t=row/12,x=cx+Math.sin(t*3.2)*10,y=sy+span*.73*t;puff(x,y,29+20*t,row,.88);}
  for(let row=0;row<4;row++){const half=w*(.22+.035*row),y=base-span*.18+row*17;
   for(let k=-6;k<=6;k++){const x=cx+half*k/6;puff(x,y-Math.cos(k/6*Math.PI)*24,65+9*Math.sin(k+row),row*17+k+6,.84);}}
 }
 // Small attached cap joins the projector instead of a detached floating shape.
 puff(cx,sy+8,24,2,.90);
}
try{
 for(const id of ['a','b']){const r=new StageRenderer(document.querySelector('#'+id+'-world'));const asset=await r.load('assets/robot.glb');const root=model(0,2.04,0,.48,0,-.03,-.10);r.begin([0,1.72,4.5],[0,1.15,0],0);r.asset(asset,root);
  const q=[0,1,2].map(i=>root[12+i]-1.04*root[4+i]);smoke(document.querySelector('#'+id+'-fx'),r.project(q),id);}
 document.body.dataset.ready='true';
}catch(e){document.body.dataset.error=String(e);throw e;}
