// Soft bottom-to-top room conversion begins just before MOA turns upward.
const clamp=x=>Math.max(0,Math.min(1,x)),smooth=x=>{x=clamp(x);return x*x*(3-2*x)};
export class Hologram{
 constructor(canvas){this.canvas=canvas;this.ctx=canvas.getContext('2d');this.mode='off';this.p=0;}
 set(mode,p,reduced=false){this.mode=mode;this.p=p;this.reduced=reduced;}
 clear(){this.mode='off';this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);}
 draw(points,time){const c=this.canvas,g=this.ctx,w=c.clientWidth,h=c.clientHeight,d=Math.min(devicePixelRatio||1,1.3);
  if(c.width!==Math.round(w*d)||c.height!==Math.round(h*d)){c.width=Math.round(w*d);c.height=Math.round(h*d);}g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);
  if(this.mode==='off')return;const p=this.p,exit=this.mode==='exit';
  if(this.reduced||exit){g.fillStyle='rgba(15,28,47,'+Math.sin(Math.PI*p)*.74+')';g.fillRect(0,0,w,h);return;}
  if(p<=.27||p>=.78)return;
  const progress=clamp((p-.27)/.38),fade=1-smooth((p-.68)/.10),on=smooth(progress/.16),front=h*(1.12-1.58*progress);
  // Broad feathered edge: lower rows change first, then the change travels upward.
  // A gentle curved leading edge avoids a hard horizontal curtain.
  const bands=96;
  for(let i=0;i<bands;i++){
   const x=(i+.5)/bands,y=front+h*.035*Math.cos(x*Math.PI*2),v=g.createLinearGradient(0,y-h*.20,0,y+h*.43);
   v.addColorStop(0,'rgba(25,47,75,0)');v.addColorStop(.26,'rgba(32,55,86,'+on*fade*.10+')');v.addColorStop(.54,'rgba(27,48,77,'+on*fade*.55+')');v.addColorStop(.82,'rgba(20,38,62,'+on*fade*.91+')');v.addColorStop(1,'rgba(14,28,47,'+on*fade+')');g.fillStyle=v;const l=Math.round(i*w*d/bands)/d,r=Math.round((i+1)*w*d/bands)/d;g.fillRect(l,0,r-l,h);
  }
  // Preserve the subject's readability until it has flown out of the frame.
  if(p<.55&&points.length>3){const [_,center,left,top]=points,rx=Math.abs(center[0]-left[0])*1.24,ry=Math.max(rx*.79,Math.abs(center[1]-top[1])*1.24);
   if(rx>0&&ry>0){g.save();g.globalCompositeOperation='destination-out';g.translate(center[0],center[1]);g.scale(rx,ry);const v=g.createRadialGradient(0,0,0,0,0,1);v.addColorStop(0,'rgba(0,0,0,.72)');v.addColorStop(.55,'rgba(0,0,0,.60)');v.addColorStop(1,'rgba(0,0,0,0)');g.fillStyle=v;g.fillRect(-1,-1,2,2);g.restore();}}
 }
}
