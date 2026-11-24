// Screen-space optical layers anchored to the robot's projected eye positions.
// No circular clipping mask; the underlying 3D environment changes with the projection.
const sat=x=>Math.max(0,Math.min(1,x));
const smooth=x=>{x=sat(x);return x*x*(3-2*x)};
export class Hologram {
  constructor(canvas){this.canvas=canvas;this.ctx=canvas.getContext('2d');this.mode='off';this.p=0;this.reduced=false;}
  set(mode,p,reduced=false){this.mode=mode;this.p=p;this.reduced=reduced;}
  clear(){this.mode='off';this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);}
  draw(eyes,time){
    const c=this.canvas,g=this.ctx,w=c.clientWidth,h=c.clientHeight,d=Math.min(devicePixelRatio||1,1.5);
    if(c.width!==Math.round(w*d)||c.height!==Math.round(h*d)){c.width=Math.round(w*d);c.height=Math.round(h*d);}
    g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);
    if(this.mode==='off'||this.reduced)return;
    const p=this.p;
    if(this.mode==='exit'){
      const y=(-.1+1.2*smooth(p))*h,fade=Math.sin(Math.PI*p);
      const band=g.createLinearGradient(0,y-110,0,y+30);band.addColorStop(0,'rgba(190,216,239,0)');band.addColorStop(.85,`rgba(190,216,239,${.13*fade})`);band.addColorStop(1,'rgba(190,216,239,0)');g.fillStyle=band;g.fillRect(0,y-110,w,140);
      g.strokeStyle=`rgba(221,237,250,${fade*.6})`;g.lineWidth=1;g.beginPath();g.moveTo(0,y);g.lineTo(w,y);g.stroke();
      for(let i=0;i<36;i++){const x=(i+.5)*w/36,py=y-((i*43)%150);g.fillStyle=`rgba(215,232,247,${.2*fade})`;g.fillRect(x,py,2,7+((i*7)%14));}
      return;
    }
    const power=smooth(p/.22)*(1-smooth((p-.78)/.22)), spread=smooth((p-.15)/.48), cx=(eyes[0][0]+eyes[1][0])/2, cy=(eyes[0][1]+eyes[1][1])/2;
    // Two actual sources, with soft pools of light, then broad rays approaching the viewer.
    g.globalCompositeOperation='screen';
    eyes.forEach(([x,y],i)=>{
      const radius=12+power*26;const glow=g.createRadialGradient(x,y,0,x,y,radius);glow.addColorStop(0,`rgba(243,250,255,${power*.95})`);glow.addColorStop(.17,`rgba(198,228,255,${power*.65})`);glow.addColorStop(1,'rgba(173,216,255,0)');g.fillStyle=glow;g.fillRect(x-radius,y-radius,radius*2,radius*2);
      g.strokeStyle=`rgba(206,232,255,${power*.4})`;g.lineWidth=.8;g.beginPath();g.ellipse(x,y,8+power*8,10+power*9,0,0,Math.PI*2);g.stroke();
      if(spread<=0)return;
      const left=x-(x+w*.35)*spread, right=x+(w-x+w*.35)*spread, endY=y+(h-y+h*.30)*spread;
      g.save();g.beginPath();g.moveTo(x-4,y);g.lineTo(left,endY);g.lineTo(right,endY);g.lineTo(x+4,y);g.closePath();g.clip();
      const light=g.createLinearGradient(x,y,x,endY);light.addColorStop(0,`rgba(214,238,255,${power*.18})`);light.addColorStop(.22,`rgba(151,204,245,${power*.09})`);light.addColorStop(.85,`rgba(148,200,239,${power*.15})`);light.addColorStop(1,'rgba(136,196,244,0)');g.fillStyle=light;g.fillRect(-w,y,w*3,h*2);
      // Distinct, gently moving optical ribs communicate a projected volume, not a flat wipe.
      for(let k=0;k<11;k++){const r=k/10,tx=left+(right-left)*r;g.strokeStyle=`rgba(180,222,252,${power*(k%3===0?.13:.045)})`;g.beginPath();g.moveTo(x,y);g.lineTo(tx,endY);g.stroke();}
      for(let k=0;k<8;k++){const z=((k/8+time*.24)%1),e=z*z,yy=y+(endY-y)*e,half=(right-left)*e*.5;g.strokeStyle=`rgba(192,227,253,${power*.22*z})`;g.beginPath();g.moveTo(x-half,yy);g.quadraticCurveTo(cx,yy+16*e,x+half,yy);g.stroke();}
      g.restore();
    });
    // A translucent perspective projection plane traverses the screen and resolves into the room.
    const sheet=smooth((p-.36)/.4)*(1-smooth((p-.79)/.21));
    if(sheet>0){
      const z=smooth((p-.36)/.48),hw=w*(.13+z*.76),hh=h*(.10+z*.61),centerY=cy+(h*.5-cy)*z;
      g.strokeStyle=`rgba(183,222,251,${sheet*.42})`;g.lineWidth=1;g.strokeRect(cx-hw,centerY-hh,hw*2,hh*2);
      const haze=g.createLinearGradient(0,centerY-hh,0,centerY+hh);haze.addColorStop(0,`rgba(145,197,237,${sheet*.03})`);haze.addColorStop(1,`rgba(145,197,237,${sheet*.13})`);g.fillStyle=haze;g.fillRect(cx-hw,centerY-hh,hw*2,hh*2);
      for(let i=1;i<13;i++){const yy=centerY-hh+i*hh*2/13;g.strokeStyle=`rgba(184,223,251,${sheet*.12})`;g.beginPath();g.moveTo(cx-hw,yy);g.lineTo(cx+hw,yy);g.stroke();}
      for(let i=0;i<42;i++){const seed=(i*97)%101/101,xx=cx+(seed-.5)*hw*2,yy=centerY+((((i*37)%97)/97-.5)*hh*2);g.fillStyle=`rgba(208,236,255,${sheet*(.12+seed*.24)})`;g.fillRect(xx,yy,2,2);}
    }
    g.globalCompositeOperation='source-over';
  }
}
