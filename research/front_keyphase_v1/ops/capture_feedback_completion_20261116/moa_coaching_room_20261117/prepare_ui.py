from pathlib import Path
D=Path(__file__).resolve().parent
p=D/'site/index.html';s=p.read_text('utf-8')
s=s.replace('</head>','<link rel="stylesheet" href="companion.css"></head>')
start=s.index('<a id="g5-review-link"');end=s.index('<button id="motion"',start)
s=s[:start]+s[end:]
s=s.replace('눈빛','빛').replace('<h2>모아와 함께, 천천히</h2>','<h2>모아와 함께 연습하기</h2>')
s=s.replace('<div id="npc-scan"', '<section id="angle-views" hidden aria-label="같은 시범 동작의 좌우 45도 보기"><div class="angle-heading">다른 각도에서 보기 <span>동시에 재생돼요</span></div><div class="angle-pair"><figure id="angle-left"><figcaption>왼쪽 45°</figcaption></figure><figure id="angle-right"><figcaption>오른쪽 45°</figcaption></figure></div></section>\n  <div id="npc-scan"')
s=s.replace('<div class="capture-buttons">','<div class="capture-buttons"><button id="capture-toggle" disabled><span id="capture-label">촬영</span><kbd>Space</kbd></button>')
s=s.replace('확인할 순간 보기 →','분석 결과 보기 →')
s=s.replace('촬영 중 뽑은 관절을 재사용하고 있어요.','조금만 기다려 줘. 손모양과 움직임을 확인하고 있어.')
s=s.replace('<span class="key-tip">Space · 문장 바로 보기 / 다음</span>','<span class="key-tip">Space · 문장 바로 보기 / 다음</span><button id="skip-intro">바로 연습하기 ↗</button>')
s=s.replace('<footer>','<footer hidden>')
p.write_text(s,encoding='utf-8')

p=D/'site/scene.js';s=p.read_text('utf-8')
s=s.replace("let captureLocked=false;","let captureLocked=false,previewActive=false,lastDraw=0,mood='idle',moodUntil=0;")
s=s.replace("'avatar-controls'])","'avatar-controls','angle-views'])")
s=s.replace("setScene('projecting','practice');label('HOLOGRAM PROJECTOR','모아가 공간을 열고 있어요.','눈빛을 따라, 새로운 연습 공간으로.');", "setScene('projecting','practice');label('PRACTICE ROOM','함께 연습하러 가 볼까?','');")
start=s.index('    transitionP=p;dark=');end=s.index('\n  });',start)
s=s[:start]+'''    transitionP=p;dark=ease((p-.12)/.56);
    // Approach the viewer, then sweep up and out. Re-enter from the upper right.
    if(p<.62){const a=ease(p/.62);robotX=-.10*Math.sin(a*Math.PI);robotY=1.48-.16*Math.sin(a*Math.PI)+2.7*a*a;robotZ=1.15*Math.sin(a*Math.PI*.7);robotScale=.65+.16*Math.sin(a*Math.PI);robotYaw=-.08*Math.sin(a*Math.PI);}
    else{const a=ease((p-.62)/.38);robotX=mix(2.5,1.13,a);robotY=mix(2.9,1.61,a);robotZ=mix(-1.4,-.85,a);robotScale=mix(.2,.32,a);robotYaw=mix(-.65,-.3,a);}
    const move=ease((p-.55)/.45);zoom=mix(0,1.7,move);targetY=mix(1.15,1.49,move);
    if(p>.36)document.body.classList.add('dark');
    if(p>.73){npcVisible=true;npcReveal=ease((p-.73)/.27);}
''' +s[end:]
s=s.replace("robotYaw=0;document.body.classList.add('dark','focus-scene');", "robotYaw=-.3;document.body.classList.add('dark','focus-scene');")
s=s.replace("$('#avatar-controls').hidden=false;", "$('#avatar-controls').hidden=false;$('#angle-views').hidden=false;")
s=s.replace("say('참고 동작을 보고 따라 해 봐. 촬영을 마치면 확인할 부분을 함께 살펴볼게.');", "say('먼저 가운데 동작을 살펴봐. 준비되면 Space를 눌러 줘. 끝나고 손을 내리면 내가 확인해 줄게.');")
s=s.replace("setScene('quest');button", "robotYaw=-.3;setScene('quest');button")
s=s.replace("if(e.code==='Space'&&!['BUTTON'].includes", "if(e.code==='Space'&&!['practice','quest'].includes(scene)&&!['BUTTON'].includes")
s=s.replace("const pause3D=captureLocked||document.hidden", "const pause3D=captureLocked||document.hidden")
s=s.replace("const dt=(now-last)/1000;", "if(previewActive&&now-lastDraw<1000/12){last=now;raf=requestAnimationFrame(frame);return;}lastDraw=now;\n      const dt=(now-last)/1000;")
s=s.replace("const robotRoot=model(rx,ry+bob,robotZ,rs,robotYaw,tilt);", "const response=now<moodUntil?Math.sin((moodUntil-now)*.009)*.06:0;\n      const robotRoot=model(rx,ry+bob+(mood==='success'?Math.abs(response):0),robotZ,rs,robotYaw+(typing?.025*Math.sin(t*4):0),tilt+(mood==='uncertain'?response:response*.3));")
s=s.replace("      optics.set(transitionMode,transitionP,reduced);", "      const neck=renderer.project([robotRoot[12],robotRoot[13]-.65*rs,robotRoot[14]]);\n      if(scene==='practice'&&demoAsset&&!$('#angle-views').hidden){for(const [id,side] of [['angle-left',-1],['angle-right',1]]){const rect=$('#'+id).getBoundingClientRect();renderer.inset([side*1.7,1.8,1.7],[0,1.55,0],rect,1,()=>renderer.asset(demoAsset,model(0,0,0,1.12)));}}\n      optics.set(transitionMode,transitionP,reduced);")
start=s.index('      // Eye centers');end=s.index('      raf=requestAnimationFrame(frame);',start)
s=s[:start]+"      optics.draw([neck,neck],t);\n"+s[end:]
s=s.replace("window.addEventListener('coach:stop-gesture',stopGesture);", "window.addEventListener('coach:stop-gesture',stopGesture);\nwindow.addEventListener('coach:preview',e=>{previewActive=!!e.detail;});\nwindow.addEventListener('coach:reaction',e=>{mood=e.detail;moodUntil=performance.now()+2400;});\n$('#skip-intro').onclick=openPractice;")
p.write_text(s,encoding='utf-8')

p=D/'site/renderer.js';s=p.read_text('utf-8');i=s.index('\nproject(p)')
s=s[:i]+'''
inset(eye,target,rect,dark,draw){const g=this.g,d=this.canvas.width/this.canvas.clientWidth,old=this.vp;
const x=Math.round(rect.left*d),y=Math.round((this.canvas.clientHeight-rect.bottom)*d),w=Math.round(rect.width*d),h=Math.round(rect.height*d);
g.enable(g.SCISSOR_TEST);g.scissor(x,y,w,h);g.viewport(x,y,w,h);g.clearColor(.045,.065,.10,1);g.clear(g.COLOR_BUFFER_BIT|g.DEPTH_BUFFER_BIT);
this.vp=mul(perspective(w/h),look(eye,target));g.uniformMatrix4fv(this.u.vp,false,this.vp);g.uniform3fv(this.u.eye,eye);g.uniform1f(this.u.dark,dark);draw();
g.disable(g.SCISSOR_TEST);g.viewport(0,0,this.canvas.width,this.canvas.height);this.vp=old;g.uniformMatrix4fv(this.u.vp,false,old);}
''' +s[i:];p.write_text(s,encoding='utf-8')
