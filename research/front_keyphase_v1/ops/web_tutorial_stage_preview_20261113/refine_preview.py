from pathlib import Path
import json,struct
U=Path(__file__).resolve().parent
p=U/'site/renderer.js'
s=p.read_text(encoding='utf-8')
s=s.replace('layout(location=3) in vec4 joints;layout(location=4) in vec4 weights;', 'layout(location=3) in vec4 joints;layout(location=4) in vec4 weights;layout(location=5) in vec4 joints2;layout(location=6) in vec4 weights2;')
s=s.replace('bones[int(joints.w)]*weights.w):node;', 'bones[int(joints.w)]*weights.w+bones[int(joints2.x)]*weights2.x+bones[int(joints2.y)]*weights2.y+bones[int(joints2.z)]*weights2.z+bones[int(joints2.w)]*weights2.w):node;')
s=s.replace("['weights',4,4]", "['weights',4,4],['joints2',5,4],['weights2',6,4]")
s=s.replace('if(!p[key])continue;', 'if(!p[key]){g.disableVertexAttribArray(loc);g.vertexAttrib4f(loc,0,0,0,0);continue;}')
s=s.replace('index:p.indices!==undefined?acc(p.indices):null,material:base', 'joints2:a.JOINTS_1!==undefined?acc(a.JOINTS_1):null,weights2:a.WEIGHTS_1!==undefined?acc(a.WEIGHTS_1):null,index:p.indices!==undefined?acc(p.indices):null,material:base')
p.write_text(s,encoding='utf-8')
p=U/'site/app.js';s=p.read_text(encoding='utf-8')
s=s.replace('gesture=false;npcVisible=false;', 'gesture=false;gestureTime=0;npcVisible=false;')
s=s.replace('gesture=false;robotVisible=true;', 'gesture=false;gestureTime=0;robotVisible=true;')
s=s.replace('mix(0,1.22,e)', 'mix(0,1.10,e)').replace('mix(0,-.65,e)', 'mix(0,-.95,e)').replace('mix(.65,.42,e)', 'mix(.65,.34,e)').replace('zoom=mix(0,.65,e)', 'zoom=mix(0,1.6,e)')
s=s.replace("zoom=zoom>.8?.65:1.1;$('#zoom').textContent=zoom>.8?'−':'＋'", "zoom=zoom>1.8?1.6:2.0;$('#zoom').textContent=zoom>1.8?'−':'＋'")
s=s.replace('renderer.begin([0,1.72,4.5-zoom],[0,1.15,0],dark);', 'renderer.begin([0,1.72,4.5-zoom],[0,1.15+Math.min(zoom/1.6,1)*.17,0],dark);')
s=s.replace("document.body.classList.add('ready');", "document.body.classList.add('ready');$('#loading').setAttribute('aria-hidden','true');")
# Eye origin in the existing robot's face, two light sources rather than a mouth glow.
s=s.replace('renderer.project([robotX,1.5,robotZ+.45])', 'renderer.project([robotX,1.61,robotZ+.45])')
s=s.replace("$('#eye-light').style.left=pt[0]+'px';", "$('#eye-light').style.left=pt[0]+'px';")
p.write_text(s,encoding='utf-8')
p=U/'site/style.css';s=p.read_text(encoding='utf-8')
s+='\n#eye-light{background:transparent;box-shadow:none;width:75px;height:16px}#eye-light:before,#eye-light:after{content:"";position:absolute;top:0;width:12px;height:16px;border-radius:50%;background:#dceeff;box-shadow:0 0 28px 13px #c3dffb80}#eye-light:before{left:0}#eye-light:after{right:0}#avatar-controls{left:7%;transform:none;bottom:230px}#avatar-controls small{left:0;transform:none}@media(max-width:850px){#avatar-controls{left:5%}}@media(max-width:560px){#avatar-controls{left:50%;transform:translateX(-50%)}#avatar-controls small{left:50%;transform:translateX(-50%)}}\n'
p.write_text(s,encoding='utf-8')
for name in ['robot','npc']:
    b=(U/'site/assets'/f'{name}.glb').read_bytes();n=struct.unpack_from('<I',b,12)[0];j=json.loads(b[20:20+n]);print(name, 'nodes',len(j['nodes']),'attrs',j['meshes'][0]['primitives'][0]['attributes'].keys(),'skins',[len(x['joints']) for x in j.get('skins',[])])
