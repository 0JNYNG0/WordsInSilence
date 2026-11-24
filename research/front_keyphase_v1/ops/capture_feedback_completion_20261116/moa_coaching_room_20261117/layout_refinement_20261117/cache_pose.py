from pathlib import Path
p=Path(__file__).resolve().parent/'site/renderer.js'
t=p.read_text('utf-8-sig')
a='pose(a,time){for(const c of a.animations[0]?.channels||[])'
b='pose(a,time){if(a.poseTime===time)return;a.poseTime=time;a.skinMatrices=null;for(const c of a.animations[0]?.channels||[])'
assert t.count(a)==1;t=t.replace(a,b)
a='bones=new Float32Array(s.joints.length*16);s.joints.forEach((j,i)=>bones.set(mul(a.nodes[j].world,s.ib.slice(i*16,i*16+16)),i*16));'
b='a.skinMatrices??=new Map();bones=a.skinMatrices.get(n.skin);if(!bones){bones=new Float32Array(s.joints.length*16);s.joints.forEach((j,i)=>bones.set(mul(a.nodes[j].world,s.ib.slice(i*16,i*16+16)),i*16));a.skinMatrices.set(n.skin,bones);}'
assert t.count(a)==1;t=t.replace(a,b)
p.write_text(t,encoding='utf-8')
print('Skin matrices shared across the three views; invalidated only when the pose changes')
