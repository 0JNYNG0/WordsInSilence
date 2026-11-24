from pathlib import Path
import json
U=Path(__file__).resolve().parent;WEB=U/'web'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def sources():return read(WEB/'SOURCES.json')if(WEB/'SOURCES.json').exists()else{}
def catalog():return[dict(label=r['label'],url=f'/hello-examples/{k}.json')for k,r in sources().items()]
def cards():
 from presentation import LABELS
 r=read(WEB/'REFERENCE.json')['clauses'];return[dict(id=k,title=v[0],instruction=v[2],hands=v[3],reference=r['sweep'if k.startswith('sweep_')else'whole'if k=='order'else'fists'])for k,v in LABELS.items()]
def handle_get(handler,route):
 if route=='/ref/SEN0354':
  r=read(WEB/'REFERENCE.json');r.pop('local_video',None);handler.reply(r);return True
 if route=='/hello-media/reference.mp4':handler.file(Path(read(WEB/'REFERENCE.json')['local_video']));return True
 if route=='/public-replay/SEN0354.mp4':handler.file(Path(sources()['normal']['video']));return True
 for k,r in sources().items():
  if route==f'/hello-media/{k}.mp4':handler.file(Path(r['video']));return True
  if route==f'/hello-examples/{k}.json':handler.reply(read(WEB/f'{k}.json'));return True
 return False
