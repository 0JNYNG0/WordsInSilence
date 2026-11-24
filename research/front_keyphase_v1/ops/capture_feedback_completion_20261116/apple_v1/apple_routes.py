from pathlib import Path
import json
U=Path(__file__).resolve().parent;WEB=U/'web'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def sources():return read(WEB/'SOURCES.json')if(WEB/'SOURCES.json').exists()else{}
def catalog():return[dict(label=r['label'],url=f'/apple-examples/{k}.json')for k,r in sources().items()]
def cards():
 from apple_presentation import LABELS
 r=read(WEB/'REFERENCE.json')['clauses'];return[dict(id=k,title=v[0],instruction=v[2],hands=v[3],reference=r['mouth'if k in['index_shape','mouth_motion']else'whole'if k=='sequence'else'wipe'])for k,v in LABELS.items()]
def handle_get(handler,route):
 if route=='/ref/WORD0099':
  r=read(WEB/'REFERENCE.json');r.pop('local_video',None);handler.reply(r);return True
 if route=='/apple-media/reference.mp4':handler.file(Path(read(WEB/'REFERENCE.json')['local_video']));return True
 if route=='/public-replay/WORD0099.mp4':handler.file(Path(sources()['normal']['video']));return True
 for k,r in sources().items():
  if route==f'/apple-media/{k}.mp4':handler.file(Path(r['video']));return True
  if route==f'/apple-examples/{k}.json':handler.reply(read(WEB/f'{k}.json'));return True
 return False
