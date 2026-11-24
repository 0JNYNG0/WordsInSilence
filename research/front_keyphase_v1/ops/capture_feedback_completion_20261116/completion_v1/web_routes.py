from pathlib import Path
import json
V=Path(__file__).resolve().parent;WEB=V/'web'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def sources():return read(WEB/'SOURCES.json')
def catalog():
 return [dict(label=r['label'],url=f'/basic-examples/{k}.json')for k,r in sources().items()]
def cards():
 from presentation import LABELS
 return [dict(id=k,title=v[0],instruction=v[2],hands=v[3],reference=read(WEB/'REFERENCE.json')['action_span'])for k,v in LABELS.items()]
def handle_get(handler,route):
 if route=='/ref/WORD1290':
  r=read(WEB/'REFERENCE.json');r.pop('local_video',None);handler.reply(r);return True
 if route=='/public-replay/WORD1290.mp4':handler.file(Path(sources()['normal']['video']));return True
 if route=='/basic-media/reference.mp4':handler.file(Path(read(WEB/'REFERENCE.json')['local_video']));return True
 for key,r in sources().items():
  if route==f'/basic-media/{key}.mp4':handler.file(Path(r['video']));return True
  if route==f'/basic-examples/{key}.json':handler.reply(read(WEB/f'{key}.json'));return True
 return False
