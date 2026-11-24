"""Public front-camera data for feedback; multi-view data belongs to avatars."""
from pathlib import Path
import re

def require_front(row):
    cid=row.get('clip_id') or row.get('id') or row.get('clip')
    if not isinstance(cid,str) or not re.fullmatch(r'NIA_SL_(?:WORD|SEN)\d+_REAL\d+_F',cid):
        raise ValueError(f'Feedback source requires an explicit public front-camera id: {cid}')
    for key in ['raw','raw_path','video','channels','channel_path']:
        path=str(row.get(key,''))
        if any(x in path.lower() for x in ['avatar_teaching','video_only','user_inference','wis_action_demo','temp\\','temp/']):
            raise ValueError(f'Feedback source must not come from avatar/private/capture storage: {key}')
    if row.get('role') in ['user','private'] or row.get('source')=='user_inference_only':
        raise ValueError('User recordings cannot be learning, tuning or selection sources')
    return cid

def check_rows(rows):
    for row in rows:require_front(row)
    return len(rows)
