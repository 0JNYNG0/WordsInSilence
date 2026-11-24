"""Larger source-RGB evidence panels; no measurement or label changes."""
from audit import *


def main():
    dest = U / 'ZOOM_MANIFEST.json'
    assert not dest.exists()
    result = rd(U / 'RESULTS.json')
    sources = {r['clip_id']: r for r in rd(OPS / 'word100_rtmw_joint_learning_20261108/SOURCES.json') if r['role'] == 'fit'}
    manifest = []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    crop = (720, 250, 1320, 850)
    for case in result['cases']:
        src = sources[case['clip_id']]
        assert sha(src['feedback_source']) == case['source_sha256']
        cap = cv2.VideoCapture(src['feedback_source'])
        assert cap.isOpened()
        cols = 4
        image = Image.new('RGB', (cols*400, 50+((len(case['frames'])+cols-1)//cols)*425), '#e6eaf0')
        ImageDraw.Draw(image).text((10, 10), case['id']+' | elbow RED, body wrist ORANGE, hand root BLUE, MCP CYAN', font=font, fill='#14243a')
        for i, r in enumerate(case['measurements']):
            cap.set(cv2.CAP_PROP_POS_FRAMES, r['frame'])
            ok, bgr = cap.read()
            assert ok
            im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop(crop).resize((400, 400))
            d = ImageDraw.Draw(im)
            pts = {k: ((v[0]-crop[0])*2/3, (v[1]-crop[1])*2/3) for k, v in r['points_px'].items()}
            d.line([pts['elbow'], pts['body_wrist']], fill='#d35a1e', width=2)
            d.line([pts['elbow'], pts['hand_root']], fill='#226dde', width=1)
            for name, color in [('elbow', '#e63440'), ('body_wrist', '#e59c1c'), ('hand_root', '#206fe5'), ('mcp_centroid', '#08c3d7'), ('moving_palm', '#119b68')]:
                if r['masks'][name]:
                    x, y = pts[name]
                    d.ellipse((x-4, y-4, x+4, y+4), fill=color, outline='white', width=1)
            tile = Image.new('RGB', (400, 425), 'white')
            tile.paste(im, (0, 25))
            ImageDraw.Draw(tile).text((10, 4), 'Original frame '+str(r['frame']), font=font, fill='#14243a')
            image.paste(tile, ((i%cols)*400, 50+(i//cols)*425))
        cap.release()
        path = U / (case['id']+'_zoom.jpg')
        assert not path.exists()
        image.save(path, quality=95)
        manifest.append(dict(id=case['id'], path=str(path), sha256=sha(path), frames=case['frames'], crop=crop))
    with dest.open('x', encoding='utf-8') as f:
        json.dump(dict(rows=manifest, result_sha256=sha(U / 'RESULTS.json'), code_sha256=sha(__file__)), f, indent=2)
    print('Prepared', len(manifest), 'larger FIT evidence sheets')


if __name__ == '__main__':
    main()
