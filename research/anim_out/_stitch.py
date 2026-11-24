
import cv2, os, numpy as np
views = ['front', 'side', 'angle']
rd = r"C:\MyUnityProjects\WordsInSilence\research\anim_out\render_review"
out = r"C:\MyUnityProjects\WordsInSilence\research\anim_out\Anim_SIGN001_Greeting_review.mp4"
w = None
for fi in range(1, 124):
    row = []
    for v in views:
        p = os.path.join(rd, v, f"{fi:04d}.png")
        img = cv2.imread(p)
        if img is None: img = np.zeros((480,640,3), dtype=np.uint8)
        cv2.putText(img, v, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f"f={fi}/123", (10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        row.append(img)
    canvas = np.hstack(row)
    if w is None:
        h2, w2 = canvas.shape[:2]
        w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w2, h2))
    w.write(canvas)
w.release()
print(f"Encoded: {out} ({os.path.getsize(out)//1024} KB)")
