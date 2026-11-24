
import cv2, os, glob, numpy as np
views = ['front', 'side', 'angle45']
render_dir = r"C:\MyUnityProjects\WordsInSilence\Tools\s04_out_v3a\render_review"
out_path = r"C:\MyUnityProjects\WordsInSilence\Tools\s04_out_v3a\SIGN001_blender_render_review.mp4"
fps = 30
total = 157
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None
for fi in range(1, total + 1):
    row = []
    for view in views:
        pattern = os.path.join(render_dir, view, f"{fi:04d}.png")
        img = cv2.imread(pattern)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add view label
        cv2.putText(img, view, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f"f={fi}/{total}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        row.append(img)
    canvas = np.hstack(row)
    if writer is None:
        h, w = canvas.shape[:2]
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    writer.write(canvas)
writer.release()
print(f"Encoded: {out_path} ({os.path.getsize(out_path)//1024} KB)")
