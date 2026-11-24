"""
Tools/s04_encode_mp4.py — PNG 시퀀스 → MP4 (cv2 VideoWriter)

사용:
  python s04_encode_mp4.py <frame_dir> <output_mp4> [fps]

frame_dir: 0001.png, 0002.png ... 형식의 이미지 시퀀스 디렉터리
"""
import cv2, glob, os, sys

def encode(frame_dir, out_mp4, fps=30):
    pngs = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not pngs:
        print(f"[오류] PNG 없음: {frame_dir}")
        return 1
    first = cv2.imread(pngs[0])
    if first is None:
        print(f"[오류] 이미지 읽기 실패: {pngs[0]}")
        return 1
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    for p in pngs:
        img = cv2.imread(p)
        if img is not None:
            writer.write(img)
    writer.release()
    sz = os.path.getsize(out_mp4) // 1024
    print(f"[저장] {out_mp4} ({sz}KB, {len(pngs)}프레임, {fps}fps)")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용: python s04_encode_mp4.py <frame_dir> <out_mp4> [fps]")
        sys.exit(1)
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    sys.exit(encode(sys.argv[1], sys.argv[2], fps))
