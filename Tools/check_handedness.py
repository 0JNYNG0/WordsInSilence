import json, os

folder = os.path.join(os.path.dirname(__file__), "videos")
for fname in sorted(os.listdir(folder)):
    if not fname.endswith(".json"):
        continue
    d = json.load(open(os.path.join(folder, fname), encoding="utf-8"))
    frames = d.get("Frames", [])
    left  = sum(1 for f in frames if f.get("HasLeftHand"))
    right = sum(1 for f in frames if f.get("HasRightHand"))
    both  = sum(1 for f in frames if f.get("HasLeftHand") and f.get("HasRightHand"))
    total = len(frames)
    print(f"{d['MotionId']}: Total={total}  Left={left}  Right={right}  Both={both}")
