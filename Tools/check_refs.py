import json, os

folder = os.path.join(os.path.dirname(__file__), "videos")
for f in sorted(os.listdir(folder)):
    if f.endswith(".json"):
        d = json.load(open(os.path.join(folder, f), encoding="utf-8"))
        print(f"{d['MotionId']}: Total={d['TotalFrames']} Hand={d['ValidHandFrames']} Combined={d['ValidCombinedFrames']}")
