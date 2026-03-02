import cv2
from pathlib import Path

VIDEO_DIR = Path("data/Train/1")
frame_paths = sorted(VIDEO_DIR.glob("*.jpg"))

for p in frame_paths:
    frame = cv2.imread(str(p))
    if frame is None:
        continue
    cv2.imshow("clip", frame)

    # 500 ms = 0.5 s
    if cv2.waitKey(500) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()