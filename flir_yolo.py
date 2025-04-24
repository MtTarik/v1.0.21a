# FLIR 16-bit + YOLO Detection
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

WIDTH, HEIGHT = 640, 512
DEVICE_NAME = "FLIR Video"
FFMPEG_BIN = "ffmpeg"

model = YOLO("yolov8n.pt")

ffmpeg_cmd = [
    FFMPEG_BIN, "-f", "dshow",
    "-video_size", f"{WIDTH}x{HEIGHT}",
    "-pixel_format", "gray16le",
    "-i", f"video={DEVICE_NAME}",
    "-vframes", "1", "-f", "rawvideo",
    "-pix_fmt", "gray16le", "-"
]

print("🚀 Стартуємо ffmpeg pipe...")
pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
frame_size = WIDTH * HEIGHT * 2
raw_frame = pipe.stdout.read(frame_size)
pipe.kill()

if len(raw_frame) != frame_size:
    print("❌ Недостатньо даних у потоці")
    exit()

frame16 = np.frombuffer(raw_frame, dtype=np.uint16).reshape((HEIGHT, WIDTH))
frame_norm = ((frame16 - frame16.min()) / (frame16.ptp() + 1e-5) * 255).astype(np.uint8)
frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

results = model.predict(source=frame_rgb, conf=0.5, device="cuda" if model.device.type == "cuda" else "cpu")
annotated = results[0].plot()

cv2.imshow("🧠 FLIR Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
