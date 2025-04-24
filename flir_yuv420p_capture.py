
import subprocess
import numpy as np
import cv2

WIDTH, HEIGHT = 640, 512

# Команда ffmpeg для формату yuv420p
CMD = [
    "ffmpeg",
    "-f", "dshow",
    "-video_size", f"{WIDTH}x{HEIGHT}",
    "-pixel_format", "yuv420p",
    "-i", "video=FLIR Video",
    "-frames:v", "1",
    "-vcodec", "rawvideo",
    "-f", "rawvideo",
    "pipe:1"
]

print("🚀 Захоплення 1 кадру з FLIR через yuv420p...")
pipe = subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

frame_size = int(WIDTH * HEIGHT * 3 / 2)  # Y + UV
raw = pipe.stdout.read(frame_size)

if len(raw) < frame_size:
    print("❌ Кадр не зчитано повністю")
    exit()

# Витягуємо тільки Y-площину
y_plane = np.frombuffer(raw[:WIDTH*HEIGHT], dtype=np.uint8).reshape((HEIGHT, WIDTH))

# Псевдоколір
colored = cv2.applyColorMap(y_plane, cv2.COLORMAP_JET)

# Показ кадру
cv2.imshow("FLIR Boson (Y-pixel only)", colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
