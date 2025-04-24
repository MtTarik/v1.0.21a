
import subprocess
import numpy as np
import cv2

WIDTH, HEIGHT = 640, 512

# Команда ffmpeg для потокового формату yuv420p
CMD = [
    "ffmpeg",
    "-f", "dshow",
    "-video_size", f"{WIDTH}x{HEIGHT}",
    "-pixel_format", "yuv420p",
    "-i", "video=FLIR Video",
    "-vcodec", "rawvideo",
    "-f", "rawvideo",
    "pipe:1"
]

print("🎥 Потокове зчитування з FLIR Boson через yuv420p...")
pipe = subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

frame_size = int(WIDTH * HEIGHT * 3 / 2)  # Y + UV

while True:
    raw = pipe.stdout.read(frame_size)
    if len(raw) < frame_size:
        print("❌ Потік обірвано або недостатньо байтів.")
        break

    # Витягуємо Y-площину
    y_plane = np.frombuffer(raw[:WIDTH*HEIGHT], dtype=np.uint8).reshape((HEIGHT, WIDTH))

    # Псевдоколір
    color = cv2.applyColorMap(y_plane, cv2.COLORMAP_JET)

    # Відображення
    cv2.imshow("🔥 FLIR Thermal Stream (Y-channel)", color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.terminate()
cv2.destroyAllWindows()
