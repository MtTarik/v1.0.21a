
import subprocess
import numpy as np
import cv2

# Параметри відео з FLIR Boson
WIDTH, HEIGHT = 640, 512
PIX_FMT = "gray16le"
CMD = [
    "ffmpeg",
    "-f", "dshow",
    "-video_size", f"{WIDTH}x{HEIGHT}",
    "-pixel_format", "y16",
    "-i", "video=FLIR Video",
    "-vcodec", "rawvideo",
    "-an",
    "-sn",
    "-f", "rawvideo",
    "pipe:1"
]

print("🚀 Стартуємо ffmpeg pipe...")
pipe = subprocess.Popen(CMD, stdout=subprocess.PIPE, bufsize=10**8)

while True:
    # Читаємо точно один кадр
    raw = pipe.stdout.read(WIDTH * HEIGHT * 2)
    if len(raw) < WIDTH * HEIGHT * 2:
        print("❌ Недостатньо даних у потоці")
        break

    # Преобразуємо з 16-бітного формату
    frame = np.frombuffer(raw, dtype=np.uint16).reshape((HEIGHT, WIDTH))

    # Масштабуємо для перегляду (до 8 біт)
    view = cv2.convertScaleAbs(frame, alpha=(255.0 / 65535.0))
    color = cv2.applyColorMap(view, cv2.COLORMAP_JET)

    # Вивід
    cv2.imshow("🔥 FLIR Thermal Stream", color)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

pipe.terminate()
cv2.destroyAllWindows()
