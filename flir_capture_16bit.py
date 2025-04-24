import subprocess
import numpy as np
import cv2

# Параметри
WIDTH, HEIGHT = 640, 512
FFMPEG_BIN = "ffmpeg"
DEVICE_NAME = "FLIR Video"

# Команда ffmpeg
cmd = [
    FFMPEG_BIN,
    "-f", "dshow",
    "-video_size", f"{WIDTH}x{HEIGHT}",
    "-framerate", "30",
    "-pixel_format", "y16",
    "-i", f"video={DEVICE_NAME}",
    "-vframes", "1",            # тільки 1 кадр
    "-f", "rawvideo",
    "pipe:1"
]

print("🚀 Захоплення одного 16-бітного кадру з FLIR Boson...")

# Запуск ffmpeg
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# Читання точного розміру кадру
raw = process.stdout.read(WIDTH * HEIGHT * 2)
frame = np.frombuffer(raw, dtype=np.uint16)

# Перевірка
if frame.size != WIDTH * HEIGHT:
    print(f"❌ Неправильний розмір кадру: {frame.size}, очікується: {WIDTH * HEIGHT}")
    exit()

# Формування 2D зображення
frame = frame.reshape((HEIGHT, WIDTH))

# Масштабування до 8-біт для перегляду
frame_scaled = cv2.convertScaleAbs(frame, alpha=(255.0/65535.0))
colorized = cv2.applyColorMap(frame_scaled, cv2.COLORMAP_JET)

# Вивід
cv2.imshow("🌡️ FLIR Thermal Frame (16-bit Scaled)", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("🌡️ FLIR Thermal Frame (16-bit)", frame, cmap='gray' )