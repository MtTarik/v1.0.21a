# import cv2

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Помилка: Не вдалося отримати доступ до веб-камери.")
#     exit()


# while True:
    
#     ret, frame = cap.read()
#     if not ret:
#         print("Помилка: Не вдалося зчитати кадр.")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


#     cv2.imshow('Webcam', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# # тестова версія для обробкти

import subprocess
import numpy as np
import cv2

WIDTH, HEIGHT = 640, 512
DEVICE_NAME = "FLIR Video"

# Можливі формати: формат → (dtype, байтів на піксель)
PIXEL_FORMATS = {
    'gray16le': (np.uint16, 2),
    'yuv420p': (np.uint8, 1.5),
    'nv12': (np.uint8, 1.5),
}

# FFmpeg запуск для різних форматів
ffmpeg_args_variants = [
    ["-pixel_format", "gray16le"],
    ["-pixel_format", "yuv420p"],
    []
]

def start_pipe(extra_args):
    base = [
        "ffmpeg",
        "-f", "dshow",
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", "30",
        "-i", f"video={DEVICE_NAME}",
        "-vcodec", "rawvideo",
        "-an", "-sn", "-f", "rawvideo"
    ]
    return subprocess.Popen(base + extra_args + ["-"], stdout=subprocess.PIPE, bufsize=10**8)

def try_read(pipe, pixel_bytes):
    raw = pipe.stdout.read(int(WIDTH * HEIGHT * pixel_bytes))
    return raw if len(raw) == int(WIDTH * HEIGHT * pixel_bytes) else None

def detect_and_display():
    for fmt in ffmpeg_args_variants:
        print(f"🔍 Проба формату: {fmt if fmt else 'автоматично'}")
        pipe = start_pipe(fmt)
        raw_sample = pipe.stdout.read(WIDTH * HEIGHT * 2)

        dtype, pixel_bytes = None, None
        for fmt_name, (dt, bpp) in PIXEL_FORMATS.items():
            if len(raw_sample) == int(WIDTH * HEIGHT * bpp):
                dtype, pixel_bytes = dt, bpp
                print(f"✅ Формат визначено: {fmt_name}, dtype={dt}, {bpp} bytes/pixel")
                break

        if dtype is None:
            print("❌ Не вдалося визначити формат.")
            pipe.terminate()
            continue

        while True:
            raw = try_read(pipe, pixel_bytes)
            if raw is None:
                print("❌ Потік перервано або недостатньо байтів.")
                break

            if dtype == np.uint16:
                frame = np.frombuffer(raw, dtype=np.uint16).reshape((HEIGHT, WIDTH))
                display = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            elif dtype == np.uint8 and pixel_bytes == 1.5:
                y_plane = np.frombuffer(raw, dtype=np.uint8).reshape((int(HEIGHT * 1.5), WIDTH))[:HEIGHT]
                display = y_plane
            else:
                display = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH))

            cv2.imshow("🛰️ FLIR Thermal Stream", display)
            print(f"🧪 min: {display.min()}, max: {display.max()}, mean: {display.mean():.2f}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        pipe.terminate()
        cv2.destroyAllWindows()
        break

detect_and_display()
