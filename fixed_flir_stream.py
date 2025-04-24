
import subprocess
import numpy as np
import cv2

WIDTH, HEIGHT = 640, 512
DEVICE_NAME = "FLIR Video"

PIXEL_FORMATS = {
    'gray16le': (np.uint16, 2),
    'yuv420p': (np.uint8, 1.5),
    'nv12': (np.uint8, 1.5),
}

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
                raw16 = np.frombuffer(raw, dtype=np.uint16)

                if raw16.size > WIDTH * HEIGHT:
                    print(f"⚠️ Detected frame overflow: got {raw16.size}, expected {WIDTH * HEIGHT}. Trimming...")
                    raw16 = raw16[:WIDTH * HEIGHT]

                frame = raw16.reshape((HEIGHT, WIDTH))
                norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                display = cv2.applyColorMap(norm_frame.astype(np.uint8), cv2.COLORMAP_JET)

            elif dtype == np.uint8 and pixel_bytes == 1.5:
                y_plane = np.frombuffer(raw, dtype=np.uint8).reshape((int(HEIGHT * 1.5), WIDTH))[:HEIGHT]
                display = cv2.applyColorMap(y_plane, cv2.COLORMAP_JET)

            else:
                display = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH))

            print(f"🧪 min: {display.min()}, max: {display.max()}, mean: {display.mean():.2f}")
            cv2.imshow("🔥 FLIR Thermal Stream", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        pipe.terminate()
        cv2.destroyAllWindows()
        break

detect_and_display()
