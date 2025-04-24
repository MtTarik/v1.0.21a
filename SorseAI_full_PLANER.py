# import cv2

# obj_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_forehead_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 

# obj_cascade = cv2.CascadeClassofer(cv2.data.haarcascades + 'haarcascade_horizont_flight.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# if not obj_cascade.load(cv2.data.haarccascades) or not eye_cascade.load(cv2.data.haarccascades)
# print('Error loading cascade -- classifiers')
 

# 1. Визначити об'єкти в полі зору камери по горизонту тангауту 

# дані за обробку тільки в полі зозу 7*8 


# обект такі як будинки дерева та інші об'єкти які можуьб заважати на приблизній1 відстані 6-10 метрів с мінімальною погрішністю дивитися по спектру піксселяґ 



# [12331],[13231],[1323],[1323],[131321],[133131],[13123],[31321]
# [12331],[13231],[1323],[1323],[131321],[133131],[13123],[31321]
# [12331],[13231],[1323],[1323],[131321],[133131],[13123],[31321]
# [12331],[13231],[1323],[1323],[131321],[133131],[13123],[31321]
# [12331],[13231],[1323],[1323],[131321],[133131],[13123],[31321]
# [12331],[13231],[1323],[1323],[131321],[133131],[13123],[31321]
# [12331],[13231],[1323],[1323],[131321],[133131],[13123],[31321]
 
# 7*8 = поле на горизонті яке бачитиме тільки огбекти в межах 7*8
# тобто обробляемо дані тільки тоді коли щось в полі ми міняемо направлення польоту взаемодіею с ArduPilot --
# взаемодіє з ArduPilot 
# на основі дангихх які ми отримали з камери на борту,
#  простими if-else конструкціями ми будемо в момент спокою тобто якщо піксель нее темнй -
#  ми не прораховуемо тригер буде піксель або світлилий або темнй в задлежності яка камера
# в нас є ІЧ камера та RGB камера протестити на веб камері по відаленості мінімальна відстань 6-10 метрів від об'єкта
# використовуємо 2 камери ІЧ та RGB
 

# if not obj_cascade.load(cv2.data.haarccascades) or not eye_cascade.load(cv2.data.haarccascades):
#     print("Error loading cascade classifiers")
#     exit() 

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():



#  v1


# import cv2
# import numpy as np
# import time
# from ultralytics import YOLO


# GRID_ROWS, GRID_COLS = 7, 8
# MODEL_PATH = "yolov8n.pt"
# CLASSES = {"person", "car", "truck", "tree", "house", "building"}

# # YOLOv8n на GPU
# model = YOLO(MODEL_PATH).to("cuda")

# def get_grid_index(x, y, w, h):
#     return int(y / (h / GRID_ROWS)), int(x / (w / GRID_COLS))

# def draw_grid(frame):
#     h, w = frame.shape[:2]
#     for i in range(1, GRID_ROWS):
#         y = int(i * h / GRID_ROWS)
#         cv2.line(frame, (0, y), (w, y), (180, 180, 180), 1)
#     for j in range(1, GRID_COLS):
#         x = int(j * w / GRID_COLS)
#         cv2.line(frame, (x, 0), (x, h), (180, 180, 180), 1)

# def generate_code(label: str):
#     codes = {
#         "person": "1",
#         "car": "2",
#         "truck": "3",
#         "tree": "4",
#         "house": "5",
#         "building": "6"
#     }
#     base = codes.get(label, "0")
#     count = np.random.randint(2, 6)
#     return base * count

# def estimate_distance_px(object_height_px: int) -> float:
#     """Покращена модель оцінки відстані по висоті об'єкта (в пікселях)"""
#     if object_height_px <= 0:
#         return -1.0
#     a = 320  # емпіричний коефіцієнт, підібраний під 640×480- важливо що він буде підлаштованй під камеру ІЧ та РGB
#     # a = 320 * (640 / object_height_px)  # для адаптації до різних роздільних здатностей
#     b = 0.5  # невелике зміщення
#     return round(a / object_height_px + b, 2)


# # поки для тесту юзаю вебку і в подальшому буду тестити на ІЧ камері та RGB камері в паралельному режимі 
# # з ArduPilot
# # Пробблема може виникнути в роботі с ІЧ та RGB камерами тобто для кожної камери потрібно буде підбирати
# # свої параметри для обробки даних, є варіант що ІЧ камера буде краще розпізнавати об'єкти

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# prev_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Камера не зчиталась")
#         break

#     h, w = frame.shape[:2]
#     results = model.predict(frame, conf=0.5, verbose=False, device="cuda")[0]
#     grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

#     if results.boxes is not None:
#         for box in results.boxes:
#             cls_id = int(box.cls[0])
#             label = model.names[cls_id]

#             if label not in CLASSES:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             obj_height = max(y2 - y1, 1)  # захист від /0
#             distance_est = estimate_distance_px(obj_height)

#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             row, col = get_grid_index(cx, cy, w, h)

#             if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
#                 obj_code = generate_code(label)
#                 grid[row][col].append(f"{obj_code}:{distance_est}")

        
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {distance_est}m", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

#     draw_grid(frame)


#     now = time.time()
#     fps = 1 / (now - prev_time)
#     prev_time = now
#     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     cv2.imshow("AI FULL PLANER | GPU + RANGE", frame)

#     print("📊 GRID:")
#     for row in grid:
#         formatted = ["[" + ",".join(cell) + "]" if cell else "[0]" for cell in row]
#         print(formatted)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import numpy as np
# import time
# from ultralytics import YOLO

# GRID_ROWS, GRID_COLS = 7, 8
# MODEL_PATH = "yolov8n.pt"
# CLASSES = {"person", "car", "truck", "tree", "house", "building"}


# model = YOLO(MODEL_PATH).to("cuda")

# def get_grid_index(x, y, w, h):
#     return int(y / (h / GRID_ROWS)), int(x / (w / GRID_COLS))

# def draw_grid(frame):
#     h, w = frame.shape[:2]
#     for i in range(1, GRID_ROWS):
#         y = int(i * h / GRID_ROWS)
#         cv2.line(frame, (0, y), (w, y), (180, 180, 180), 1)
#     for j in range(1, GRID_COLS):
#         x = int(j * w / GRID_COLS)
#         cv2.line(frame, (x, 0), (x, h), (180, 180, 180), 1)

# def generate_code(label: str):
#     codes = {
#         "person": "1",
#         "car": "2",
#         "truck": "3",
#         "tree": "4",
#         "house": "5",
#         "building": "6"
#     }
#     base = codes.get(label, "0")
#     count = np.random.randint(2, 6)
#     return base * count

# def estimate_distance_by_class(object_height_px: int, label: str) -> float:
#     class_params = {
#         "person": (300, 0.5),
#         "car": (420, 0.6),
#         "truck": (500, 0.8),
#         "tree": (700, 1.0),
#         "house": (800, 1.2),
#         "building": (900, 1.5)
#     }
#     a, b = class_params.get(label, (320, 0.5))
#     if object_height_px <= 0:
#         return -1.0
#     return round(a / object_height_px + b, 2)


# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# prev_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Камера не зчиталась")
#         break

#     h, w = frame.shape[:2]
#     results = model.predict(frame, conf=0.5, verbose=False, device="cuda")[0]
#     grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

#     if results.boxes is not None:
#         for box in results.boxes:
#             cls_id = int(box.cls[0])
#             label = model.names[cls_id]

#             if label not in CLASSES:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             obj_height = max(y2 - y1, 1)
#             distance_est = estimate_distance_by_class(obj_height, label)

#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             row, col = get_grid_index(cx, cy, w, h)

#             if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
#                 obj_code = generate_code(label)
#                 grid[row][col].append(f"{obj_code}:{distance_est}")


#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {distance_est}m", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

#     draw_grid(frame)

#     now = time.time()
#     fps = 1 / (now - prev_time)
#     prev_time = now
#     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     cv2.imshow("AI FULL PLANER | GPU + CLASS-BASED RANGE", frame)
#     print("📊 GRID:")
#     for row in grid:
#         formatted = ["[" + ",".join(cell) + "]" if cell else "[0]" for cell in row]
#         print(formatted)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import time
# from ultralytics import YOLO


# GRID_ROWS, GRID_COLS = 7, 8
# MODEL_PATH = "yolov8n.pt"
# CLASSES = {"person", "car", "truck", "tree", "house", "building"}


# model = YOLO(MODEL_PATH).to("cuda")



# def get_grid_index(x, y, w, h):
#     return int(y / (h / GRID_ROWS)), int(x / (w / GRID_COLS))

# def draw_grid(frame):
#     h, w = frame.shape[:2]
#     for i in range(1, GRID_ROWS):
#         y = int(i * h / GRID_ROWS)
#         cv2.line(frame, (0, y), (w, y), (180, 180, 180), 1)
#     for j in range(1, GRID_COLS):
#         x = int(j * w / GRID_COLS)
#         cv2.line(frame, (x, 0), (x, h), (180, 180, 180), 1)

# def generate_code(label: str):
#     codes = {
#         "person": "1",
#         "car": "2",
#         "truck": "3",
#         "tree": "4",
#         "house": "5",
#         "building": "6"
#     }
#     base = codes.get(label, "0")
#     count = np.random.randint(2, 6)
#     return base * count

# def estimate_distance_by_class(object_height_px: int, label: str) -> float:
#     class_params = {
#         "person": (300, 0.5),
#         "car": (420, 0.6),
#         "truck": (500, 0.8),
#         "tree": (700, 1.0),
#         "house": (800, 1.2),
#         "building": (900, 1.5)
#     }
#     a, b = class_params.get(label, (320, 0.5))
#     if object_height_px <= 0:
#         return -1.0
#     return round(a / object_height_px + b, 2)

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# prev_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Камера не зчиталась")
#         break

#     h, w = frame.shape[:2]
#     results = model.predict(frame, conf=0.5, verbose=False, device="cuda")[0]
#     grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

#     if results.boxes is not None:
#         for box in results.boxes:
#             cls_id = int(box.cls[0])
#             label = model.names[cls_id]

#             if label not in CLASSES:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             obj_height = max(y2 - y1, 1)
#             distance_est = estimate_distance_by_class(obj_height, label)

#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             row, col = get_grid_index(cx, cy, w, h)

#             if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
#                 obj_code = generate_code(label)
#                 grid[row][col].append(f"{obj_code}:{distance_est}")

           
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {distance_est}m", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

#     draw_grid(frame)

    
#     now = time.time()
#     fps = 1 / (now - prev_time)
#     prev_time = now
#     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     cv2.imshow("AI FULL PLANER | GPU + CLASS-BASED RANGE", frame)


#     print("📊 GRID:")
#     for row in grid:
#         formatted = ["[" + ",".join(cell) + "]" if cell else "[0]" for cell in row]
#         print(formatted)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import time 
# from ultralytics import YOLO

# GRID_ROWS, GRID_COLS = 7, 8
# MODEL_PATH = "yolov8n.pt"

# model = YOLO(MODEL_PATH).to("cuda")

# def find_cameras(max_index=10):
#     available = []
#     for i in range(max_index):
#         cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
#         if cap.isOpened():
#             ret, _ = cap.read()
#             if ret:
#                 available.append(i)
#             cap.release()
#     return available

# def get_grid_index(x, y, w, h):
#     return int(y / (h / GRID_ROWS)), int(x / (w / GRID_COLS))

# def draw_grid(frame):
#     h, w = frame.shape[:2]
#     for i in range(1, GRID_ROWS):
#         y = int(i * h / GRID_ROWS)
#         cv2.line(frame, (0, y), (w, y), (180, 180, 180), 1)
#     for j in range(1, GRID_COLS):
#         x = int(j * w / GRID_COLS)
#         cv2.line(frame, (x, 0), (x, h), (180, 180, 180), 1)

# def generate_code(label: str):
#     return str(abs(hash(label)) % 99999)

# def estimate_distance_px(object_height_px: int, label: str) -> float:
#     class_params = {
#         "person": (300, 0.5),
#         "car": (420, 0.6),
#         "truck": (500, 0.8),
#         "tree": (700, 1.0),
#         "house": (800, 1.2),
#         "building": (900, 1.5)
#     }
#     a, b = class_params.get(label, (350, 0.7))
#     if object_height_px <= 0:
#         return -1.0
#     return round(a / object_height_px + b, 2)

# #  Сканування камер 
# camera_ids = find_cameras()
# print(f"🎥 Знайдено камер: {len(camera_ids)} -> {camera_ids}")
# caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in camera_ids]

# for cap in caps:
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# prev_time = time.time()

# while True:
#     for cam_index, cap in enumerate(caps):
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         h, w = frame.shape[:2]
#         results = model.predict(frame, conf=0.5, verbose=False, device="cuda")[0]
#         grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

#         if results.boxes is not None:
#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 label = model.names.get(cls_id, "unknown")

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 obj_height = max(y2 - y1, 1)
#                 distance_est = estimate_distance_px(obj_height, label)

#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#                 row, col = get_grid_index(cx, cy, w, h)

#                 if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
#                     obj_code = generate_code(label)
#                     grid[row][col].append(f"{obj_code}:{distance_est}")

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label} {distance_est}m", (x1, max(y1 - 10, 10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         draw_grid(frame)

        
#         now = time.time()
#         fps = 1 / (now - prev_time)
#         prev_time = now
#         cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        
#         window_name = f"CAM {camera_ids[cam_index]}"
#         cv2.imshow(window_name, frame)

    
#         print(f"\n📊 GRID CAM {camera_ids[cam_index]}:")
#         for row in grid:
#             formatted = ["[" + ",".join(cell) + "]" if cell else "[0]" for cell in row]
#             print(formatted)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# for cap in caps:
#     cap.release()

# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import subprocess
# import time
# from ultralytics import YOLO

# # Константи
# WIDTH, HEIGHT = 640, 512
# GRID_ROWS, GRID_COLS = 7, 8
# DEVICE_NAME = 'video=FLIR BOSON USB'
# MODEL_PATH = 'yolov8n.pt'

# # YOLO модель
# model = YOLO(MODEL_PATH).to("cuda")

# # GRID координати
# def get_grid_index(x, y, w, h):
#     return int(y / (h / GRID_ROWS)), int(x / (w / GRID_COLS))

# # Малювання сітки
# def draw_grid(frame):
#     for i in range(1, GRID_ROWS):
#         y = int(i * frame.shape[0] / GRID_ROWS)
#         cv2.line(frame, (0, y), (frame.shape[1], y), (150, 150, 150), 1)
#     for j in range(1, GRID_COLS):
#         x = int(j * frame.shape[1] / GRID_COLS)
#         cv2.line(frame, (x, 0), (x, frame.shape[0]), (150, 150, 150), 1)

# # Код + відстань
# def generate_code(label):
#     return str(abs(hash(label)) % 99999)

# def estimate_distance(height, label):
#     class_params = {
#         "person": (300, 0.5),
#         "car": (420, 0.6),
#         "truck": (500, 0.8),
#         "tree": (700, 1.0),
#         "house": (800, 1.2),
#         "building": (900, 1.5)
#     }
#     a, b = class_params.get(label, (350, 0.7))
#     return round(a / height + b, 2) if height > 0 else -1

# # FFmpeg команда
# def get_16bit_frame():
#     ffmpeg_cmd = [
#         'ffmpeg',
#         '-f', 'dshow',
#         '-video_size', f'{WIDTH}x{HEIGHT}',
#         '-pixel_format', 'gray16le',
#         '-i', DEVICE_NAME,
#         '-vframes', '1',
#         '-f', 'rawvideo', '-'
#     ]
#     proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
#     raw = proc.stdout.read(WIDTH * HEIGHT * 2)
#     proc.terminate()
#     return np.frombuffer(raw, dtype=np.uint16).reshape((HEIGHT, WIDTH))

# # Основний цикл
# while True:
#     try:
#         frame16 = get_16bit_frame()
#         if frame16 is None:
#             continue

#         # Нормалізація для візуалізації
#         frame8 = cv2.convertScaleAbs(frame16, alpha=(255.0 / frame16.max()))
#         frame_colored = cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)

#         # YOLO детекція
#         results = model.predict(frame_colored, conf=0.5, verbose=False)[0]
#         grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

#         for box in results.boxes:
#             label = model.names[int(box.cls[0])]
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             h = max(y2 - y1, 1)
#             d = estimate_distance(h, label)
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             row, col = get_grid_index(cx, cy, WIDTH, HEIGHT)
#             code = generate_code(label)
#             if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
#                 grid[row][col].append(f"{code}:{d}")
#             cv2.rectangle(frame_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame_colored, f"{label} {d}m", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         draw_grid(frame_colored)
#         cv2.imshow("FLIR Boson 16-bit + YOLO", frame_colored)

#         print("📊 GRID:")
#         for row in grid:
#             print(["[" + ",".join(cell) + "]" if cell else "[0]" for cell in row])

#         if cv2.waitKey(1) & 0xFF == ordй("q"):
#             break

#     except Exception as e:
#         print("⚠️", e)
#         continue

# cv2.destroyAllWindows()



import cv2
import numpy as np
import time
import subprocess
from ultralytics import YOLO

# Константи
WIDTH, HEIGHT = 640, 512
GRID_ROWS, GRID_COLS = 7, 8
FFMPEG_PATH = r"C:\Users\Admin\Desktop\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
DEVICE_NAME = 'video=FLIR Video'
MODEL_PATH = "yolov8n.pt"

# Завантаження YOLO
model = YOLO(MODEL_PATH).to("cuda")

def get_grid_index(x, y, w, h):
    return int(y / (h / GRID_ROWS)), int(x / (w / GRID_COLS))

def draw_grid(frame):
    h, w = frame.shape[:2]
    for i in range(1, GRID_ROWS):
        y = int(i * h / GRID_ROWS)
        cv2.line(frame, (0, y), (w, y), (180, 180, 180), 1)
    for j in range(1, GRID_COLS):
        x = int(j * w / GRID_COLS)
        cv2.line(frame, (x, 0), (x, h), (180, 180, 180), 1)

def generate_code(label: str):
    return str(abs(hash(label)) % 99999)

def estimate_distance_by_class(object_height_px: int, label: str) -> float:
    class_params = {
        "person": (300, 0.5),
        "car": (420, 0.6),
        "truck": (500, 0.8),
        "tree": (700, 1.0),
        "house": (800, 1.2),
        "building": (900, 1.5)
    }
    a, b = class_params.get(label, (320, 0.5))
    return round(a / object_height_px + b, 2) if object_height_px > 0 else -1.0

def get_16bit_frame():
    cmd = [
        FFMPEG_PATH,
        '-f', 'dshow',
        '-video_size', f'{WIDTH}x{HEIGHT}',
        '-pixel_format', 'gray16le',
        '-i', DEVICE_NAME,
        '-vframes', '1',
        '-f', 'rawvideo', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout.read(WIDTH * HEIGHT * 2)
    proc.terminate()
    if not raw:
        return None
    return np.frombuffer(raw, dtype=np.uint16).reshape((HEIGHT, WIDTH))

prev_time = time.time()

while True:
    frame16 = get_16bit_frame()
    if frame16 is None:
        print("⚠️ Не вдалося зчитати кадр з FLIR")
        continue

    # Нормалізуємо 16-біт для виводу
    frame_norm = cv2.convertScaleAbs(frame16, alpha=(255.0 / frame16.max()))
    frame = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

    h, w = frame.shape[:2]
    results = model.predict(frame, conf=0.5, verbose=False, device="cuda")[0]
    grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_height = max(y2 - y1, 1)
            distance_est = estimate_distance_by_class(obj_height, label)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            row, col = get_grid_index(cx, cy, w, h)

            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                obj_code = generate_code(label)
                grid[row][col].append(f"{obj_code}:{distance_est}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {distance_est}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    draw_grid(frame)

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("FLIR Boson 16-bit | YOLO + GRID", frame)

    print("📊 GRID:")
    for row in grid:
        formatted = ["[" + ",".join(cell) + "]" if cell else "[0]" for cell in row]
        print(formatted)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
