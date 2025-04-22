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


import cv2
import numpy as np
import time
from ultralytics import YOLO


GRID_ROWS, GRID_COLS = 7, 8
MODEL_PATH = "yolov8n.pt"
CLASSES = {"person", "car", "truck", "tree", "house", "building"}


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
    codes = {
        "person": "1",
        "car": "2",
        "truck": "3",
        "tree": "4",
        "house": "5",
        "building": "6"
    }
    base = codes.get(label, "0")
    count = np.random.randint(2, 6)
    return base * count

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
    if object_height_px <= 0:
        return -1.0
    return round(a / object_height_px + b, 2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Камера не зчиталась")
        break

    h, w = frame.shape[:2]
    results = model.predict(frame, conf=0.5, verbose=False, device="cuda")[0]
    grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label not in CLASSES:
                continue

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

    cv2.imshow("AI FULL PLANER | GPU + CLASS-BASED RANGE", frame)


    print("📊 GRID:")
    for row in grid:
        formatted = ["[" + ",".join(cell) + "]" if cell else "[0]" for cell in row]
        print(formatted)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
