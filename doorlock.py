import cv2
import face_recognition
import pickle
import numpy as np
import requests
import time

# ============== KONFIG ==============
ESP_IP = "172.20.10.2"
ROOT_URL = f"http://{ESP_IP}/"
UNLOCK_URL = f"http://{ESP_IP}/unlock"

ENC_FILE = "encodings.pkl"
SAFE_MODE = True           # True = stabil, False = cepat
MATCH_THRESHOLD = 0.62

FRAME_W, FRAME_H = 640, 480
DOWNSCALE = 0.5
PROCESS_EVERY = 2

DEBOUNCE_SEC = 5.0
STATUS_HOLD_SEC = 4.0

# ============== LOAD DATA ==============
with open(ENC_FILE, "rb") as f:
    data = pickle.load(f)
known_encs = np.array(data["encodings"], dtype="float32")
known_names = np.array(data["names"])
if len(known_encs) == 0:
    raise RuntimeError("Encodings kosong. Jalankan train_faces.py dulu.")

# ============== UTIL ESP ==============
def esp_check():
    try:
        requests.get(ROOT_URL, timeout=2)
        return True
    except Exception:
        return False

def esp_unlock():
    for _ in range(2):
        try:
            requests.get(UNLOCK_URL, timeout=2.5)
            return True
        except Exception:
            time.sleep(0.4)
    return False

# ============== CAMERA ==============
cap = cv2.VideoCapture(0)
if not SAFE_MODE:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

status_text = "LOCKED"
status_color = (0, 0, 255)
esp_ok = esp_check()
if not esp_ok:
    status_text, status_color = "ESP OFFLINE", (0, 165, 255)

last_unlock_ts = -1e9
status_change_ts = -1e9
last_boxes, last_names = [], []
frame_count = 0

print("[INFO] Running Face Recognition...")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if SAFE_MODE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(
            rgb, number_of_times_to_upsample=1, model="hog"
        )
        encs = face_recognition.face_encodings(rgb, boxes)

        names = []
        for (top, right, bottom, left), enc in zip(boxes, encs):
            dists = face_recognition.face_distance(known_encs, enc)
            if len(dists):
                idx = int(np.argmin(dists))
                name = known_names[idx] if dists[idx] <= MATCH_THRESHOLD else "Unknown"
            else:
                name = "Unknown"

            names.append(name)
            color = (0,255,0) if name!="Unknown" else (0,0,255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = max(top - 8, 20)
            cv2.putText(frame, name, (left + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if any(n != "Unknown" for n in names):
            now = time.monotonic()
            if (now - last_unlock_ts) >= DEBOUNCE_SEC:
                ok_unlock = esp_unlock()
                last_unlock_ts = now
                status_change_ts = now
                if ok_unlock:
                    status_text, status_color = "UNLOCKED", (0,255,0)
                else:
                    status_text, status_color = "ESP TIMEOUT", (0,165,255)
        else:
            now = time.monotonic()
            if (now - status_change_ts) >= STATUS_HOLD_SEC:
                status_text = "LOCKED" if esp_ok else "ESP OFFLINE"
                status_color = (0,0,255) if esp_ok else (0,165,255)

    else:
        if frame_count % PROCESS_EVERY == 0:
            small = cv2.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            loc_small = face_recognition.face_locations(
                rgb_small, number_of_times_to_upsample=1, model="hog"
            )
            if loc_small:
                encs_small = face_recognition.face_encodings(rgb_small, loc_small)
            else:
                encs_small = []

            last_boxes = [
                (int(t / DOWNSCALE), int(r / DOWNSCALE),
                 int(b / DOWNSCALE), int(l / DOWNSCALE))
                for (t, r, b, l) in loc_small
            ]
            last_names = []
            someone_matched = False
            for enc in encs_small:
                dists = face_recognition.face_distance(known_encs, enc)
                if len(dists):
                    idx = int(np.argmin(dists))
                    name = known_names[idx] if dists[idx] <= MATCH_THRESHOLD else "Unknown"
                else:
                    name = "Unknown"
                last_names.append(name)
                if name != "Unknown":
                    someone_matched = True

            if not loc_small:
                last_boxes, last_names = [], []

            now = time.monotonic()
            if someone_matched and (now - last_unlock_ts) >= DEBOUNCE_SEC:
                ok_unlock = esp_unlock()
                last_unlock_ts = now
                status_change_ts = now
                if ok_unlock:
                    status_text, status_color = "UNLOCKED", (0,255,0)
                else:
                    status_text, status_color = "ESP TIMEOUT", (0,165,255)
            elif not someone_matched and (now - status_change_ts) >= STATUS_HOLD_SEC:
                status_text = "LOCKED" if esp_ok else "ESP OFFLINE"
                status_color = (0,0,255) if esp_ok else (0,165,255)

        for (top, right, bottom, left), name in zip(last_boxes, last_names):
            color = (0,255,0) if name!="Unknown" else (0,0,255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = max(top - 8, 20)
            cv2.putText(frame, name, (left + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        frame_count += 1

    cv2.rectangle(frame, (0,0), (frame.shape[1], 36), (30,30,30), -1)
    cv2.putText(frame, f"Door: {status_text}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
    conn_txt = f"ESP: {'OK' if esp_ok else 'OFFLINE'}"
    conn_col = (0,255,0) if esp_ok else (0,165,255)
    cv2.putText(frame, conn_txt, (frame.shape[1]-160, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, conn_col, 2)

    cv2.imshow("Door Lock Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        esp_ok = esp_check()
        if not esp_ok:
            status_text, status_color = "ESP OFFLINE", (0,165,255)
        else:
            status_text, status_color = "LOCKED", (0,0,255)

cap.release()
cv2.destroyAllWindows()
