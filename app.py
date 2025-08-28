from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os, cv2, pickle, time, threading, requests
import face_recognition
import numpy as np
from collections import deque

# ================== KONFIG ==================
ESP_IP = "10.254.46.39"  # pastikan benar
UNLOCK_URL = f"http://{ESP_IP}/unlock"
ESP_TIMEOUT = 3.0
ESP_RETRIES = 2

DATASET_DIR = "dataset"
ENC_FILE = "encodings.pkl"

# Kamera & performa
FRAME_W, FRAME_H = 640, 480        # capture native (biar autofocus lebih oke)
PROCESS_SCALE = 0.5                 # proses di 50% (lebih cepat)
PROCESS_EVERY_N_FRAMES = 6          # proses tiap N frame
FPS_TARGET = 24
LABEL_PERSIST_TIME = 1.0            # detik

# Recognition
UPSAMPLE = 0                        # upsample=0 → cepat
MATCH_THRESHOLD = 0.5               # threshold jarak
DEBOUNCE = 5.0                      # detik antar unlock
REGISTER_SAMPLES = 20            # jumlah foto saat register

# Update these constants at top of file
UNLOCK_DURATION = 5.0    # How long door stays unlocked
LABEL_TIMEOUT = 5.0     # How long labels persist
PROCESS_EVERY_N = 6      # Process every N frames

os.makedirs(DATASET_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "face-doorlock-secret"

# ======= STATE =======
recognition_thread = None
recognition_running = False
recognition_lock = threading.Lock()

# =========================================================
# Kamera thread (selalu menyediakan frame terbaru → anti-lag)
# =========================================================
class CameraStream:
    def __init__(self, index_candidates=(0,1)):
        self.cap = None
        for idx in index_candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                self.cap = cap
                break
        if self.cap is None:
            raise RuntimeError("Kamera tidak ditemukan")

        # Kurangi lag buffer
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.t = threading.Thread(target=self.update, daemon=True)
        self.t.start()

    def update(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

# ======= UTIL =======
def load_encodings():
    if not os.path.exists(ENC_FILE):
        return np.empty((0,128), dtype="float32"), np.array([])
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
    encs = np.array(data.get("encodings", []), dtype="float32")
    names = np.array(data.get("names", []))
    return encs, names

def save_encodings(encs, names):
    with open(ENC_FILE, "wb") as f:
        pickle.dump({"encodings": encs, "names": names}, f)

def train_dataset():
    enc_list, name_list = [], []
    for user in os.listdir(DATASET_DIR):
        udir = os.path.join(DATASET_DIR, user)
        if not os.path.isdir(udir):
            continue
        for img_name in os.listdir(udir):
            path = os.path.join(udir, img_name)
            try:
                image = face_recognition.load_image_file(path)
            except Exception:
                continue
            boxes = face_recognition.face_locations(image, number_of_times_to_upsample=UPSAMPLE, model="hog")
            if not boxes:
                continue
            encs = face_recognition.face_encodings(image, boxes)
            if encs:
                enc_list.append(encs[0])
                name_list.append(user)
    encs = np.array(enc_list, dtype="float32") if enc_list else np.empty((0,128), dtype="float32")
    names = np.array(name_list) if name_list else np.array([])
    save_encodings(encs, names)
    return len(encs), set(names.tolist())

def register_user(username, samples=REGISTER_SAMPLES):
    user_dir = os.path.join(DATASET_DIR, username)
    os.makedirs(user_dir, exist_ok=True)

    # pakai kamera terbaik yang ketemu
    cam = None
    try:
        cam = CameraStream(index_candidates=(0,1))
    except RuntimeError:
        return 0, 0, set()

    saved = 0
    while saved < samples:
        frame = cam.read()
        if frame is None:
            continue

        preview = frame.copy()
        h, w = preview.shape[:2]
        cv2.putText(preview, f"Posisikan wajah | Foto {saved+1}/{samples}",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Register - Tekan 's' untuk foto, 'q' batal", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            fname = os.path.join(user_dir, f"{username}_{int(time.time()*1000)}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        elif key == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
    total, users = train_dataset()  # auto-train
    return saved, total, users

def distance_to_confidence(dist, match_threshold=MATCH_THRESHOLD, max_dist=0.6):
    """
    Map jarak (semakin kecil semakin mirip) → ~confidence 0..1.
    Linear mapping mulai 100% di threshold, turun ke 0% di max_dist.
    """
    if dist <= match_threshold:
        return 1.0
    if dist >= max_dist:
        return 0.0
    # skala linear
    return float(1.0 - (dist - match_threshold) / (max_dist - match_threshold))

def esp_unlock_async():
    # panggil unlock di thread terpisah → tidak blokir loop kamera
    threading.Thread(target=esp_unlock, daemon=True).start()

def esp_unlock():
    for attempt in range(ESP_RETRIES):
        try:
            r = requests.get(UNLOCK_URL, timeout=ESP_TIMEOUT)
            if r.status_code == 200:
                print("[ESP] Unlock OK")
                return True
            print(f"[ESP] HTTP {r.status_code}")
        except requests.exceptions.Timeout:
            print("[ESP] timeout")
        except Exception as e:
            print("[ESP] err:", e)
        time.sleep(0.3)
    return False

# ======= LOOP RECOGNITION =======
def recognition_loop():
    global recognition_running

    encs, names = load_encodings()
    if len(encs) == 0:
        print("No encodings found. Silakan train dulu.")
        return

    # Mulai kamera non-blocking
    try:
        cam = CameraStream(index_candidates=(0,1))
    except RuntimeError as e:
        print(str(e))
        return

    frame_count = 0
    last_unlock = 0
    unlock_end_time = 0
    door_status = "LOCKED"
    face_labels = {}  # Store persistent labels

    try:
        while recognition_running:
            frame = cam.read()
            if frame is None:
                continue

            current_time = time.time()
            
            # Check if unlock period is over
            if door_status == "UNLOCKED" and current_time > unlock_end_time:
                door_status = "LOCKED"
                print("Door auto-locked")

            # Clean old labels
            face_labels = {k:v for k,v in face_labels.items() 
                         if current_time - v['last_seen'] < LABEL_TIMEOUT}

            frame_count += 1
            if frame_count % PROCESS_EVERY_N != 0:
                # Draw existing labels even when skipping processing
                for face_info in face_labels.values():
                    left, top = face_info['pos']
                    name = face_info['name']
                    draw_face_label(frame, name, left, top)

                draw_status_banner(frame, door_status)
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Face detection and recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            enc_curr = face_recognition.face_encodings(rgb, boxes)

            for (top, right, bottom, left), e in zip(boxes, enc_curr):
                dists = face_recognition.face_distance(encs, e)
                if len(dists) > 0:
                    idx = np.argmin(dists)
                    min_dist = dists[idx]
                    detected_name = names[idx] if min_dist <= MATCH_THRESHOLD else "Unknown"
                    
                    # Update face label
                    face_key = f"{left}_{top}"
                    face_labels[face_key] = {
                        'name': detected_name,
                        'pos': (left, top),
                        'last_seen': current_time
                    }

                    # Handle unlock
                    if detected_name != "Unknown" and door_status == "LOCKED":
                        if esp_unlock():
                            door_status = "UNLOCKED"
                            unlock_end_time = current_time + UNLOCK_DURATION
                            print(f"Door unlocked for {detected_name}")

            # Draw all current labels
            for face_info in face_labels.values():
                left, top = face_info['pos']
                name = face_info['name']
                draw_face_label(frame, name, left, top)

            draw_status_banner(frame, door_status)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error in recognition loop:", e)
    finally:
        cam.stop()
        cv2.destroyAllWindows()

def draw_face_label(frame, name, left, top):
    """Helper function to draw consistent face labels"""
    color = (0,255,0) if name != "Unknown" else (0,0,255)
    
    # Draw label background
    label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame,
                 (left, top - 30),
                 (left + label_size[0] + 10, top),
                 color, -1)
    
    # Draw name
    cv2.putText(frame, name,
                (left + 5, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def draw_status_banner(frame, status):
    """Helper function to draw status banner"""
    cv2.rectangle(frame, (0,0), (frame.shape[1], 35), (30,30,30), -1)
    color = (0,255,0) if status == "UNLOCKED" else (0,0,255)
    cv2.putText(frame, f"Door: {status}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ================== ROUTES ==================
@app.route("/")
def index():
    users = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    return render_template("index.html", users=users)

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "").strip()
    if not username:
        flash("Nama user tidak boleh kosong.", "error")
        return redirect(url_for("index"))

    saved, total, users = register_user(username)
    if saved == 0:
        flash("Registrasi dibatalkan atau kamera gagal.", "error")
    else:
        flash(f"Registrasi {username}: tersimpan {saved} foto. Training selesai. Total embedding: {total}.", "success")
    return redirect(url_for("index"))

@app.route("/train", methods=["POST"])
def train():
    total, users = train_dataset()
    flash(f"Training selesai. Total embedding: {total}. Users: {', '.join(users) if users else '-'}", "success")
    return redirect(url_for("index"))

@app.route("/start", methods=["POST"])
def start():
    global recognition_thread, recognition_running
    if recognition_thread and recognition_thread.is_alive():
        flash("Recognition sudah berjalan", "info")
        return redirect(url_for("index"))

    with recognition_lock:
        recognition_running = True

    recognition_thread = threading.Thread(target=recognition_loop, daemon=True)
    recognition_thread.start()

    flash("Recognition started", "success")
    return redirect(url_for("index"))

@app.route("/stop", methods=["POST"])
def stop():
    global recognition_running, recognition_thread
    with recognition_lock:
        recognition_running = False
    if recognition_thread:
        recognition_thread.join(timeout=3.0)
        recognition_thread = None
    flash("Recognition stopped", "info")
    return redirect(url_for("index"))

@app.route("/unlock", methods=["GET"])
def unlock():
    if esp_unlock():
        return jsonify({"status": "success", "message": "Door unlocked"})
    return jsonify({"status": "error", "message": "Failed to unlock"}), 500

if __name__ == "__main__":
    # (opsional) kurangi thread internal OpenCV kalau perlu:
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    app.run(host="0.0.0.0", port=5000, debug=True)
