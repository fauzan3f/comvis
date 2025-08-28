from flask import Flask, render_template, request, redirect, url_for, flash
import os, cv2, pickle, time, threading, requests
import face_recognition
import numpy as np

# ================== KONFIG ==================
ESP_IP = "172.20.10.2"
UNLOCK_URL = f"http://{ESP_IP}/unlock"
DATASET_DIR = "dataset"
ENC_FILE = "encodings.pkl"

MATCH_THRESHOLD = 0.62         # ambang cocok
REGISTER_SAMPLES = 20           # banyak foto saat register
FRAME_W, FRAME_H = 640, 480     # resolusi kamera
UPSAMPLE = 1                    # 0=cepat, 1=lebih sensitif

os.makedirs(DATASET_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "face-doorlock-secret"  # untuk flash message

# ======= STATE GLOBAL RECOGNITION =======
recognition_thread = None
recognition_running = False
recognition_lock = threading.Lock()

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

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    saved = 0
    while saved < samples:
        ok, frame = cap.read()
        if not ok:
            break
        # tampilkan panduan kecil di window (opsional)
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

    cap.release()
    cv2.destroyAllWindows()
    # otomatis TRAIN setelah register
    total, users = train_dataset()
    return saved, total, users

def esp_unlock():
    try:
        requests.get(UNLOCK_URL, timeout=2.5)
        return True
    except Exception:
        return False

# ======= LOOP RECOGNITION (JALAN DI THREAD) =======
def recognition_loop():
    global recognition_running
    encs, names = load_encodings()
    if encs.shape[0] == 0:
        flash("Belum ada encodings. Lakukan Register/Train dulu.", "error")
        with recognition_lock:
            recognition_running = False
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    last_unlock = 0.0
    DEBOUNCE = 5.0

    while True:
        with recognition_lock:
            if not recognition_running:
                break

        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=UPSAMPLE, model="hog")
        enc_curr = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), e in zip(boxes, enc_curr):
            dists = face_recognition.face_distance(encs, e)
            if len(dists) == 0:
                name = "Unknown"
            else:
                idx = int(np.argmin(dists))
                name = names[idx] if dists[idx] <= MATCH_THRESHOLD else "Unknown"

            color = (0,255,0) if name != "Unknown" else (0,0,255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = max(top - 8, 20)
            cv2.putText(frame, name, (left + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if name != "Unknown" and (time.time() - last_unlock) >= DEBOUNCE:
                ok_unlock = esp_unlock()
                status = "UNLOCKED" if ok_unlock else "ESP TIMEOUT"
                banner_color = (0,255,0) if ok_unlock else (0,165,255)
                last_unlock = time.time()
                cv2.rectangle(frame, (0,0), (frame.shape[1], 36), (30,30,30), -1)
                cv2.putText(frame, f"Door: {status}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, banner_color, 2)

        # banner status default
        cv2.rectangle(frame, (0,0), (frame.shape[1], 36), (30,30,30), -1)
        cv2.putText(frame, "Door: LOCKED", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        cv2.imshow("Realtime Unlock (tekan 'q' untuk stop juga)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            with recognition_lock:
                recognition_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ================== ROUTES ==================
@app.route("/")
def index():
    # ambil list user terdaftar (untuk info saja)
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
    with recognition_lock:
        if recognition_running:
            flash("Recognition sudah berjalan.", "info")
            return redirect(url_for("index"))
        recognition_running = True
    recognition_thread = threading.Thread(target=recognition_loop, daemon=True)
    recognition_thread.start()
    flash("Recognition dimulai. Lihat jendela kamera. Klik Stop atau tekan 'q' di jendela kamera untuk berhenti.", "success")
    return redirect(url_for("index"))

@app.route("/stop", methods=["POST"])
def stop():
    global recognition_running
    with recognition_lock:
        recognition_running = False
    flash("Recognition dihentikan.", "info")
    return redirect(url_for("index"))

if __name__ == "__main__":
    # Jalankan Flask
    app.run(host="0.0.0.0", port=5000, debug=True)
