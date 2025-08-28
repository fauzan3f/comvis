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

# ========== KONFIGURASI OPTIMAL UNTUK PERFORMA ==========
# Kamera & performa - OPTIMIZED FOR SMOOTH PERFORMANCE
FRAME_W, FRAME_H = 640, 480         # Resolusi rendah untuk speed maksimal
PROCESS_SCALE = 0.5                 # Balance antara speed dan akurasi
PROCESS_EVERY_N_FRAMES = 2          # Proses setiap 2 frame saja
FPS_TARGET = 15                     # Target FPS realistis
LABEL_PERSIST_TIME = 1.5            # Persist label lebih singkat

# Recognition - ADJUSTED FOR BETTER DETECTION
UPSAMPLE = 1                        # Increase for better accuracy
MATCH_THRESHOLD = 0.45              # Lower threshold for easier matching
UNKNOWN_THRESHOLD = 0.8             # Threshold untuk unknown
DEBOUNCE = 1.0                      # Faster debounce
REGISTER_SAMPLES = 15               # Kurangi samples untuk training lebih cepat
MIN_FACE_SIZE = 40                  # Minimum face size
CONFIDENCE_THRESHOLD = 0.3          # Lower confidence threshold

# Timing
UNLOCK_DURATION = 3.0
LABEL_TIMEOUT = 2.0                 # Timeout label
PROCESS_EVERY_N = 2                 # Konsisten dengan PROCESS_EVERY_N_FRAMES
SHOW_LABEL_DURATION = 3.0           # Show user label for 3 seconds after recognition
STOP_CAMERA_AFTER_UNLOCK = True     # Stop camera after successful unlock

# Add these constants at the top
UNKNOWN_LABEL_DURATION = 1.0  # Duration in seconds for Unknown label
UNKNOWN_CONFIDENCE_THRESHOLD = 0.3  # Threshold for unknown detection

os.makedirs(DATASET_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "face-doorlock-secret"

# Global vars
recognition_thread = None
recognition_running = False
recognition_lock = threading.Lock()

# ========== CAMERA STREAM OPTIMIZED ==========
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

        # Optimize camera settings for brightness
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Add these lines to adjust brightness
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)     # Increase brightness (0-255)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 120)       # Adjust contrast (0-255)
        self.cap.set(cv2.CAP_PROP_GAIN, 100)          # Increase gain if available
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)    # Auto exposure on
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -3)        # Adjust exposure if manual (-7 to 0)
        
        print("Camera settings:")
        print(f"Brightness: {self.cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
        print(f"Contrast: {self.cap.get(cv2.CAP_PROP_CONTRAST)}")
        print(f"Exposure: {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.frame_count = 0
        self.t = threading.Thread(target=self.update, daemon=True)
        self.t.start()

    def update(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)  # Prevent busy waiting
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

# ========== HELPER FUNCTIONS ==========
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
    """Train face encodings dari dataset folder"""
    all_encodings = []
    all_names = []
    
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Training {person_name}...")
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(person_dir, img_file)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            
            if encodings:
                all_encodings.append(encodings[0])
                all_names.append(person_name)
    
    if all_encodings:
        save_encodings(all_encodings, all_names)
        print(f"Training selesai: {len(all_encodings)} encodings untuk {len(set(all_names))} orang")
        return True
    return False

def register_user(username, samples=REGISTER_SAMPLES):
    """Register user baru dengan mengambil foto dari kamera"""
    user_dir = os.path.join(DATASET_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    
    try:
        cam = CameraStream()
    except RuntimeError as e:
        return False, str(e)
    
    count = 0
    while count < samples:
        frame = cam.read()
        if frame is None:
            continue
            
        cv2.putText(frame, f"Foto {count+1}/{samples} - Tekan SPACE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Register User", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            timestamp = int(time.time() * 1000)
            filename = f"{username}_{timestamp}.jpg"
            filepath = os.path.join(user_dir, filename)
            cv2.imwrite(filepath, frame)
            count += 1
            print(f"Saved: {filename}")
        elif key == ord('q'):
            break
    
    cam.stop()
    cv2.destroyAllWindows()
    return count == samples, f"Berhasil mengambil {count} foto"

def distance_to_confidence(dist, match_threshold=MATCH_THRESHOLD, max_dist=1.0):
    """Convert face distance to confidence percentage - IMPROVED"""
    if dist > max_dist:
        return 0.0
    else:
        # Linear mapping: closer distance = higher confidence
        confidence = 1.0 - (dist / max_dist)
        return max(0.0, confidence)

def esp_unlock_async():
    threading.Thread(target=esp_unlock, daemon=True).start()

def esp_unlock():
    """Kirim perintah unlock ke ESP32"""
    for attempt in range(ESP_RETRIES):
        try:
            response = requests.get(UNLOCK_URL, timeout=ESP_TIMEOUT)
            if response.status_code == 200:
                print(f"ESP unlock berhasil (attempt {attempt+1})")
                return True
        except requests.exceptions.RequestException as e:
            print(f"ESP unlock gagal (attempt {attempt+1}): {e}")
            if attempt < ESP_RETRIES - 1:
                time.sleep(0.5)
    return False

def draw_face_label(frame, name, left, top, right, bottom, confidence=0.0):
    """Helper function to draw consistent face labels with confidence"""
    # Pastikan koordinat dalam batas frame
    h, w = frame.shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    
    # Warna berdasarkan status dan confidence
    if name == "Unknown":
        color = (0, 0, 255)  # Merah untuk unknown
        label_text = "UNKNOWN"
    else:
        # Hijau untuk dikenal, intensitas berdasarkan confidence
        green_intensity = int(255 * min(confidence, 1.0))
        color = (0, green_intensity, 0)
        label_text = f"{name} ({confidence:.2f})"
    
    # Draw rectangle around face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Draw label background
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    label_y = max(35, top)  # Pastikan tidak keluar dari frame atas
    
    cv2.rectangle(frame,
                 (left, label_y - 35),
                 (left + label_size[0] + 10, label_y),
                 color, -1)
    
    # Draw name with confidence
    cv2.putText(frame, label_text,
                (left + 5, label_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def draw_status_banner(frame, status):
    """Helper function to draw status banner"""
    cv2.rectangle(frame, (0,0), (frame.shape[1], 40), (30,30,30), -1)
    color = (0,255,0) if status == "UNLOCKED" else (0,0,255)
    
    # Tambahkan timestamp
    timestamp = time.strftime("%H:%M:%S")
    status_text = f"Door: {status} | {timestamp}"
    
    cv2.putText(frame, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def draw_user_label(frame, username, unlock_time_remaining=0):
    """Draw user label with unlock countdown"""
    h = frame.shape[0]
    if unlock_time_remaining > 0:
        text = f"WELCOME {username.upper()}! Unlocking....... )"
        color = (0, 255, 0)  # Green
    else:
        text = f"Welcome back, {username}!"
        color = (0, 200, 200)  # Yellow-green
    
    # Background rectangle
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(frame, (10, h - 60), (text_size[0] + 20, h - 10), (0, 0, 0), -1)
    
    # Text
    cv2.putText(frame, text, (15, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# ========== RECOGNITION LOOP - OPTIMIZED VERSION ==========
def recognition_loop():
    global recognition_running

    encs, names = load_encodings()
    print(f"Loaded {len(encs)} encodings for names: {names}")
    if len(encs) == 0:
        print("No encodings found. Silakan train dulu.")
        return

    # Start camera
    try:
        cam = CameraStream(index_candidates=(0,1))
    except RuntimeError as e:
        print(str(e))
        return

    # State variables
    frame_count = 0
    last_unlock = 0
    unlock_end_time = 0
    door_status = "LOCKED"
    
    # Recognition state - SIMPLIFIED
    recognized_user = None
    show_label_until = 0
    camera_stop_time = 0
    unlock_in_progress = False
    
    # Add these variables inside the function
    unknown_faces = {}  # Store unknown face locations and their timestamps
    current_time = time.time()
    
    try:
        while recognition_running:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            display_frame = frame.copy()
            
            # Check if we should stop camera after successful unlock
            if STOP_CAMERA_AFTER_UNLOCK and camera_stop_time > 0 and current_time >= camera_stop_time:
                print("Stopping camera after successful unlock...")
                break
            
            # Update door status
            if unlock_end_time > 0 and current_time >= unlock_end_time:
                door_status = "LOCKED"
                unlock_end_time = 0
                unlock_in_progress = False
            
            # Process face detection every N frames
            frame_count += 1
            if frame_count % PROCESS_EVERY_N == 0 and not unlock_in_progress:
                # Clean old unknown faces first
                unknown_faces = {k:v for k,v in unknown_faces.items() 
                               if current_time - v['timestamp'] < UNKNOWN_LABEL_DURATION}

                # Detect faces
                small_frame = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, 
                                                              number_of_times_to_upsample=UPSAMPLE,
                                                              model="hog")
                
                for (top, right, bottom, left) in face_locations:
                    # Scale coordinates back
                    top = int(top / PROCESS_SCALE)
                    right = int(right / PROCESS_SCALE)
                    bottom = int(bottom / PROCESS_SCALE)
                    left = int(left / PROCESS_SCALE)

                    # Get face encoding
                    face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])
                    if not face_encoding:
                        # Add to unknown faces with timestamp
                        face_key = f"{left}_{top}"
                        unknown_faces[face_key] = {
                            'box': (left, top, right, bottom),
                            'timestamp': current_time
                        }
                        continue
                    
                    face_encoding = face_encoding[0]
                    min_distance = float('inf')  # Initialize default distance
                    
                    if len(encs) > 0:
                        face_distances = face_recognition.face_distance(encs, face_encoding)
                        min_distance = np.min(face_distances)
                        confidence = distance_to_confidence(min_distance)
                        
                        if min_distance <= MATCH_THRESHOLD:
                            # Known face detected
                            best_match_index = np.argmin(face_distances)
                            name = names[best_match_index]
                            
                            # Handle unlock process
                            if not unlock_in_progress and (current_time - last_unlock) >= DEBOUNCE:
                                if esp_unlock():
                                    door_status = "UNLOCKED"
                                    recognized_user = name
                                    show_label_until = current_time + SHOW_LABEL_DURATION
                                    unlock_end_time = current_time + UNLOCK_DURATION
                                    last_unlock = current_time
                                    unlock_in_progress = True
                                    print(f"Door unlocked for {name}")
                            
                            # Draw recognized face label
                            draw_face_label(display_frame, name, left, top, right, bottom, confidence)
                        else:
                            # Unknown face detected
                            face_key = f"{left}_{top}"
                            unknown_faces[face_key] = {
                                'box': (left, top, right, bottom),
                                'timestamp': current_time
                            }

            # Draw all unknown faces
            for face_info in unknown_faces.values():
                left, top, right, bottom = face_info['box']
                draw_face_label(display_frame, "Unknown", left, top, right, bottom, 0)

            # Always draw status banner
            draw_status_banner(display_frame, door_status)
            
            # Draw user welcome label if recently recognized
            if recognized_user and current_time <= show_label_until:
                unlock_remaining = max(0, unlock_end_time - current_time) if unlock_end_time > 0 else 0
                draw_user_label(display_frame, recognized_user, unlock_remaining)
            elif current_time > show_label_until:
                recognized_user = None
            
            # Show frame
            cv2.imshow("Face Recognition", display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in recognition loop: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        recognition_running = False
        print("Recognition stopped")

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
    
    try:
        register_user(username)
        flash(f"User {username} berhasil diregistrasi.", "success")
    except Exception as e:
        flash(f"Error: {e}", "error")
    
    return redirect(url_for("index"))

@app.route("/train", methods=["POST"])
def train():
    try:
        # Cek apakah ada data untuk training
        if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
            flash("Tidak ada data untuk training. Lakukan Register terlebih dahulu.", "error")
            return redirect(url_for("index"))

        print("[INFO] Memulai training data wajah...")
        known_encodings, known_names = [], []

        for user in os.listdir(DATASET_DIR):
            user_dir = os.path.join(DATASET_DIR, user)
            if not os.path.isdir(user_dir):
                continue

            print(f"[INFO] Training user {user}...")
            for img_name in os.listdir(user_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(user_dir, img_name)
                try:
                    image = face_recognition.load_image_file(img_path)
                    boxes = face_recognition.face_locations(image, model="hog")
                    
                    if not boxes:
                        print(f"[SKIP] No face detected: {img_name}")
                        continue

                    encodings = face_recognition.face_encodings(image, boxes)
                    if not encodings:
                        print(f"[SKIP] Could not encode: {img_name}")
                        continue

                    known_encodings.append(encodings[0])
                    known_names.append(user)
                    print(f"[OK] Processed {user}/{img_name}")

                except Exception as e:
                    print(f"[ERROR] Failed to process {img_name}: {str(e)}")
                    continue

        if not known_encodings:
            flash("Tidak ada wajah yang dapat dideteksi dalam dataset.", "error")
            return redirect(url_for("index"))

        # Save encodings
        data = {"encodings": known_encodings, "names": known_names}
        with open(ENC_FILE, "wb") as f:
            pickle.dump(data, f)

        flash(f"Training berhasil: {len(known_encodings)} encoding dari {len(set(known_names))} user", "success")
        print(f"[SUCCESS] Saved {len(known_encodings)} encodings to {ENC_FILE}")

    except Exception as e:
        flash(f"Error during training: {str(e)}", "error")
        print(f"[ERROR] Training failed: {str(e)}")
    
    return redirect(url_for("index"))

@app.route("/start", methods=["POST"])
def start():
    global recognition_thread, recognition_running
    
    if recognition_running:
        flash("Recognition sudah berjalan", "warning")
        return redirect(url_for("index"))
    
    with recognition_lock:
        recognition_running = True

    recognition_thread = threading.Thread(target=recognition_loop, daemon=True)
    recognition_thread.start()

    flash("Recognition started", "success")
    return redirect(url_for("index"))

@app.route("/stop", methods=["POST"])
def stop():
    global recognition_running
    
    with recognition_lock:
        recognition_running = False
    
    flash("Recognition stopped", "success")
    return redirect(url_for("index"))

@app.route("/unlock", methods=["GET"])
def unlock():
    esp_unlock_async()
    return jsonify({"status": "unlock_sent"})

if __name__ == "__main__":
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    app.run(host="0.0.0.0", port=5000, debug=True)