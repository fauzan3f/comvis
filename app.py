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
FRAME_W, FRAME_H = 320, 240        # Resolusi rendah untuk speed maksimal
PROCESS_SCALE = 0.5                 # Balance antara speed dan akurasi
PROCESS_EVERY_N_FRAMES = 2          # Proses setiap 2 frame saja
FPS_TARGET = 15                     # Target FPS realistis
LABEL_PERSIST_TIME = 1.5            # Persist label lebih singkat

# Recognition - BALANCED SETTINGS
UPSAMPLE = 0                        # tetap 0 untuk speed
MATCH_THRESHOLD = 0.6               # Threshold seimbang
UNKNOWN_THRESHOLD = 0.8             # Threshold untuk unknown
DEBOUNCE = 2.0                      # Debounce lebih cepat
REGISTER_SAMPLES = 15               # Kurangi samples untuk training lebih cepat
MIN_FACE_SIZE = 40                  # Minimum face size
CONFIDENCE_THRESHOLD = 0.5          # Confidence threshold seimbang

# Timing
UNLOCK_DURATION = 5.0
LABEL_TIMEOUT = 2.0                 # Timeout label
PROCESS_EVERY_N = 2                 # Konsisten dengan PROCESS_EVERY_N_FRAMES

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

        # Optimasi kamera untuk performa maksimal - FIXED
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer untuk mengurangi lag
        
        # Disable auto settings untuk konsistensi
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        
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
            
            # HAPUS frame skipping di sini - biarkan recognition_loop yang handle
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

def distance_to_confidence(dist, match_threshold=MATCH_THRESHOLD, max_dist=0.6):
    """Convert face distance to confidence percentage"""
    if dist > max_dist:
        return 0.0
    elif dist < match_threshold:
        return (max_dist - dist) / max_dist
    else:
        return 0.0

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
    """Helper function to draw consistent face labels with confidence - FIXED"""
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
    
    # Draw rectangle around face - FIXED: gunakan koordinat yang benar
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Draw label background - FIXED: posisi yang lebih aman
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    label_y = max(35, top)  # Pastikan tidak keluar dari frame atas
    
    cv2.rectangle(frame,
                 (left, label_y - 35),
                 (left + label_size[0] + 10, label_y),
                 color, -1)
    
    # Draw name with confidence - FIXED: posisi yang tepat
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

# ========== RECOGNITION LOOP - PINDAHKAN KE SINI ==========
def recognition_loop():
    global recognition_running

    encs, names = load_encodings()
    print(f"Loaded {len(encs)} encodings for names: {names}")
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
    
    # Tambahkan FPS counter untuk monitoring
    fps_counter = 0
    fps_start_time = time.time()

    try:
        while recognition_running:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
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
            
            # ALWAYS draw existing labels first - FIXED
            for face_info in face_labels.values():
                left, top, right, bottom = face_info['bbox']
                name = face_info['name']
                confidence = face_info.get('confidence', 0)
                draw_face_label(frame, name, left, top, right, bottom, confidence)
            
            # OPTIMIZED: Proses face recognition hanya setiap N frame
            if frame_count % PROCESS_EVERY_N == 0:
                # Resize frame untuk processing yang lebih cepat
                small_frame = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Face detection dengan model HOG (lebih cepat)
                face_locations = face_recognition.face_locations(rgb_small_frame, 
                                                               number_of_times_to_upsample=UPSAMPLE,
                                                               model="hog")
                
                print(f"Found {len(face_locations)} faces")  # Debug
                
                # Filter faces berdasarkan ukuran minimum
                valid_boxes = []
                for (top, right, bottom, left) in face_locations:
                    # Scale back ke ukuran asli
                    top = int(top / PROCESS_SCALE)
                    right = int(right / PROCESS_SCALE)
                    bottom = int(bottom / PROCESS_SCALE)
                    left = int(left / PROCESS_SCALE)
                    
                    # Check minimum face size
                    face_width = right - left
                    face_height = bottom - top
                    print(f"Face size: {face_width}x{face_height}")  # Debug
                    
                    if face_width >= MIN_FACE_SIZE and face_height >= MIN_FACE_SIZE:
                        valid_boxes.append((top, right, bottom, left))
                        print(f"Valid face at: ({left}, {top}, {right}, {bottom})")  # Debug

                if len(valid_boxes) > 0:
                    # Encode faces yang valid saja
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, 
                        [(int(t*PROCESS_SCALE), int(r*PROCESS_SCALE), int(b*PROCESS_SCALE), int(l*PROCESS_SCALE)) 
                         for t,r,b,l in valid_boxes])

                    for i, face_encoding in enumerate(face_encodings):
                        top, right, bottom, left = valid_boxes[i]
                        
                        name = "Unknown"
                        confidence = 0.0
                        
                        if len(encs) > 0:
                            # Calculate distances
                            face_distances = face_recognition.face_distance(encs, face_encoding)
                            min_distance = np.min(face_distances)
                            confidence = distance_to_confidence(min_distance)
                            
                            # Debug output
                            print(f"Face distances: {face_distances}")
                            print(f"Min distance: {min_distance:.3f}, Confidence: {confidence:.3f}")
                            
                            if min_distance <= MATCH_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
                                best_match_index = np.argmin(face_distances)
                                name = names[best_match_index]
                                print(f"RECOGNIZED: {name} with confidence {confidence:.3f}")
                                
                                # Unlock logic dengan debounce
                                if current_time - last_unlock > DEBOUNCE:
                                    if door_status == "LOCKED":
                                        print(f"Attempting to unlock for {name}")
                                        if esp_unlock():
                                            door_status = "UNLOCKED"
                                            unlock_end_time = current_time + UNLOCK_DURATION
                                            last_unlock = current_time
                                            print(f"Door unlocked for {name}")
                            else:
                                print(f"UNKNOWN: confidence {confidence:.3f} < threshold {CONFIDENCE_THRESHOLD}")
                        
                        # Store face info untuk persistent labeling - FIXED
                        face_key = f"{left}_{top}_{frame_count}"
                        face_labels[face_key] = {
                            'name': name,
                            'bbox': (left, top, right, bottom),  # FIXED: include right, bottom
                            'confidence': confidence,
                            'last_seen': current_time
                        }
                        
                        print(f"Stored label: {name} at ({left}, {top}, {right}, {bottom})")  # Debug
                else:
                    print("No valid faces found")  # Debug

            # Draw status banner
            draw_status_banner(frame, door_status)
            
            # Show frame
            cv2.imshow("Face Recognition", frame)
            
            # Kurangi delay untuk responsivitas
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in recognition loop: {e}")
        import traceback
        traceback.print_exc()  # Print full error untuk debugging
    finally:
        cam.stop()
        cv2.destroyAllWindows()
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
    train_dataset()
    flash("Training completed", "success")
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
