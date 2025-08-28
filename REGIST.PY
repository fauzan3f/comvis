import cv2
import os
from datetime import datetime

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

username = input("Masukkan nama user: ").strip()
user_dir = os.path.join(DATASET_DIR, username)
os.makedirs(user_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print("[INFO] Tekan 's' untuk simpan foto, 'q' untuk selesai.")
saved = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # panduan sederhana di layar
    h, w = frame.shape[:2]
    cv2.putText(frame, "Tekan 's' untuk simpan, 'q' untuk selesai",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Registrasi Wajah", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        fname = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(os.path.join(user_dir, fname), frame)
        saved += 1
        print(f"[SAVED] {fname} ({saved} foto)")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Total foto {username}: {saved}")
