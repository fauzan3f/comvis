import os
import pickle
import face_recognition

DATASET_DIR = "dataset"
ENC_FILE = "encodings.pkl"

known_encodings, known_names = [], []

print("[INFO] Training data wajah...")
for user in os.listdir(DATASET_DIR):
    user_dir = os.path.join(DATASET_DIR, user)
    if not os.path.isdir(user_dir):
        continue

    for img_name in os.listdir(user_dir):
        img_path = os.path.join(user_dir, img_name)
        try:
            image = face_recognition.load_image_file(img_path)
        except Exception as e:
            print(f"[SKIP] gagal load {img_name}: {e}")
            continue

        # deteksi wajah (lebih sensitif sedikit)
        boxes = face_recognition.face_locations(
            image, number_of_times_to_upsample=1, model="hog"
        )
        if not boxes:
            print(f"[SKIP] no face: {img_name}")
            continue

        encs = face_recognition.face_encodings(image, boxes)
        if not encs:
            print(f"[SKIP] no encoding: {img_name}")
            continue

        known_encodings.append(encs[0])
        known_names.append(user)
        print(f"[OK] {user} <- {img_name}")

data = {"encodings": known_encodings, "names": known_names}
with open(ENC_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"[DONE] {len(known_encodings)} embeddings disimpan ke {ENC_FILE}")
