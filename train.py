import os
import cv2
import pickle
import face_recognition
import numpy as np

DATASET_DIR = "dataset"
ENC_FILE = "encodings.pkl"

def train_dataset():
    print("[INFO] Starting face training...")
    known_encodings = []
    known_names = []

    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print("[ERROR] Dataset directory not found")
        return False

    # Process each user directory
    for user in os.listdir(DATASET_DIR):
        user_dir = os.path.join(DATASET_DIR, user)
        if not os.path.isdir(user_dir):
            continue

        print(f"[INFO] Processing user {user}...")
        
        # Process each image
        for img_name in os.listdir(user_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(user_dir, img_name)
            print(f"[INFO] Processing {img_path}")

            try:
                # Load and encode face
                image = face_recognition.load_image_file(img_path)
                boxes = face_recognition.face_locations(image, model="hog")

                if not boxes:
                    print(f"[WARN] No face found in {img_name}")
                    continue

                # Get face encodings
                encodings = face_recognition.face_encodings(image, boxes)
                
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(user)
                    print(f"[OK] Encoded {user}/{img_name}")

            except Exception as e:
                print(f"[ERROR] Failed to process {img_name}: {str(e)}")

    # Save results if we found any faces
    if known_encodings:
        data = {
            "encodings": known_encodings,
            "names": known_names
        }
        with open(ENC_FILE, "wb") as f:
            pickle.dump(data, f)
        
        print(f"[SUCCESS] Saved {len(known_encodings)} encodings to {ENC_FILE}")
        return True
    else:
        print("[ERROR] No faces were encoded")
        return False

if __name__ == "__main__":
    train_dataset()
