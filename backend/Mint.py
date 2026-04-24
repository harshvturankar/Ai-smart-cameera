import cv2
import os
import numpy as np

# -------------------- PATHS --------------------
DATASET_PATH = r"C:\Users\Asus\Desktop\ai-surveillance-system\backend\authorization\known_faces"
MODEL_FILE = r"C:\Users\Asus\Desktop\ai-surveillance-system\backend\authorization\face_model.yml"

# -------------------- INITIALIZATION --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map = {} # You will need to save/load this separately in a real app (e.g., to a JSON file or DB)

# -------------------- LOAD OR TRAIN MODEL --------------------
if os.path.exists(MODEL_FILE):
    print("🔄 Loading existing face model...")
    recognizer.read(MODEL_FILE)
    print("✅ Model loaded successfully.")
    
    # Note: If loading from a file, you'd also need to load your label_map here 
    # so you know which ID corresponds to which name!
else:
    print("⚙️ No saved model found. Training from dataset...")
    faces = []
    labels = []
    current_label = 0

    if os.path.exists(DATASET_PATH):
        for person_name in os.listdir(DATASET_PATH):
            person_path = os.path.join(DATASET_PATH, person_name)

            if not os.path.isdir(person_path):
                continue

            label_map[current_label] = person_name

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in detected_faces:
                    faces.append(gray[y:y+h, x:x+w])
                    labels.append(current_label)

            current_label += 1
            
        if len(faces) > 0:
            recognizer.train(faces, np.array(labels))
            recognizer.save(MODEL_FILE) # Save the model so we don't have to train again!
            print(f"✅ Face model trained on {len(faces)} faces and saved to disk.")
        else:
            print("❌ No faces were found in the dataset to train on.")
    else:
        print(f"❌ Error: Dataset path '{DATASET_PATH}' does not exist.")


# ================== MAIN FUNCTION ==================
def recognize_face(person_crop):
    try:
        if person_crop is None or person_crop.size == 0:
            return "unknown"
            
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        
        # Try to find a face in the crop
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If Haar finds a face, use it
        if len(faces_detected) > 0:
            for (fx, fy, fw, fh) in faces_detected:
                face_img = gray[fy:fy+fh, fx:fx+fw]
                
                # Resize to a standard size for better LBPH accuracy (optional but recommended)
                face_img = cv2.resize(face_img, (200, 200)) 
                
                id_, confidence = recognizer.predict(face_img)
                if confidence < 100:
                    return "known" 
                    
        # FALLBACK: If Haar failed to find a face, but we KNOW this is a person crop from YOLO
        else:
            # Assume the whole crop is the face (or head), resize, and attempt prediction anyway
            face_img = cv2.resize(gray, (200, 200))
            id_, confidence = recognizer.predict(face_img)
            
            # Require a stricter confidence threshold here since we aren't 100% sure it's a perfectly cropped face
            if confidence < 80: 
                return "known"

        return "unknown"

    except Exception as e: 
        print(f"⚠️ Recognition Error: {e}")
        return "unknown"