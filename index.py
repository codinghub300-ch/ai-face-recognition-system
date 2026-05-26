import cv2
from deepface import DeepFace
import numpy as np
import os
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
from datetime import datetime

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Recognition System")
        self.root.geometry("950x750")
        ctk.set_appearance_mode("dark")

        self.cap = None
        self.running = False
        self.frame_count = 0
        self.last_verified = {}  #cache for names
        self.known_embeddings = {}  # cache for known faces

        # UI 
        self.title_label = ctk.CTkLabel(root, text="🧠 Face Detection + Recognition", font=("Arial", 28, "bold"))
        self.title_label.pack(pady=15)

        self.video_label = ctk.CTkLabel(root, text="")
        self.video_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(root, text="Status: Idle", font=("Arial", 18))
        self.status_label.pack(pady=10)

        # Buttons 
        self.start_btn = ctk.CTkButton(root, text="▶ Start Recognition", command=self.start_recognition)
        self.start_btn.pack(side="left", padx=25, pady=20)

        self.stop_btn = ctk.CTkButton(root, text="⏸ Stop", command=self.stop_recognition)
        self.stop_btn.pack(side="left", padx=25, pady=20)

        self.capture_btn = ctk.CTkButton(root, text="📸 Capture Snapshot", command=self.capture_snapshot)
        self.capture_btn.pack(side="left", padx=25, pady=20)

        self.register_btn = ctk.CTkButton(root, text="🧍 Register New Face", command=self.register_new_face)
        self.register_btn.pack(side="left", padx=25, pady=20)

        # Load known faces and compute embeddings 
        self.known_faces_dir = "known_faces"
        os.makedirs(self.known_faces_dir, exist_ok=True)
        self.known_faces = [f for f in os.listdir(self.known_faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for face_file in self.known_faces:
            path = os.path.join(self.known_faces_dir, face_file)
            name = os.path.splitext(face_file)[0]
            try:
                embedding = DeepFace.represent(path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                self.known_embeddings[name] = embedding
            except:
                print(f"⚠️ Failed to process {face_file}")

        self.status_label.configure(text=f"✅ Loaded {len(self.known_embeddings)} known faces")

    def start_recognition(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.recognition_loop, daemon=True).start()

    def stop_recognition(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.status_label.configure(text="⏹ Recognition stopped")

    def recognition_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            if self.frame_count % 2 != 0:
                continue  

            frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            try:
                detections = DeepFace.extract_faces(frame_small, detector_backend='opencv', enforce_detection=False)
            except:
                detections = []

            for det in detections:
                region = det['facial_area']
                x, y, w, h = region['x']*2, region['y']*2, region['w']*2, region['h']*2
                face = frame[y:y+h, x:x+w]
                name = "Unknown"

                face_id = (x, y, w, h)
                if face_id in self.last_verified:
                    name = self.last_verified[face_id]
                else:
                    try:
                        face_embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                        min_dist = float("inf")
                        for known_name, known_emb in self.known_embeddings.items():
                            dist = np.linalg.norm(np.array(face_embedding) - np.array(known_emb))
                            if dist < min_dist and dist < 10:  
                                min_dist = dist
                                name = known_name
                    except:
                        name = "Unknown"

                    self.last_verified[face_id] = name

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
                cv2.putText(frame, f"{name}", (x+6, y+h-6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.status_label.configure(text=f"Faces Detected: {len(detections)}")

        if self.cap:
            self.cap.release()

    def capture_snapshot(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                filename = f"snapshot_{datetime.now().strftime('%H-%M-%S')}.jpg"
                cv2.imwrite(filename, frame)
                self.status_label.configure(text=f"📷 Snapshot saved: {filename}")

    def register_new_face(self):
        name_window = ctk.CTkInputDialog(text="Enter name for the new face:", title="Register New Face")
        name = name_window.get_input()
        if not name:
            self.status_label.configure(text="❌ Registration canceled (no name).")
            return

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.status_label.configure(text="⚠️ Failed to capture image.")
            return

        detections = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        if not detections:
            self.status_label.configure(text="❌ No face detected. Try again.")
            return

        face_data = detections[0]['face']
        face_image = np.array(face_data * 255, dtype=np.uint8)
        filename = os.path.join(self.known_faces_dir, f"{name}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

        try:
            embedding = DeepFace.represent(filename, model_name='Facenet', enforce_detection=False)[0]["embedding"]
            self.known_embeddings[name] = embedding
        except:
            pass

        self.known_faces.append(f"{name}.jpg")
        self.status_label.configure(text=f"✅ New face '{name}' registered.")

# Run 
if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceRecognitionApp(root)
    root.mainloop()
