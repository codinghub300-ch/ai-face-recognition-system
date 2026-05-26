# AI Face Recognition System 

A real-time AI-powered Face Detection and Recognition application built with Python, OpenCV, DeepFace, and CustomTkinter.

---

##  Features

- Real-time face detection using webcam
- Face recognition using Deep Learning
- Register new faces dynamically
- Capture image snapshots
- Modern GUI interface
- Live recognition status updates
- Known faces database
- Automatic facial embedding comparison

---

##  Technologies Used

- Python
- OpenCV
- DeepFace
- NumPy
- CustomTkinter
- Pillow (PIL)
- Threading

---

##  AI Model Used

The system uses:

- FaceNet Model (via DeepFace)

For:
- Face Embedding Extraction
- Face Matching & Recognition

---

## Application Features
▶ Start Recognition

Starts real-time webcam face recognition.

⏸ Stop Recognition

Stops webcam processing.

📸 Capture Snapshot

Saves the current webcam frame as an image.

🧍 Register New Face

Allows users to register and save a new face into the system database.

-- 
## Recognition Workflow
. Webcam captures live video
. Faces are detected using OpenCV
. DeepFace extracts facial embeddings
. System compares embeddings with stored known faces
. Matching faces are identified and displayed

--

## GUI Features
. Dark mode interface
. Live video feed
. Detection status updates
. Interactive control buttons
. Clean modern UI using CustomTkinter

 --
 
 ## Concepts Used
. Artificial Intelligence
. Deep Learning
. Computer Vision
. Face Recognition
. Real-Time Video Processing
. GUI Development
. Multithreading

--
##  How to Run

### 1. Install Required Libraries

```bash
pip install opencv-python deepface numpy customtkinter pillow
