# SecureNest - AI-Powered Home Security System

A real-time AI-powered home security system that combines face recognition, weapon detection, and loitering detection to monitor and secure your home.

---

## Features
- **Face Recognition** — Identifies known family members and flags unknown individuals using KNN algorithm trained on custom face data
- **Weapon Detection** — Detects pistols and knives in real-time using YOLOv8n fine-tuned on 5,000+ images (86.3% mAP50)
- **Loitering Detection** — Triggers automated alert when an unknown person is present beyond a set duration
- **Email Alerts** — Sends automated SMTP email notifications to the homeowner on suspicious activity (coming soon)
- **Mobile App** — Flutter app for live monitoring and remote alerts (coming soon)

---

## Project Structure
```
SecureNest/
├── data/
│   └── known_faces/        # .npy face data files (gitignored)
├── face_recognition/
│   ├── haarcascade_frontalface_alt.xml
│   ├── face_recognition.py
│   └── collect_faces.py
├── weapon_detection/
│   └── best.pt             # Fine-tuned YOLOv8n model
├── .gitignore
├── requirements.txt
└── README.md
```
---

## How It Works

### Face Recognition
- Collects face samples at 10-frame intervals to ensure diversity and prevent overfitting
- Flattens face images into numpy arrays and stores as `.npy` files
- KNN classifier compares incoming face against known faces using Euclidean distance
- Distance threshold of 4300 separates known from unknown faces — tuned experimentally on real data

### Weapon Detection
- YOLOv8n fine-tuned on Sohas Weapon Detection dataset (5,000+ images)
- Detects pistols and knives in real-time camera feed
- Trained on Google Colab with T4 GPU, achieving 86.3% mAP50

### Loitering Detection
- Timer starts when an unknown face is detected
- Alert triggered if unknown person remains beyond threshold duration
- Grace period prevents false resets when face briefly leaves frame

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/prajna1606/SecureNest.git
cd SecureNest
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Collect face data
```bash
python face_recognition/collect_faces.py
```

### 4. Run face recognition
```bash
python face_recognition/face_recognition.py
```

### 5. Run weapon detection
```bash
python weapon_detection/weapon_detection.py
```

---

## Tech Stack
- Python
- OpenCV
- NumPy
- Ultralytics YOLOv8
- KNN (custom implementation)
- Flutter (mobile app — coming soon)

---

## Model Details
| Component | Approach | Performance |
|-----------|----------|-------------|
| Face Recognition | Custom KNN + Euclidean Distance | Threshold tuned at 4300 |
| Weapon Detection | YOLOv8n fine-tuned | 86.3% mAP50 |

---

## Roadmap
- [x] Face recognition with unknown detection
- [x] Weapon detection
- [x] Loitering detection with alerts
- [ ] Email notifications via SMTP
- [ ] Flutter mobile app
- [ ] Raspberry Pi deployment