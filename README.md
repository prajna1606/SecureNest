# SecureNest

**Secure Nest** is a smart home security project that explores multiple face recognition approaches to detect known individuals and send alerts for unusual activity at the door.

## Features

- Real-time face recognition using webcam
- Multiple models explored:
  - `face_recognition` (lightweight, fast)
  - `DeepFace` with FaceNet (accurate and robust)
  - Manual `.h5` model loading (experimental)
- Detects if someone stays for more than 15 seconds
- Sends email alerts for unknown or suspicious presence
- Project is evolving: future additions include alerts, UI, and door unlocking logic


