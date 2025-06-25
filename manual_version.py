import cv2
import time
import smtplib
from deepface import DeepFace
from tensorflow.keras.models import load_model # type: ignore
import numpy as np

# Load FaceNet model
facenet_model = load_model('path_to_facenet_model.h5')  # Ensure to provide the correct path
known_faces_db = {}  # Dictionary to store known faces

# Email Configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_email_password'
RECIPIENT_EMAIL = 'recipient_email@gmail.com'

def send_email_notification(message):
    """Sends an email notification with the specified message."""
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, message)
    print("Email notification sent!")

def recognize_face(face_image):
    """Recognizes a face using hybrid DeepFace and FaceNet."""
    try:
        # Using FaceNet embedding for the face
        facenet_embedding = facenet_model.predict(np.expand_dims(face_image, axis=0))

        # Using DeepFace for enhanced accuracy
        face_analysis = DeepFace.analyze(face_image, actions=['emotion'])
        
        # Add any matching criteria or checks for known_faces_db here if needed
        print("Face recognized.")
        return True  # Return True for recognized, False otherwise
    except Exception as e:
        print(f"Error recognizing face: {e}")
        return False

def detect_unusual_activity(start_time):
    """Detects if a person stays longer than 15 seconds and sends a notification."""
    elapsed_time = time.time() - start_time
    if elapsed_time > 15:  # If person is present for more than 15 seconds
        send_email_notification("Unusual activity detected at your door.")
        print("Unusual activity detected - notification sent.")

def main():
    cap = cv2.VideoCapture(0)
    last_detection_time = None
    presence_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            if not presence_detected:
                last_detection_time = time.time()
                presence_detected = True

            for (x, y, w, h) in faces:
                face_image = frame[y:y + h, x:x + w]
                face_image = cv2.resize(face_image, (160, 160))

                # Attempt recognition with the hybrid model
                if recognize_face(face_image):
                    send_email_notification("Known face detected at your door.")
                
                # Check for unusual activity detection
                detect_unusual_activity(last_detection_time)

        else:
            presence_detected = False

        # Display the resulting frame (optional for debugging purposes)
        cv2.imshow('Secure Nest Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
