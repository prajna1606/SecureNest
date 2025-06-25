# Import necessary libraries
from deepface import DeepFace
import cv2
import time
import smtplib
from email.mime.text import MIMEText
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize parameters
THRESHOLD = 0.9
ALERT_DELAY = 15  # Time in seconds for unusual activity
face_detected_time = None
email_sent = False

# Placeholder for known face embeddings
known_embeddings = []  # Populate this with actual embeddings from known faces

# Function to get embedding for a captured image using DeepFace/FaceNet hybrid
def get_embedding(frame):
    try:
        # Get embedding using DeepFace with FaceNet
        representation = DeepFace.represent(frame, model_name='Facenet', enforce_detection=True)
        return np.array(representation[0]['embedding'])
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

# Function to send email notification
def send_email_alert():
    global email_sent
    sender_email = "prajnaasati@gmail.com"
    receiver_email = "iotp256@gmail.com"
    password = "prajna123102043"  # Ensure to handle this securely in production

    msg = MIMEText("Unusual activity detected outside your door.")
    msg['Subject'] = "SecureNest Alert!"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Alert email sent.")
        email_sent = True
    except Exception as e:
        print(f"Error sending email: {e}")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Get the embedding for the current frame
    embedding = get_embedding(frame)

    # Check similarity with known faces
    if embedding is not None:
        matches = [cosine_similarity([embedding], [known_emb])[0][0] > THRESHOLD for known_emb in known_embeddings]

        if any(matches):
            if face_detected_time is None:
                face_detected_time = time.time()
            elif time.time() - face_detected_time > ALERT_DELAY and not email_sent:
                print("Unusual activity detected.")
                send_email_alert()
        else:
            # Reset if no matching face is detected
            face_detected_time = None
            email_sent = False

    # Display the video feed
    cv2.imshow('SecureNest Feed', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
