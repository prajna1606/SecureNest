from ultralytics import YOLO
import cv2
import time
import os
import smtplib
from email.message import EmailMessage

# Load model
model = YOLO("weapon_detection/best.pt")

# Webcam
cap = cv2.VideoCapture(0)

# Cooldown settings
last_alert_time = 0
ALERT_COOLDOWN = 30  # seconds


# ================= EMAIL FUNCTION =================
def send_email_alert(image_path, weapon_name):

    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

    RECEIVER_EMAIL = "prajnaasati@example.com"

    msg = EmailMessage()

    msg["Subject"] = f"SecureNest Alert - {weapon_name} Detected"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECEIVER_EMAIL

    msg.set_content(
        f"SecureNest detected a suspicious weapon: {weapon_name}.\n"
        "Attached is the captured image."
    )

    # Attach image
    with open(image_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(image_path)

    msg.add_attachment(
        file_data,
        maintype="image",
        subtype="jpeg",
        filename=file_name
    )

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print("Email alert sent successfully!")


# ================= MAIN LOOP =================
while True:

    ret, frame = cap.read()

    if ret == False:
        continue

    # Run YOLO inference
    results = model(frame)

    # Annotated frame
    annotated_frame = results[0].plot()

    # Current time
    current_time = time.time()

    # Process detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            class_name = model.names[cls]

            # Confidence score
            confidence = float(box.conf[0])

            # Detect only weapons
            if class_name in ["knife", "pistol"] and confidence > 0.5:

                cv2.putText(
                    annotated_frame,
                    "WEAPON DETECTED!",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Cooldown check
                if current_time - last_alert_time > ALERT_COOLDOWN:

                    print(f"ALERT! {class_name} detected")

                    # Save image
                    image_path = f"weapon_{int(time.time())}.jpg"

                    cv2.imwrite(image_path, annotated_frame)

                    # Send email
                    send_email_alert(image_path, class_name)

                    # Update timer
                    last_alert_time = current_time

    # Show output
    cv2.imshow("Weapon Detection", annotated_frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()