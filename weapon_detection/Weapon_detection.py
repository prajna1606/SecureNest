from ultralytics import YOLO
import cv2
model = YOLO("weapon_detection/best.pt")
cap = cv2.VideoCapture(0)
while True:
    ret, frame =cap.read()
    if ret == False:
        continue
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Weapon Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()