import cv2
import numpy as np
import os
import time
import smtplib
from email.message import EmailMessage

def send_email_alert(image_path):

    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = "Alert - Unknown Person Detected"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = "prajna@example.com"

    msg.set_content(
        "Unknown person detected loitering near your home. "
        "Attached is the captured image."
    )

    with open(image_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(image_path)

    msg.add_attachment(
        file_data,
        maintype="image",
        subtype="jpeg",
        filename=file_name
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print("Email sent successfully!")

def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist=[]
    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=distance(test,ix)
        dist.append([d,iy])
    dk=sorted(dist,key=lambda x:x[0])[:k]
    labels=np.array(dk)[:,-1]

    output=np.unique(labels,return_counts=True)
    index=np.argmax(output[1])
    if( dk[0][0]> 4300) :
        return -1
    return output[0][index]

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("face_recognition/haarcascade_frontalface_alt.xml")
dataset_path=r"data\known_faces"
face_data=[]
labels=[]
class_id=0
names={}
for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        names[class_id]=fx[:-4]
        data_item=np.load(os.path.join(dataset_path,fx))
        face_data.append(data_item)

        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_labels.shape)
print(face_dataset.shape)
trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

font=cv2.FONT_HERSHEY_SIMPLEX
unknown_start_time=None
unknown_detected=False
alert_triggered=False
while True:
    ret, frame=cap.read()
    if ret == False:
        continue
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        out=knn(trainset,face_section.flatten())
        if(out==-1):
            pred_name="Unknown"
            unknown_detected= True
            if unknown_start_time is None:
                unknown_start_time = time.time()
            elif unknown_detected and time.time() - unknown_start_time > 10 and not alert_triggered:
                print("ALERT!! Unknown person loitering detected")
                image_path = "alert.jpg"
                cv2.imwrite(image_path, frame)
                # Send email
                send_email_alert(image_path)
                alert_triggered = True
        else:   
            pred_name=names[int(out)]
            unknown_detected = False
            unknown_start_time = None
            alert_triggered = False
        cv2.putText(frame,pred_name,(x,y-10),font,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Faces",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()