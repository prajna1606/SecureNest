import cv2
import numpy as np
import os
import time
import smtplib
from email.message import EmailMessage
from ultralytics import YOLO
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("face_recognition/haarcascade_frontalface_alt.xml")
dataset_path=r"data\known_faces"
face_data=[]
labels=[]
class_id=0
names={}
model = YOLO("weapon_detection/best.pt")
