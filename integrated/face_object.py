from imutils.video import VideoStream
from imutils.video import FPS
import pytesseract
import numpy as np
import imutils
import pickle
import time
import cv2
import os
import threading
import pyttsx3

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"bed", "train", "tvmonitor"]

engine=pyttsx3.init()
def spk(string):
	rate = engine.getProperty('rate')
	engine.setProperty('rate', rate-5)	
	engine.say(string)
	engine.runAndWait()
	print(string)

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("prototxt.txt", "m.caffemodel")
detector = cv2.dnn.readNetFromCaffe("face_detection_model/deploy.prototxt","face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

print("Starting..................................................................................")
spk("Welcome!")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
	f_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

	net.setInput(blob)
	detections = net.forward()
	detector.setInput(f_blob)
	detection = detector.forward()

	key = cv2.waitKey(1) & 0xFF

	if key == ord("o"):
		string=""
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.1:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				obj = CLASSES[idx]
				angle=(box[0]+box[2]-650)/10
				angle=int(angle)
				if abs(angle)<5:
					st=obj+" at centre."
				elif angle<0:
					angle=abs(angle)
					st=obj+" at "+str(angle)+" degree the right."
				else:
					st=obj+" at "+str(angle)+" degree the left."
				if obj!="person":
					string=string+st+"\n"
		for i in range(0, detection.shape[2]):
			confidence = detection[0, 0, i, 2]
			if confidence > 0.5:
				box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				angle=(box[0]+box[2]-650)/10
				angle=int(angle)
				if abs(angle)<5:
					st=name+" at centre."
				elif angle<0:
					angle=abs(angle)
					st=name+" at "+str(angle)+" degree the right."
				else:
					st=name+" at "+str(angle)+" degree the left."
				string=string + st + "\n"
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(frame, name, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		t1 = threading.Thread(target=spk, args=(string,))
		t1.start()
	elif key==ord("s"):	
		ocr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
		string=(pytesseract.image_to_string(ocr))
		print(string)
		if string is "":
			string="can not recognize."
		t1 = threading.Thread(target=spk, args=(string,))
		t1.start()

	elif key==ord("q"):
		break

	else:
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.1:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				obj =CLASSES[idx]
				if obj != "person":
					cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, obj, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
		for i in range(0, detection.shape[2]):
			confidence = detection[0, 0, i, 2]
			if confidence > 0.5:
				box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(frame, name, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)				

	cv2.imshow("Frame", frame)
	fps.update()
fps.stop()
cv2.destroyAllWindows()
vs.stop()