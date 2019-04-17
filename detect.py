from imutils.video import VideoStream
from imutils.video import FPS
import win32com.client as wincl
from PIL import Image
import pytesseract
import threading
import numpy as np
import argparse
import imutils
import time
import cv2
from threading import Thread


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.01,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
speak = wincl.Dispatch("SAPI.SpVoice")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"bed", "train", "tvmonitor"]

def spk(st):
	speak.Speak(st)

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
k=0
# loop over the frames from the video stream

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	k=k+1
	print(k)
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	key = cv2.waitKey(1) & 0xFF
	if key == ord("o"):
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				ax=(box[2]-box[0])/2
				ax=(box[0]+ax-200)/3
				ax=int(ax)
				if abs(ax)<5:
					st=CLASSES[idx]+" at centre"
				elif ax<0:
					ax=abs(ax)
					st=CLASSES[idx]+" at "+str(ax)+"degree the right"
				else:
					st=CLASSES[idx]+" at "+str(ax)+"degree the left"

				t1 = threading.Thread(target=spk, args=(st,))
				t1.start()
				label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	else:
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				ax=(box[2]-box[0])/2
				ax=(box[0]+ax-200)/3
				ax=int(ax)
				if abs(ax)<5:
					st=CLASSES[idx]+" at centre"
				elif ax<0:
					ax=abs(ax)
					st=CLASSES[idx]+" at "+str(ax)+" degree the right"
				else:
					st=CLASSES[idx]+" at "+str(ax)+" degree the left"				

				# draw the prediction on the frame
				label =CLASSES[idx]+"{:.2f}% ".format(confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	# loop over the detections
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()