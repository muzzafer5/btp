from pykinect import nui
from scipy.misc import toimage
import numpy
import cv2

video = numpy.empty((480,640,4),numpy.uint8)

def video_handler_function(frame):
    video = numpy.empty((480,640,4),numpy.uint8)
    frame.image.copy_bits(video.ctypes.data)
    cv2.imshow('KINECT Video Stream', video)

kinect = nui.Runtime()
kinect.video_frame_ready += video_handler_function
kinect.video_stream.open(nui.ImageStreamType.Video, 2,nui.ImageResolution.Resolution640x480,nui.ImageType.Color)

#cv2.namedWindow('KINECT Video Stream', cv2.WINDOW_AUTOSIZE)

while True:
	toimage(video).show()
	cv2.imshow('camera',img)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

kinect.close()
cv2.destroyAllWindows()