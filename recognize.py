from imutils.video import VideoStream
from threading import Thread
import pygame
import argparse
import time
import cv2
import dlib
from utils.imageprocessing import Image
from utils.imageprocessing import Face
from utils.modelmanager import Model

from imutils import face_utils
import numpy as np
import imutils

from joblib import dump, load


def most_common(lst):
	return max(set(lst), key=lst.count)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

ap.add_argument("-m", "--model", required=True, default='models/emotion.joblib',help="path to serialized emotion model.")

args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

print("[INFO] loading emotions model...")
model = load(args['model'])

buff = []

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=562)
	image = Image(frame, detector)
	rects = image.detect_faces()

	for (i, rect) in enumerate(rects):
		face = Face(image.gray, rect, predictor)

		prediction = model.predict([face.extract_features()])
		buff.insert(0 ,prediction[0])

		if len(buff) >= 10:
			buff.pop()
			prediction = [most_common(buff)]


		#print(model.le.inverse_transform(prediction))

		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

		cv2.putText(frame, "###{}".format(model.le.inverse_transform(prediction)[0]), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()