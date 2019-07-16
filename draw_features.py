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

ap.add_argument("-o", "--output", default='output.avi',help="Output file for video.")

args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None
(h, w) = (None, None)

print("[INFO] loading emotions model...")
model = load(args['model'])

buff = []

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=562)
	image = Image(frame, detector)
	rects = image.detect_faces()
	if writer is None:
		(h_f, w_f) = frame.shape[:2]
		writer = cv2.VideoWriter(args["output"], fourcc, 10,
				(w_f, h_f), True)

	for (i, rect) in enumerate(rects):
		face = Face(image.gray, rect, predictor)

		prediction = model.predict([face.extract_features()])
		buff.insert(0 ,prediction[0])

		if len(buff) >= 5:
			buff.pop()
			prediction = [most_common(buff)]

		def draw_brews_f():
			cv2.line(frame, tuple(face.left_eyebrow[0]), tuple(face.left_eye.mean(axis=0).astype("int")), (0,255,0), 2)
			cv2.line(frame, tuple(face.left_eyebrow[2]), tuple(face.left_eye.mean(axis=0).astype("int")), (0,255,0), 2)
			cv2.line(frame, tuple(face.left_eyebrow[4]), tuple(face.left_eye.mean(axis=0).astype("int")), (0,255,0), 2)

			cv2.line(frame, tuple(face.right_eyebrow[0]), tuple(face.right_eye.mean(axis=0).astype("int")), (0,255,0), 2)
			cv2.line(frame, tuple(face.right_eyebrow[2]), tuple(face.right_eye.mean(axis=0).astype("int")), (0,255,0), 2)
			cv2.line(frame, tuple(face.right_eyebrow[4]), tuple(face.right_eye.mean(axis=0).astype("int")), (0,255,0), 2)
		
		def draw_ear():
			cv2.line(frame, tuple(face.right_eye[1]), tuple(face.right_eye[5]), (0,255,0), 2)
			cv2.line(frame, tuple(face.right_eye[2]), tuple(face.right_eye[4]), (0,255,0), 2)
			cv2.line(frame, tuple(face.right_eye[0]), tuple(face.right_eye[3]), (0,255,0), 2)
			cv2.line(frame, tuple(face.left_eye[1]), tuple(face.left_eye[5]), (0,255,0), 2)
			cv2.line(frame, tuple(face.left_eye[2]), tuple(face.left_eye[4]), (0,255,0), 2)
			cv2.line(frame, tuple(face.left_eye[0]), tuple(face.left_eye[3]), (0,255,0), 2)

		def draw_mouth():
			cv2.line(frame, tuple(face.mouth[3]), tuple(face.gravity_point), (0,255,0), 2)
			cv2.line(frame, tuple(face.mouth[9]), tuple(face.gravity_point), (0,255,0), 2)
			cv2.line(frame, tuple(face.mouth[6]), tuple(face.gravity_point), (0,255,0), 2)
			cv2.line(frame, tuple(face.mouth[0]), tuple(face.gravity_point), (0,255,0), 2)


		def draw_mar():
			cv2.line(frame, tuple(face.mouth[13]), tuple(face.mouth[19]), (0,255,0), 2)
			cv2.line(frame, tuple(face.mouth[14]), tuple(face.mouth[18]), (0,255,0), 2)
			cv2.line(frame, tuple(face.mouth[15]), tuple(face.mouth[17]), (0,255,0), 2)
			cv2.line(frame, tuple(face.mouth[12]), tuple(face.mouth[16]), (0,255,0), 2)

		def draw_norm():
			cv2.line(frame, tuple(face.left_eye.mean(axis=0).astype("int")), tuple(face.gravity_point), (0,255,0), 2)
			cv2.line(frame, tuple(face.right_eye.mean(axis=0).astype("int")), tuple(face.gravity_point), (0,255,0), 2)

		draw_norm()
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

		#cv2.putText(frame, "###{}".format(model.le.inverse_transform(prediction)[0]), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


	cv2.imshow("Frame", frame)
	writer.write(frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
writer.release()