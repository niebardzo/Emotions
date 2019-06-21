from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import cv2
import math


def eye_aspect_ratio(eye):
	"""Method returns Eye Aspect Ratio."""
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0],eye[3])
	ear = (A+B)/(2.0*C)
	return ear


def mouth_aspect_ratio(mouth):
	"""Method returns Mouth Aspect Ratio."""
	A = dist.euclidean(mouth[13], mouth[19])
	B = dist.euclidean(mouth[14], mouth[18])
	C = dist.euclidean(mouth[15], mouth[17])
	F = dist.euclidean(mouth[12], mouth[16])
	mar = (A+B+C)/(3.0*F)
	return mar


class Image(object):
	"""
	A class used to represent the Image provided.
	
	Attributes:
	-----------
	image: object.
		Opencv object representing the loaded image.

	gray: object.
		Opencv object representing the loaded image in gray scale.

	detector: object
		Dlib frontal face detector object.

	"""

	def __init__(self, image, detector):
		"""Consutructor of the class that handles images."""
		self.image = imutils.resize(image, width=562)
		self.gray = self.generate_gray()
		self.detector = detector
		
	def detect_faces(self):
		"""Method that returns faces detected on the image itself."""
		rects = self.detector(self.gray, 1)
		return rects

	def generate_gray(self):
		"""Image that generate the grayscale object opencv."""
		return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)



class Face(object):
	"""
	A class used to represent the Face.

	Attributes:
	-----------
	predictor: object
		A landmark predictor used to get the face landmark.
	shape: array
		Array represents face landmark.
	face_parts: arrays
		Subarrays of shape describing face parts.
	gravity_point: array
		X,Y array represents gravity point of the face.
	normalizer: float
		Value to normalize features.
	features: array
		Array with all face features.

	"""

	def __init__(self, gray, rect, predictor):
		"""Constructor of the class describing face. Setting up all necassary attributes."""
		self.predictor = predictor
		self.shape = self.get_landmark(gray, rect)
		self.left_eye = self.extract_part("left_eye")
		self.right_eye = self.extract_part("right_eye")
		self.left_eyebrow = self.extract_part("left_eyebrow")
		self.right_eyebrow = self.extract_part("right_eyebrow")
		self.mouth = self.extract_part("mouth")

		self.gravity_point = self.calculate_face_gravity_center()
		self.normalizer = self.calculate_normalizer()

		self.features = []

	def get_landmark(self, gray, rect):
		"""Method returns face landmark."""
		shape = self.predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		return shape


	def calculate_face_gravity_center(self):
		"""Method returns the face gravity center."""
		return self.shape.mean(axis=0).astype("int")

	def extract_part(self, part):
		"""Method returns the subarray of face landmark."""
		(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS[part]
		return self.shape[Start:End]

	def calculate_normalizer(self):
		"""Method returns face normalizer."""
		left_eye_center = self.left_eye.mean(axis=0).astype("int")
		right_eye_center = self.right_eye.mean(axis=0).astype("int")
		A = dist.euclidean(left_eye_center, self.gravity_point)
		B = dist.euclidean(right_eye_center, self.gravity_point)
		return (A+B)/2.0

	def get_eyes_features(self):
		"""Method that appends to the features attribute all eyes features."""
		left_eye_center = self.left_eye.mean(axis=0).astype("int")
		
		left_1 = dist.euclidean(self.left_eyebrow[0], left_eye_center)/self.normalizer
		left_2 = dist.euclidean(self.left_eyebrow[2], left_eye_center)/self.normalizer
		left_3 = dist.euclidean(self.left_eyebrow[4], left_eye_center)/self.normalizer

		self.features.append(eye_aspect_ratio(self.left_eye))
		self.features.append(left_1)
		self.features.append(left_2)
		self.features.append(left_3)

		right_eye_center = self.right_eye.mean(axis=0).astype("int")

		right_3 = dist.euclidean(self.right_eyebrow[0], right_eye_center)/self.normalizer
		right_2 = dist.euclidean(self.right_eyebrow[2], right_eye_center)/self.normalizer
		right_1 = dist.euclidean(self.right_eyebrow[4], right_eye_center)/self.normalizer

		self.features.append(eye_aspect_ratio(self.right_eye))
		self.features.append(right_1)
		self.features.append(right_2)
		self.features.append(right_3)

	def get_mouth_features(self):
		"""Method that appends to the features attributes all mouth features."""
		self.features.append(mouth_aspect_ratio(self.mouth))

		mouth_1 = dist.euclidean(self.mouth[3], self.gravity_point)/self.normalizer
		mouth_2 = dist.euclidean(self.mouth[9], self.gravity_point)/self.normalizer
		mouth_3 = dist.euclidean(self.mouth[6], self.gravity_point)/self.normalizer
		mouth_4 = dist.euclidean(self.mouth[0], self.gravity_point)/self.normalizer

		self.features.append(mouth_1)
		self.features.append(mouth_2)
		self.features.append(mouth_3)
		self.features.append(mouth_4)


	def extract_features(self):
		"""Method returns the features of the face."""
		self.get_eyes_features()
		self.get_mouth_features()
		return np.array(self.features)
		
		

