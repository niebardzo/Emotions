from tqdm import tqdm
from imutils import paths
import numpy as np
import argparse
import os
import dlib
import imutils
import cv2
from imutils import face_utils
from utils.imageprocessing import Image
from utils.imageprocessing import Face
from utils.modelmanager import Model
from joblib import dump, load


ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", type=str, default="databases/KDEF/", help="Path to the directory containing the 'KDEF' dataset.")
ap.add_argument("-m", "--model", type=str, default="knn", help="Name of the ML model to be used.")
ap.add_argument("-s", "--serialized", type=str, help="Path to the file with serialized model.")
ap.add_argument("-i", "--image", type=str, help="Path to the image file to analyze")
ap.add_argument("-a", "--action", type=str,choices=["ts", "tt", "si", "d"], required=True, help="What action to pefrom (ts - train and serialize, tt - train and test,"+
	" si - use serialized to anaylze the image, d - save test and train dataset in HDF5.")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor", default="models/shape.dat")

args = vars(ap.parse_args())

emotion_labels = ['AF','AN','DI','HA','NE','SA','SU']

def change_label(label):
	if label == 'AF':
		return 'Affraid'
	if label == 'AN':
		return 'Angry'
	if label == 'DI':
		return 'Disgust'
	if label == 'HA':
		return 'Happy'
	if label == 'NE':
		return 'Neutral'
	if label == 'SA':
		return 'Sad'
	if label == 'SU':
		return 'Surprised'


if args["action"][:1] == "t":
	data = load('databases/data.joblib')
	labels = load('databases/labels.joblib')
	if args["action"] == "ts":
		if args["model"] == "":
			print("[ERROR] NO MODEL PROVIDED.")
			exit()

		model = Model(args["model"], data=data, labels=labels)
		model.split_dataset(0.01)

		print("[INFO] Training the Model...")
		model.train()

		dump(model, 'models/emotion.joblib')
		print("[INFO] Model successfully serialized under models/emotion.joblib .")
	
	elif args["action"] == "tt":
		model_names = [
				"knn",
				"naive_bayes",
				"svm",
				"decision_tree",
				"random_forest",
				"extra_tree"
			]

		for model_name in model_names:
			model = Model(model_name, data=data, labels=labels)
			model.split_dataset()
			model.train()
			print("[INFO] testing model: {}".format(model_name))
			print(model.test())


elif args["action"] == "si":
	if args["serialized"] == "" or args["image"] == "":
		print("[ERROR] NO SERIALIZED MODEL OR IMAGE PATH PROVIDED.")
		exit()
	#
	# Write this!
	#
	pass



elif args["action"] == "d":
	if args["dataset"] == "":
		print("[ERROR] NO DATASET PATH PROVIDED")
		exit()

	data = []
	labels = []

	imagePaths = paths.list_images(args["dataset"])
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	print("[INFO] extracting features from the database...")
	for imagePath in tqdm(imagePaths):
		if imagePath.split(os.path.sep)[-1][6] != "S":
			continue
		loaded_image = cv2.imread(imagePath)
		loaded_image = imutils.resize(loaded_image, width=562)
		image = Image(loaded_image, detector)
		rects = image.detect_faces()

		for rect in rects:
			face = Face(image.gray, rect, predictor)
			label = imagePath.split(os.path.sep)[-1][4:6]
			if label not in emotion_labels:
				continue
			data.append(face.extract_features())
			labels.append(change_label(label))


	dump(data, 'databases/data.joblib')
	dump(labels, 'databases/labels.joblib')