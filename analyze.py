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
from utils.modelanalytics import Analytics, EstimatorSelectionHelper
from joblib import dump, load

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", type=str, default="databases/KDEF/", help="Path to the directory containing the 'KDEF' dataset.")
ap.add_argument("-m", "--model", type=str, default="knn", help="Name of the ML model to be used.")
ap.add_argument("-s", "--serialized", type=str, help="Path to the file with serialized model.")
ap.add_argument("-i", "--image", type=str, help="Path to the image file to analyze")
ap.add_argument("-a", "--action", type=str,choices=["ts","tsv", "tt", "si", "d"], required=True, help="What action to pefrom (ts - train and serialize, tt - train and test,"+
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
	if args["action"][:2] == "ts":
		if args["model"] == "":
			print("[ERROR] NO MODEL PROVIDED.")
			exit()

		model = Model(args["model"], data=data, labels=labels)
		if args["action"] == "tsv":
			model.use_voting_classifier()
		model.split_dataset(0.01)

		print("[INFO] Training the Model...")
		model.train()

		dump(model, 'models/emotion.joblib')
		print("[INFO] Model successfully serialized under models/emotion.joblib .")
	
	elif args["action"] == "tt":

		#analytics = Analytics("naive_bayes", data=data, labels=labels)
		#analytics.feature_selection("select_k_best", "f_classif", 8)
		#analytics.draw_cross_validation_scores(cv=StratifiedKFold(5))
		
		ms = [
			"knn",
			"naive_bayes",
			"svm",
			"decision_tree",
			"random_forest",
			"extra_tree",
			"gradient_boost",
			"mlp"
		]

		
		def print_validation_curve(data,labels):
			analytics = Analytics("knn", data=data, labels=labels)
			analytics.feature_selection("select_k_best", "f_classif", 8)
			analytics.draw_validation_curve(param_name="n_neighbors", param_range=np.arange(2,20,2), cv=StratifiedKFold(5))

			analytics = Analytics("svm", data=data, labels=labels)
			analytics.feature_selection("select_k_best", "f_classif", 8)
			analytics.draw_validation_curve(param_name="C", param_range=np.arange(0.1,50,2), cv=StratifiedKFold(5))


			analytics = Analytics("decision_tree", data=data, labels=labels)
			analytics.feature_selection("select_k_best", "f_classif", 8)
			analytics.draw_validation_curve(param_name="max_depth", param_range=np.arange(20,300,10), cv=StratifiedKFold(5))


			analytics = Analytics("random_forest", data=data, labels=labels)
			analytics.feature_selection("select_k_best", "f_classif", 8)
			analytics.draw_validation_curve(param_name="n_estimators", param_range=np.arange(20, 200,10), cv=StratifiedKFold(5))

			analytics = Analytics("extra_tree", data=data, labels=labels)
			analytics.feature_selection("select_k_best", "f_classif", 8)
			analytics.draw_validation_curve(param_name="n_estimators", param_range=np.arange(20, 200,10), cv=StratifiedKFold(5))

			analytics = Analytics("gradient_boost", data=data, labels=labels)
			analytics.feature_selection("select_k_best", "f_classif", 8)
			analytics.draw_validation_curve(param_name="n_estimators", param_range=np.arange(5,60,2), cv=StratifiedKFold(5))
			analytics.draw_validation_curve(param_name="learning_rate", param_range=np.arange(0.03,0.2,0.01), cv=StratifiedKFold(5))

			analytics = Analytics("mlp", data=data, labels=labels)
			analytics.feature_selection("select_k_best", "f_classif", 8)
			analytics.draw_validation_curve(param_name="hidden_layer_sizes", param_range=np.array([i for i in range(6,20,1)]), cv=StratifiedKFold(5))


		#print_validation_curve(data,labels)

		#for m in ms:
		#	analytics = Analytics(m, data=data, labels=labels)
		#	analytics.feature_selection("select_k_best", "f_classif", 8)
		##	analytics.print_cross_val_score(cv=StratifiedKFold(5))
		#	analytics.draw_learning_curve(cv=StratifiedKFold(5))
		#	analytics.draw_cross_validation_scores(cv=StratifiedKFold(5))



		params = {
		"knn": {"n_neighbors": np.arange(1,20,1), "weights": ["uniform", "distance"],
		"algorithm" : ["auto", "ball_tree", "kd_tree", "brute"]},
		"naive_bayes": {},
		"svm": {"kernel":["linear", "rbf"], "C": np.arange(0.1,50,0.5),
		"gamma": ['auto', 'scale']},
		"decision_tree": {"criterion": ["gini", "entropy"], "splitter": ["best", "random"], "max_depth": np.arange(5,300,1)},
		"random_forest": {"n_estimators": np.arange(20,300,3),"criterion": ["gini", "entropy"]},
		"extra_tree": {"n_estimators": np.arange(20,300,3),"criterion": ["gini", "entropy"]},
		"gradient_boost": {"n_estimators": np.arange(5,60,2), "learning_rate": np.arange(0.03,0.2,0.01)},
		"mlp": [{"hidden_layer_sizes": [ (i, ) for i in np.arange(6,20,1)], "alpha": np.arange(5e-06,5e-05,5e-06), "solver": ["lbfgs"]},
		{"hidden_layer_sizes": [ (i, j, ) for i in np.arange(4,18,1) for j in np.arange(8,20,1)],"alpha": np.arange(5e-06,5e-05,5e-06), "solver": ["lbfgs"]}
		]
		}

		#selection = EstimatorSelectionHelper("knn", params, data=data, labels=labels)
		#selection.fit(scoring="accuracy", n_jobs=12, cv=5)
		#print(selection.score_summary(sort_by='max_score'))



		model = Model("naive_bayes", data=data, labels=labels)
		model.use_voting_classifier()
		model.split_dataset(0.30)
		model.univariate_feature_selection("select_k_best", "mutual_info_classif", 8)
		model.train()
		print("[INFO] testing model: {}".format("voting"))
		print(model.test())


		analytics = Analytics("knn", data=data, labels=labels)
		analytics.use_voting_classifier()
		analytics.feature_selection("select_k_best", "f_classif", 8)
		analytics.print_cross_val_score(cv=StratifiedKFold(5))
		analytics.draw_learning_curve(cv=StratifiedKFold(5))
		analytics.draw_cross_validation_scores(cv=StratifiedKFold(5))


elif args["action"] == "si":
	if args["serialized"] == "" or args["image"] == "":
		print("[ERROR] NO SERIALIZED MODEL OR IMAGE PATH PROVIDED.")
		exit()

	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
	print("[INFO] detecting emoton on image...")

	model = load(args['serialized'])
	load_image = cv2.imread(args["image"])
	load_image = imutils.resize(load_image, width=562)
	image = Image(load_image, detector)
	rects = image.detect_faces()
	for rect in rects:
		face = Face(image.gray, rect, predictor)
		prediction = model.predict([face.extract_features()])
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(load_image, (x,y), (x+w, y+h), (0,255,0),2)
		cv2.putText(load_image, "###{}".format(model.le.inverse_transform(prediction)[0]), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
	cv2.imshow("Frame", load_image)
	print("[INFO] press Enter to exit.")
	cv2.waitKey(0)


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