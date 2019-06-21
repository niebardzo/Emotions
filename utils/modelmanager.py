from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE



class Model(object):
	"""
	Class that representats the model.

	Attributes:
	le: object
		Object of Label Encoder
	model: object
		Object of sklearn model of choice.
	training_data: array
		Array of data.
	training_labels: array
		Encoded labels for training data.
	test_data: array
		Array of test data. Initially empty.
	test_labels: array
		Array of test labels. Initially empty.

	"""

	def __init__(self, model, data=None, labels=None):
		"""Constructor for model class."""
		if data is None or labels is None:
			raise AttributeError("No Data in a constructor provided.")


		self.models = {
			"knn": KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree", weights="distance"),
			"naive_bayes": GaussianNB(),
			"svm": SVC(C=10, gamma="scale", kernel="rbf"),
			"decision_tree": DecisionTreeClassifier(criterion="gini", max_depth=246, splitter="random"),
			"random_forest": RandomForestClassifier(n_estimators=158, criterion="gini"),
			"extra_tree": ExtraTreesClassifier(n_estimators=41, criterion="entropy"),
			"gradient_boost": GradientBoostingClassifier(n_estimators=23, learning_rate=0.11),
			"mlp":  MLPClassifier(solver="lbfgs", hidden_layer_sizes=(16,), alpha=1.5E-05)

		}

		self.le = LabelEncoder()
		self.model = self.models[model]

		self.training_data = data
		self.training_labels = self.le.fit_transform(labels)
		self.feature_names = ['EARL','L1','L2','L3', 'EARR', 'R1', 'R2', 'R3', 'MAR', 'M1', 'M2', 'M3', 'M4']
		self.feature_mask = [True,True,True,True,True,True,True,True,True,True,True,True,True]


	def use_voting_classifier(self):
		self.model = VotingClassifier(estimators=[('nb', self.models["naive_bayes"]), ('et', self.models["extra_tree"]), ('mlp', self.models["mlp"])], voting='soft', weights=[1.5,3,2.5])

	def split_dataset(self, test_size=0.20):
		"""Method for spliting dataset to the training and test."""
		(self.training_data, self.test_data, self.training_labels, self.test_labels) = train_test_split(self.training_data, self.training_labels, test_size=test_size)

	def train(self):
		"""Method for training a model with the training dataset."""
		self.model.fit(self.training_data, self.training_labels)

	def test(self):
		"""Method returns the classification report."""
		return classification_report(self.test_labels, self.predict(self.test_data), target_names=self.le.classes_)

	def predict(self, to_predict):
		"""Method returns the prefiction for new data."""
		return self.model.predict(to_predict)

	def univariate_feature_selection(self, method, scoring, number):
		
		self.scoring_functions = {
			"f_classif": f_classif,
			"mutual_info_classif": mutual_info_classif,
			"chi2": chi2
		}

		self.selection_methods = {
			"select_k_best": SelectKBest(self.scoring_functions[scoring], k=number),
			"select_percentile": SelectPercentile(self.scoring_functions[scoring], percentile=number)
		}

		
		self.model = Pipeline([
			('feature_selection', self.selection_methods[method]),
			('classification', self.model)
			])



	def recursive_feature_elimination(self):
		svc = SVC(kernel="linear")
		self.model = Pipeline([
			('feature_selection', RFE(estimator=svc, n_features_to_select=8, step=10)),
			('classification', self.model)
			])

	def get_feature_labels(self):
		feature_labels = []
		for feature, i in zip(self.feature_names,self.feature_mask):
			if i == True:
				feature_labels.append(feature)
		return feature_labels
		