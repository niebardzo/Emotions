from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVR

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel



class Model(object):

	def __init__(self, model, data=None, labels=None):
		if data is None or labels is None:
			raise AttributeError("No Data in a constructor provided.")


		self.models = {
			"knn": KNeighborsClassifier(n_neighbors=4),
			"naive_bayes": GaussianNB(),
			"svm": SVC(kernel="linear"),
			"decision_tree": DecisionTreeClassifier(),
			"random_forest": RandomForestClassifier(n_estimators=20),
		}

		self.le = LabelEncoder()
		self.model = self.models[model]

		self.training_data = data
		self.training_labels = self.le.fit_transform(labels)
		self.test_data = []
		self.test_labels = []


	def split_dataset(self, test_size=0.20):
		(self.training_data, self.test_data, self.training_labels, self.test_labels) = train_test_split(self.training_data, self.training_labels, test_size=test_size)

	def train(self):
		self.model.fit(self.training_data, self.training_labels)

	def test(self):
		return classification_report(self.test_labels, self.predict(self.test_data), target_names=self.le.classes_)

	def predict(self, to_predict):
		return self.model.predict(to_predict)

	def univariate_feature_selection(self, method, scoring):
		
		self.scoring_functions = {
			"f_classif": f_classif,
			"mutual_info_classif": mutual_info_classif
		}

		self.selection_methods = {
			"select_k_best": SelectKBest(self.scoring_functions[scoring], k=40),
			"select_percentile": SelectPercentile(self.scoring_functions[scoring], percentile=10)
		}

		
		self.model = Pipeline([
			('feature_selection', self.selection_methods[method]),
			('classification', self.model)
			])

	def recursive_feature_elimination(self):
		svc = SVC(kernel="linear")
		self.model = Pipeline([
			('feature_selection', RFE(estimator=svc, n_features_to_select=30, step=50)),
			('classification', self.model)
			])


	def select_from_model(self):
		pass

