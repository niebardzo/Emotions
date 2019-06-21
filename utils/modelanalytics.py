from .modelmanager import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from yellowbrick.features.rankd import Rank1D, Rank2D
from yellowbrick.features.radviz import RadViz
from yellowbrick.features.pcoords import ParallelCoordinates
from yellowbrick.features.importances import FeatureImportances
from yellowbrick.features.rfecv import RFECV

from yellowbrick.target import FeatureCorrelation, ClassBalance

from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ConfusionMatrix

from sklearn.model_selection import GridSearchCV

from yellowbrick.model_selection import CVScores
from yellowbrick.model_selection import ValidationCurve
from yellowbrick.model_selection import LearningCurve

from yellowbrick.style import set_palette

set_palette('paired')


class Analytics(Model):

	def __init__(self, model, data=None, labels=None):
		super().__init__(model, np.array(data), np.array(labels))


	def draw_rad_viz(self):
		visualizer = RadViz(classes=self.le.classes_, features=self.get_feature_labels(), alpha=0.4)
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.transform(self.training_data)
		visualizer.poof()

	def draw_rank_1d(self):
		visualizer = Rank1D(features=self.get_feature_labels(), algorithm='shapiro')
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.transform(self.training_data)
		visualizer.poof()


	def draw_rank_2d(self, algo='pearson'):
		visualizer = Rank2D(features=self.get_feature_labels(), algorithm=algo)
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.transform(self.training_data)
		visualizer.poof()

	def draw_parallel_coordinates(self):
		visualizer = ParallelCoordinates(classes=self.le.classes_, features=self.get_feature_labels(), sample=0.2, shuffle=True, fast=True)
		visualizer.fit_transform(self.training_data, self.training_labels)
		visualizer.poof()


	def draw_feature_importance(self):
		visualizer = FeatureImportances(self.model, labels=self.get_feature_labels())
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.poof()


	def draw_recurisive_feature_elimination(self, model):
		visualizer = RFECV(model)
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.poof()

	def draw_feature_correlation(self):
		visualizer = FeatureCorrelation(method='mutual_info-classification',
									labels=self.get_feature_labels(), sort=True)
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.poof()

	def draw_class_balance(self):
		visualizer = ClassBalance(labels=self.le.classes_)
		visualizer.fit(self.training_labels)
		visualizer.poof()

	def draw_classification_report(self):
		visualizer = ClassificationReport(self.model,classes=self.le.classes_)
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.score(self.test_data, self.test_labels)
		visualizer.poof()

	def draw_confusion_matrix(self):
		visualizer = ConfusionMatrix(self.model,classes=self.le.classes_, label_encoder=self.le)
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.score(self.test_data, self.test_labels)
		visualizer.poof()


	def draw_prediction_error(self):
		visualizer = ClassPredictionError(self.model,classes=self.le.classes_)
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.score(self.test_data, self.test_labels)
		visualizer.poof()


	def draw_validation_curve(self, param_name, param_range, cv, logx=False, scoring="f1_weighted", n_jobs=5):
		visualizer = ValidationCurve(self.model, param_name=param_name, param_range=param_range,
    					logx=logx, cv=cv, scoring=scoring, n_jobs=n_jobs)
		visualizer.fit(self.training_data, self.test_data)
		visualizer.poof()

	def draw_learning_curve(self, cv, scoring='f1_weighted', n_job=5):
		visualizer = LearningCurve(self.model, cv=cv,scoring=scoring, n_jobs=n_jobs)
		visualizer.fit(self.training_data, self.test_data)
		visualizer.poof()

	def draw_cross_validation_scores(self, cv, scoring='f1_weighted'):
		visualizer = CVScores(model=self.model, cv=cv, scoring=scoring)
		visualizer.fit(self.training_data, self.training_labels)
		visualizer.poof()

	def feature_selection(self, method, scoring, number):
		
		self.scoring_functions = {
			"f_classif": f_classif,
			"mutual_info_classif": mutual_info_classif,
			"chi2": chi2
		}

		self.selection_methods = {
			"select_k_best": SelectKBest(self.scoring_functions[scoring], k=number),
			"select_percentile": SelectPercentile(self.scoring_functions[scoring], percentile=number)
		}

		
		selector = self.selection_methods[method]
		self.training_data = selector.fit_transform(self.training_data, self.training_labels)
		self.feature_mask = selector.get_support()

	def print_cross_val_score(self,cv, scoring='accuracy'):
		scores = cross_val_score(self.model, self.training_data, self.training_labels,cv=cv,scoring=scoring)
		print("Accuracy: {} +/- {}".format(scores.mean(), scores.std()))



class EstimatorSelectionHelper(Model):

	def __init__(self, model, params, data=None, labels=None):
		super().__init__(model, np.array(data), np.array(labels))
		#if not set(self.models.keys()).issubset(set(params.keys())):
		#	missing_params = list(set(self.models.keys()) - set(params.keys()))
		#	raise ValueError("Some estimators are missing parameters: %s" % missing_params)
		self.params = params
		self.keys = ["knn", "naive_bayes","svm"]
		#self.keys = models.keys()
		self.grid_searches = {}

	def fit(self, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
		for key in self.keys:
			print("Running GridSearchCV for %s." % key)
			model = self.models[key]
			params = self.params[key]
			gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
								verbose=verbose, scoring=scoring, refit=refit,
								return_train_score=True, iid=False)
			gs.fit(self.training_data,self.training_labels)
			self.grid_searches[key] = gs

	def score_summary(self, sort_by='mean_score'):
		def row(key, scores, params):
			d = {
					'estimator': key,
					'min_score': min(scores),
					'max_score': max(scores),
					'mean_score': np.mean(scores),
					'std_score': np.std(scores),
				}
			return pd.Series({**params,**d})

		rows = []
		for k in self.grid_searches:
			params = self.grid_searches[k].cv_results_['params']
			scores = []
			for i in range(self.grid_searches[k].cv):
				key = "split{}_test_score".format(i)
				r = self.grid_searches[k].cv_results_[key]        
				scores.append(r.reshape(len(params),1))

			all_scores = np.hstack(scores)
			for p, s in zip(params,all_scores):
				rows.append((row(k, s, p)))

		df = pd.concat(rows, axis=1, sort=True).T.sort_values([sort_by], ascending=False)

		columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
		columns = columns + [c for c in df.columns if c not in columns]

		return df[columns]