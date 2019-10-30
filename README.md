# Emotions

## Introduction

The project was realized as a part of a Master Thesis on AGH. The purpose of the project was to create the emotion recognition application with the usage of Machine Learning methods. To achieve the goal the part of Karolinska Directed Emotional Faces dataset ([KDEF](http://www.kdef.se/)) have been used. The dataset contains the photos of each emotions representation (only the frontal face image have been used - 980 samples; 140 samples for each emotion). To extract the facial landmark dlib library have been used with the pre-trained model for facial landmark detection.

The facial landmark detector included in the dlib library is an implementation of the One Millisecond Face Alignment with an Ensemble of Regression Trees paper by Kazemi and Sullivan (2014).
The landmark can be download [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), also it is included in the repository under **models/shape.dat**.

The method for extracting features and normalization have been presented in the Features Extraction chapter.
Features extracted with the labels are numpy objects serialized in the folder **databases**. The final emotion classification model is serialized under the **models/emotion.joblib**.

## Requirements

The software was tested on Ubuntu 18.04.* LTS with Python 3.6.* installed.



All other requirements are specified in Pipfile or requirements.txt. The best way to install is using **pipenv** being in the folder with Pipfile.

```
pipenv install --dev
pipenv shell
```

Or use standard pip3:

```
pip3 install --requirements
```

To properly install dlib and OpenCV libraries please refer to the guidelines prepared by Adrian Rosebrock from [Pyimagesearch.com](https://www.pyimagesearch.com). 

https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/
https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

## Usage

To recognize the emotions use **recognize.py**  landmark predictor path and serialized emotion classification model path. Example usage:

```
python recognize.py -p models/shape.dat -m models/emotion.joblib
```

which results in the following demo.

![RadViz_Init](../master/static/Demo.gif)


To train the different model using serialized extracted features use **analyze.py** script. Example usage:

```
python analyze.py --action ts -p models/shape.dat -m naive_bayes
```

To train voting classifier model yourself use:
```
python analyze.py --action tsv -p models/shape.dat
```


For additional options in **analyze.py** see help:

```
python analyze.py --help
```

**Emotion Matrix**:
![Emotion Matrix](../master/static/emotion_matrix.png)


## Solution Overview

The chapter presents the solution overview. Each chapter describes the method and/or workflow of how each step was implemented. Please see the diagrams describing the overall emotion detection procedure and the diagram describing the workflow of training the model.

![ARCH_DIAGRAM](../master/static/arch_diagram.png)

### Features Extraction

To properly analyze the image, the face features have to be extracted. The class Face has been implemented to extract the face features. There are 3 features extracted for each eyebrows region, 1 feature for each eye region and 5 features extracted for the mouth region which gives 13 features in total.

The features describing the eyebrows have been presented on the image below:


![Eyebrows](../master/static/eyebrows_f.png)

Based on the work by Soukupová and Čech in their 2016 paper, Real-Time Eye Blink Detection using Facial Landmarks, the features for eyes have been define as EAR:

![EAR](../master/static/EAR_f.jpg)

Similar feature have been used to describe the mouth, called MAR described in the A. Singh, C. Chandewar, P. Pattarkine, Driver Drowsiness Alert System with Effective Feature Extraction, International Journal for Research in Emerging Science and Technology Volume-5 Issue-4, 2018.

![MAR](../master/static/MAR_f.png)

Additionally, for the mouth region there have been addtional features extracted:
![mouth](../master/static/mouth_f.png)

Features that are not represented by ratio are normalized by the following normalizer:

![norm](../master/static/norm_f.png)

The features have the following labels:

```
self.feature_names = ['EARL','L1','L2','L3', 'EARR', 'R1', 'R2', 'R3', 'MAR', 'M1', 'M2', 'M3', 'M4']
```


The features are normalized with the normalizer calculated based on the sum of the euclidean distance between face gravity center point and center of each eye divide by 2.0.



### Feature Engineering

The first point is to visualize the data that we have collected, for that purpose we utilize the functionality of the yellowbrick library for ML Visualization.


![RadViz_Init](../master/static/init_data.png)

![Para_Init](../master/static/init_paraller_coordinates.png)

and zoomed image:

![RadViz_Init_Zoomed](../master/static/init_data_zoomed.png)

As we can see there is not enough spread of the data to easily distinguish between the right final class. To build the right model we should proceed with feature selection. First, we evaluated which features are correlated with each other by counting the Pearson correlation between the features.

![Pearson](../master/static/init_pearson.png)

As we can see the EAR left and EAR right possesses the same information, it is because all of the image faces are symmetrical. The same approach is visible in the distance between the center of the eye and eyebrow for left and right eye. The conclusion for that is that the model will be only valid for the symmetric faces. At the moment, we could get rid of half of the features representing one side of the face. But let's proceed with feature analytics. Let's check the feature correlation with the dependent variable:

![Dependent_corelations](../master/static/init_corelations.png)

It is visible that we can get rid off 5 least significant features and then see the data spreading again for 8 features:

![8_features_data](../master/static/8_data.png)

![8_features_paral](../master/static/8_paraller.png)

and zoomed image:

![8_features_data_zoomed](../master/static/8_data_zoomed.png)

It looks better, additionally if we consider that angry, sad and afraid emotions as "dissatisfied" we could achieve better accuracy on that emotion.


Saying that always 8 features will be selected out of 13 initial features. The selection is done in the pipeline.


### Model Selection & Hyperparameter Tunning

Most of the models available in SKLearn Library for classification have been tested. A detailed list can be found below:
```
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
```

For each classification, the most significant hyperparameter has been tunned using **GridSearchCV()** class from sklearn library. To make it easier the separate class have been made which inherits from Model class and tune hyperparameters for every class in Model.classes, with the input of parameters.

```
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

```

The results are in the file under **static/results.csv**.


For better accuracy and f1 score, finally the voting classifier has been used with the following partial classifiers and weights:

```
    def use_voting_classifier(self):
        """Method for changing to VotingClassifier."""
        self.model = VotingClassifier(estimators=[('naive_bayes', self.models["naive_bayes"]), ('et', self.models["extra_tree"]), ('gb', self.models["gradient_boost"])], voting='hard', weights=[2,3,1.5])


```

Below there is Cross-Validation(score **accuracy**) and Learning Curve for Voting Classifier.

![CV_Voting](../master/static/Cross_V_Voting.png)

![LC_Voting](../master/static/Learning_Curve_Voting.png)

The learning curve of the voting classifier is pretty far from the training score curve, which means there can be easily more training data applied to the model.

### Results

In this chapter there are results of classification with Voting Classifier defined previously. The data have been split into 20% of test data and 80% of training data. Below you can see the data distribution (previously the data was distributed equally through all classes):

![Class_balance](../master/static/Class_balance_after_splitting.png)

Classification report below shows the final results of the classification.

![Voting_Classification_report](../master/static/Voting_Classification_report.png)

and the prediction error chart for each class.

![Voting_pred_error](../master/static/Voting_class_prediction_error.png)

The final confusions matrix presents the results of emotion recoginition for each emotion.

![RadViz_Init](../master/static/Confusion_Matrix.png)

## Contact

If you would like to ask me a question, please contact me: pat049b@gmail.com.

## Kudos

Kudos to Adrian Rosebrock from [Pyimagesearch.com](https://www.pyimagesearch.com) for showing the simple and effective way of extracting the face landmark and for great tutorials that he does.

## To Do

- [x] Write a docstring for each class and each method in utils.
- [x] Implement Feature Selection, implement checking feature importance with plotting.
- [x] Implement good model choosing with metrics and plotting - analyze.py -a tt
- [x] Finish README - record a demo, add drawings and diagrams.
