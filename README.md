# Emotions

## Introduction

The project was reliazied as a part of Master Thesis on AGH. The purpose of the project was to create the emotion recognition application with the usage of Machine Learning methods. To achive the goal the part of Karolinska Directed Emotional Faces dataset ([KDEF](http://www.kdef.se/)) have been used. The dataset contains the photos of each emotions representation (only the frontal face image have been used - 980 samples; 140 samples for each emotion). To extract the facial landmark dlib library have been used with the pretrained model for facial landmark detection.

The facial landmark detector included in the dlib library is an implementation of the One Millisecond Face Alignment with an Ensemble of Regression Trees paper by Kazemi and Sullivan (2014).
The landmark can be download [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), also it is included in the repository under **models/shape.dat**.

The method for extracting features and normalization have been presented in the Features Extraction chapter.
Features extracted with the labels are numpy objects serialized in the folder **databases**. The final emotion classification model is serialized under the **models/emotiion.joblib**.

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

To properly install dlib and opencv libraries please refer to the guidelines prepared by Adrian Rosebrock from [Pyimagesearch.com](https://www.pyimagesearch.com). 

https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/
https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

## Usage

To recognize the emotions use **recognize.py** specifing landmark predictor path and serialized emotion classification model path. Example usage:

```
python recognize.py -p models/shape.dat -m models/emotion.joblib
```

To train different model using serialized extracted features use **analyze.py** script. Example usage:

```
python analyze.py --action ts -p models/shape.dat -m naive_bayes
```

For additional options in **analyze.py** see help:

```
python analyze.py --help
```


## Demo

Please see below the demo of final emotion prediction.

TBF

## Solution Overview

The chapter presents the solution overview. Each chapter describes the method and/or workflow how each step was implemented. Please see the diagrams describing the overall emotion detection procedure and the diagram describing the workflow of training the model.

TBF

### Features Extraction

To properly analyze the image, the face features have to be extracted. The class Face have been implemented to extract the face features. There are 4 features extracted for each eye region and 5 features extracted for mouth region which gives 13 features in total. The features are normalized with the normalizer caluculated based on the sum of the eucilidean distance between face gravity center point and center of each eye divide by 2.0.

IMAGE


### Feature Engineering

The first point is to visualize the data that we have collected, for that purpose we utilize the functionallity of yellowbrick library for ML Visualization.


![RadViz_Init](../master/static/init_data.png)

![Para_Init](../master/static/init_paraller_coordinates.png)

and zoomed image:

![RadViz_Init_Zoomed](../master/static/init_data_zoomed.png)

As we can see there is not enough spread of the data to easly distinguish between the right final class. To build the right model we should proceed with feature selection. First we evalued which features are correlated with each other by couting the pearson correlation between the features.

![Pearson](../master/static/init_pearson.png)

As we can see the EAR left and EAR right possess the same information, it is because all of image faces are symetrical. The same approach is visiable in the distance between the center of the eye and eyebrow for left and right eye. The conclusion for that is that the model will be only valid for the symetric faces. At the moment, we could get rid of half of the features representing one side of the face. But let's proceed with feature analytics. Let's check the feature corelation with dependant variable:

![Dependent_corelations](../master/static/init_corelations.png)

It is clearly visiable that we can get rid off 5 least significat features and then see the data spreading again for 8 features:

![8_features_data](../master/static/8_data.png)

![8_features_paral](../master/static/8_paraller.png)

and zoomed image:

![8_features_data_zoomed](../master/static/8_data_zoomed.png)

It looks better, additionally if we consider that angry, sad and affraid emotions as "dissatisfied" we could achive better accuracy on that emotion.


Saying that always 8 features will be selected out of 13 initial features. The selection is done in the pipeline.


### Model Selection & Hyperparameter Tunning

TBF



### Results

TBF

## Contact

If you would like to ask me a question, please contact me: niebardzo@gmail.com.

## Kudos

Kudos to Adrian Rosebrock from [Pyimagesearch.com](https://www.pyimagesearch.com) for showing the simple and effective way of extracting the face landmark and for great tutorials that he does.

## To Do

- [ ] Write a docsting for each class and each method in utils.
- [x] Implement Feature Selection, implement checking feature importance with plotting.
- [x] Implement good model choosing with metrics and plotting - analyze.py -a tt
- [ ] Finish README - record a demo, add drawings and diagrams.
