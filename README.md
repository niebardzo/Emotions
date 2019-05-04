# Emotions

## Introduction

The project was reliazied as a part of Master Thesis on AGH. The purpose of the project was to create the emotion recognition application with the usage of Machine Learning methods. To achive the goal the part of Karolinska Directed Emotional Faces dataset ([a KDEF](http://www.kdef.se/)) have been used. The dataset contains the photos of each emotions representation (only the frontal face image have been used - 980 samples; 140 samples for each emotion). To extract the facial landmark dlib library have been used with the pretrained model for facial landmark detection.
The facial landmark detector included in the dlib library is an implementation of the One Millisecond Face Alignment with an Ensemble of Regression Trees paper by Kazemi and Sullivan (2014).
The landmark can be download [a here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), also it is included in the repository under **models/shape.dat**. The method for extracting features and normalization have been presented in the Features Extraction chapter.

Features extracted with the labels are numpy objects serialized in the folder **databases**. The final emotion classification model is serialized under the **models/emotiion.joblib**.

## Requirements

All requirements are specified in Pipfile. The best way to install is using **pipenv** being in the folder with Pipfile.

```
pipenv install --dev
pipenv shell
```

Or use standard pip3:

```
pip3 install --requirements
```

To properly install dlib library please refer to the guideline prepared by Adrian from [a Pyimagesearch.com](https://www.pyimagesearch.com). 

https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/

## Usage


## Demo


## Solution Overview

### Features Extraction

### Choosing the right model

### Feature Importance/Selection

### Results

## Contact

If you would like to ask me a question, please contact me: pat049b@gmail.com.