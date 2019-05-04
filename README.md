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

TBF

### Choosing the right model

TBF

### Feature Importance/Selection

TBF

### Results

TBF

## Contact

If you would like to ask me a question, please contact me: pat049b@gmail.com.

## To Do

- [ ] Write a docsting for each class and each method in utils.
- [ ] Implement Feature Selection, implement checking feature importance with plotting.
- [ ] Implement good model choosing with metrics and plotting - analyze.py -a tt
- [ ] Finish README - record a demo, add drawings and diagrams.