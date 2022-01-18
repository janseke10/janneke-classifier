# Pose Detector + Classifier Android Application

## Introduction

This ML Kit Quickstart app demonstrates how to use and integrate various vision based ML Kit features into your app.

## Feature List
Features that are included in this Quickstart app:
* [Pose Detection](https://developers.google.com/ml-kit/vision/pose-detection/android) - Detect the position of the human body in real time.


<img src="../screenshots/quickstart-pose-detection.png" width="220"/>

## Getting Started

* Clone this repository
* Open the project using android studio
* Run the application on a connected android device, or on an AVD


## Some extra explanation
In the output of the application can be seen when the test data is run by the classification model at the start of the application, 
what it should be, and what was actually found to have the highest probability.
When the running of the test data is done, the camera starts and you can actually detect poses and read on the screen what the classifier thinks it is!


Most of the additions I made to the application can be found in java/posedetector/classification/DeepLearningClassification.java

