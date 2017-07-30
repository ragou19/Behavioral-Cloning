## Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<p align="center">
  <img width="320" height="160" src="./pictures/clip.gif">
</p>

Overview
---
In this project, we use what we've learned about deep neural networks and convolutional neural networks to clone driving behavior with the training, validation, and testing of a model using Keras. The neural network takes as input images from the cameras onboard the front of a simulated car and outputs a single steering angle to essentially create an autonomous vehicle in Unity. The goals and steps of this project are the following:
* Use a simulator to collect data of good driving behavior.
* Design, train and validate a model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle remains on the road for an entire loop around the track. In our case, we get a vehicle which can run indefinitely.
* Summarize the results with a written report.

The following files are included in the repository: 
* [Writeup Report](./writeup_report.md) with a full explanation of the project.
* [model.py](./model.py) Python script used to create and train the model.
* [drive.py](./drive.py) script to drive the car - created by Udacity staff and modified only to maximize the speed of the vehicle. Requires their simulator to run - see below in the Dependencies section.
* [model.h5](./model.h5), a trained Keras model output from the model.py file and used by the drive.py file.
* [video.mp4](./video.mp4) recording of the vehicle driving autonomously around the track using the generated model.

### Dependencies
Running the simulator (not included; see above) and associated files on your own machine requires the following:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with the CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

Note that the simulator itself is not available for public consumption without access to their [Self-Driving Car Nanodegree](http://www.udacity.com/drive) materials. Interested parties may consult Udacity's SDCND Project 3 pages directly for more information.

The simulator is necessary for gathering training images and the related driving_log.csv file containing filenames of each of the car hood cameras' images (left, center, and right cameras), along with the associated steering angle for each center image.
