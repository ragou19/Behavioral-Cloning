# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

In this project, we use what we've learned about deep neural networks and convolutional neural networks to clone driving behavior with the training, validation, and testing of a model using Keras. The neural network takes as input images from the cameras onboard the front of a simulated car and outputs a single steering angle to essentially create an autonomous vehicle in Unity. The goals and steps of this project are the following:
* Use a simulator to collect data of good driving behavior.
* Design, train and validate a model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle remains on the road for an entire loop around the track. In our case, we get a vehicle which can run indefinitely.
* Summarize the results with a written report.


[//]: # (Image References)

[image0]: ./pictures/center_2017_07_21_07_34_54_931.jpg "Video Preview"
[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./pictures/trig.png "Trigonometric Derivation"
[image3]: ./pictures/center_2017_07_21_07_34_54_931.jpg "Center Camera Image"
[image4]: ./pictures/left_2017_07_21_07_34_54_931.jpg "Left Camera Image"
[image5]: ./pictures/right_2017_07_21_07_34_54_931.jpg "Right Camera Image"

## Rubric Points
### Here I consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) for this project individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode.

My project includes the following files:
* [model.py](http://) script which creates and trains the model.
* [drive.py](http://) file for driving the car in autonomous mode.
* [model.h5](http://), a trained convolution neural network in Keras's HDF5 format.
* [video.mp4](http://) recording showing the car being driven autonomously by the generated model.
* [README](http://) introducing readers to the project.
* A writeup report (this document) summarizing the results.

Note that the simulator itself is not included. Please see the [Readme](http://) for more details.

#### 2. Submission includes functional code
Using the Udacity-provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Starting up the simulator once the script has been fully loaded, we are able to drive the car indefinitely as shown for one lap around the track in the following video:

[![alt text][image0]](./video.mp4 "Autonomous Driving Video")

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with five convolutional layers, each employing either a 5x5 or 3x3 filter, and an associated ReLU. It is very similar in concept to and inspired by the deep learning network posted on NVIDIA's blog by Bojarski, Firner, and others:

[NVIDIA: End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

![alt text][image1]

The five ReLU layers introduce nonlinearity to the model and are included in the conv2d function in line 54 of the model.py file:
```sh
def conv2d(f, kw, kh, strides = (1,1), relu=True):
	model.add(Convolution2D(f, kw, kh, subsample = strides))
	if relu == True:
		model.add(Activation('relu'))
``` 

The data is further normalized and cropped for the model using a Keras lambda layer and cropping layer (code lines 61 and 62):

```sh
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((64, 24), (0,0))))
```

Normalization entailed taking pixel values between 0 and 255, and modifying them to range between -0.5 and 0.5. The cropping layer chops off the top 64 pixels and bottom 24 pixels of each image, which displayed unnecessary foliage or scenery and the front part of the hood of the car. Naturally, these portions of the image did nothing to reliably improve our model's performance on the track (and arguably diminished it), and so were excluded.

#### 2. Attempts to reduce overfitting in the model

Curiously enough, the model initially suffered from dramatically decreased accuracy on account of its dropout layers being present alongside each convolutional layer. Once removed, not only had mean-squared-error decreased and tolerable performance been achieved, but also overfitting became no real concern. This is perhaps due to the robustness of the NVIDIA model in general and to the presence of sufficient data after flipping and accounting for the different cameras' images. (see #4 below)

As explained further below, the images were flipped along the vertical midline axis in order to account for directionality along the track. Also, a trigonometric model was devised to account for the difference of steering angle between the different cameras' images, making for a far better self-correcting model when tested on the track that actively recovers to the center of the road and avoids lane boundaries in all instances:

```sh
sign = (-1) ^ (-i  + 1)
	angle = np.deg2rad(-25 * measurement)
	new_angle = np.arctan(1/4 + (sign) * np.sign(measurement) \
				* (np.tan(angle))) 
	measurement = (0.9825 * measurement) \
				+ (0.0175 * (np.rad2deg(-new_angle)) / 25)
```

For more information on the trigonometric solution, see the "Model Architecture and Training Strategy" section below. Incorporating the left and right camera angles through these equations, as well as flipping each resultant image and driving around the track numerous times in different ways, made for a large bank of image data that prevented the model from overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually originally (model.py, line 74).

However, upon insertion of the devised trigonometric equations (lines 36-42), one new hyperparameters emerged: one related to the distance from the car to the point of observation for judging how much to turn, and two others related to the relative proportion the steering angle resulting from these equations were given weight in comparison to the original measurement.

Tuning these new hyperparameters eventually caused a fairly robust and accurate model to emerge overall.

#### 4. Appropriate training data

The small bank of training data was augmented with two additional laps around the track in both the clockwise and counterclockwise directions, then evened out further by flipping each image. The car was driven manually as close to the center line as possible, especially during curves. No attempts were made to record data with the car recovering from the side of the road exclusively; instead, the left and right camera images were incorporated with improved measurement estimates.

For details on how I modified the left and right camera steering angle measurements, please see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to enlarge the training set as much as possible while making good use of an existing architecture well-suited for the purpose.

First, the NVIDIA architecture above was replicated and flipped versions of the provided images incorporated. This convolutional neural network model was chosen because it was purposefully built by a reputable authority in order to help drive self-driving cars, which is this project's overall objective. Although weights for the network were not known beforehand, the order and shape of most of the provided layers assisted with hastening the construction of a reliable model. 

Two laps were driven around the track in each direction with the provided test images flipped before initial testing of the NVIDIA model. With the testing yielding wildly inaccurate driving, this led me to alter the architecture's content in a few ways. 

First, the sizes of the flatten layers were lessened to allow for the model to run completely on my machine without batching (I was adamant about being able to run the complete set of data at once :) ). Second, the dropout layers were altogether eliminated due to their effect on the overall loss. Third, a pooling layer was added to reduce overall mean squared error on the measurement by relating nearby weights in the first layer (no more pooling was added due to insufficient dimensions to work with).

My second overall strategy was to incorporate the left and right camera images, which led to a threefold increase in the data available. Doing so successfully required deriving the relationship between the original measurement of the steering angle, which existed solely for the center camera in the driving_log.csv file, and tuning the resulting hyperparameters which emerged from the derivation.

To see this in depth, observe the following diagram elucidating the relationships between the left and right camera steering angles, and the central camera' steering angle:

![alt text][image2]

The letters and items in the diagram signify the following:
* A = left camera
* B = center camera
* C = right camera
* P = point of reaction, or the point observed on the road in front of the car which determines the resulting center camera's steering angle (equivalent to the car's steering angle)
* R = "radius" of the car, equal to half the distance between the left and right cameras and thus similar to half the car's width
* d = direct distance from the center camera to the point of reaction
* \alpha = steering angle to be calculated for the left camera. Each left camera image will take this angle and feed it into the model as if it were the center camera, thereby improving reactivity when the car approaches either side of the road too much
* \beta = steering angle provided for the center camera and thus the car itself
* \gamma = steering angle to be calculated for the right camera

Negating intricacies related to rigid body mechanics or integral relationships between differential amounts of steering and vehicle velocity, we can take this relatively simple model and use it to predict what the steering angles for the left and right cameras should be had their corresponding images been located at the center of the vehicle.

All told, we end up with the following equations:

tan(\alpha) = (R/dcos(\beta)) - tan(\beta)
(P left of center)

tan(\alpha) = (R/dcos(\beta)) + tan(\beta)
(P right of center)

tan(\gamma) = (R/dcos(\beta)) + tan(\beta)
(P left of center)

tan(\gamma) = (R/dcos(\beta)) - tan(\beta)
(P right of center)

All four equations were incorporated as shown below by relating the camera's orientation (encapsulated in the index i) to the algebraic sign in the middle via the "sign" variable:

```sh
for i in range(3):
if i == 0: 
	pass
else: 
	sign = (-1) ^ (-i  + 1)
	angle = np.deg2rad(-25 * measurement)
	new_angle = np.arctan(1/4 \
				+ (sign) * np.sign(measurement) \
				* (np.tan(angle))) 
	measurement = (0.9825 * measurement) \
				+ (0.0175 * (np.rad2deg(-new_angle)) / 25) 
```

Since the quantity dcos(\beta) represents the perpendicular distance from the front bumper to the parallel line containing the point of reaction, it is a rough equivalent to the length it takes the car to travel with the new steering angle. Assuming (quite poorly) that the tires change direction instantaneously, this allows us to relate distance in car width units.

For instance, if dcos(\beta) = 4R = 2 * 2R, the point of reaction is two car width's away from the front of the car, and the first term in each equation becomes

R/dcos(\beta) = R/4R = 1/4

Since a typical car is about [5.6 feet wide](https://en.wikipedia.org/wiki/Vehicle_size_class), we can use this and a vehicle speed of 30 mph to predict what dcos(\beta) should be for the average human with a [reaction time of 0.2476 seconds to visual stimuli](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4456887/):

0.2476 seconds * (1 hour/3600 seconds) * (30 miles/1 hour) * (5280 feet/1 mile) 

= 10.89 feet = 1.945 car widths = 3.891 * R

which is practically equal to 2 car widths, or 4R. Thus, one hyperparameter could have been found for the trigonometric equations.

However, I mistakenly brute-forced various guesses for this number across applications of the model itself in tandem with another hyperparameter yet to be discussed in order to arrive at this same conclusion....

The other hyperparameter was found to be necessary due to the erratic sinusoidal driving behavior of the correction when applied. Instinctively, one figured that the model was overfitting to the corrective factor, and so a weighted comparison needed to be done with the original measurement:

```sh
measurement = (0.9825 * measurement) \
			+ (0.0175 * (np.rad2deg(-new_angle)) / 25)
```

Although small, a proportionality constant of 0.0175 for the trig equations (equal to a 1.75% contribution to the final measurement for the left and right camera steering angles) was eventually used once the 1/4 constant up above was found (this one due to inspection, once the model stayed reliably close to center). The proportionality constant allowed the car to stop driving sinusoidally along its overall course of trajectory. With the previous constant keeping it close to center, the data from the left and right camera angles allowed the car to continuously self-correct its course in any event where the car veered off-center - be it curves, initial placement, intentionally veering off the track beforehand, etc.

Further tests showed robustness to both reversing the direction of the car's travel and moderate displacement from the road itself. Along the track, it would stay as close to center as possible, or otherwise remain in the lane lines during curves and restore its central position afterward.

Before running the model in the simulator, a rough measure of its accuracy was calculated by using a train/test split upon the input data of 20%. The resulting figures of test and validation loss were initally very helpful in predicting performance and under/overfitting, but as they got smaller and smaller would not bear significant similarity to the car's ability to drive successfully. Due to the absence of either overfitting or underfitting upon building the NVIDIA archtecture and incorporating flips and other cameras' images for data, these figures were not used for much else than to determine the proper number of epochs for which to train.

Again, by the very end of training and inclusion of the left and right camera images into the model, the car stayed satisfactorily close to center, self-corrected manual small displacements on or off the road, and was able to drive within the lane lines indefinitely in either direction along the track.

#### 2. Final Model Architecture

The final model architecture (lines 60-75) consisted of a convolution neural network with the following layers and sizes:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Lambda 				| Normalization and input dimension setting		|
| Cropping 				| Remove top 64 and bottom 24 pixels 			|
| Convolution 5x5     	| 2x2 stride, valid padding, 24 filters		 	|
| ReLU                  | extends model space & reduces sparsity		|
| Max pooling           | 2x2 stride adn pool size, outputs local maxes	|
| Convolution 5x5     	| 2x2 stride, valid padding, 36 filters			|
| ReLU                  | extends model space & reduces sparsity		|
| Convolution 3x3     	| 1x1 stride, valid padding, 48 filters		 	|
| ReLU                  | extends model space & reduces sparsity		|
| Convolution 3x3     	| 1x1 stride, valid padding, 64 filters		 	|
| ReLU                  | extends model space & reduces sparsity		|
| Convolution 3x3     	| 1x1 stride, valid padding, 64 filters		 	|
| ReLU                  | extends model space & reduces sparsity		|
| Flatten               | make 1D output for further processing 		|
| Fully connected		| outputs 150        							|
| Fully connected		| outputs 100        							|
| Fully connected		| outputs 50        							|
| Fully connected		| outputs 1 number: the	steering angle  		|

Here is a visualization of the architecture (created on [NVIDIA's blog](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)):

![alt text][image1]

####3. Creation of the Training Set & Training Process

Two laps were done in either direction, driving as close to the center of the road as possible (especially on curves). Here is an example of what it looked like from the center camera angle:

![alt text][image3]

I then devised steering angles for the left and right camera images and incorporated them into the model. Here are examples of each corresponding to the image above:

![alt text][image4]
![alt text][image5]

I decided not to touch track two since driving on it is beyond what I can achieve in a reasonable amount of time with my own motor capabilities. 

To augment the data sat, I also flipped images and angles thinking that this would enrich the amount of images that were available.

After the collection and augmentation processes, the model had something like 66003 separate data points to work with. The data was then preprocessed by normalizing the pixel values around a mean of 1 and cropping each image to discard useless data.


Finally, the data was shuffled and 20% randomly chosen to be set aside for validation purposes during training. 

I used this training data for training the model while employing mean squared error as my loss function and the ADAM optimizer for stochastic gradient descent. The validation set helped determine early on if the model was overfitting or underfitting. The ideal number of epochs was 2 as evidenced by the typical trend of validation loss increasing after 2 epochs, but decreasing between the first and second. Toggling the learning rate was not necessary since the ADAM optimizer was employed, but then again, as stated above, use of trigonometry to measure left and right steering angles introduced two new hyperparameters to play around with during training.

<p align="center">
  <img width="320" height="160" src="./pictures/clip.gif">
</p>