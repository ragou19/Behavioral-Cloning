import numpy as np
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
import csv
import cv2
import re
import matplotlib.image as mpimg

# The next code block opens the csv file containing image names and associated steering angles.
lines = [] 
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Extracts image data and steering data.
images = []
measurements = []
for line in lines:
	if line[0] == "center":
		pass
	else:
		for i in range(3): # Adds center, left, and right camera images to list.
			source_path = line[0 + i]
			filename = source_path.split('/')[-1].split('\\')[-1]
			current_path = './IMG/' + filename
			image = mpimg.imread(current_path)
			images.append(image)
			measurement = float(line[3])
			if i == 0: # Do not modify the steering angle for the center camera.
				pass
			else: 
			#The indented block modifies the steering angle slightly for left and right camera images. The next block after adds this new image and steering data to the lists.
				sign = (-1) ^ (-i  + 1)
				angle = np.deg2rad(-25 * measurement)
				new_angle = np.arctan(1/4 \
							+ (sign) * np.sign(measurement) \
							* (np.tan(angle))) # The 1/4 represents steering reaction to curves 2 car widths away from the front of the vehicle.
				measurement = (0.9825 * measurement) \
							+ (0.0175 * (np.rad2deg(-new_angle)) / 25) # The numerical coefficients represent the weight of the trig compensation relative to the original steering measurement.
			measurements.append(measurement)
			image_flipped = np.fliplr(image)
			measurement_flipped = -measurement
			images.append(image_flipped)
			measurements.append(measurement_flipped)

# Convert lists to arrays.
X_train = np.array(images)
y_train = np.array(measurements)

# Define convolutional layers with relus by default.
def conv2d(f, kw, kh, strides = (1,1), relu=True):
	model.add(Convolution2D(f, kw, kh, subsample = strides))
	if relu == True:
		model.add(Activation('relu'))

# Keras model with normalization, cropping, 5 convolutional layers, 1 pooling layer, 1 flatten layer, and 4 dense layers before compiling and fitting.
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((64, 24), (0,0))))
conv2d(24, 5, 5, strides = (2, 2))
model.add(MaxPooling2D())
conv2d(36, 5, 5, strides = (2, 2))
conv2d(48, 3, 3)
conv2d(64, 3, 3)
conv2d(64, 3, 3)
model.add(Flatten())
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')


