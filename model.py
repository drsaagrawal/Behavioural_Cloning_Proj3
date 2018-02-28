
# coding: utf-8

# ## Loading CSV

# In[ ]:

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.utils import shuffle
lines = []
correction=0.1
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# ## Augmenting the image

# In[ ]:

def augment(images,measurements):
    augmented_images, augmented_measurements = [], []
    #print("cc",images)
    for image,measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(np.fliplr(image))
        augmented_measurements.append(measurement*-1.0)
    return augmented_images,augmented_measurements


# In[ ]:




# ## Training data using generator in batches and augmenting the image

# In[ ]:

import tensorflow as tf
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            aug_images = []
            angles = []
            
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                   # print("b"+str(counter))
                    filename = source_path.split('/')[-1]
                    #filename=filename.split('/')[-1]
                    #print(filename)
                    current_path = 'data/IMG/' + filename
                    image = plt.imread(current_path)
                    images.append(image)
                    if(i==1):
                        measurement = (float(batch_sample[3])+correction)
                        measurements.append(measurement)
                    elif(i==2):
                        measurement = (float(batch_sample[3])-correction)
                        measurements.append(measurement)
                    else:
                        measurement = (float(batch_sample[3]))
                        measurements.append(measurement)
            #print("bb",measurements)
            aug_images,angles = augment(images,measurements)
            #print("aa",images)

            X_train = np.array(aug_images)
            y_train = np.array(angles)
            #print("X_train:", len(X_train))
            yield shuffle(X_train, y_train)


# ## Splitting data into training and validation set and calling generator

# In[ ]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)



# ## Model Architecture

# In[ ]:

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x : x / 255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(68,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator,nb_val_samples=len(validation_samples)*6, nb_epoch=3,verbose=1)
model.save('model.h5')


# In[ ]:

model.summary()

