
# Behavioral Cloning Project
### The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

# Files Submitted & Code Quality
## 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:

model.py containing the script to create and train the model

drive.py for driving the car in autonomous mode

model.h5 containing a trained convolution neural network

writeup_report.md or writeup_report.pdf summarizing the results

video.mp4 of the running car autonomously

## 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5

## 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

# Model Architecture and Training Strategy

## 1. An appropriate model architecture has been employed

I had used NVIDIA model to train my model which has 5 conv layer and 4 fully connected layer. The detailed architecture can be found in the below points. 

## 2. Attempts to reduce overfitting in the model

The model contains augmentation in order to reduce overfitting (model.py Cell 2).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py Cell 3). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

## 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py Cell 5).

Instead I used and tuned the correction factor for steering angle to compensate if car is moving only to particular direction.

## 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
For details about how I created the training data, see the next section.

# Model Architecture and Training Strategy

## 1. Solution Design Approach
The overall strategy for deriving a model architecture was to keep the loss low and try to get a trained model that can keep car on track.
My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because of its previous research work.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (As per thumb rule of 20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.
To combat the overfitting, I augmented and shuffled the images for training.
Then I also used the correction factor for steering angle to compensate if car is moving only to particular direction.
I also cropped the unwanted portion of image to train my model well.
I used generator to combat memory issue.
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and to improve the driving behavior in these cases, I had trained the model for some more recovery cases.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

## 2. Final Model Architecture
The final model architecture (model.py Cell 5) consisted of a convolution neural network with the following layers and layer sizes-
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 68)     29444       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     39232       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 352,255
Trainable params: 352,255
Non-trainable params: 0
____________________________________________________________________________________________________
## 3. Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving: ![center_2016_12_01_13_32_42_143.jpg](attachment:center_2016_12_01_13_32_42_143.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center if it is moving out of track. These images show what a recovery looks like starting from ... : ![left_2016_12_01_13_43_23_427.jpg](attachment:left_2016_12_01_13_43_23_427.jpg)![left_2016_12_01_13_43_23_529.jpg](attachment:left_2016_12_01_13_43_23_529.jpg)![left_2016_12_01_13_43_23_631.jpg](attachment:left_2016_12_01_13_43_23_631.jpg)![left_2016_12_01_13_43_23_733.jpg](attachment:left_2016_12_01_13_43_23_733.jpg)![left_2016_12_01_13_43_23_836.jpg](attachment:left_2016_12_01_13_43_23_836.jpg)![left_2016_12_01_13_43_23_936.jpg](attachment:left_2016_12_01_13_43_23_936.jpg)![left_2016_12_01_13_43_24_039.jpg](attachment:left_2016_12_01_13_43_24_039.jpg)![left_2016_12_01_13_43_24_141.jpg](attachment:left_2016_12_01_13_43_24_141.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would combat overfitting.For example, here is an image that has then been flipped: ![flip_image.jpg](attachment:flip_image.jpg)

After the collection process, I had 36180 number of data points. I then preprocessed this data by lambda function around 0.
I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by my model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
