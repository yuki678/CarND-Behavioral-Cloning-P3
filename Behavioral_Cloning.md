# Behavioral Cloning
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/training_center.jpg "Center lane driving"
[image3]: ./images/recovery1.jpg "Recovery Image"
[image4]: ./images/recovery2.jpg "Recovery Image"
[image5]: ./images/recovery3.jpg "Recovery Image"
[image6]: ./images/recovery4.jpg "Recovery Image"
[image7]: ./images/loss.png "Loss chart"

---
## 1. Files Submitted & Code Quality

### 1-1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* models/model.h5 containing a trained convolution neural network 
* Behavioral_Cloning.md or writeup_report.pdf summarizing the results

### 1-2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 1-3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
## 2. Model Architecture and Training Strategy

### 2-1. An appropriate model architecture has been employed

I used [NVidia Autonomous Car Group model](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/).
The model can be found in `model.py`

The summary of the model is as follows.
```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```


### 2-2. Attempts to reduce overfitting in the model

The model didn't contains dropout layers but I limited the epoch to 3 in order to reduce overfitting. I split the training data to the train dataset and validation dataset by 80%:20% to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 2-3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### 2-4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The simulator provide three difference images at each time as center, right and left cameras, so I used all with adjustment.
I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

## 3. Model Architecture and Training Strategy

### 3-1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from well-known architectures.
My first step was to use [LeNet](http://yann.lecun.com/exdb/lenet/) model. This original model could learn how to keep the car within the lane to some extent but often went off and stuck.

Then, I added a Lambda layer at the beginning to to normalize the input, followed by a Cropping Layer to crop the images as the above part of each picture is more or less scenery and thought those would become noises. Tried several times with adjustment of parameters, this model with the additional layers could move the car much better within the lane up to the bridge of the track.

Here shows the model architecture at this stage.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 86, 316, 6)        456       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 43, 158, 6)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 39, 154, 6)        906       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 19, 77, 6)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8778)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 120)               1053480   
_________________________________________________________________
dense_2 (Dense)              (None, 84)                10164     
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 85        
=================================================================
Total params: 1,065,091
Trainable params: 1,065,091
Non-trainable params: 0
_________________________________________________________________
```

Then, I tried [NVidia Autonomous Car Group model](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) instead of LeNet based architecture.
After some trials, this model could almost complete the first track but still failed at the end where the curve was steeper.
The summary of the model is as follows.
```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

I saw that left turn work good but not right turn, which could be due to the anti-clockwise track. Therefore, I added data augmentation by horizontally flipping the image, expecting this has the same effect of taking additional recording to drive the track in reverse direction with a fewer data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for two rounds.

### 3-2. Final Model Architecture

The final model architecture is as follows:

![alt text][image1]

### 3-3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to go back to the road when it is off. These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles, while keeping the number of data per epoch as original the number of original data.

After the collection process, I had X number of data points. I then preprocessed this data by ...

As a result of training, the MSE error was reduced for both training set and validation set as follows and the car could drive without fall off the road.

![alt text][image7]