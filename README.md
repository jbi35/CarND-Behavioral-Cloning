
# Behavioral Cloning Project

[//]: # (Image References)

[image1]: ./example_images/cnn-architecture.png "Model Visualization"
[image2]: ./example_images/center-driving.jpg "center driving"
[image3]: ./example_images/right_1.jpg "Recovery Image"
[image4]: ./example_images/right_2.jpg "Recovery Image"
[image5]: ./example_images/right_3.jpg "Recovery Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* ImageProcessor for preprocessing the images and implementation of the generator
* model.h5 containing a trained convolution neural network
* model.json containing the description of the network
* writeup_report.md summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.json
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The python generator is implemented in the file ImageProcessor along with some other preprocessing functions.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with five convolutional layers and 3 fully connected layers as described in this [paper](https://arxiv.org/abs/1604.07316). Before the data is passed to the first convolutional layer, the images are normalized using Keras lambda layer.

The model uses [ELU layers](https://arxiv.org/abs/1511.07289) to introduce nonlinearity, which turned out to work better than RELU activations.

The model is best summarized by its Keras code:

```sh
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(66,200,3),output_shape=(66,200,3)))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='linear', subsample=(2,2),input_shape=(66,200,3)))
model.add(ELU())
model.add(Convolution2D(36,5,5,border_mode='valid', activation='linear', subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(48,5,5,border_mode='valid', activation='linear', subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(64,3,3,border_mode='valid', activation='linear', subsample=(1,1)))
model.add(ELU())
model.add(Convolution2D(64,3,3,border_mode='valid', activation='linear', subsample=(1,1)))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100, activation='linear'))
model.add(ELU())
model.add(Dense(50, activation='linear'))
model.add(ELU())
model.add(Dense(10, activation='linear'))
model.add(ELU())
model.add(Dense(1))```

and the following visualization of the architecture from the original [paper](https://arxiv.org/abs/1604.07316)

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers to reduce overfitting. However,
the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 46-48 model.py). In addition, the training data is randomly augmentend to avoid overfitting.

Specifically, the training images were:

* flipped with a probability of 0.5 (steering angles is then multiplied with -1)
* rotated by a random angle between -5 and 5 degrees (steering angle is adjusted accordingly

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 32).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving for roughly for laps in combination with recoveries from the left and the right side. For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look at existing architectures for this task. Specifically, I investigated the [Comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) model and the model proposed by researchers from [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).  

I found that the NVIDIA architecture worked rather well. I started varying the activation functions and found that elu layers worked better than linear or relu activations.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set and in addition, I augmented the training set as described above. Since the augmentation of the data is to some extend random, the training data is slightly different in each training epoch, which prevents overfitting.

Using the ModelCheckpoint (line 52 model.py ) I made sure to only save the model if performance measured on the validation set improved.

The final step was to run the simulator to see how well the car was driving around track one. I first tried this only with the training data provided by Udacity. This, however, did not yield satisfactory results. After generating additional own training data, the results were satisfactory.

#### 2. Final Model Architecture

The final model architecture (model.py lines 11-30) is already shown above and not repeated here.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded roughly four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to reclaim a centered position on the road after getting close to the sides of the road. The following images show a recovery from the right.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would improve performance in right turns, since most of the curves on track one go to the left. In addition, I made use of the images from the left and right camera. Here, I adjusted the steering angle by 5 degrees to adjust for the off-center viewpoint. In total I have collected ca. 80.000 images without augmentation.

Since the dataset contained a lot of images with very small steering angle
I randomly removed images with steering angle below 0.01 from the training set
(ImageProcessor.py line 48)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I stopped the training after 50 Epochs because the validation loss did non longer significantly improve. Moreover, the adam optimizer had reduced the learning rate close to 1e-7 which I considered to be the minimum below which significant improvement could not be expected. Also the autonomous driving performance of the car on track one was very good at this point. A limitation of the current model is that the car does not perform as well on the second track, where the car occasionally hits the sidewalls.
