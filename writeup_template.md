# **Traffic Sign Recognition** 

Bolin ZHAO
Dec, 3rd, 2018
bolinzhao@yahoo.com

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/example_1.jpg "data_set_exp1"
[image2]: ./examples/example_2.jpg "data_set_exp2"
[image3]: ./examples/example_3.jpg "data_set_exp3"
[image4]: ./examples/example_4.jpg "data_set_exp4"
[image5]: ./DE_trf_sign/1.jpg "Traffic Sign 1"
[image6]: ./DE_trf_sign/2.jpg  "Traffic Sign 2"
[image7]: ./DE_trf_sign/3.jpg  "Traffic Sign 3"
[image8]: ./DE_trf_sign/4.jpg  "Traffic Sign 4"
[image9]: ./DE_trf_sign/5.jpg  "Traffic Sign 5"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/berlala/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) on my Github.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
 34799
* The size of the validation set is ?
4410
* The size of test set is ?
12630
* The shape of a traffic sign image is ?
 (32,32,3)
* The number of unique classes/labels in the data set is ?
 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.  The second figure shows the overall distribution of the training pictures.  

![Example of random train pictures][image1]  

![The distribution of the labels][image2]  

![Example of label 4: Speed Limit 80Km/h][image3]  

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To simplized the process, only two key steps are applied on the original picture for the try.   
The first measure is convert the color picture to gray by cv2, and the second step is to normalize the pictures.   
The different between the input and output can be found in the following picture.
![Example of pre-process][image4]  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers, it is very similar to the LeNET,

| Layer         		|     Description	        			| Output |
|:---------------------:|:---------------------------------------------:|-----------------------------------------------|
| Input         		| pre-processed 32x32x1 Gray image |32x32x1|
| Convolution 1st | filter size 5x5, filter num 6, stride 1x1, VALID padding  |28x28x6|
| RELU					|												|28x28x6|
| Average pooling	| filter size 2x2,  stride 2x2, VALID padding |14x14x6|
| Convolution	    | filter size 5x5, filter num 16, stride 1x1, VALID padding |10x10x16|
| RELU	|         									|10x10x16|
| Average pooling	| filter size 2x2,  stride 2x2, VALID padding |5x5x16|
| Flatten |												|400|
| Fully Connected |												|120|
| RELU |  |120|
| Fully Connected | |84|
| RELU | |84|
| Fully Connected | |43|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The epochs is set as  50 and batch size is 128. The optimizer is adaptive moment estimation(AdamOptimizer). For every iteration, the model is valid by the validation set. It can be found that the accuray is increasing from 0.8 to 0.93 around 30 iterations. And the accuracy is floating around 0.93.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
99.9%
* validation set accuracy of ? 
94.3%
* test set accuracy of ?
93.2%(lucky ^_^)

* The LeNET structure is applied for the try. The first reason is due to that this is the first and most familiar model for me. The second reason is that the LeNET is proved to work on the number detection. While from the characteristic point of view, the sign in fact is very close to the number in a picture. They can both divide into small patches of characteristic. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]

The last image might be difficult to classify because it contain some backgournd color and information. (in fact I try other images with trees or house in the background, and the model cannot determine the right label of the sign).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed Limit 60Km/h      		| Speed Limit 60Km/h      									|
| Stop sign     			| Stop sign 										|
| priority road				| priority road											|
| Turn left ahead       		| Turn left ahead 					 				|
|General Caution		| General Caution     							|


The model was able to correctly all traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Speed Limit 60Km/h |
| .99     		| Stop sign 	|
| .99			| Priority road	|
| 1.0	   | Turn left ahead	|
| 1.0				 | General Caution |




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Sorry do not finish this part yet.


