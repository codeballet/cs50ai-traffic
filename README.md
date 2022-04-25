# Traffic

## Introduction

Traffic is an AI that identifies which traffic sign appears in a photograph.

The AI explores a common, current field of research in the field of computer vision, allowing self-driving cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, I am using TensorFlow to build a neural network to classify road signs based on an image of those signs. The dataset I am using is the [German Traffic Sign Recognition Benchmark (GTSRB) dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html), which contains thousands of images of 43 different kinds of road signs.

## The `load_data` function

The `load_data` function accepts as an argument `data_dir`, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.

The `data_dir` should contain one directory named after each category, numbered `0` through `NUM_CATEGORIES - 1`. Inside each category should be some number of image files for that category.

I am using the OpenCV-Python module `cv2` to read each image as a `numpy.ndarray`. The images are read with the `IMREAD_COLOR` flag, ensuring that images always have the BGR color channel values.

Each image is then resized to a square, the dimensions specified by the variables `IMG_WIDTH` and `IMG_HEIGHT`. The images from the dataset are neither necessarily square, nor are the road signs necessarily in the centre. Hence, in order not to lose any of the roadsigns, I am not cropping the images. This may result in some pictures ending up with a different scaling than the original.

The function returns a tuple `(images, labels)`. `images` is a list of all of the images in the data set, each image represented as a `numpy.ndarray`. `labels` is a list of integers, representing the category number for each of the corresponding images in the `images` list.

## The `get_model` function

The `get_model` function returns a compiled neural network model.

The input to the neural network should be of shape `(IMG_WIDTH, IMG_HEIGHT, 3)`. The output should have `NUM_CATEGORIES` units, one for each of the traffic sign categories.

### Numbers of convolutional and pooling layers

I did experiment with different convolutional and pooling layers, and I found that the number and size of convolutional and pooling layers did affect the overall outcome.

### Numbers and sizes of filters for convolutional layers

For the convolutional layers, the size of the layer matters. Given that the images are rather small, 30x30 pixels, having too large kernel filters lead to worse result. It seems that having larger kernel matrices may result in too much loss of information. Also, I found that adding several convolutional filters did not significantly improve the accuracy. Once again, I believe this is due to an increased loss of information.

### Pool sizes for pooling layers

For the pool sizes, my conclusion is in accordance with the above: Having too large pooling sizes, or more than one pooling layer, seems to worsen the accuracy of the neural network. I believe that has to do with too much loss of information.

### Numbers and sizes of hidden layers

Working with only one hidden layer, I found that using 128 units produced surprisingly good results. While the training accuracy only achieved an accuracy around 0.935, the testing data usually gave better result, with accuracy around 0.973. It seems that only using one layer prevented overfitting, and the testing data typically gave better accuracy than the training data.

However, increasing the number of hidden layers, and units in the layers, I achieved a better balance between the accuracy of the training and testing data. For instance, using three hidden layers with 512, 256, and 128 nodes respectively, I typically achieved an accuracy on the training data around 0.988, and on the testing data 0.973.

### Dropout

Adjusting the dropout seved to create a more consistent result between the accuracies of the training and the testing data. If the dropout was too small, the data typically became overfitted, especially in the case of having several hidden layers in the neural network. However, when only using one layer, the overfitting problem was not as noticeable, and the dropout did not have such a dramatic effect.

Using three layers with 512, 256, and 128 layers, the dropout rate seemed to be important. Anything smaller than a dropout of 0.4 seemed to lead to overfitting. However, higher dropout rates appeared to result in worse accuracy of both training and testing data.
