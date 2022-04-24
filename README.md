# Traffic

## Introduction

Traffic is an AI that identifies which traffic sign appears in a photograph.

The AI explores a common, current field of research in the field of computer vision, allowing self-driving cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, I am using TensorFlow to build a neural network to classify road signs based on an image of those signs. The dataset I am using is the [German Traffic Sign Recognition Benchmark (GTSRB) dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html), which contains thousands of images of 43 different kinds of road signs.

## The `load_data` function

The `load_data` function accepts as an argument `data_dir`, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.

The `data_dir` should contain one directory named after each category, numbered `0` through `NUM_CATEGORIES - 1`. Inside each category should be some number of image files for that category.

I am using the OpenCV-Python module `cv2` to read each image as a `numpy.ndarray`. Each image is then resized to a square, the dimensions specified by the variables `IMG_WIDTH` and `IMG_HEIGHT`.

The images from the dataset are neither necessarily square, nor are the road signs necessarily in the centre. Hence, in order not to lose any of the roadsigns, I am not cropping the images. This may result in some pictures ending up with a different scaling than the original.

The function returns a tuple `(images, labels)`. `images` is a list of all of the images in the data set, each image represented as a `numpy.ndarray`. `labels` is a list of integers, representing the category number for each of the corresponding images in the `images` list.

## The `get_model` function
