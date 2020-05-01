# In this Repo:
## SSD-[MobileNetV1, VGG] in tensorflow.keras v1.15
---
### Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Examples](#examples)
4. [Dependencies](#dependencies)
5. [How to use it](#how-to-use-it)
6. [Download the convolutionalized VGG-16 weights](#download-the-convolutionalized-vgg-16-weights)
7. [Download the original trained model weights](#download-the-original-trained-model-weights)
8. [How to fine-tune one of the trained models on your own dataset](#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)
9. [ToDo](#todo)
10. [Important notes](#important-notes)
11. [Terminology](#terminology)

### Overview

A fork of the original keras implementation by pierluigiferrari at https://github.com/pierluigiferrari/ssd_keras, modified to be able to achieve the same results with tf.keras v1.15. An implementation of MobileNetV1-SSD is also added with promising results. Additionally contains experiments on binarization with larq when applied to Object Detection models. 

The repository currently provides the following network architectures:
* SSD300-VGG: [`tfkeras_ssd_vgg.py`](models/tfkeras_ssd_vgg.py)
* SSD300-MobileNetV1: [`tfkeras_ssd_mobilenet_3x3.py`](models/tfkeras_ssd_mobilenet_3x3.py)
* SSD7: [`keras_ssd7.py`](models/keras_ssd7.py) - a smaller 7-layer version custom made by pierluigiferrari. Fast, but subpar results - best used as a toy model.

Includes 3 implementations of SSD-MobileNetV1:
1) tfkeras_ssd_mobilenet_3x3.py - imports tf.keras.applications MobileNetV1, and uses kernel size 3 on the detection heads.
2) tfkeras_ssd_mobilenet_official.py - imports tf.keras.applications MobileNetV1, and uses kernel size 1 on the detection heads.
2) tfkeras_ssd_mobilenet_beta.py - uses a defined MobileNetV1 at mobilenet_v1.py, and uses kernel size 1 on the detection heads.

If you would like to build an SSD with your own base network architecture, you can use [`keras_ssd7.py`](models/keras_ssd7.py) as a template, it provides documentation and comments to help you.

### Performance

In this section mAP evaluation results of models trained with this repository are compared with existing SSD implementations. All models were evaluated using the official Pascal VOC test server (for 2012 `test`) or the official Pascal VOC Matlab evaluation script (for 2007 `test`).

Note that training for the models trained with this repository are currently halted at 20 epochs/20,000 steps in the interest of time, as mAP already at this point looks promising and will achieve close to the expected performance.

Training the SSD-VGG model on PASCAL VOC 7+12 for 20,000 steps yields the same mAP achieved by pierluigiferrari's implementation at this stage. You can find a summary of the training by pierluigiferrari [here](training_summaries/ssd300_pascal_07+12_training_summary.md).

<table width="95%">
  <tr>
    <td></td>
    <td colspan=3 align=center>Mean Average Precision</td>
  </tr>
  <tr>
    <td></td>
    <td align=center>Ours@20k steps</td>
    <td align=center>Pierlugiferrari's @ 20k steps</td>
    <td align=center>Pierlugiferrari's @ 102k steps</td>
  </tr>
  <tr>
    <td><b>SSD300 "07+12"</td>
    <td align=center width="26%"><b>0.682</td>
    <td align=center width="26%"><b>0.696</td>
    <td align=center width="26%"><b>0.771</td>
  </tr>
</table>

### Dependencies

* Python 3.6 (Not confident if this implementation works with other python versions)
* Numpy
* TensorFlow 1.15.x
* OpenCV
* Beautiful Soup 4.x

### How to use it

I have provided the Google Colab ipython notebooks used to train the models, and should contain all the necessary code for 
- preparing a colab environment
- training a model
- evaluating a model
- loading a model for inference

However, some features were not used in these colab notebooks (such as the DecodeDetections layers). To compensate...

The original repository by pierluigiferrari contains well-documented ipython notebooks for training, inference, evaluation, and the many features included in the repo. DO NOTE that these notebooks are unchanged and will not work with this repo - ONLY USE THEM AS A GUIDE.

How to use a trained model for inference:
* [`ssd300_inference.ipynb`](old_keras_notebooks/ssd300_inference.ipynb)
* [`ssd512_inference.ipynb`](old_keras_notebooks/ssd512_inference.ipynb)

How to train a model:
* [`ssd300_training.ipynb`](old_keras_notebooks/ssd300_training.ipynb)
* [`ssd7_training.ipynb`](old_keras_notebooks/ssd7_training.ipynb)

How to use one of the provided trained models for transfer learning on your own dataset:
* [Read below](#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)

How to evaluate a trained model:
* In general: [`ssd300_evaluation.ipynb`](old_keras_notebooks/ssd300_evaluation.ipynb)
* On MS COCO: [`ssd300_evaluation_COCO.ipynb`](old_keras_notebooks/ssd300_evaluation_COCO.ipynb)

How to use the data generator:
* The data generator used here has its own repository with a detailed tutorial [here]
(https://github.com/pierluigiferrari/data_generator_object_detection_2d)

#### Training details

The general training setup is layed out and explained in [`ssd7_training.ipynb`](old_keras_notebooks/ssd7_training.ipynb) and in [`ssd300_training.ipynb`](old_keras_notebooks/ssd300_training.ipynb). The setup and explanations are similar in both notebooks for the most part, so it doesn't matter which one you look at to understand the general training setup, but the parameters in [`ssd300_training.ipynb`](old_keras_notebooks/ssd300_training.ipynb) are preset to copy the setup of the original Caffe implementation for training on Pascal VOC, while the parameters in [`ssd7_training.ipynb`](old_keras_notebooks/ssd7_training.ipynb) are preset to train on the [Udacity traffic datasets](https://github.com/udacity/self-driving-car/tree/master/annotations).

To train the original SSD300 model on Pascal VOC:

1. Download the datasets:
  ```c
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  ```
2. Download the weights for the convolutionalized VGG-16 or for one of the trained original models provided below.
3. Set the file paths for the datasets and model weights accordingly in [`ssd300_training.ipynb`](ssd300_training.ipynb) and execute the cells.

The procedure for training SSD512 is the same of course. It is imperative that you load the pre-trained VGG-16 weights when attempting to train an SSD300 or SSD512 from scratch, otherwise the training will probably fail. Here is a summary of a full training of the SSD300 "07+12" model for comparison with your own training:

* [SSD300 Pascal VOC "07+12" training summary](training_summaries/ssd300_pascal_07+12_training_summary.md)

#### Encoding and decoding boxes

The [`ssd_encoder_decoder`](ssd_encoder_decoder) sub-package contains all functions and classes related to encoding and decoding boxes. Encoding boxes means converting ground truth labels into the target format that the loss function needs during training. It is this encoding process in which the matching of ground truth boxes to anchor boxes (the paper calls them default boxes and in the original C++ code they are called priors - all the same thing) happens. Decoding boxes means converting raw model output back to the input label format, which entails various conversion and filtering processes such as non-maximum suppression (NMS).

In order to train the model, you need to create an instance of `SSDInputEncoder` that needs to be passed to the data generator. The data generator does the rest, so you don't usually need to call any of `SSDInputEncoder`'s methods manually.

Models can be created in 'training' or 'inference' mode. In 'training' mode, the model outputs the raw prediction tensor that still needs to be post-processed with coordinate conversion, confidence thresholding, non-maximum suppression, etc. The functions `decode_detections()` and `decode_detections_fast()` are responsible for that. The former follows the original Caffe implementation, which entails performing NMS per object class, while the latter performs NMS globally across all object classes and is thus more efficient, but also behaves slightly differently. Read the documentation for details about both functions. If a model is created in 'inference' mode, its last layer is the `DecodeDetections` layer, which performs all the post-processing that `decode_detections()` does, but in TensorFlow. That means the output of the model is already the post-processed output. In order to be trainable, a model must be created in 'training' mode. The trained weights can then later be loaded into a model that was created in 'inference' mode.

A note on the anchor box offset coordinates used internally by the model: This may or may not be obvious to you, but it is important to understand that it is not possible for the model to predict absolute coordinates for the predicted bounding boxes. In order to be able to predict absolute box coordinates, the convolutional layers responsible for localization would need to produce different output values for the same object instance at different locations within the input image. This isn't possible of course: For a given input to the filter of a convolutional layer, the filter will produce the same output regardless of the spatial position within the image because of the shared weights. This is the reason why the model predicts offsets to anchor boxes instead of absolute coordinates, and why during training, absolute ground truth coordinates are converted to anchor box offsets in the encoding process. The fact that the model predicts offsets to anchor box coordinates is in turn the reason why the model contains anchor box layers that do nothing but output the anchor box coordinates so that the model's output tensor can include those. If the model's output tensor did not contain the anchor box coordinates, the information to convert the predicted offsets back to absolute coordinates would be missing in the model output.

#### Using a different base network architecture

If you want to build a different base network architecture, you could use [`tfkeras_ssd7.py`](models/tfkeras_ssd7.py) as a template. It provides documentation and comments to help you turn it into a different base network. Put together the base network you want and add a predictor layer on top of each network layer from which you would like to make predictions. Create two predictor heads for each, one for localization, one for classification. Create an anchor box layer for each predictor layer and set the respective localization head's output as the input for the anchor box layer. The structure of all tensor reshaping and concatenation operations remains the same, you just have to make sure to include all of your predictor and anchor box layers of course.

### Download the convolutionalized VGG-16 weights

In order to train an SSD300 or SSD512 from scratch, download the weights of the fully convolutionalized VGG-16 model trained to convergence on ImageNet classification here:

[`VGG_ILSVRC_16_layers_fc_reduced.h5`](https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox).

As with all other weights files below, this is a direct port of the corresponding `.caffemodel` file that is provided in the repository of the original Caffe implementation.

### How to fine-tune one of the trained models on your own dataset

If you want to fine-tune one of the provided trained models on your own dataset, chances are your dataset doesn't have the same number of classes as the trained model. The following tutorial explains how to deal with this problem:

[`weight_sampling_tutorial.ipynb`](old_keras_notebooks/weight_sampling_tutorial.ipynb)

### Important notes

* All trained models that were trained on MS COCO use the smaller anchor box scaling factors provided in all of the Jupyter notebooks. In particular, note that the '07+12+COCO' and '07++12+COCO' models use the smaller scaling factors.

### Terminology

* "Anchor boxes": The paper calls them "default boxes", in the original C++ code they are called "prior boxes" or "priors", and the Faster R-CNN paper calls them "anchor boxes". All terms mean the same thing, but I slightly prefer the name "anchor boxes" because I find it to be the most descriptive of these names. I call them "prior boxes" or "priors" in `keras_ssd300.py` and `keras_ssd512.py` to stay consistent with the original Caffe implementation, but everywhere else I use the name "anchor boxes" or "anchors".
* "Labels": For the purpose of this project, datasets consist of "images" and "labels". Everything that belongs to the annotations of a given image is the "labels" of that image: Not just object category labels, but also bounding box coordinates. "Labels" is just shorter than "annotations". I also use the terms "labels" and "targets" more or less interchangeably throughout the documentation, although "targets" means labels specifically in the context of training.
* "Predictor layer": The "predictor layers" or "predictors" are all the last convolution layers of the network, i.e. all convolution layers that do not feed into any subsequent convolution layers.
