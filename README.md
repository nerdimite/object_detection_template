# Tensorflow Custom Object Detection Template

#### Easy to use notebooks for preprocessing data locally to training on the cloud with colab

This repository is an easy to template for using Tensorflow Object Detection API on custom datasets. The preprocessing can be done with the provided notebook locally and then training can be done easily on colab.

## Required Libraries

- Tensorflow 1.15
- object_detection
- opencv

It is best to create a separate python/conda environment

## Usage

1. `git clone https://github.com/theneuralbeing/object_detection_template.git`
2. [Read this](data_preprocessing/README.md) for gathering and annotating data
3. Run the [Preprocess_Data.ipynb](data_preprocessing/Preprocess_Data.ipynb) notebook on your computer
4. After your data is ready, you can directly start [training on this colab notebook](https://colab.research.google.com/github/theneuralbeing/object_detection_template/blob/master/object_detection_training.ipynb) The colab notebook contains all the further steps.
5. After training, the trained inference graph will be downloaded to your computer and you can use the [`inference_webcam.py`](inference_webcam.py). (the steps for inference are also mentioned in the colab notebook)

## Resources
* [Tensorflow Object Detection API Documentation by Lyudmil Vladimirov](https://tensorflow-object-detection-api-tutorial.readthedocs.io/)
* [This Medium Post](https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d)
* [Racoon Detector by datitrain](https://github.com/datitran/raccoon_dataset)
