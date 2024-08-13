
# CNN Models and Transfer Learning

This project demonstrates the implementation of several Convolutional Neural Network (CNN) models, including LeNet, AlexNet, and VGG13, with a focus on transfer learning techniques applied to AlexNet and VGG13.

## Project Overview

The project includes implementations of the following CNN models:

- **LeNet**
- **AlexNet**
- **VGG13**

### Transfer Learning

Transfer learning is applied to the AlexNet and VGG13 models. The pre-trained versions of these models are fine-tuned on a specific target dataset to improve performance with reduced training time and data requirements.

### Structure

The project is structured as follows:

- **Data Preparation:** Scripts for downloading, preprocessing, and splitting the dataset into training, validation, and test sets.
- **Model Implementation:** Implementation of the LeNet, AlexNet, and VGG13 models.
- **Transfer Learning:** Fine-tuning of the AlexNet and VGG13 models using pre-trained weights.
- **Training and Evaluation:** Scripts for training the models, monitoring performance, and evaluating the results.

### Results

The performance of the models is evaluated based on accuracy, loss, and other relevant metrics. The results of the transfer learning experiments with AlexNet and VGG13 show improvements in model performance compared to training from scratch.

### References

- PyTorch documentation for [Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- Original research papers for [LeNet](http://yann.lecun.com/exdb/lenet/), [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), and [VGG](https://arxiv.org/abs/1409.1556).

##

