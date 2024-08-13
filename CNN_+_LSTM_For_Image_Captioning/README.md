# Image Captioning with CNN and LSTM

This project demonstrates an image captioning system using a Convolutional Neural Network (CNN) as an encoder and a Long Short-Term Memory (LSTM) network as a decoder. The CNN uses the ResNet152 model to extract image features, and the LSTM generates captions based on these features. The model is trained and evaluated on the COCO dataset.

## Project Overview

### Model Architecture

- **Encoder:** The encoder is a pre-trained ResNet152 model, which is used to extract features from images. These features are then fed into the LSTM decoder.
- **Decoder:** The decoder is an LSTM network that generates captions based on the features provided by the CNN encoder. The captions are generated word by word.

### Dataset

The project uses the [COCO dataset](http://cocodataset.org/), a large-scale dataset for object detection, segmentation, and captioning. The dataset provides a rich set of images along with their corresponding captions, making it suitable for training image captioning models.

### Vocabulary

A vocabulary of words is created from the captions in the COCO dataset. This vocabulary is used by the LSTM decoder to generate the captions. The vocabulary includes special tokens such as `<start>`, `<end>`, and `<pad>` to handle the beginning and end of sentences and to pad sequences to a fixed length.

## Project Structure

- **Data Preparation:** Scripts for downloading and preprocessing the COCO dataset, including image resizing and caption tokenization.
- **Vocabulary Creation:** A script to build a vocabulary from the COCO captions, including the addition of special tokens.
- **Model Implementation:** The ResNet152-based encoder and the LSTM decoder are implemented in PyTorch.
- **Training:** Scripts to train the CNN-LSTM model on the COCO dataset, including loss computation and optimization.
- **Evaluation:** Scripts to evaluate the model's performance on the test set, using metrics like BLEU scores.

## References

- [COCO Dataset](http://cocodataset.org/)
- [ResNet152 Paper](https://arxiv.org/abs/1512.03385)


