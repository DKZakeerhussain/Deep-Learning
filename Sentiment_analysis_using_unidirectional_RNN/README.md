# Sentiment Analysis on IMDb Dataset using Unidirectional RNN

This repository contains code to perform sentiment analysis on the IMDb dataset using a unidirectional RNN model implemented in PyTorch.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training and evaluation](#training-and-evaluation)
- [Further Improvements](#further-improvements)


## Introduction
This project implements a sentiment analysis model using a unidirectional Recurrent Neural Network (RNN). The model is trained to classify movie reviews from the IMDb dataset as either positive or negative.

## Dataset

The dataset has been preprocessed by removing punctuation marks. This basic preprocessing step helps to reduce noise in the text data. To further improve the model's performance, we could also remove stop words, which are common words (like "the", "is", "in") that do not contribute significant meaning to the sentiment of the text. Removing stop words can help the model focus on the more meaningful words in the dataset.
For more dataset information, please go through the following link,[https://ai.stanford.edu/~amaas/data/sentiment/]

## Preprocessing
Dataset is preprocess by removing the punctuations only.Further we can remove the other stop words also.

## Model architecture
We implemented a small unidirectional RNN model with a batch size of 32 and an embedding dimension of 100. The model can be extended by stacking multiple RNN layers to enhance its learning capacity and capture more complex patterns in the data.

## Training and evaluation
The model was trained on 75% of the IMDb dataset with a learning rate of 0.001 using the Adam optimizer. The loss function employed was Binary Cross-Entropy with Logits Loss. The training process ran for 10 epochs, resulting in a training accuracy of 91.35%. The model was then evaluated on the remaining 25% of the dataset, achieving a validation accuracy of 66.97%.

## Further Improvements
-Increase Model Complexity:
  Stack Multiple RNN Layers: Adding more RNN layers can help the model capture more complex patterns in the data.
  Use a More Powerful RNN Variant: Switching to LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit) layers can help the model remember important information over longer sequences.

-Optimize Hyperparameters
Learning Rate: Fine-tuning the learning rate can lead to better convergence. You can use a learning rate scheduler or conduct a grid search to find the optimal value.
Batch Size: Experiment with different batch sizes. Smaller batches might help the model generalize better, while larger batches can provide more stable gradients.
Number of Epochs: Training for more epochs could improve performance, but be cautious of overfitting. Early stopping can be employed to halt training when validation performance stops improving.

-  Data Augmentation and Preprocessing
Augment Data: Use techniques like synonym replacement, random insertion, or back-translation to generate more diverse training samples.
Use Pre-trained Word Embeddings: Leveraging pre-trained embeddings like GloVe or FastText can provide the model with a better starting point.
Text Cleaning: Ensure that the text data is properly preprocessed—removing stop words, handling negations, and normalizing text can improve the quality of the input data.

- Fine-tuning on Pretrained Models
Transfer Learning: Use a pre-trained model (e.g., BERT, GPT) and fine-tune it on the IMDb dataset. Pre-trained transformers often outperform RNNs on NLP tasks.


