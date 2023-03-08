# Time-Series-Air-Quality-Index-Prediction-and-Analysis-using-Multiple-Deep-Learning-Models

This repository contains code for predicting and analyzing time series data for air quality index using multiple deep learning models. The repository currently contains three different models namely LSTM, BiLSTM, and BiLSTMConv1D.

###**Models**

* LSTM: Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture, which is capable of capturing long-term dependencies in time series data.

* BiLSTM: Bidirectional LSTM (BiLSTM) is an extension of the LSTM architecture that processes the input sequence in both forward and backward directions.

* BiLSTMConv1D: Bidirectional LSTM with 1D convolutional layers (BiLSTMConv1D) is a combination of BiLSTM and 1D convolutional layers. This model is effective in capturing both local and global temporal patterns in time series data.

###**Coming Soon**

* Support vector regression: Support vector regression (SVR) is a powerful machine learning technique used for regression tasks. It will be added to the repository shortly.

* GRU: Gated Recurrent Unit (GRU) is another type of RNN architecture that is similar to LSTM but has fewer parameters. It will also be added to the repository shortly.

###**Dataset**

The air quality index dataset used in this repository is publicly available on Kaggle. The dataset contains hourly air pollution data from 12 different stations in the city of Beijing, China, from January 1st, 2010 to December 31st, 2014.

###**Usage**

To use the code in this repository, clone the repository and run the Jupyter notebook files (.ipynb) in Google Colab or any other Python environment that supports Jupyter notebooks. The code is organized by model, with each model in a separate notebook.

###**Requirements**

The code in this repository requires the following Python packages:

* Tensorflow 2.x
* Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
