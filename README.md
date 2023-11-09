# Deep-Learning

Over the last decade, so-called “deep learning” techniques have become very popular in various application domains such as computer vision, automatic speech recognition, natural language processing, and bio-informatics where they produce state-of-the-art results on various challenging tasks. A crucial success factor of deep neural networks is their ability to learn hierarchies of increasingly complex features from raw input data. Such representations often outperform traditionally hand-crafted features that require expensive human effort and expertise. But they do not come as a free lunch. Designing and training deep neural networks such that they actually learn meaningful and useful feature representations of data is an art itself. Mastering it requires practice and experience.

The programs in this repository corresponds to the work done throughout the curriculum "Introduction to Deep Learning" by Prof. Dr. Sebastian Stober at Otto-von-Guericke University, Magdeburg. These programs can be run on Jupyter notebook and uses Tensorflow and Keras for building, visualising and training deep neural networks.

## Deep MLP Model for MNIST dataset
MNIST is a collection of handwritten digits and a popular (now trivialized) benchmark for image classification models. Before building a deep neural network, a linear MLP was build to classify the MNIST images. This linear model was turned into a deep model by adding hidden layers and few experiments were performed by changing the hyperparameters such as - number of layers, number of units in a hidden layer, activation function, learning rate, etc.

## Visualisation and Datasets
Five simple MLP training scripts for MNIST were provided, which fail at training. The idea behind this exercise was to diagnose the underlying problem using visualisation. Tensorborad was used to visualise the computation graph of the model and the trainable variables. Along with the visualisation a way of handling datasets using tf.data was introduced. Tensorflow often uses the three operations shuffle, batch and repeat for handling of datasets. Some experiments were performed by changing the order of these operations and the most sensible order was recorded.

## Convolutional Neural Network

