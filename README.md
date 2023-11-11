# Deep-Learning

Over the last decade, so-called “deep learning” techniques have become very popular in various application domains such as computer vision, automatic speech recognition, natural language processing, and bio-informatics where they produce state-of-the-art results on various challenging tasks. A crucial success factor of deep neural networks is their ability to learn hierarchies of increasingly complex features from raw input data. Such representations often outperform traditionally hand-crafted features that require expensive human effort and expertise. But they do not come as a free lunch. Designing and training deep neural networks such that they actually learn meaningful and useful feature representations of data is an art itself. Mastering it requires practice and experience.

The programs in this repository corresponds to the work done throughout the curriculum "Introduction to Deep Learning" by Prof. Dr. Sebastian Stober at Otto-von-Guericke University, Magdeburg. These programs can be run on Jupyter notebook and uses Tensorflow and Keras for building, visualising and training deep neural networks.

## Deep MLP Model for MNIST dataset
MNIST is a collection of handwritten digits and a popular (now trivialized) benchmark for image classification models. Before building a deep neural network, a linear MLP was build to classify the MNIST images. This linear model was turned into a [deep model](https://github.com/nishad-pawaskar/Deep-Learning/tree/d603ad441c0e94befffdb7f4c5d29753e875e9d9/MNIST_Deep_Model) by adding hidden layers and few experiments were performed by changing the hyperparameters such as - number of layers, number of units in a hidden layer, activation function, learning rate, etc.

## Visualisation and Datasets
Five simple MLP training scripts for MNIST were provided, which fail at training. The idea behind this exercise was to diagnose the underlying problem using visualisation. Tensorborad was used to visualise the computation graph of the model and the trainable variables. Along with the visualisation a way of handling datasets using tf.data was introduced. Tensorflow often uses the three operations shuffle, batch and repeat for handling of datasets. Some experiments were performed by changing the order of these operations and the most sensible order was recorded. The corresponding program is given in [Visualisation and Datasets](https://github.com/nishad-pawaskar/Deep-Learning/tree/10a4b1c8b8f3435c03462ed02acc403c2e71b429/Visualisation%20and%20Datasets). 

## Convolutional Neural Network

Convolutional networks, also known as Convolutional Neural Networks, or CNNs, are a specialized kind of neural network for processing data that has a known grid-like topology. Convolutional networks have been tremendously successful in practical applications. The name “convolutional neural network” indicates that the network employs a mathematical operation called convolution. Convolution is a specialized kind of linear operation. Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers. The basic MLP programmed in [Deep MLP Model](https://github.com/nishad-pawaskar/Deep-Learning/tree/d603ad441c0e94befffdb7f4c5d29753e875e9d9/MNIST_Deep_Model) was modified by adding convolutional layers before the dense MLP layer for classification. These convolutional layers help in extracting and learning features using the convolutional kernel from the dataset. [Convolutional Neural Network (CNN)](https://github.com/nishad-pawaskar/Deep-Learning/tree/d603ad441c0e94befffdb7f4c5d29753e875e9d9/Convolutional%20Nueral%20Networks) trains the CNN model for classifying the FashionMNIST and CIFAR10 dataset.

## DenseNet

convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. The Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. To build a DenseNet, a convolutional block is created, the input and output of which is concatenated to give the output of the corresponding layer, thus forming a [Dense Convolutional Network](https://github.com/nishad-pawaskar/Deep-Learning/tree/9dc3fa9b2620eafbb0d11112af11008859d2f0ba/DenseNets). The program also includes a graph-based execution of the conventional CNN using tf.function. 

## Residual Network

When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error. The degradation problem was addressed by
introducing a deep residual learning framework. Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping. Formally, denoting the desired underlying mapping as H(x), we let the stacked non-linear layers fit another mapping of F(x) := H(x)−x. The original mapping is thus recast into F(x)+x. The [Residual Network]() implements a ResNet18 architecture consisting of 18 residual convolutional layers.


## Recurrent Neural Networks

Recurrent neural networks, or RNNs, are a family of neural networks for processing sequential data. Much as a convolutional network is a neural network that is specialized for processing a grid of values X such as an image, a recurrent neural network is a neural network that is specialized for processing a sequence of values x(1), . . . , x(τ). Recurrent networks can scale to much longer sequences than would be practical for networks without sequence-based specialization. Most recurrent networks can also process sequences of variable length. RNNs can be seen as the basic building blocks of language models. The RNNs are also known to suffer from the problems of vanishing gradients, exploding gradients and long-term dependencies. Hence various methods have been developed such as LSTM, GRU, FastRNN, FastGRNN, etc. In the current language models, attention mechanism is added to enhance the ability of the model to focus on a certain region, semantics, etc. A simple RNN implementation is illustrated in [Rcurrent Neural Networks](https://github.com/nishad-pawaskar/Deep-Learning/tree/f18592dd22971e15ffeb9268bc7f449c9d4023d7/Recurrent%20Neural%20Networks) whereas Language models with GRU is illustrated in [RNN with variable length sequences](https://github.com/nishad-pawaskar/Deep-Learning/tree/f18592dd22971e15ffeb9268bc7f449c9d4023d7/RNN_variableLength_Sequence).

## Autoencoders

Autoencoders are an unsupervised learning technique in which we leverage neural networks for the task of representation learning. Specifically, we'll design a neural network architecture such that we impose a bottleneck in the network which forces a compressed knowledge representation of the original input. In other words, an autoencoder is a neural network that is trained to attempt to copy its input to its output. The network consists of two parts - an encoder function and a decoder function that produces a reconstruction. In order to build an autoencoder, you need to define an encoding based on the input, a decoding based on the encoding, and a loss function that measures the distance between decoding and input. Note that you are in no way obligated to choose the “reverse” encoder architecture for your decoder; i.e. you could use a 10-layer MLP as an encoder and a single layer as a decoder if you wanted, or the filter sizes need not be same in case of convolutional autoencoder. The [autoencoder](https://github.com/nishad-pawaskar/Deep-Learning/tree/b14a752bd8dfe258d852dfc796223896c00f4327/Autoencoders) program consists of a Convolutional autoencoder which works with image data, and unsupervised pretraining approach such as - 
- Train autoencoder – freeze encoder – train classifier on top.
- Train autoencoder – train classifier on top of encoder. Do not freeze the encoder, i.e. the encoder is “fine-tuned” on the labeled subset of data as well.
- Train a classifier directly on the labeled subset; no pretraining. For fairness, it should have the same architecture as the encoder + classifier above.

## Data Validation

The dataset archive contains triplets of training, validation and test sets. For each, a model is trained on the training set, making sure it “works” by checking its performance on the validation set. The models were then fine-tuned to achieve as good of a validation set performance as possible, and the performance on the test set was checked whether the performance on the test set was close to that on the validation set. For each triplet, through inspection or computing statistics of the dataset it was found out what was going wrong. Typical things to look out for which decreases the test accuracy include:

- Do the train, validation and test sets follow the same distribution?
- Are the different subsets disjunct?
- Are the sets balanced?
- Were the sets processed in the same way?

The corresponding program is given in [Data Validation](https://github.com/nishad-pawaskar/Deep-Learning/tree/0113b8072dbcd6c10a042dbb6a771e406272af3a/Data%20Validation). 

## Adversarial Examples

A central problem in machine learning is how to make an algorithm that will perform well not just on the training data, but also on new inputs. Many strategies used in machine 
learning are explicitly designed to reduce the test error, possibly at the expense of increased training error. These strategies are known collectively as regularization. In 
[Adversarial examples](https://github.com/nishad-pawaskar/Deep-Learning/tree/eae23be236d1cb9fe98639ca812510e986980cda/Adversarial%20Examples) we will explore the phenomenon of adversarial examples and how to defend against them. The adversarial examples are split into two parts - Creating 
Adversarial examples and Adversarial training. In order to create adversarial examples, the input is changed such that the loss is increased. This type of adversarial phenomenon is 
known as "_untargeted attack_" where we simply increase the loss, no matter what. Another type is the "_targeted attack_" where the goal is to make the network misclassify an input in a specific way – e.g. classify every MNIST digit as a 3. Adding the gradients to the inputs will give a batch of adversarial examples. These gradients are multiplied with a small constant such that the inputs aren't changed too much. 

In order to defend against such adversarial attacks, a defense method known as "_Adversarial Training_" is employed, where the models are explicitly trained to classify the adversarial examples correctly. This is done by integrating adversarial examples in the training batch at each step.