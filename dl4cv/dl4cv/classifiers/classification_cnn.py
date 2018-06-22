"""ClassificationCNN"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    chanels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################

        # compute number of padded zeros
        pad = int((kernel_size-1) / 2)
        # compute height and width of input to first fully connected layer
        height_fc1 = int((1 + (height + 2*pad - kernel_size)/stride_conv) / pool)
        width_fc1 = int((1 + (width + 2*pad - kernel_size)/stride_conv) / pool)


        # initialize layers
        #-------------------

        # Convolutional layer
        self.conv = nn.Conv2d(channels, num_filters, kernel_size, stride_conv, pad)
        #nn.init.xavier_uniform(self.conv.weight.data)
        self.conv.weight.data.mul_(weight_scale)

        # Fully Connected Layers
        self.fc1 = nn.Linear(num_filters * height_fc1 * width_fc1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Other Layers
        self.dropout = nn.Dropout(p=dropout)
        self.maxpool = nn.MaxPool2d(pool, stride=stride_pool)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################

        #1 Convolutional Layer
        x = self.conv(x)

        #2 ReLU Layer
        x = F.relu(x)

        #3 Max Pool Layer
        x = self.maxpool(x)

        #4 Fully Connected Layer
        x = x.view(x.size()[0], -1)     # flatten data for fc
        x = self.fc1(x)

        #5 Dropout Layer
        x = self.dropout(x)

        #6 ReLu Layer
        x = F.relu(x)

        #7 Fully Connected Layer
        x = self.fc2(x)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
