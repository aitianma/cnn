# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classifies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network¡¯s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let¡¯s define this network:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


mnist=input_data.read_data_sets("data/",one_hot=True)

for i in range(55000):
	x=mnist.train.images[i]
	a=np.random.randn(32,32)
	a[2:30,2:30]=np.reshape(x,[28,28])
	input=torch.from_numpy(np.reshape(a,[1,1,32,32]))
	output=net(input.float())

	target=torch.from_numpy(np.reshape(mnist.train.labels[i],[1,10]))

	criterion = nn.MSELoss()
	loss = criterion(output, target.float())
	#print loss



	net.zero_grad()     # zeroes the gradient buffers of all parameters


	loss.backward()

		
	learning_rate = 0.01
	for f in net.parameters():
	    f.data.sub_(f.grad.data * learning_rate)


n=0

for i in range(1000):


	x=mnist.test.images[i]
        a=np.random.randn(32,32)
        a[2:30,2:30]=np.reshape(x,[28,28])
        input=torch.from_numpy(np.reshape(a,[1,1,32,32]))
        output=net(input.float())


	if output.argmax()==mnist.test.labels[i].argmax():
		n=n+1


print n






