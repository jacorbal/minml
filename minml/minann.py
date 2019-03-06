#!/usr/bin/env python
# vim: set ft=python fenc=utf-8 tw=72:

# MINML :: Minimal machine learning algorithms
# Copyright (c) 2019, J. A. Corbal
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Minimal artificial neural network with backpropagation.

In machine learning and cognitive science, artificial neural networks
(ANNs) are a family of models inspired by biological neural networks
(the central nervous systems of animals, in particular the brain) and
are used to estimate or approximate functions that can depend on a
large number of inputs and are generally unknown. Artificial neural
networks are generally presented as systems of interconnected
"neurons" which exchange messages between each other. The connections
have numeric weights that can be tuned based on experience, making
neural nets adaptive to inputs and capable of learning.

Backpropagation, an abbreviation for "backward propagation of errors",
is a common method of training artificial neural networks used in
conjunction with an optimization method such as gradient descent. The
method calculates the gradient of a loss function with respect to all
the weights in the network. The gradient is fed to the optimization
method which in turn uses it to update the weights, in an attempt to
minimize the loss function.

.. note:: This module is based on Toby Segaran's model published in
          *Programming Collective Intelligence*, O'Reilly, 2007.
"""

import math
import random


__all__ = ['sigmoid', 'dsigmoid', 'MinAnn']


class MinAnn(object):
    """Instantiates an neural network object ready to be trained.

    Example for a NN that solves an ``AND`` gate::

        >>> topology = [2, 4, 1]
        >>> p = [ [[0,0],[0]], [[0,1],[0]], [[1,0],[0]], [[1,1],[1]]]
        >>> net = MinAnn(topology)
        >>> net.train(p, max_iterations=4000, verbose=False, test=True)
        Inputs: [0,0] -> [-7.255498785979596e-05]  Target: [0]
        Inputs: [0,1] -> [-6.731854924635882e-05]  Target: [0]
        Inputs: [1,0] -> [1.1761479715612379e-05]  Target: [0]
        Inputs: [1,1] -> [0.9941687241240065]      Target: [1]

        >>> test = [1,0]
        >>> print("Testing", test, "->", net.feed_forward(test))
        Testing [1,0] -> [1.1761479715612379e-05]
    """
    def __init__(self, topology):
        """Default constructor. Sets up the network.

        :param topology: Topology of the net in the form of a list:
                         [inputs_num, hidden_layers_num, outputs_num]
        :type topology: list
        """
        # Number of nodes in layers
        self._ni = topology[0] + 1  # "+1" for bias
        self._nh = topology[1]
        self._no = topology[2]

        # Initialize node-activations
        self._ai, self._ah, self._ao = [], [], []
        self._ai = [1.0] * self._ni
        self._ah = [1.0] * self._nh
        self._ao = [1.0] * self._no

        # Create node weight matrices
        self._wi = make_matrix(self._ni, self._nh)
        self._wo = make_matrix(self._nh, self._no)

        # Initialize node weights to random values
        randomize_matrix(self._wi, -0.2, 0.2)
        randomize_matrix(self._wo, -2.0, 2.0)

        # Create last change in weights matrices for momentum
        self._ci = make_matrix(self._ni, self._nh)
        self._co = make_matrix(self._nh, self._no)

    def feed_forward(self, inputs):
        """Takes a list of inputs, pushes them through the network,
        and returns the output of all nodes in the output layer.

        :param inputs: Feeding forward values
        :type inputs: list
        :return: Output values
        :rtype: list
        """
        if (len(inputs) != self._ni - 1):
            raise ValueError("Incorrect number of inputs.")

        for i in range(self._ni - 1):
            self._ai[i] = inputs[i]

        for j in range(self._nh):
            sum = 0.0
            for i in range(self._ni):
                sum += (self._ai[i] * self._wi[i][j])
            self._ah[j] = sigmoid(sum)

        for k in range(self._no):
            sum = 0.0
            for j in range(self._nh):
                sum += (self._ah[j] * self._wo[j][k])
            self._ao[k] = sigmoid(sum)

        return self._ao

    def back_propagate(self, targets, N, M):
        """Executes the backpropagation algorithm updating the weights
        using target values.

        Calculates the error in advance and then adjusts the weights,
        because all the calculations rely on knowing the current
        weights rather than the updated weights.

        :param targets: List of target values
        :type targets: list
        :param N: Overall learning rate (*eta*)
        :type N: float
        :param M: Momentum, multiplier of last weight change (*alpha*)
        :type M: float
        """
        # Calc output deltas
        # We want to find the instantaneous rate of change of(error
        # with respect to weight from node j to node k) output_delta is
        # defined as an attribute of each ouput node. It is not the
        # final rate we need.
        # To get the final rate we must multiply the delta by the
        # activation of the hidden layer node in question. This
        # multiplication is done according to the chain rule as we are
        # taking the derivative of the activation function
        # of the ouput node.
        # dE/dw[j][k] =(t[k] - ao[k]) * s'(SUM(w[j][k]*ah[j])) * ah[j]
        output_deltas = [0.0] * self._no
        for k in range(self._no):
            error = targets[k] - self._ao[k]
            output_deltas[k] = error * dsigmoid(self._ao[k])

        # Update output weights
        for j in range(self._nh):
            for k in range(self._no):
                # Output_deltas[k] * self._ah[j]
                # is the full derivative of dError/dweight[j][k]
                change = output_deltas[k] * self._ah[j]
                self._wo[j][k] += N * change + M * self._co[j][k]
                self._co[j][k] = change

        # Calculate hidden deltas
        hidden_deltas = [0.0] * self._nh
        for j in range(self._nh):
            error = 0.0
            for k in range(self._no):
                error += output_deltas[k] * self._wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self._ah[j])

        # Update input weights
        for i in range(self._ni):
            for j in range(self._nh):
                change = hidden_deltas[j] * self._ai[i]
#                print('Activation:',self._ai[i],'Synapse:',i,j,'Change:',change)
                self._wi[i][j] += N * change + M * self._ci[i][j]
                self._ci[i][j] = change

        # Calculate combined error
        # 1/2 for differential convenience & **2 for modulus
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self._ao[k])**2
        return error

    def test(self, patterns):
        """Test the net with a new set of inputs and target values and
        outputs the result values.

        :param patterns: List of inputs and target values
        :type patterns: list
        """
        for p in patterns:
            inputs = p[0]
            print("Inputs:", p[0], "->", self.feed_forward(inputs), " \t",
                  "Target:", p[1])

    def train(self, patterns, max_iterations=1000, N=0.5, M=0.1,
              verbose=False, test=False):
        """Train the net with a new set of inputs and target values.

        :param patterns: List of inputs and target values
        :type patterns: list
        :param max_iterations: Iterations for the same pattern
        :type max_iterations: int
        :param N: Overall learning rate (*eta*)
        :type N: float
        :param M: Momentum, multiplier of last weight change (*alpha*)
        :type M: float
        :param verbose: If ``True`` outputs the combined error on each
                        iteration
        :type verbose: bool
        :param test: If ``True`` outputs a test after training
        :type test: bool
        """
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feed_forward(inputs)
                error = self.back_propagate(targets, N, M)
            if verbose and i % 50 == 0:
                print("Combined error", error)
        if test:
            self.test(patterns)

    def get_weights(self):
        """Return weights in the format: [Input weights, Output
        weights] where both Input weights and Output weights are
        lists.
        """
        weights = [[], []]
        for i in range(self._ni):
            weights[0].append(self._wi[i])
        for j in range(self._nh):
            weights[1].append(self._wo[j][0])

        return weights

    # Properties
    weights = property(get_weights, None)

def sigmoid(x):
    """A sigmoid function is a bounded differentiable real function
    that is defined for all real input values and has a positive
    derivative at each point.

    A sigmoid function is a mathematical function having an "S" shape
    (sigmoid curve). Often, sigmoid function refers to the special
    case of the logistic function shown in the first figure and
    defined by the formula: :math:`s(x) = 1 / (1 + exp(-x))`, but also
    :math:`tanh(x)=(exp(x) - exp(-x)) / (exp(x) + exp(-x))` is a
    sigmoid.

    Generally speaking, :math:`s(x)` is used for values with a
    non-negative domain [0,1],  while :math:`tanh(x)` is in the range
    [-1,1].
    """
    return math.tanh(x)

def dsigmoid(y):
    """Derivative function of the function represented in
    :py:func:`sigmoid`.

      * If :math:`y = tanh(x)`, then :math:`Dy = 1 - y^2`,
      * if :math:`y = s(x)`, then :math:`Ds = y - y^2`.
      * There are infinite sigmoid functions. Just put here the
        derivative of the ``sigmoid`` function.
    """
    return 1 - y**2

def make_matrix(rows, cols, fill=0.0):
    """Returns a matrix (list of list of floats) using a default
    value.

    :param rows: Number of rows
    :type rows: int
    :param cols: Number of columns
    :type cols: int
    :param fill: Default value for each element in the matrix
    :type fill: float
    """
    m = []
    for i in range(rows):
        m.append([fill] * cols)
    return m

def randomize_matrix(matrix, a, b):
    """Randomizes the values of a matrix in the range [a,b].

    :param matrix: Matrix to randomize
    :type matrix: list
    :param a: Start-point value
    :type a: float
    :param b: End-point value
    :type b: float

    .. note:: The end-point value ``b`` may or may not be included in
              the range depending on floating-point rounding in the
              equation ``a + (b-a) * random()``.
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)
