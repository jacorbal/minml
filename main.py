#!/usr/bin/env python3
# vim: set ft=python fenc=utf-8 tw=72:

# MINML :: Minimal machine learning algorithms
# Copyright (c) 2019-2020, J. A. Corbal
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# “Software”), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from minml import MinAnn


def round_list(float_list):
    """Rounds a list of floats and returns a list of integers."""
    return [round(i) for i in float_list]


if __name__ == '__main__':
    # Topology: 4 inputs; 4 outputs
    # TODO(Optimize). Check best number of hidden layers (3 for now)
    net = MinAnn([4, 3, 4])

    # Example (random, as an example)
    pattern = [
        [[True,  False, False, 0.35], [False, 1, -1,  0]],
        [[True,  False, False, 0.00], [False, 1,  1,  1]],
        [[True,  False, False, 1.00], [False, 1,  1, -1]],
        [[False, False, False, 0.35], [True,  1, -1,  0]],
        [[False, False, False, 0.00], [True,  1,  1,  1]],
        [[False, False, False, 1.00], [True,  1,  1, -1]],
    ]

    # Train the previous values (`pattern`) 8000 times
    print("Training...")
    net.train(pattern, 8000)

    # Echo results
    print(round_list(net.feed_forward([1, 0, 0, 0.9])))
    print(round_list(net.feed_forward([0, 0, 0, 0.7])))
    print(round_list(net.feed_forward([1, 0, 0, 0.5])))
    print(round_list(net.feed_forward([0, 0, 0, 0.4])))
    print(round_list(net.feed_forward([1, 0, 0, 0.3])))
    print(round_list(net.feed_forward([0, 0, 0, 0.1])))
    print(round_list(net.feed_forward([1, 0, 0, 0.0])))
