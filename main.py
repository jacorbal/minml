#!/usr/bin/env python

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
