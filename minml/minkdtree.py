#!/usr/bin/env python3
# vim: set ft=python fenc=utf-8 tw=72:

# Heavily based on ``scpipy.spatial.KDTree``, this minimal KD-tree has
# only the minimum functions to make a query.
#
# The original ``scpipy.spatial.KDTree`` was released under the scipy
# license with Copyright Anne M. Archibald, 2008


# On why using ``scpipy.spatial`` instead of ``sklearn.neighbors``
# regarding ``KDTree`` structure:
#
# The following code uses returns a list of values regarding the
# primary emotion, secondary (if any) and third (if any) by using
# ``scipy.spatial``'s KDTree.
#
# The code changes in order to use ``sklearn`` module requires to
# change the line::
#
#   d, i = self._tree.query(query_point, k=k)
#
# to::
#
#   d, i = self._tree.query(np.array(query_point).reshape(1,-1), k=k)
#
# as well as lines #247 and #252 and #277: change ``i.tolist()`` and
# ``d.tolist()`` to ``i.tolist()[0]`` and ``d.tolist()[0]``; and also
# ignore the body of ``if isinstance(i, np.int32)``, since it has no
# effect on ``sklearn.neighbors.KDTree`` for it always returns an
# array.
#
# Basically, the return value of ``sklearn.neighbors.KDTree`` is
# always of type ``np.ndarray`` whilst in ``scipy.spatial.KDTree``
# depends: it could be a numeric value (integer or double), or an
# array of numeric values (integers or doubles).  This code is not yet
# optimized to use ``sklearn.neighbors``, so the result, even
# technically correct (more or less) gives just the primary emotion
# instead of a set of them. I haven't made the appropriate changes to
# the code in order to use ``sklearn.neighbors`` library to achieve
# the same output.


import numpy as np
from heapq import heappush, heappop


__all__ = ['minkowski_distance_p', 'MinKDTree']


class MinKDTree:
    """
    """
    def __init__(self, data, leafsize=10):
        self.data = np.asarray(data)
        self.n, self.m = np.shape(self.data)
        self.leafsize = int(leafsize)
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = np.amax(self.data, axis=0)
        self.mins = np.amin(self.data, axis=0)

        self.tree = self.__build(np.arange(self.n),
                                 self.maxes, self.mins)

    class node(object):
        def __lt__(self, other):
            return id(self) < id(other)

        def __gt__(self, other):
            return id(self) > id(other)

        def __le__(self, other):
            return id(self) <= id(other)

        def __ge__(self, other):
            return id(self) >= id(other)

        def __eq__(self, other):
            return id(self) == id(other)

    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(idx)

    class innernode(node):
        def __init__(self, split_dim, split, less, greater):
            self.split_dim = split_dim
            self.split = split
            self.less = less
            self.greater = greater
            self.children = less.children+greater.children

    def __build(self, idx, maxes, mins):
        if len(idx) <= self.leafsize:
            return MinKDTree.leafnode(idx)
        else:
            data = self.data[idx]
            # maxes = np.amax(data,axis=0)
            # mins = np.amin(data,axis=0)
            d = np.argmax(maxes-mins)
            maxval = maxes[d]
            minval = mins[d]
            if maxval == minval:
                # all points are identical; warn user?
                return MinKDTree.leafnode(idx)
            data = data[:, d]

            # sliding midpoint rule; see Maneewongvatana and Mount
            # 1999 for arguments that this is a good idea.
            split = (maxval+minval)/2
            less_idx = np.nonzero(data <= split)[0]
            greater_idx = np.nonzero(data > split)[0]
            if len(less_idx) == 0:
                split = np.amin(data)
                less_idx = np.nonzero(data <= split)[0]
                greater_idx = np.nonzero(data > split)[0]
            if len(greater_idx) == 0:
                split = np.amax(data)
                less_idx = np.nonzero(data < split)[0]
                greater_idx = np.nonzero(data >= split)[0]
            if len(less_idx) == 0:
                # _still_ zero? all must have the same value
                if not np.all(data == data[0]):
                    raise ValueError("Troublesome data array: %s" % data)
                split = data[0]
                less_idx = np.arange(len(data)-1)
                greater_idx = np.array([len(data)-1])

            lessmaxes = np.copy(maxes)
            lessmaxes[d] = split
            greatermins = np.copy(mins)
            greatermins[d] = split
            return MinKDTree.innernode(d, split,
                    self.__build(idx[less_idx], lessmaxes, mins),
                    self.__build(idx[greater_idx], maxes, greatermins))


    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        side_distances = np.maximum(0,
                                    np.maximum(x-self.maxes, self.mins-x))
        if p != np.inf:
            side_distances **= p
            min_distance = np.sum(side_distances)
        else:
            min_distance = np.amax(side_distances)

        # priority queue for chasing nodes
        # entries are:
        #  minimum distance between the cell and the target
        #  distances between the nearest side of the cell and the
        #  target the head node of the cell
        q = [(min_distance,
              tuple(side_distances),
              self.tree)]
        # priority queue for the nearest neighbors
        # furthest known neighbor first
        # entries are (-distance**p, i)
        neighbors = []

        if eps == 0:
            epsfac = 1
        elif p == np.inf:
            epsfac = 1/(1+eps)
        else:
            epsfac = 1/(1+eps)**p

        if p != np.inf and distance_upper_bound != np.inf:
            distance_upper_bound = distance_upper_bound**p

        while q:
            min_distance, side_distances, node = heappop(q)
            if isinstance(node, MinKDTree.leafnode):
                # brute-force
                data = self.data[node.idx]
                ds = minkowski_distance_p(data, x[np.newaxis, :], p)
                for i in range(len(ds)):
                    if ds[i] < distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-ds[i], node.idx[i]))
                        if len(neighbors) == k:
                            distance_upper_bound = -neighbors[0][0]
            else:
                # we don't push cells that are too far onto the queue
                # at all, but since the distance_upper_bound
                # decreases, we might get here even if the cell's too
                # far
                if min_distance > distance_upper_bound*epsfac:
                    # since this is the nearest cell, we're done, bail out
                    break
                # compute minimum distances to the children and push
                # them on
                if x[node.split_dim] < node.split:
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less

                # near child is at the same distance as the current node
                heappush(q, (min_distance, side_distances, near))

                # far child is further by an amount depending only on
                # the split value
                sd = list(side_distances)
                if p == np.inf:
                    min_distance = max(min_distance,
                                       abs(node.split-x[node.split_dim]))
                elif p == 1:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])
                    min_distance = min_distance - \
                        side_distances[node.split_dim] + \
                        sd[node.split_dim]
                else:
                    sd[node.split_dim] = \
                        np.abs(node.split-x[node.split_dim])**p
                    min_distance = min_distance - \
                        side_distances[node.split_dim] + \
                        sd[node.split_dim]

                # far child might be too far, if so, don't bother
                # pushing it
                if min_distance <= distance_upper_bound*epsfac:
                    heappush(q, (min_distance, tuple(sd), far))

        if p == np.inf:
            return sorted([(-d, i) for (d, i) in neighbors])
        else:
            return sorted([((-d)**(1./p), i) for (d, i) in neighbors])

    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        """Query the kd-tree for nearest neighbors

        :param x: An array of points to query.
        :type x: np.array (array_like, last dimension ``self.m``)
        :param k: The number of nearest neighbors to return.
        :type k: int
        :param eps:Return approximate nearest neighbors; the ``k``th
                    returned value is guaranteed to be no further than
                    ``(1 + eps)`` times the distance to the real
                    ``k``th nearest neighbor.
        :type esp: float (non-negative)
        :param p: Which Minkowski ``p-norm`` to use.
                      * `1`` is the sum-of-absolute-values "Manhattan"
                        distance
                      * ``2`` is the usual Euclidean distance
                      * ``infinity`` is the maximum-coordinate-difference
                        distance
        :type p: float (1 <= ``p`` <= infinity)
        :param distance_upper_bound: Return only neighbors within this
                    distance. This is used to prune tree searches, so
                    if you are doing a series of nearest-neighbor
                    queries, it may help to supply the distance to the
                    nearest neighbor of the most recent point.
        :type distance_upper_bound: float (non-negative)

        :return: Tuple composed of two arrays. The first holds the
                    distances to the nearest neighbors. If ``x`` has
                    shape ``tuple+(self.m,)``, then d has shape tuple
                    if ``k`` is one, or ``tuple+(k,)`` if ``k`` is
                    larger than one. Missing neighbors are indicated
                    with infinite distances. If ``k`` is ``None``,
                    then ``d`` is an object array of shape tuple,
                    containing lists of distances. In either case the
                    hits are sorted by distance (nearest first).
                    The second array of the tuple, holds the locations
                    of the neighbors in ``self.data``.  ``i`` is the
                    same shape as ``d``.

        :rtype: tuple ((np.array of floats, np.array of integers))
        """
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
            raise ValueError("x must consist of vectors of length %d \
            but has shape %s" % (self.m, np.shape(x)))
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape != ():
            if k is None:
                dd = np.empty(retshape, dtype=np.object)
                ii = np.empty(retshape, dtype=np.object)
            elif k > 1:
                dd = np.empty(retshape+(k,), dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape+(k,), dtype=np.int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape, dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; \
                acceptable numbers are integers greater than or equal \
                to one, or None")
            for c in np.ndindex(retshape):
                hits = self.__query(x[c], k=k, eps=eps, p=p,
                                    distance_upper_bound=distance_upper_bound)
                if k is None:
                    dd[c] = [d for (d, i) in hits]
                    ii[c] = [i for (d, i) in hits]
                elif k > 1:
                    for j in range(len(hits)):
                        dd[c+(j,)], ii[c+(j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(x, k=k, eps=eps, p=p,
                                distance_upper_bound=distance_upper_bound)
            if k is None:
                return [d for (d, i) in hits], [i for (d, i) in hits]
            elif k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(k, dtype=np.int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; \
                acceptable numbers are integers greater than or equal \
                to one, or None")


def minkowski_distance_p(x, y, p=2):
    """Compute the ``p``-th power of the ``L**p`` distance between two
    arrays.

    For efficiency, this function computes the ``L**p`` distance but
    does not extract the ``p``th root. If ``p`` is 1 or infinity, this
    is equal to the actual ``L**p`` distance.

    :param x: Input array
    :type x: np.arary ((M, K) array like)
    :param y: Input array
    :type y: np.arary ((N, K) array like)
    :param p: Which Minkowski p-norm to use
    :type p: float (1 <= ``p`` <= infinity)

    Example::

    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)
