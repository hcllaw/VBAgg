# Code based on Dougal Sutherland
'''
Some utilities for handling bag features taken from skl-groups.
'''
from __future__ import division, print_function

from copy import deepcopy
import warnings

import numpy as np
from sklearn.externals.six import iteritems, string_types
from sklearn.externals.six.moves import xrange


################################################################################
### Type utilities for numpy

def is_integer_type(x):
    "Checks whether the array is of an integral type."
    return issubclass(np.asanyarray(x).dtype.type, np.integer)


def is_categorical_type(ary):
    "Checks whether the array is either integral or boolean."
    ary = np.asanyarray(ary)
    return is_integer_type(ary) or ary.dtype.kind == 'b'


def is_integer(x):
    "Checks whether the argument is a single integer."
    return np.isscalar(x) and is_integer_type(x)


def is_categorical(x):
    "Checks whether the argument is a single integer or boolean."
    return np.isscalar(x) and is_categorical_type(x)


def as_integer_type(ary):
    '''
    Returns argument as an integer array, converting floats if convertable.
    Raises ValueError if it's a float array with nonintegral values.
    '''
    ary = np.asanyarray(ary)
    if is_integer_type(ary):
        return ary
    rounded = np.rint(ary)
    if np.any(rounded != ary):
        raise ValueError("argument array must contain only integers")
    return rounded.astype(int)


################################################################################
### Main features class

class Mal_features(object):
    '''
    A wrapper class for storing bags of features for bag data (A *bag* is a set of feature
    vectors corresponding to a single "object.")

    Supports storing data in two major ways:

    - As a list of pointers to a numpy array per bag. This is the default,
      because it usually doesn't require copying all of your data. Note that
      the sub-arrays are not enforced to be row-major or even contiguous.

    - As a single big row-major array. This lets you do certain things more
      easily (e.g. run PCA).

    The main usage API is the same for both versions; you can distinguish them
    with the `stacked` property, and convert from pointers to stacked with the
    `make_stacked()` method.

    Supports the following operations:

    * ``len(features)`` gives the number of bags.
    * ``for bag in features:`` loops over bags.
    * ``features[4]`` gives the fifth bag.
    * ``features[[4, 8, 7]]`` makes a new Mal_features object with only the \
       passed indices, preserving metadata.
    * ``feats1 == feats2`` checks that all of the features `and metadata` are \
      the same.
    * ``feats1 + feats2`` concatenates the two Mal_features objects. Metadata is \
      preserved if both features have that key, thrown out if not.


    Parameters
    ----------
    bags : list of numpy arrays, single array, or Mal_features object
        The feature data. If a list of numpy arrays, should be one array per 
        bag, each of shape [n_pts, dim], where dim is consistent between bags
        but n_pts need not be (though it cannot ever be 0). If a single numpy
        array, it should be of shape [sum(n_pts), dim] and contain the features
        from the first bag, then the next bag, .... In this case you must also
        pass n_pts. If a Mal_features object, "copies" it (but only actually copies
        any data if ``copy=True``).

    n_pts : array-like of positive integers, only if bags is a single array
        If bags is passed as a single array, a list of positive integers
        defining the size of each bag.

    stack : boolean, optional, default False
        If true, stack the features. Otherwise, only stack them if a stacked
        array of features is passed in.

    copy : boolean, optional, default False
        If true, always make a copy of the data (so that direct modifications)
        don't modify the original arrays. If false, make a copy only if
        necessary (i.e. stack=True for an unstacked argument).

    bare : boolean, optional, default False
        If true, and ``bags`` is a Mal_features instance, don't include its
        metadata.

    any other keyword argument : array-like with first dimension num_bags
        Metadata for each bag. Just stored along with the features, nothing
        in particular done with it. (If ``bags`` is a Mal_features instance, its
        metadata is merged with any keyword arguments, with keywords taking
        precedence.)

    Attributes
    ----------
    features : list of arrays of shape ``[n_pts[i], dim]``
        A list of the contained features.
        If ``stacked``, each array is a slice of ``stacked_features``.

    stacked : boolean
        Whether the features are stacked.

    stacked_features : array of shape ``[sum(n_pts), dim]``
        All of the features, concatenated together. Only present if ``stacked``.

    n_pts : integer array of shape ``[len(self)]``
        The number of points in each bag.

    meta : dictionary mapping strings to arrays of shape ``[len(self)]``
        The stored metadata. ``meta['foo']`` is also accessible as ``self.foo``.
    '''

    def __init__(self, bags, pop=False, indiv=False, true_indiv=False, n_pts=None, stack=False, copy=False, bare=False, **meta):
        if isinstance(bags, Mal_features):
            if n_pts is not None:
                raise TypeError("can't pass n_pts if copying a Mal_features object")
            raise NotImplementedError
            oth = bags
            if oth.stacked:
                bags = oth.stacked_features
                n_pts = oth.n_pts
            else:
                bags = oth.features
                n_pts = None
            if not bare:
                for k, v in iteritems(oth.meta):
                    meta.setdefault(k, v)

        if isinstance(bags, np.ndarray) and bags.ndim == 2:
            raise NotImplementedError
            if n_pts is None:
                raise TypeError("must pass n_pts if passing stacked array of "
                                "features")

            n_pts = np.asarray(n_pts)
            if n_pts.ndim != 1:
                raise TypeError("n_pts must be 1-dimensional")
            if n_pts.size == 0:
                raise TypeError("must have at least one bag")
            if np.any(n_pts <= 0):
                raise TypeError("n_pts must all be positive")
            try:
                n_pts = as_integer_type(n_pts)
            except ValueError:
                raise TypeError("n_pts must be an array of integers.")

            bags = np.array(bags, order='C', copy=copy)
            if bags.ndim != 2 or bags.shape[0] != np.sum(n_pts):
                raise TypeError("bags must have shape sum(n_pts) x dim")
            if bags.shape[1] == 0:
                raise TypeError("bags must have dimension > 0")
            dim = bags.shape[1]
            self.stacked = True
            self.n_pts = n_pts
            self.stacked_features = bags
            self._boundaries = bounds = np.r_[0, np.cumsum(n_pts)]

            self.features = np.empty(len(n_pts), object)
            self.features[:] = [bags[bounds[i-1]:bounds[i]]
                                for i in xrange(1, len(bounds))]
        else:
            if n_pts is not None:
                raise TypeError("n_pts should only be passed if bags is a "
                                "single stacked array")

            dim = None
            dtype = None
            new_bags = np.empty(len(bags), dtype=object)
            n_pts = np.empty(len(bags), dtype=int)
            for i, bag in enumerate(bags):
                a = np.array(bag, copy=copy)

                if a.ndim == 1:
                    a = a[None, :]
                if a.ndim != 2:
                    raise TypeError("bag {} not two-dimensional".format(i))

                if dim is None:
                    dim = a.shape[1]
                elif a.shape[1] != dim:
                    msg = "bags' second dimension must be consistent: " \
                          "{} is {}, expected {}"
                    raise TypeError(msg.format(i, a.shape[1], dim))

                if dtype is None:
                    dtype = a.dtype
                    if dtype.kind not in 'fiu':
                        msg = "can't handle features of type {}"
                        raise TypeError(msg.format(a.dtype.name))
                elif a.dtype != dtype:
                    msg = "bags' dtype is inconsistent: {} is {}, expected {}"
                    raise TypeError(msg.format(i, a.dtype.name, dtype.name))

                if a.shape[0] == 0:
                    raise TypeError("bag {} has no points".format(i))

                new_bags[i] = a
                n_pts[i] = a.shape[0]

            self.stacked = False
            self.n_pts = n_pts
            self.features = new_bags
            try:
                del self._boundaries
            except AttributeError:
                pass
            try:
                del self.stacked_features
            except AttributeError:
                pass

            if stack:
                self.make_stacked()

        # handle metadata
        self.pop = pop
        self.indiv = indiv
        self.true_indiv = true_indiv
        self.meta = {}
        for name, val in iteritems(meta):
            self.set_meta(name, val, copy=copy)

    def make_stacked(self):
        "If unstacked, convert to stacked. If stacked, do nothing."
        if self.stacked:
            return

        self._boundaries = bounds = np.r_[0, np.cumsum(self.n_pts)]
        self.stacked_features = stacked = np.vstack(self.features)
        self.features = np.empty(len(bounds) - 1, dtype=object)
        self.features[:] = [
            stacked[bounds[i-1]:bounds[i]] for i in xrange(1, len(bounds))
        ]
        self.stacked = True

    def set_meta(self, name, val, copy=False):
        if len(val) != len(self):
            msg = "Have {} bags but {} values for {}"
            raise ValueError(msg.format(len(self), len(val), name))

        val = np.array(val, copy=copy)
        self.meta[name] = val
        if hasattr(self, name):
            msg = "Mal_features already has an attribute named '{}'; won't " \
                  "be accessible as an attribute"
            warnings.warn(msg.format(name))
        else:
            setattr(self, name, val)

    ############################################################################
    ## Properties to get at basic metadata

    @property
    def total_points(self):
        "The total number of points in all bags."
        return self.n_pts.sum()

    @property
    def dim(self):
        "The dimensionality of the features."
        return self.features[0].shape[1]

    @property
    def dtype(self):
        "The data type of the feature vectors."
        return self.features[0].dtype

    ############################################################################
    ## Copying / pickling utilities

    def copy(self, stack=False, copy_meta=False, memo=None):
        '''
        Copies the Feature object. Makes a copy of the features array.

        Parameters
        ----------
        stack : boolean, optional, default False
            Whether to stack the copy if this one is unstacked.

        copy_meta : boolean, optional, default False
            Also copy the metadata. If False, metadata in both points to the
            same object.
        '''
        if self.stacked:
            fs = deepcopy(self.stacked_features, memo)
            n_pts = self.n_pts.copy()
        elif stack:
            fs = np.vstack(self.features)
            n_pts = self.n_pts.copy()
        else:
            fs = deepcopy(self.features, memo)
            n_pts = None

        meta = deepcopy(self.meta, memo) if copy_meta else self.meta
        return Mal_features(fs, n_pts, copy=False, **meta)

    def __copy__(self):
        return self.copy(stack=False, copy_meta=False)

    def __deepcopy__(self, memo=None):
        return self.copy(stack=False, copy_meta=True, memo=memo)

    def __getstate__(self):
        if self.stacked:
            return (self.stacked_features, self.n_pts, self.meta)
        else:
            return (self.features, None, self.meta)

    def __setstate__(self, state):
        feats, n_pts, meta = state
        self.__init__(feats, n_pts, **meta)

    ############################################################################ 
    ## General magic methods for basic behavior

    __hash__ = None

    def __eq__(self, oth):
        if self is oth:
            return True
        elif isinstance(oth, Mal_features):
            return (len(self) == len(oth) and
                    set(self.meta) == set(oth.meta) and
                    all(np.all(self_b == oth_b)
                        for self_b, oth_b in zip(self, oth)) and
                    all(np.all(self.meta[k] == oth.meta[k])
                        for k in self.meta))
        elif self.meta:
            return False
        else:
            return (len(self) == len(oth) and 
                    all(np.all(self_b == oth_b)
                        for self_b, oth_b in zip(self, oth)))

    def __ne__(self, oth):
        return not (self == oth)

    def __repr__(self):
        true_dim = self.dim
        if self.pop:
            pop_mess = 'population'
            true_dim = true_dim - 1
        else:
            pop_mess = 'None'
        if self.indiv:
            indiv_mess = 'individual'
            true_dim = true_dim - 1
        else:
            indiv_mess = 'None'
        s = '<Mal_features: {:,} bags with {} {}-dimensional points ({:,} total)>'
        min_p = self.n_pts.min()
        max_p = self.n_pts.max()
        if min_p == max_p:
            pts = "{:,}".format(min_p)
        else:
            pts = '{:,} to {:,}'.format(min_p, max_p)
        #return s.format(len(self), pts, true_dim, pop_mess, indiv_mess, self.total_points)
        return s.format(len(self), pts, true_dim, self.total_points)

    def __len__(self):
        return self.n_pts.size

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, key):
        if (isinstance(key, string_types) or
                (isinstance(key, (tuple, list)) and
                 any(isinstance(x, string_types) for x in key))):
            msg = "Mal_features indexing only subsets rows, but got {!r}"
            raise TypeError(msg.format(key))

        if np.isscalar(key):
            return self.features[key]
        else:
            return type(self)(self.features[key], copy=False, stack=False, pop=self.pop, indiv=self.indiv, true_indiv=self.true_indiv,
                              **{k: v[key] for k, v in iteritems(self.meta)})

    def __add__(self, oth):
        if isinstance(oth, Mal_features):
            meta = {k: np.r_[self.meta[k], oth.meta[k]]
                    for k in self.meta if k in oth.meta}
            oth_features = oth.features
        elif isinstance(oth, list):
            meta = {}
            oth_features = np.empty(len(oth), object)
            oth_features[:] = oth
        else:
            return NotImplemented

        return Mal_features(np.r_[self.features, oth_features],
                        stack=False, copy=True, **meta)

    def __radd__(self, oth):
        if isinstance(oth, list):
            oth_features = np.empty(len(oth), object)
            oth_features[:] = oth
        else:
            return NotImplemented
        return Mal_features(np.r_[oth_features, self.features],
                        stack=False, copy=True)

    ############################################################################
    ## Others

    def bare(self):
        "Make a Mal_features object with no metadata; points to the same features."
        if not self.meta:
            return self
        elif self.stacked:
            return Mal_features(self.stacked_features, self.n_pts, copy=False)
        else:
            return Mal_features(self.features, copy=False)

    def means(self):
        '''
        Returns a kind-of-hacky Mal_features objects with one point per bag, the
        mean of the features. Sets the 'n' property to our n_pts; maintains
        other metadata.
        '''
        out = np.empty((len(self), self.dim), dtype=self.dtype)
        for i, bag in enumerate(self):
            out[i] = bag.mean(axis=0)
        feats = Mal_features(out, np.ones(len(self), dtype=int), **self.meta)
        feats.set_meta('n', self.n_pts)
        return feats


def as_features(X, stack=False, bare=False):
    '''
    Returns a version of X as a :class:`Mal_features` object.

    Parameters
    ----------
    stack : boolean, default False
        Make a stacked version of X. Note that if X is a features object,
        this will stack it in-place, since that's usually what you want.
        (If not, just use the :class:`Mal_features` constructor instead.)

    bare : boolean, default False
        Return a bare version of X (no metadata).

    Returns
    -------
    feats : :class:`Mal_features`
        A version of X. If X is already a :class:`Mal_features` object, the original
        X may be returned, depending on the arguments.
    '''
    if isinstance(X, Mal_features):
        if stack:
            X.make_stacked()
        return X.bare() if bare else X
    return Mal_features(X, stack=stack, bare=bare)
