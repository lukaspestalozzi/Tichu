import abc
from collections import abc as collectionsabc
from decorator import contextmanager
from time import time
from datetime import timedelta
import os
import errno

# constants
# all primes smaller than 200
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199)

# TODO unicode characters:


# functions

def raiser(ex):
    """
    :param ex: the exception to be raised
    :raise raises the given Exception
    """
    raise ex


def flatten(iterable):
    """
    >>> list(flatten(1))
    [1]
    >>> list(flatten(None))
    [None]
    >>> list(flatten("abc"))
    ['abc']
    >>> list(flatten(['abc', 12, ['a', ['c', ['i']], ('xyz','tu')]]))
    ['abc', 12, 'a', 'c', 'i', 'xyz', 'tu']
    >>> list(flatten([1]))
    [1]
    >>> list(flatten([1, "abc", [["de", 3]], "fg", 9]))
    [1, 'abc', 'de', 3, 'fg', 9]
    >>> list(flatten([1, 2, [3, [4, 5, 6], 7], [8, 9], (10, 11, {12}), 13, 14, [[[[[[15, 16]]], 17]], 18]]))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    >>> list(flatten([[[[[1]]]]]))
    [1]


    Flattens the given iterable. Does treat strings and bytes not as iterables.
    if iterable is not an iterable, yields the element

    :param iterable: any iterable
    :return: a generator yielding all non iterable elements in the iterable.
    """
    # handle strings
    if isinstance(iterable, (str, bytes)):
        yield iterable
        return

    try:
        # try to iterate over the input
        for item in iterable:
            yield from flatten(item)
    except TypeError:
        # 'iterable' is not iterable and therefore a 'terminal' item
        yield iterable


def indent(n, s="-"):
    """
    :param n: number >= 0
    :param s: string
    :return: string containing a copy of n times the string s
    """
    return s.join("" for _ in range(n))


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def check_true(expr, ex=AssertionError, msg="expr was not True"):
    """
    Raises an Exception when expr evaluates to False.

    :param expr: The expression
    :param ex: (optional, default AssertionError) exception to raise.
    :param msg: (optional) The message for the exception
    :return: True otherwise
    """
    if not bool(expr):
        raise ex(msg)
    else:
        return True


def check_all_isinstance(iterable, clazzes):
    """
    :return: True
    :raises: TypeError when for one of the items in iterable isinstance(item, clazzes) is False
    """
    for item in iterable:
        check_isinstance(item, clazzes)
    return True


def check_isinstance(item, clazzes, msg=None):
    """
    >>> check_isinstance(1, int)
    True
    >>> check_isinstance(1.1, (int, float))
    True
    >>> check_isinstance(1.1, [int, float])
    Traceback (most recent call last):
        ...
    TypeError: isinstance() arg 2 must be a type or tuple of types
    >>> check_isinstance(1.1, int, float)  # note that msg=float in this case
    Traceback (most recent call last):
        ...
    TypeError: <class 'float'>
    >>> check_isinstance(1.1, 1)
    Traceback (most recent call last):
        ...
    TypeError: isinstance() arg 2 must be a type or tuple of types
    >>> check_isinstance(1, float)
    Traceback (most recent call last):
        ...
    TypeError: item must be instance of at least one of: [<class 'float'>], but was 1

    :param item:
    :param clazzes: must be a type or tuple of types
    :param msg: (optional) The message for the exception
    :return: True
    :raises: TypeError when isinstance(item, clazzes) is False
    """
    if not isinstance(item, clazzes):
        import inspect
        message = (msg if msg is not None
                   else
                   "Item must be instance of at least one of: {}, but was: {} (from {})"
                        .format(', '.join("{} from {}".format(c, inspect.getfile(c)) for c in clazzes),
                                item.__class__,
                                inspect.getfile(item.__class__)))
        raise TypeError(message)
    else:
        return True


def check_param(expr, param="[No Parameter given]", msg=None):
    """
    :param expr:
    :param param: optional, the parameter to be checked, is only used in the error message
    :param msg: optional, In case the expr is False, show this message instead of the default one
    :return: True
    :raises: ValueError when the expr is Falsy (bool(expr) is False)
    """
    if not bool(expr):
        message = msg if msg is not None else "The Expression must be true, but was False. (param:{})".format(param)
        raise ValueError(message)
    else:
        return True


def crange(start, stop, modulo):
    """
    Circular range generator.
    :param start: int, start integer (included)
    :param stop: stop integer (excluded), If start == stop, then a whole circle is returned. ie. crange(0, 0, 4) -> [0, 1, 2, 3]
    :param modulo: the modulo of the range
    >>> list(crange(0, 5, 10))
    [0, 1, 2, 3, 4]
    >>> list(crange(7, 3, 10))
    [7, 8, 9, 0, 1, 2]
    >>> list(crange(0, 10, 4))
    [0, 1]
    >>> list(crange(13, 10, 4))
    [1]
    >>> list(crange(0, 0, 4))
    [0, 1, 2, 3]
    >>> list(crange(6, 6, 4))
    [2, 3, 0, 1]
    >>> list(crange(1, 2, 4))
    [1]
    >>> list(crange(1, 4, 4))
    [1, 2, 3]
    >>> list(crange(3, 2, 4))
    [3, 0, 1]
    >>> list(crange(3, 0, 4))
    [3]
    >>> list(crange(2, 3, 4))
    [2]
    >>> list(crange(2, 1, 4))
    [2, 3, 0]
    """
    startmod = start % modulo
    stopmod = stop % modulo
    yield startmod
    k = (startmod + 1) % modulo
    while k != stopmod:
        yield k
        k = (k + 1) % modulo


def time_since(since: int)->timedelta:
    """
    
    :param since: time since epoche (in seconds)
    :return: a datetime.timedelta element: timedelta(seconds=time() - since)
    """
    return timedelta(seconds=time() - since)

@contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass


@contextmanager
def error_logged(logger):
    try:
        yield
    except Exception as ex:
        logger.exception(ex)
        raise


# Classes
class TypedFrozenCollection(collectionsabc.Collection, metaclass=abc.ABCMeta):

    def __init__(self, sequence=()):
        super().__init__()
        with ignored(StopIteration):
            dtype = type(next(iter(sequence)))
            if not all((isinstance(e, dtype) for e in sequence)):
                raise TypeError("All elements must be of the same type, but were not: {}".format(
                        ','.join(['{t}({se})'.format(se=repr(e), t=type(e)) for e in sequence]))
                )


class TypedFrozenSet(frozenset, TypedFrozenCollection):
    """frozenset containing only elements of one type (or subclasses thereof).

    >>> TypedFrozenSet((1, 3, 4))
    TypedFrozenSet({1, 3, 4})
    
    >>> TypedFrozenSet((1, 3, 4, 's'))
    Traceback (most recent call last):
    ...
    TypeError: All elements must be of the same type, but were not: <class 'int'>(1),<class 'int'>(3),<class 'int'>(4),<class 'str'>('s')
    
    >>> TypedFrozenSet(('b', 1, 3, 4, 's'))
    Traceback (most recent call last):
    ...
    TypeError: All elements must be of the same type, but were not: <class 'str'>('b'),<class 'int'>(1),<class 'int'>(3),<class 'int'>(4),<class 'str'>('s')
    
    >>> TypedFrozenSet((1, 3, 4)) | TypedFrozenSet((5, 6, 7))
    frozenset({1, 3, 4, 5, 6, 7})
    
    >>> TypedFrozenSet((1, 3, 4)).union( TypedFrozenSet((5, 6, 7)))
    frozenset({1, 3, 4, 5, 6, 7})
    """
    __slots__ = ()


class TypedTuple(tuple, TypedFrozenCollection):
    """tuple containing only elements of one type (or subclasses thereof).

        >>> TypedTuple((1, 3, 4))
        TypedTuple<class 'int'>(1, 3, 4)

        >>> TypedTuple((1, 3, 4, 's'))
        Traceback (most recent call last):
        ...
        TypeError: All elements must be of the same type, but were not: <class 'int'>(1),<class 'int'>(3),<class 'int'>(4),<class 'str'>('s')

        >>> TypedTuple(('b', 1, 3, 4, 's'))
        Traceback (most recent call last):
        ...
        TypeError: All elements must be of the same type, but were not: <class 'str'>('b'),<class 'int'>(1),<class 'int'>(3),<class 'int'>(4),<class 'str'>('s')

        >>> TypedTuple((1, 3, 4)) + TypedTuple((5, 6, 7))
        (1, 3, 4, 5, 6, 7)
        
        """
    __slots__ = ()

    def __repr__(self):
        if len(self) > 0:
            return '{name}{dtype}({elems})'.format(name=self.__class__.__name__,
                                                   dtype=repr(type(self[0])),
                                                   elems=', '.join(repr(e) for e in self))
        else:
            return '{name}()'.format(name=self.__class__.__name__)
