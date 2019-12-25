import os
from collections import namedtuple, OrderedDict
from datetime import datetime
from itertools import chain
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm


__all__ = ['parallel', 'parallel_chunks', 'chunks', 
           'unique', 'count', 'order', 'named_tuple',
           'as_list', 'savediv', 'init_dict']


def parallel(func, rows, num_workers=cpu_count()):
    with Parallel(num_workers) as p:
        results = p(delayed(func)(row) for row in rows)
    return results

def parallel_chain(func, rows, num_workers=cpu_count()):
    return list(chain(*parallel(func, rows, num_workers)))


def chunks(arr, chunk_size=4):
    n = len(arr)
    n_chunks = int(np.ceil(n / chunk_size))
    for i in range(n_chunks):
        yield arr[i*chunk_size:(i+1)*chunk_size]

        
def unique(dataframes, column):
    """Returns list of unique items from a column among all provided datasets."""
    return list(set(chain(*[df[column].unique().tolist() for df in dataframes])))


def count(dataframe, column):
    """Counts values of column in dataframe."""
    return dataframe[column].value_counts()


def order(dataframe, column):
    """Returns list of unique of values of a column, in the decreasing order
    of their frequency.
    """
    return count(dataframe, column).index.tolist()


def named_tuple(name, **params):
    """Converts a dictionary into a named tuple."""
    return namedtuple(name, params.keys())(**params)


def as_list(seq): 
    return list(sorted(seq))


def savediv(a, b, fallback=0): 
    return a/b if b != 0 else fallback


def init_dict(keys, init_value=0):
    return OrderedDict([(k, init_value) for k in keys])


def default(value, fallback=0):
    return value if value is not None else fallback


def combine(func, *funcs):
    from functools import reduce
    return lambda arg: reduce(lambda x, f: f(x), [func] + list(funcs), arg)


def on_kaggle():
    return 'KAGGLE_URL_BASE' in os.environ


def log(*args, **kwargs):
    if on_kaggle():
        message = ' '.join([str(x) for x in args])
        log_message = datetime.now().strftime(f'[Kernel][%Y-%m-%d %H:%M:%S] {message}') 
        os.system(f'echo \"{log_message}\"')
    print(*args, **kwargs)