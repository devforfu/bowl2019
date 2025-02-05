import os
import re
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
    

def prefix_keys(od, prefix):
    return OrderedDict([(f'{prefix}{k}', v) for k, v in od.items()])


def now(utc=True, fmt='%a_%d_%h_%Y__%H_%M'):
    now_fn = datetime.utcnow if utc else datetime.now
    now_ts = now_fn()
    return now_ts.strftime(fmt)


def filter_nan(x, fallback=0):
    return fallback if np.isnan(x) else x


def guard_false(func, x, fallback=0):
    return func(x) if x else fallback


def set_nested(d, key, value, sep='.'):
    if sep not in key:
        d[key] = value
    else:
        *keys, last = key.split(sep)
        for part in keys:
            if part not in d:
                raise KeyError(f'cannot resolve key: {key}')
            d = d[part]
        d[last] = value


def get_nested(d, key, sep='.'):
    if sep not in key:
        return d[key]
    else:
        value = d
        for part in key.split(sep):
            if part not in d:
                raise KeyError(f'cannot resolve key: {key}')
            value = value[part]
        return value
    
    
def dict_format(d, joined=', '):
    formatted = []
    for key, value in d.items():
        if isinstance(value, (float, np.float16, np.float32, np.float64)):
            fmt = f'{value:.4f}'
        elif isinstance(value, (int, np.int16, np.int32, np.int64)):
            fmt = f'{value:d}'
        else:
            fmt = str(value)
        formatted.append(f'{key}={fmt}')
    if joined is not None:
        formatted = joined.join(formatted)
    return formatted


def match_cols(cols, regex):
    return [col for col in cols if re.match(regex, col)]


def starts_with(cols, prefix): 
    return match_cols(cols, f'^{prefix}')


def agg_dict(d, func):
    return func(list(chain(*[[value]*count for count, value in d.items()])))


def show_all(dataframe):
    import pandas as pd
    from IPython.display import display
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        display(dataframe)

        
def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def trim_common_prefix(strings):
    prefix = os.path.commonprefix(strings)
    n = len(prefix)
    trimmed = [string[n:] for string in strings]
    return prefix, trimmed


def starts_with_any(string, prefixes):
    for prefix in prefixes:
        if string.startswith(prefix):
            return True
    return False