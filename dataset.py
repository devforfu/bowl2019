from enum import Enum
import pandas as pd
from basedir import TEST, TRAIN, TRAIN_LABELS, TRAIN_SPECS, SAMPLE


class Subset(Enum):
    Train = 1
    Test = 2
    Sample = 3
    

def load(subset: Subset, chunksize: int=None):
    if subset == Subset.Train:
        trn_data = pd.read_csv(TRAIN, chunksize=chunksize)
        trn_target = pd.read_csv(TRAIN_LABELS, chunksize=chunksize)
        trn_specs = pd.read_csv(TRAIN_SPECS, chunksize=chunksize)
        datasets = trn_data, trn_target, trn_specs
    elif subset == Subset.Test:
        datasets = [pd.read_csv(TEST, chunksize=chunksize)]
    elif subset == Subset.Sample:
        datasets = [pd.read_csv(SAMPLE, chunksize=chunksize)]
    else:
        raise ValueError(subset)
    if chunksize is not None:
        print(f'Reading data in chunk mode with chunk size: {chunksize}')
    else:
        for dataset in datasets:
            print(dataset.shape, end=' ')
    return datasets


def load_sample(subset: Subset, size: int=10000):
    datasets = load(subset, chunksize=size)
    chunks = [next(dataset) for dataset in datasets]
    return chunks


def n_unique(dataset):
    for column in dataset.columns:
        print(f'{column}: {dataset[column].nunique()}')


def missing_info(dataset):
    total = dataset.isnull().count()
    pct = (dataset.isnull().sum()/total)*100
    table = pd.concat([total, pct], axis=1, keys=['Total', 'Percent'])
    table['Types'] = dataset.dtypes.values
    return table.T


def existing_info(dataset):
    total = dataset.isnull().count() - dataset.isnull().sum()
    pct = 100*(1 - dataset.isnull().sum()/dataset.isnull().count())
    table = pd.concat([total, pct], axis=1, keys=['Total', 'Percent'])
    table = table.sort_values(['Total'], ascending=False)
    return table.T


def to_accuracy_group(accuracy):
    return (0 if accuracy == 0 else 
            3 if accuracy == 1 else 
            2 if accuracy == 0.5 else
            1)