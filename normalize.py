from tqdm.auto import tqdm

import utils as U


def normalize(dataset, columns, grouping_key='installation_id', method='min-max'):
    assert method in ('min-max', 'mean-std'), f'Wrong method: {method}'
    def _normalize(x):
        if method == 'min-max':
            return U.savediv(x - x.min(), (x.max() - x.min()))
        elif method == 'mean-std':
            m, s = x.mean(), x.std()
            return U.savediv(x - m, s + 1e-8)
    groups = dataset.groupby(grouping_key)
    for column in tqdm(columns):
        dataset[column] = groups[column].transform(_normalize)
