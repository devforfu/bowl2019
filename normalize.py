from tqdm.auto import tqdm

import utils as U


def normalize(dataset, columns, grouping_key='installation_id'):
    def _normalize(x):
        m, s = x.mean(), x.std()
        return U.savediv(x - m, s + 1e-8)
    groups = dataset.groupby(grouping_key)
    for column in tqdm(columns):
        dataset[column] = groups[column].transform(_normalize)
