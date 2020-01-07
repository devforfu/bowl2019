import numpy as np
from tqdm.auto import tqdm
import utils as U

class FeatureSelection:
    def __init__(self, rules, ignore_cols=None):
        self.rules = rules
        self.ignore_cols = ignore_cols or []
        self.selected = None
    def select(self, dataset):
        relevant = {}
        total = len(dataset.columns)
        if self.ignore_cols:
            U.log(f'Excluding from consideration: {self.ignore_cols}')
            dataset = dataset.drop(columns=self.ignore_cols)
        for name, rule in self.rules:
            U.log(f'Applying feature selection rule: {name}')
            features = rule(dataset)
            relevant[name] = set(features)
            U.log(f'Selected features: {len(features)} of {total}')
        U.log(f'Keeping only features, selected by every rule.')
        features = set.intersection(*relevant.values())
        U.log(f'Final number of features changed from {total} to {len(features)}')
        return sorted(list(features))
        
def non_zero_rows_and_cols(dataset):
    def nonzero(x): return not np.allclose(x, 0)
    nonzero_rows = dataset.sum(axis=1).map(nonzero)
    nonzero_cols = dataset.sum(axis=0).map(nonzero)
    features = dataset.loc[nonzero_rows, nonzero_cols].columns.tolist()
    return features

def non_correlated_cols(dataset, threshold=0.995):
    from itertools import combinations
    correlated = set()
    columns = dataset.columns
    pairs = combinations(columns, 2)
    n_pairs = len(columns)*(len(columns) - 1)//2
    for a, b in tqdm(pairs, total=n_pairs):
        if a in correlated: continue
        if b in correlated: continue
        c = np.corrcoef(dataset[a], dataset[b])[0][1]
        if c > threshold:
            correlated.add(b)
    return [c for c in columns if c not in correlated]