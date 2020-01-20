from collections import Counter

import numpy as np
import pandas as pd
import scipy

from metric import qwk


class Rounder:
    def __new__(cls, method='optimal', **params):
        if issubclass(cls, Rounder):
            if method == 'optimal':
                cls = OptimizationRounder
            elif method == 'dist':
                cls = DistributionRounder
            else:
                raise ValueError(f'unknown rounding method: {method}')
        return object.__new__(cls)
    
    def __init__(self, method, **params):
        self.method = method
    
    def fit(self, y_true, y_pred): pass
    
    def predict(self, y_pred):
        return list(map(self.classify, y_pred))
    
    def classify(self, x):
        return (0 if x <= self.bounds[0] else 
                1 if x <= self.bounds[1] else
                2 if x <= self.bounds[2] else
                3)

class DistributionRounder(Rounder):
    def fit(self, y_true, y_pred):
        dist = Counter(y_true)
        for k in dist:
            dist[k] /= len(y_true)
        acc, bounds = 0, []
        for i in range(3):
            acc += dist[i]
            perc =  np.percentile(y_pred, acc * 100)
            bounds.append(perc)
        self.bounds = bounds

        
class OptimizationRounder(Rounder):
    def __init__(self, method, init_coef=(0.5, 1.5, 2.5)):
        super().__init__(method)
        self.init_coef = list(init_coef)
        
    def fit(self, y_true, y_pred):
        def _loss(coef):
            bins = [-np.inf] + list(np.sort(coef)) + [np.inf]
            buckets = pd.cut(y_pred, bins, labels=[0, 1, 2, 3])
            return -qwk(y_true, buckets)
        opt_coef = scipy.optimize.minimize(_loss, self.init_coef, method='nelder-mead')
        self.bounds = opt_coef['x'].tolist()
