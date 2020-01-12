from collections import Counter

import numpy as np
import pandas as pd
import scipy
from numba import jit 
from sklearn.metrics import cohen_kappa_score


@jit
def qwk(a1, a2, max_rat=3):
    assert len(a1) == len(a2)
    
    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


class RegressionCappa:
    def __init__(self, bounds):
        self.bounds = bounds
    def __call__(self, y_true, y_pred):
        y_rounded = round_regressor_predictions(y_pred, self.bounds)
        y_true = np.asarray(y_true, dtype=int)
        y_rounded = np.asarray(y_rounded, dtype=int)
        metric = qwk(y_true, y_rounded)
        return metric
    def lightgbm(self, y_true, y_pred):
        return 'cappa', self(y_true, y_pred), True
    
    
def round_regressor_predictions(preds, coefs):
    x = preds.copy()
    for i, (lo, hi) in enumerate(zip(coefs[:-1], coefs[1:])):
        x[(x > lo) & (x <= hi)] = i
    return x


def optimize_rounding_bounds(X, y):
    def _loss(coef):
        buckets = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])
        return -qwk(y, buckets)
    
    init_coef = [0.5, 1.5, 2.5]
    opt_coef = scipy.optimize.minimize(_loss, init_coef, method='nelder-mead')
    optimized = opt_coef['x']
    return [-np.inf] + optimized.tolist() + [np.inf]


class PredictionsRoudner:
    def __init__(self, train_target):
        dist = Counter(train_target)
        size = len(train_target)
        for k in dist:
            dist[k] /= size
        self.dist = dist
    def __call__(self, y_pred):
        acc, bounds = 0, []
        for i in range(3):
            acc += self.dist[i]
            perc = np.percentile(y_pred, acc*100)
            bounds.append(perc)
        rounded = pd.cut(y_pred, [-np.inf] + sorted(bounds) + [np.inf], labels=[0, 1, 2, 3])
        return rounded.tolist(), bounds
    
    
def make_cappa_metric(train_target):
    rounder = PredictionsRoudner(train_target)
    def cappa(y_true, y_pred):
        nonlocal rounder
        y_pred_rounded, _ = rounder(y_pred)
        score = cohen_kappa_score(y_true, y_pred_rounded, weights='quadratic')
        return score
    return cappa