from collections import OrderedDict

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import utils as U
from dataset import load, Subset


class LightGBM:
    def __init__(self, config):
        self.model = lgb.LGBMRegressor(**config.get('model_params', {}))
        self.config = config
        self.features = None
    def fit(self, train_data, valid_data, metric):
        x_trn, y_trn = train_data
        x_val, y_val = valid_data
        params = self.config.get('fit_params', {}).copy()
        params['eval_set'] = [(x_trn, y_trn), (x_val, y_val)]
        params['eval_names'] = ['trn', 'val']
        params['eval_metric'] = metric
        params['X'] = x_trn
        params['y'] = y_trn
        self.features = x_trn.columns.tolist()
        self.model.fit(**params)
    def predict(self, X):
        if self.features is not None:
            assert list(X.columns) == self.features, 'Features do not match!'
        return self.model.predict(X)
    @property
    def has_feature_importance(self):
        return True
    @property
    def feature_importances(self):
        feat_imp = pd.Series(OrderedDict(
            zip(self.features, self.model.feature_importances_)))
        feat_imp.sort_values(inplace=True, ascending=False)
        return feat_imp

    
class XGBoost:
    def __init__(self, config):
        self.model = xgb.XGBRegressor(**config.get('model_params', {}))
        self.config = config
        self.features = None
    def fit(self, train_data, valid_data, metric):
        x_trn, y_trn = train_data
        x_val, y_val = valid_data
        params = self.config.get('fit_params', {}).copy()
        params['eval_set'] = [(x_trn, y_trn), (x_val, y_val)]
        params['eval_metric'] = metric
        params['X'] = x_trn
        params['y'] = y_trn
        self.features = x_trn.columns.tolist()
        self.model.fit(**params)
    def predict(self, X):
        if self.features is not None:
            assert list(X.columns) == self.features, 'Features do not match!'
        return self.model.predict(X)
    @property
    def has_feature_importance(self):
        return True
    @property
    def feature_importances(self):
        feat_imp = pd.Series(OrderedDict(
            zip(self.features, self.model.feature_importances_)))
        feat_imp.sort_values(inplace=True, ascending=False)
        return feat_imp


# ------------
# Configurator
# ------------


MODEL_CONFIG = dict(
    lightgbm=dict(
        model_params=dict(
            n_estimators=5000,
            max_depth=15,
            boosting_type='gbdt',
            metric='rmse',
            objective='regression',
            learning_rate=1e-2,
            subsample=0.75,
            subsample_freq=1,
            feature_fraction=0.9,
            lambda_l1=1,
            lambda_l2=1
        ),
        fit_params=dict(
            early_stopping_rounds=100,
            verbose=100,
            categorical_feature='auto'
        )
    ),
    xgboost=dict(
        model_params=dict(
            n_estimators=5000,
            max_depth=10,
            learning_rate=0.01,
            subsample=1,
            min_child_weight=5,
            gamma=0.25,
            objective='reg:squarederror',
            colsample_bytree=0.8,
            gpu_id=0,
            tree_method='gpu_hist'
        ),
        fit_params=dict(
            early_stopping_rounds=100,
            verbose=50
        )
    )
)

def get_default_config(name):
    assert name in MODEL_CONFIG, f'Config entry is not found: {name}'
    return MODEL_CONFIG[name]

def get_model_class(name):
    if name.startswith('lightgbm'): return LightGBM
    if name.startswith('xgboost'): return XGBoost
    raise ValueError(f'unknown model class: {name}')