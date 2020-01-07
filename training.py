from collections import OrderedDict

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
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
    def feature_importances(self):
        feat_imp = pd.Series(OrderedDict(
            zip(self.features, self.model.feature_importances_)))
        feat_imp.sort_values(inplace=True, ascending=False)
        return feat_imp

    
# --------
# Training
# --------


def train(dataset, features, reg_metric, algo='lightgbm', n_folds=5, config=None):
    models = []
    folds = GroupKFold(n_splits=n_folds)
    groups = dataset['installation_id']
    X = dataset[features].copy()
    y = dataset['accuracy_group']
    oof = np.zeros(X.shape[0], dtype=np.float32)
    cv = OrderedDict()
    model_cls = get_model_class(algo)
    metric = getattr(reg_metric, algo)
    feat_imp = np.zeros(len(features), dtype=np.float32)
    
    for i, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups), 1):
        U.log(f'Running k-fold {i} of {n_folds}')
        x_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model = model_cls(config or get_default_config(algo))
        model.fit(train_data=(x_trn, y_trn), 
                  valid_data=(x_val, y_val), 
                  metric=metric)
        oof[val_idx] = model.predict(x_val)
        cv[f'cv_cappa_{i}'] = np.mean(reg_metric(y_val, oof[val_idx]))
        models.append(model)
        feat_imp += model.feature_importances.values
    
    feat_imp /= n_folds
    feat_imp = pd.Series(OrderedDict(zip(features, feat_imp)))
    return U.named_tuple('Result', models=models, cv=cv, oof=oof, fi=feat_imp)


# -----------
# Predictions
# -----------


def inference(data, features, bounds, model='lightgbm', version='003', chunk_size=128):
    import bundle
    from metric import round_regressor_predictions
    U.log(f'Running inference on dataset of shape: {len(features)}')
    indexes = np.arange(len(data))
    U.log(f'Loading external models: {model} v{version}.')
    models = bundle.models(model=model, version=version)
    preds = {i: [] for i, _ in enumerate(models)}
    U.log('Running models on test data...')
    for chunk in U.chunks(indexes, chunk_size):
        x_test = data[features].iloc[chunk]
        for i, model in enumerate(models):
            pred = model.predict(x_test).tolist()
            preds[i].extend(pred)
    U.log('Averaging ensemble predictions.')
    avg_preds = pd.DataFrame(preds).mean(axis=1).values
    U.log('Rounding predictions using optimal bounds.')
    y_hat = round_regressor_predictions(avg_preds, bounds)
    return y_hat

def submit(predicted, filename='submission.csv'):
    U.log('Converting predictions into submission file.')
    if U.on_kaggle():
        U.log('Running on Kaggle.')
        sample = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    else:
        U.log('Running locally.')
        [sample] = load(Subset.Sample)
    sample['accuracy_group'] = predicted.astype(int)
    sample.to_csv(filename, index=False)
    return filename


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
    )
)

def get_default_config(name):
    assert name in MODEL_CONFIG, f'Config entry is not found: {name}'
    return MODEL_CONFIG[name]

def get_model_class(name):
    if name == 'lightgbm': return LightGBM
    raise ValueError(f'unknown model class: {name}')