from collections import OrderedDict

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import GroupKFold

import bundle
import utils as U
from dataset import load, Subset
from models import get_default_config, get_model_class


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


class EnsembleTrainer:
    def __init__(self, eval_metric='cappa', cv_metrics=None, algo='lightgbm'):
        self.eval_metric = eval_metric
        self.cv_metrics = cv_metrics or {}
        self.algo = algo
        
    def train(self, dataset, features, fold, 
              target='accuracy_group', grouping='installation_id', 
              config=None):
        
        assert target not in features
        assert grouping in dataset or grouping is None 
        
        groups = dataset[grouping]
        X = dataset[features]
        y = dataset[target]
        model_cls = get_model_class(self.algo)
        n_folds = fold.get_n_splits()
        
        models = []
        feat_imp = np.zeros(len(features), dtype=np.float32)
        oof = np.zeros(X.shape[0], dtype=np.float32)
        cv = OrderedDict()
        
        for i, (trn_idx, val_idx) in enumerate(fold.split(X, y, groups), 1):
            U.log(f'Running k-fold {i} of {n_folds}')
            x_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
            x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            model = model_cls(config or get_default_config(self.algo))
            model.fit(train_data=(x_trn, y_trn), 
                      valid_data=(x_val, y_val), 
                      metric=self.eval_metric)
            oof[val_idx] = model.predict(x_val)
            for name, metric in self.cv_metrics.items():
                cv[f'cv_{name}_{i}'] = metric(y_val, oof[val_idx])
            models.append(model)
            if model.has_feature_importance:
                feat_imp += model.feature_importances.values
                
        if cv:
            U.log('Fold evaluation results:')
            U.log(U.dict_format(cv))

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
