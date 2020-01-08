import json
import re
from collections import defaultdict, Counter, OrderedDict
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import utils as U
from dataset import to_accuracy_group

# ------------------------------------
# Features added before transformation
# ------------------------------------

CYCLIC_FEATURES = ('Year', 'Month', 'Week', 'Dayofweek', 'Hour', 'Minute')

def add_feature_combinations(data, pairs):
    for c1, c2 in pairs:
        assert c1 in data.columns, f'Column not found: {c1}'
        assert c2 in data.columns, f'Column not found: {c2}'
        data[f'{c1}_{c2}'] = data[c1].astype(str).str.cat(data[c2].astype(str), '_')
    return data

def add_datetime(data, column, prefix=None, with_time=True):
    data[column] = pd.to_datetime(data[column])
    prefix = U.default(prefix, re.sub('[Dd]ate$', '', column))
    attrs = ('Year', 'Month', 'Week', 'Day', 'Dayofweek')
    if with_time:
        attrs += ('Hour', 'Minute')
    for attr in attrs:
        data[f'{prefix}_{attr}'] = getattr(data[column].dt, attr.lower())
    return data

def add_cyclical(data, prefix, features=CYCLIC_FEATURES, modulo=None):
    modulo = modulo or {}
    for feature in features:
        column = f'{prefix}_{feature}'
        m = modulo.get(feature, 23.0)
        data[f'{column}_sin'] = np.sin(2*np.pi*data[column] / m)
        data[f'{column}_cos'] = np.cos(2*np.pi*data[column] / m)
    return data

# -------------------------------------
# Features extracted from user sessions
# -------------------------------------

class BaseFeatures:
    def __init__(self, meta, **params):
        self.init(meta, **params)
        
class CountersMixin:
    def update_counters(self, cnt, sess, column):
        uniq_counts = Counter(sess[column])
        for k, v in uniq_counts.items():
            if k in cnt:
                cnt[k] += v
                
class CountingFeatures(BaseFeatures, CountersMixin):
    def init(self, meta, **params):
        self.cnt_title_event_code = U.init_dict(meta.title_event_code)
        self.cnt_title = U.init_dict(meta.title)
        self.cnt_event_code = U.init_dict(meta.event_code)
        self.cnt_event_id = U.init_dict(meta.event_id)
        self.cnt_activities = U.init_dict(meta.type)
        self.last_activity = None
        
    def extract(self, session, info, meta):
        features = OrderedDict()
        if info.should_include:
            counters = OrderedDict([
                *self.cnt_title_event_code.items(),
                *self.cnt_title.items(),
                *self.cnt_event_code.items(),
                *self.cnt_event_id.items(),
                *self.cnt_activities.items()])
            features.update(counters)
        self.update_counters(self.cnt_title_event_code, session, 'title_event_code')
        self.update_counters(self.cnt_title, session, 'title')
        self.update_counters(self.cnt_event_code, session, 'event_code')
        self.update_counters(self.cnt_event_id, session, 'event_id')
        if self.last_activity is None or self.last_activity != info.session_type:
            self.cnt_activities[info.session_type] += 1
            self.last_activity = info.session_type
        return U.prefix_keys(features, 'cnt_')
    
class PerformanceFeatures(BaseFeatures):
    def init(self, meta, **params):
        self.acc_accuracy = 0
        self.acc_accuracy_group = 0
        self.acc_correct_attempts = 0
        self.acc_incorrect_attempts = 0
        self.acc_actions = 0
        self.durations = []
        self.accuracy_groups = U.init_dict([0, 1, 2, 3])
        self.last_accuracy_title = U.init_dict([f'acc_{t}' for t in meta.title], -1)
        self.n_rows = 0
    
    def extract(self, session, info, meta):
        features = OrderedDict()
        
        if info.should_include:
            features['acc_attempts_pos'] = self.acc_correct_attempts
            features['acc_attempts_neg'] = self.acc_incorrect_attempts
            self.acc_correct_attempts += info.outcomes.pos
            self.acc_incorrect_attempts += info.outcomes.neg
            
            features['acc_accuracy'] = U.savediv(self.acc_accuracy, self.n_rows)
            accuracy = U.savediv(info.outcomes.pos, info.outcomes.total)
            self.acc_accuracy += accuracy
            
            features.update(self.last_accuracy_title)
            self.last_accuracy_title[f'acc_{info.session_title}'] = accuracy
            
            features['accuracy_group'] = to_accuracy_group(accuracy)
            self.accuracy_groups[features['accuracy_group']] += 1
            
            features['acc_accuracy_group'] = U.savediv(self.acc_accuracy_group, self.n_rows)
            self.acc_accuracy_group += features['accuracy_group']

            features['acc_actions'] = self.acc_actions
            
            features['duration_mean'] = np.mean(self.durations) if self.durations else 0
            self.durations.append(info.duration_seconds)
            
            self.n_rows += 1
            
        self.acc_actions += len(session)
        
        if info.should_include:
            # hack to make sure that target variable is not included into features
            accuracy_group = features.pop('accuracy_group')
            features = U.prefix_keys(features, 'perf_')
            features['accuracy_group'] = accuracy_group
            
        return features
    
class CyclicFeatures(BaseFeatures):
    def init(self, meta, **params):
        self.acc = defaultdict(list)
    
    def extract(self, session, info, meta):
        features = OrderedDict()
        if info.should_include:           
            for dt in CYCLIC_FEATURES:
                for angle in ('sin', 'cos'):
                    key = f'ts_{dt}_{angle}'
                    acc = self.acc
                    features[f'{key}_mean'] = np.mean(acc[key]) if acc[key] else 0
                    features[f'{key}_std'] = np.std(acc[key]) if acc[key] else 0
                    self.acc[key] += session[key].tolist()
        return U.prefix_keys(features, 'cycl_')

class TimestampFeatures(BaseFeatures, CountersMixin):
    def init(self, meta, **params):
        self.cnt_month = U.init_dict([7, 8, 9, 10])
        self.cnt_dayofweek = U.init_dict(range(7))
        self.cnt_dayofmonth = U.init_dict(range(1, 32))
        self.cnt_hour = U.init_dict(range(24))
        self.cnt_minute = U.init_dict(range(60))
        
    def extract(self, session, info, meta):
        features = OrderedDict()
        
        if info.should_include:           
            features.update(U.prefix_keys(self.cnt_month, 'month_'))
            features.update(U.prefix_keys(self.cnt_dayofweek, 'dow_'))
            features.update(U.prefix_keys(self.cnt_dayofmonth, 'dom_'))
            features.update(U.prefix_keys(self.cnt_hour, 'hour_'))
            features.update(U.prefix_keys(self.cnt_minute, 'minute_'))
        
        self.update_counters(self.cnt_month, session, 'ts_Month')
        self.update_counters(self.cnt_dayofweek, session, 'ts_Dayofweek')
        self.update_counters(self.cnt_dayofmonth, session, 'ts_Day')
        self.update_counters(self.cnt_hour, session, 'ts_Hour')
        self.update_counters(self.cnt_minute, session, 'ts_Minute')
        
        return U.prefix_keys(features, 'ts_')

class VarietyFeatures(BaseFeatures, CountersMixin):
    def init(self, meta, **params):
        self.cnt_title_event_code = U.init_dict(meta.title_event_code)
        self.cnt_title = U.init_dict(meta.title)
        self.cnt_event_code = U.init_dict(meta.event_code)
        self.cnt_event_id = U.init_dict(meta.event_id)
        
    def extract(self, session, info, meta):
        features = OrderedDict()
        
        if info.should_include:
            for name in ('title_event_code', 'title', 'event_code', 'event_id'):
                cnt = getattr(self, f'cnt_{name}')
                nonzeros = np.count_nonzero(list(cnt.values()))
                features[name] = nonzeros

        self.update_counters(self.cnt_title_event_code, session, 'title_event_code')
        self.update_counters(self.cnt_title, session, 'title')
        self.update_counters(self.cnt_event_code, session, 'event_code')
        self.update_counters(self.cnt_event_id, session, 'event_id')
        
        return U.prefix_keys(features, 'var_')
    
class EventDataFeatures(BaseFeatures):
    def init(self, meta):
        self.rounds = []
        self.max_round = 0
        self.coord_x = []
        self.coord_y = []
        self.cnt_media = U.init_dict(['unknown', 'animation', 'audio'])
        
    def extract(self, session, info, meta):
        features = OrderedDict()
        
        if info.should_include:
            features['max_round'] = self.max_round
            features['avg_round'] = U.guard_false(np.mean, self.rounds)
            features['std_round'] = U.guard_false(np.std, self.rounds)
            features['avg_coord_x'] = U.guard_false(np.mean, self.coord_x)
            features['avg_coord_y'] = U.guard_false(np.mean, self.coord_y)
            features['std_coord_x'] = U.guard_false(np.std, self.coord_x)
            features['std_coord_y'] = U.guard_false(np.std, self.coord_y)
            features.update(U.prefix_keys(self.cnt_media.copy(), 'media_'))
        
        data = pd.io.json.json_normalize(session.event_data.apply(json.loads))
        self.update_round(data)
        self.update_coord(data)
        self.update_media(data)
        
        return U.prefix_keys(features, 'event_')
    
    def update_round(self, data):
        col = 'round'
        if col not in data:
            return
        rounds = data[col].fillna(0)
        self.max_round = max(self.max_round, rounds.max())
        self.rounds.extend(rounds.tolist())
                     
    def update_coord(self, data):
        col_x = 'coordinates.x'
        col_y = 'coordinates.y'
        if col_x not in data and col_y not in data:
            return
        self.coord_x.extend(data[col_x].fillna(0).astype(int).tolist())
        self.coord_y.extend(data[col_y].fillna(0).astype(int).tolist())
        
    def update_media(self, data):
        col = 'media_type'
        if col not in data:
            return
        cnt = data[col].fillna('unknown').value_counts().to_dict()
        self.cnt_media.update(cnt)

# -------------------------
# Features extraction tools
# -------------------------

class FeaturesExtractor:
    def __init__(self, steps):
        self.steps = steps
        
    def init_steps(self, meta):
        for step in self.steps:
            if hasattr(step, 'init'):
                step.init(meta)
                
    def __call__(self, user, meta, test=False):
        rows = []
        self.init_steps(meta)
        for _, session in user.groupby('game_session', sort=False):
            info = session_info(session, meta, test)
            features = OrderedDict([
                ('installation_id', info.installation_id),
                ('game_session', info.game_session),
                ('session_title', info.session_title)
            ])
            for step in self.steps:
                extracted = step.extract(session, info, meta)
                features.update(extracted)
            if info.should_include:
                rows.append(features)
        return [rows[-1]] if test else rows
    
def session_info(session, meta, test):
    """Computes information about user's session."""
    assert not session.empty, 'Session cannot be empty!'
    session_type = session['type'].iloc[0]
    assessment = session_type == 'Assessment'
    outcomes = attempt_outcomes(session, meta) if assessment else None
    should_include = (
        (assessment and test) or
        (assessment and (len(session) > 1) and outcomes.total > 0))
    duration = session.timestamp.iloc[-1] - session.timestamp.iloc[0]
    return U.named_tuple(
        name='Info', 
        installation_id=session['installation_id'].iloc[0],
        game_session=session['game_session'].iloc[0],
        session_title=session['title'].iloc[0],
        session_type=session_type,
        is_assessment=assessment,
        should_include=should_include,
        outcomes=outcomes,
        duration_seconds=duration.seconds)

def attempt_outcomes(session, meta):
    """Computes how many successful and unsuccessful attempts contains the session."""
    event_code = meta.win_codes.get(session.title.iloc[0], 4100)
    total_attempts = session.query(f'event_code == {event_code}')
    pos = total_attempts.event_data.str.contains('true').sum()
    neg = total_attempts.event_data.str.contains('false').sum()
    summary = dict(pos=pos, neg=neg, total=(pos + neg))
    return U.named_tuple('Trial', **summary)

class InMemoryAlgorithm:
    def __init__(self, extractor, meta, pbar=True, num_workers=cpu_count()):
        self.extractor = extractor
        self.meta = meta
        self.pbar = pbar
        self.num_workers = num_workers
    
    def run(self, dataset, test=False):
        mode = 'test' if test else 'train'
        U.log(f'Running algorithm in {mode} mode.')
        
        def _extract(user):
            return pd.DataFrame(self.extractor(user, self.meta, test))
        
        grouped = dataset.groupby('installation_id', sort=False)
        users = (g for _, g in grouped)
        if self.pbar:
            users = tqdm(users, total= grouped.ngroups)
        datasets = U.parallel(_extract, users, num_workers=self.num_workers)
        dataset = pd.concat(datasets, axis=0)
        dataset = dataset.reset_index(drop=True)
        return dataset
    
# ------------------------
# Post-processing features
# ------------------------

def add_user_wise_features(dataset, meta, pbar=True):
    def transform(group_obj, key, agg): 
        return group_obj[key].transform(agg)
    
    events = [f'cnt_{code}' for code in meta.event_code]
    grouped = dataset.groupby('installation_id')
    dataset['user_session_cnt'] = transform(grouped, 'cnt_Clip', 'count')
    dataset['user_duration_mean'] = transform(grouped, 'perf_duration_mean', 'mean')
    dataset['user_title_nunique'] = transform(grouped, 'session_title', 'nunique')
    dataset['user_events_sum'] = dataset[events].sum(axis=1)
    dataset['user_events_mean'] = transform(grouped, 'user_events_sum', 'mean')