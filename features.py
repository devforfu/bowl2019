import json
import os
import re
from collections import defaultdict, Counter, OrderedDict
from multiprocessing import cpu_count
from operator import itemgetter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import duration
import feedback
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
        self.cnt_worlds = U.init_dict(meta.world)
        self.cnt_types = U.init_dict(meta.type)
        self.cnt_title_worlds = U.init_dict(meta.title_world)
        self.cnt_title_types = U.init_dict(meta.title_type)
        self.cnt_world_types = U.init_dict(meta.world_type)
        self.last_activity = None
        
    def extract(self, session, info, meta):
        
        def most_freq(cnt): return max(cnt.items(), key=itemgetter(1))[0]
        def least_freq(cnt): return min(cnt.items(), key=itemgetter(1))[0]
        
        features = OrderedDict()
        if info.should_include:
            counters = OrderedDict([
                *self.cnt_title_event_code.items(),
                *self.cnt_title.items(),
                *self.cnt_event_code.items(),
                *self.cnt_event_id.items(),
                *self.cnt_activities.items(),
                *self.cnt_worlds.items(),
                *self.cnt_types.items(),
                *self.cnt_title_worlds.items(),
                *self.cnt_title_types.items(),
                *self.cnt_world_types.items()
            ])
            features.update(counters)
            features['most_freq_title'] = most_freq(self.cnt_title)
            features['least_freq_title'] = least_freq(self.cnt_title)
            features['most_freq_world'] = most_freq(self.cnt_worlds)
            features['least_freq_world'] = least_freq(self.cnt_worlds)
            features['most_freq_type'] = most_freq(self.cnt_types)
            features['least_freq_type'] = least_freq(self.cnt_types)
            features['most_freq_title_world'] = most_freq(self.cnt_title_worlds)
            features['least_freq_title_world'] = least_freq(self.cnt_title_worlds)
            features['most_freq_title_type'] = most_freq(self.cnt_title_types)
            features['least_freq_title_type'] = least_freq(self.cnt_title_types)
            features['most_freq_world_type'] = most_freq(self.cnt_world_types)
            features['least_freq_world_type'] = least_freq(self.cnt_world_types)
            
        self.update_counters(self.cnt_title_event_code, session, 'title_event_code')
        self.update_counters(self.cnt_title, session, 'title')
        self.update_counters(self.cnt_event_code, session, 'event_code')
        self.update_counters(self.cnt_event_id, session, 'event_id')
        self.update_counters(self.cnt_worlds, session, 'world')
        self.update_counters(self.cnt_types, session, 'type')
        self.update_counters(self.cnt_title_worlds, session, 'title_world')
        self.update_counters(self.cnt_title_types, session, 'title_type')
        self.update_counters(self.cnt_world_types, session, 'world_type')
        
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
        self.clip_durations = []
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
            
            event_count = session['event_count'].iloc[-1]
                
            features['duration_mean'] = np.mean(self.durations) if self.durations else 0
            features['total_duration'] = sum(self.durations)
            self.durations.append(info.duration_seconds)
        
            features['clip_duration_mean'] = U.guard_false(np.mean, self.clip_durations)
            features['clip_total_duration'] = sum(self.clip_durations)
            self.clip_durations.append(duration.CLIPS.get(info.session_title, 0))
            
            features['clip_ratio'] = U.savediv(
                features['clip_total_duration'], features['total_duration'])
            
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
    
class TimestampFeatures2(BaseFeatures):
    def init(self, meta, **params):
        self.months = []
        self.dows = []
        self.doms = []
        self.hours = []
    
    def extract(self, session, info, meta):
        features = OrderedDict()
        
        if info.should_include:
            for attr in ('months', 'dows', 'doms', 'hours'):
                for func in (min, max, np.mean, np.std):
                    arr = getattr(self, attr)
                    features[f'{func.__name__}_{attr}'] = U.guard_false(func, arr)
                    
        self.months.extend(session['ts_Month'].tolist())
        self.dows.extend(session['ts_Dayofweek'].tolist())
        self.doms.extend(session['ts_Day'].tolist())
        self.hours.extend(session['ts_Hour'].tolist())

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
    def init(self, meta, **params):
        self.rounds = []
        self.max_round = 0
        self.coord_x = []
        self.coord_y = []
        self.cnt_media = U.init_dict(['unknown', 'animation', 'audio'])
        self.cnt_source = U.init_dict([
            '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', 
            '7.0', '8.0', '9.0', '10.0', '11.0', '12.0',
            'resources', 'scale', 'left', 'middle', 'right', 
            'Lightest', 'Heavy', 'Heaviest', 'N/A'
        ])
        self.cnt_level = U.init_dict(range(6))
        self.cnt_size = U.init_dict(range(7))
        self.cnt_weight = U.init_dict(range(13))
        
    def extract(self, session, info, meta):
        features = OrderedDict()
        
        if info.should_include:
            features['var_media'] = sum([0 if not v else 1 for v in self.cnt_media.values()])
            features['var_source'] = sum([0 if not v else 1 for v in self.cnt_source.values()])
            features['var_level'] = sum([0 if not v else 1 for v in self.cnt_level.values()])
            features['var_weight'] = sum([0 if not v else 1 for v in self.cnt_weight.values()])
            features['var_size'] = sum([0 if not v else 1 for v in self.cnt_size.values()])
            features.update(U.prefix_keys(self.cnt_media.copy(), 'media_'))
            features.update(U.prefix_keys(self.cnt_source.copy(), 'source_'))
            features.update(U.prefix_keys(self.cnt_level.copy(), 'level_'))
            features.update(U.prefix_keys(self.cnt_size.copy(), 'size_'))
            features.update(U.prefix_keys(self.cnt_weight.copy(), 'weight_'))
        
        data = pd.io.json.json_normalize(session.event_data.apply(json.loads))
        self.update_round(data)
        self.update_coord(data)
        self.update_media(data)
        self.update_source(data)
        self.update_levels(data)
        self.update_sizes(data)
        self.update_weights(data)
        
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
        self.update_counters(cnt, self.cnt_media)
        
    def update_source(self, data):
        col = 'source'
        if col not in data:
            return
        cnt = data[col].fillna('N/A').value_counts().to_dict()
        self.update_counters(cnt, self.cnt_source)
        
    def update_levels(self, data):
        col = 'level'
        if col not in data:
            return
        levels = data[col].fillna(0)
        def map_to_bin(x):
            return (0 if x <= 3 else 
                    1 if x <= 5 else 
                    2 if x <= 8 else
                    3 if x <= 13 else
                    4 if x <= 21 else
                    5)
        buckets = Counter([map_to_bin(level) for level in levels])
        self.update_counters(buckets, self.cnt_level)
        
    def update_sizes(self, data):
        col = 'size'
        if col not in data:
            return
        sizes = data[col].fillna(0).astype(int).value_counts().to_dict()
        self.update_counters(sizes, self.cnt_size)
    
    def update_weights(self, data):
        col = 'weights'
        if col not in data:
            return
        weights = data[col].fillna(0).astype(int).value_counts().to_dict()
        self.update_counters(weights, self.cnt_weight)
        
    def update_counters(self, src, dst):
        for k, v in src.items():
            if k in dst:
                dst[k] += v

class FeedbackFeatures(BaseFeatures):
    def init(self, meta, **params):
        self.pos_feedback = 0
        self.neg_feedback = 0
        self.other_feedback = 0
        self.cnt_char_feedback = U.init_dict(['dot', 'buddy', 'mom', 'cleo'])
    
    def extract(self, session, info, meta):
        features = OrderedDict()
        
        if info.should_include:
            total_feedback = self.pos_feedback + self.neg_feedback + self.other_feedback
            features['pos_feedback'] = self.pos_feedback
            features['neg_feedback'] = self.neg_feedback
            features['other_feedback'] = self.other_feedback
            features['pos_neg_ratio'] = U.savediv(self.pos_feedback, self.neg_feedback, 9999)
            features['pos_all_ratio'] = U.savediv(self.pos_feedback, total_feedback, 9999)
            features['total_feedback'] = total_feedback
            features.update(U.prefix_keys(self.cnt_char_feedback, 'char_'))
        
        data = pd.io.json.json_normalize(session.event_data.apply(json.loads))        
        self.update_feedback(data)
        return U.prefix_keys(features, 'fb_')
    
    def update_feedback(self, data):
        if 'identifier' not in data:
            return
        
        def transform_identifier(x):
            if U.starts_with_any(x, ['Dot', 'Buddy', 'Mom', 'Cleo']):
                parts = x.split(',')
                if len(parts) > 1:
                    prefix = os.path.commonprefix(parts)
                    n = len(prefix)
                    trimmed = [U.camel_to_snake(part[n:]) for part in parts]
                    string = '_'.join(trimmed)
                else:
                    prefix = ''
                    string = U.camel_to_snake(x.replace('_', ''))
                result = f'{prefix}{string}'
                return result.lower()
            return x
        
        def transform_feedback(x):
            return ('positive' if x in feedback.POSITIVE else
                    'negative' if x in feedback.NEGATIVE else
                    'other')
        
        characters = 'dot', 'buddy', 'mom', 'cleo'
        normalized = data['identifier'].fillna('unknown').map(transform_identifier)
        character_identifiers = normalized.map(lambda x: U.starts_with_any(x, characters))
        edi = pd.DataFrame({'identifier': normalized[character_identifiers]})
        edi['character'] = edi['identifier'].map(lambda x: x.split('_')[0])
        edi['feedback'] = edi['identifier'].map(transform_feedback)
        
        self.pos_feedback += len(edi.query('feedback == "positive"'))
        self.neg_feedback += len(edi.query('feedback == "negative"'))
        self.other_feedback += len(edi.query('feedback == "other"'))
        
        for k, v in edi['character'].value_counts().to_dict().items():
            if k in self.cnt_char_feedback:
                self.cnt_char_feedback[k] += v

class ZFeatures(BaseFeatures):
    def init(self, meta, **params):
        self.ref_ts = meta['ref_ts']
    
    def extract(self, session, info, meta):
        features = OrderedDict()
        
        if info.should_include:
            pass
            
        breakpoint()
        delta = session['timestamp'].diff().seconds.fillna(0)
        return features
                
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