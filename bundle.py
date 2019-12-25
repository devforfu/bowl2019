import shutil

import joblib

import utils as U

if U.on_kaggle():
    root = '/kaggle/input/data-bowl-2019-external-data'
else:
    import os
    root = '/tmp/bowl2019'
    os.makedirs(root, exist_ok=True)

def save_meta(meta, key): 
    return save(meta._asdict(), key)

def load_meta(key):
    meta = load(key)
    return U.named_tuple('Meta', **meta)

def save(obj, key):
    path = f'{root}/{key}.joblib'
    joblib.dump(obj, path)
    return path

def load(key):
    return joblib.load(f'{root}/{key}.joblib')

def meta():
    return load_meta('meta')

def features():
    return load('features')

def models(model='lightgbm', version='003'):
    return load(f'models_{model}_{version}')

def bounds():
    return load('bounds')

def encoders():
    return load('encoders')

def package(folder):
    U.log('Packaging training results into dataset.')
    for filename in os.listdir(root):
        if filename.endswith('.joblib'):
            src = os.path.join(root, filename)
            dst = os.path.join(folder, filename)
            U.log(f'{src} --> {dst}')
            shutil.copy(src, dst) 
    U.log('Packaging helper scripts into dataset.')
    scripts_dir = os.path.dirname(__file__)
    for filename in os.listdir(scripts_dir):
        if filename.endswith('.py'):
            src = os.path.join(scripts_dir, filename)
            dst = os.path.join(folder, filename)
            U.log(f'{src} --> {dst}')
            shutil.copy(src, dst) 
    return folder
