from collections import OrderedDict

import pandas as pd


def encode(dataset, columns, encoders=None):
    def make_encoder(mapping):
        return lambda x: mapping.get(x, -1)
    encoders = encoders or {}
    for column in columns:
        if column in encoders:
            dataset[column] = dataset[column].map(make_encoder(encoders[column]))
        else:
            encoded, labels = pd.factorize(dataset[column])
            encoder = OrderedDict([(x, i) for i, x in enumerate(labels)])
            encoders[column] = encoder
            dataset[column] = encoded
    return dataset, encoders